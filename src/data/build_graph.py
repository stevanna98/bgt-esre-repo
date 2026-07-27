"""Convert raw fMRI arrays for one subject into a PyG Data object.

Always produces:
    edge_index  (2, 2E)   — bidirectional edges from thresholded FC
    lap_pe      (N, k)    — Laplacian positional encodings
    y           (1,)      — integer class label
    bold        (N, T)    — BOLD or FC-row node features

For economy-aware models it additionally produces:
    phi         (2E, 2)   — log-scale morphospace coordinates per edge
    FC          (2E,)     — FC weight per edge
    <seg_attr>  (2E,)     — segregation measure per edge  (e.g. E_diff)
    <int_attr>  (2E,)     — integration measure per edge  (e.g. E_rout)

The distance-bias control receives only normalised ``edge_distance``.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

from src.utils.config import BGTCCREConfig, MEASURE_CODE_TO_ATTR
from src.precompute.connectivity import proportional_threshold
from src.precompute.diffusion import compute_diffusion_efficiency
from src.precompute.routing import compute_routing_efficiency
from src.precompute.betweenness import compute_edge_betweenness_centrality
from src.precompute.clustering import compute_edge_clustering_coefficient
from src.precompute.communicability import compute_communicability
from src.precompute.participation import compute_edge_participation


# ── Measure registry ──────────────────────────────────────────────────────────

_COMPUTE_FN = {
    "E_diff": compute_diffusion_efficiency,
    "E_rout": compute_routing_efficiency,
    "EBC":    compute_edge_betweenness_centrality,
    "ECC":    compute_edge_clustering_coefficient,
    "G":      compute_communicability,
    "EP":     compute_edge_participation,
}


def atlas_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Return the Euclidean atlas distance matrix."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3), got {coords.shape}")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords contains non-finite values")
    return cdist(coords, coords, metric="euclidean")


def normalise_atlas_distances(dist_mat: np.ndarray) -> np.ndarray:
    """Divide distances by the mean over all unique non-diagonal atlas pairs."""
    dist_mat = np.asarray(dist_mat, dtype=np.float64)
    if dist_mat.ndim != 2 or dist_mat.shape[0] != dist_mat.shape[1]:
        raise ValueError(f"dist_mat must be square, got {dist_mat.shape}")
    upper = dist_mat[np.triu_indices(dist_mat.shape[0], k=1)]
    if upper.size == 0:
        raise ValueError("distance normalization requires at least two regions")
    mean_distance = float(upper.mean())
    if not np.isfinite(mean_distance) or mean_distance <= 0:
        raise ValueError("mean non-diagonal atlas distance must be positive")
    return dist_mat / mean_distance


def distance_local_adjacency(
    coords: np.ndarray,
    density: float,
) -> np.ndarray:
    """Build a deterministic exact-density undirected graph from distance only."""
    if not 0.0 < density <= 1.0:
        raise ValueError(f"local graph density must be in (0, 1], got {density}")
    dist_mat = atlas_distance_matrix(coords)
    n_regions = dist_mat.shape[0]
    rows, cols = np.triu_indices(n_regions, k=1)
    n_possible = len(rows)
    n_keep = int(round(float(density) * n_possible))
    n_keep = min(max(n_keep, 1), n_possible)

    # Primary key: distance. Secondary keys make ties deterministic.
    order = np.lexsort((cols, rows, dist_mat[rows, cols]))
    keep = order[:n_keep]
    adjacency = np.zeros((n_regions, n_regions), dtype=np.float64)
    adjacency[rows[keep], cols[keep]] = 1.0
    adjacency[cols[keep], rows[keep]] = 1.0
    return adjacency


# ── Laplacian positional encodings ────────────────────────────────────────────

def _lap_pe(A: np.ndarray, k: int) -> np.ndarray:
    """Laplacian eigenvector positional encodings.

    Computes the k smallest non-trivial eigenvectors of the unnormalised
    graph Laplacian L = D − A.

    Args:
        A: ``(N, N)`` binary symmetric adjacency.
        k: Number of eigenvectors to keep.

    Returns:
        ``(N, k)`` float32 array.  Padded with zeros if fewer than k
        non-trivial eigenvectors exist (e.g. disconnected graph).
    """
    N = A.shape[0]
    D = A.sum(axis=1)
    L = np.diag(D) - A

    try:
        eigvals, eigvecs = np.linalg.eigh(L)        # ascending order
        nontrivial = np.where(eigvals > 1e-8)[0][:k]
        pe = np.zeros((N, k), dtype=np.float32)
        pe[:, : len(nontrivial)] = eigvecs[:, nontrivial].real.astype(np.float32)
        pe = np.nan_to_num(pe, nan=0.0, posinf=0.0, neginf=0.0)
    except np.linalg.LinAlgError:
        pe = np.zeros((N, k), dtype=np.float32)

    return pe


# ── Main conversion function ──────────────────────────────────────────────────

def subject_to_data(
    bold: np.ndarray,
    connectivity: np.ndarray,
    label: int,
    coords: np.ndarray,
    cfg: BGTCCREConfig,
    bold_axes: str = "NT",
) -> Data:
    """Convert one subject's raw fMRI arrays to a PyG ``Data`` object.

    Args:
        bold:         Raw BOLD signal.  Shape depends on ``bold_axes``:
                      ``"NT"`` → ``(N_regions, T_timepoints)`` (HCP format)
                      ``"TN"`` → ``(T_timepoints, N_regions)`` (ABIDE format)
        connectivity: ``(N, N)`` functional connectivity matrix.
        label:        Integer class label (e.g. 0/1 for sex or diagnosis).
        coords:       ``(N, 3)`` MNI centroid coordinates.
        cfg:          Full ``BGTCCREConfig``.
        bold_axes:    Axis order of ``bold``.  ``"NT"`` = no transpose (HCP);
                      ``"TN"`` = transpose to ``(N, T)`` first (ABIDE).

    Returns:
        PyG ``Data`` with fields: ``x``, ``edge_index``, ``phi``, ``FC``,
        ``<seg_attr>``, ``<int_attr>``, ``lap_pe``, ``y``, ``bold``.
    """
    if bold_axes == "TN":
        bold = bold.T                                    # (T, N) → (N, T)

    N   = connectivity.shape[0]
    eps = cfg.precompute.eps

    if connectivity.ndim != 2 or connectivity.shape[0] != connectivity.shape[1]:
        raise ValueError(
            f"connectivity must be square (N, N), got shape {connectivity.shape}"
        )
    if coords.shape[0] != N:
        raise ValueError(
            f"coords has {coords.shape[0]} regions but connectivity has {N}; "
            "use matching atlas coordinates for this dataset"
        )
    if bold.ndim != 2:
        raise ValueError(f"bold must be 2-D after axis handling, got shape {bold.shape}")
    if bold.shape[0] != N:
        if bold.shape[1] == N:
            warnings.warn(
                f"BOLD shape {bold.shape} does not match graph nodes N={N}; "
                "transposing BOLD to (regions, time). Check the dataset loader "
                "or config if this warning appears unexpectedly.",
                RuntimeWarning,
            )
            bold = bold.T
        else:
            raise ValueError(
                f"BOLD shape {bold.shape} is incompatible with connectivity "
                f"shape {connectivity.shape}; one BOLD axis must equal N={N}"
            )

    # ── 1. Graph support ──────────────────────────────────────────────────
    # nan_to_num first: ABIDE has N > T so the pre-computed FC matrix can
    # contain NaN entries (zero-variance regions).  np.clip preserves NaN,
    # and nan * 0 = nan in numpy, so NaN would silently propagate into W.
    fc = np.nan_to_num(connectivity, nan=0.0, posinf=0.0, neginf=0.0)
    fc = np.clip(fc, 0.0, None)
    np.fill_diagonal(fc, 0.0)
    graph_construction = cfg.precompute.graph_construction
    if graph_construction == "fc":
        A, Aw = proportional_threshold(fc, cfg.precompute.threshold_pct)
    elif graph_construction == "distance_local":
        A = distance_local_adjacency(
            coords,
            cfg.precompute.local_graph_density,
        )
        Aw = fc * A
    else:
        raise ValueError(
            "graph_construction must be 'fc' or 'distance_local', "
            f"got {graph_construction!r}"
        )

    dist_mat = atlas_distance_matrix(coords)
    measures: dict[str, np.ndarray] = {}
    if cfg.precompute.compute_economy:
        # ── 2. Weight matrix for economy-measure computation ─────────────
        mode = cfg.precompute.weight_mode
        if mode == "binary":
            W = A
        elif mode == "fc":
            W = Aw
        elif mode == "cost_penalised":

            # λ: atlas-normalised decay constant.
            lam = cfg.precompute.eco_lambda
            if lam is None:
                connected_dists = dist_mat[A > 0]
                d_bar = connected_dists.mean() if len(connected_dists) > 0 else 1.0
                lam = 1.0 / (d_bar + 1e-12)

            # W_ij = FC_ij · exp(-λ · dist_ij)
            decay = np.exp(-lam * dist_mat)
            W = Aw * decay
            np.fill_diagonal(W, 0.0)
        else:
            raise ValueError(f"unknown weight_mode={mode!r}")

        # ── 3. Economy/topological measures ───────────────────────────────
        topo_metric_x, topo_metric_y = cfg.precompute.morphospace_pair
        topo_metric_x_attr = MEASURE_CODE_TO_ATTR[topo_metric_x]
        topo_metric_y_attr = MEASURE_CODE_TO_ATTR[topo_metric_y]

        for attr in (topo_metric_x_attr, topo_metric_y_attr):
            fn = _COMPUTE_FN[attr]
            m = fn(W, eps)
            measures[attr] = np.nan_to_num(
                m,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

    # ── 4. Build bidirectional edge list ───────────────────────────────────
    rows, cols = np.where(np.triu(A, k=1) > 0)          # upper triangle
    src = np.concatenate([rows, cols])                   # forward + reverse
    dst = np.concatenate([cols, rows])
    edge_index = np.stack([src, dst], axis=0)            # (2, 2E)

    # ── 5. Edge attributes ─────────────────────────────────────────────────
    def _edge_vals(mat: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [mat[rows, cols], mat[cols, rows]]
        ).astype(np.float32)

    fc_edges = _edge_vals(Aw)
    dist_edges = _edge_vals(dist_mat)

    # ── 6. Laplacian positional encodings ─────────────────────────────────
    lap_pe = _lap_pe(A, cfg.model.k_lap)                 # (N, k)

    # ── 7. Assemble PyG Data ───────────────────────────────────────────────
    data_kwargs = dict(
        num_nodes  = N,
        edge_index = torch.from_numpy(edge_index).long(),
        lap_pe     = torch.from_numpy(lap_pe),
        y          = torch.tensor([label], dtype=torch.long),
        bold       = torch.from_numpy(bold.astype(np.float32)),
    )
    if cfg.precompute.compute_economy:
        topo_metric_x_edges = _edge_vals(measures[topo_metric_x_attr])
        topo_metric_y_edges = _edge_vals(measures[topo_metric_y_attr])
        phi = np.stack(
            [
                np.log(topo_metric_x_edges.clip(eps, None)),
                np.log(topo_metric_y_edges.clip(eps, None)),
            ],
            axis=1,
        ).astype(np.float32)
        phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        if not cfg.precompute.use_morphospace:
            phi = np.zeros_like(phi, dtype=np.float32)
        data_kwargs.update(
            phi=torch.from_numpy(phi),
            FC=torch.from_numpy(fc_edges),
            dist=torch.from_numpy(dist_edges),
        )
    elif cfg.model.variant == "distance_bias":
        distance_norm = normalise_atlas_distances(dist_mat)
        data_kwargs["edge_distance"] = torch.from_numpy(
            _edge_vals(distance_norm)
        )

    data = Data(**data_kwargs)
    if cfg.precompute.compute_economy:
        setattr(data, topo_metric_x_attr, torch.from_numpy(topo_metric_x_edges))
        setattr(data, topo_metric_y_attr, torch.from_numpy(topo_metric_y_edges))

    return data
