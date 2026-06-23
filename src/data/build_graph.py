"""Convert raw fMRI arrays for one subject into a PyG Data object.

Produces all fields required by BGTCCREModel:
    x           (N, 5)    — hand-crafted BOLD temporal statistics
    edge_index  (2, 2E)   — bidirectional edges from thresholded FC
    phi         (2E, 2)   — log-scale morphospace coordinates per edge
    FC          (2E,)     — FC weight per edge
    <seg_attr>  (2E,)     — segregation measure per edge  (e.g. E_diff)
    <int_attr>  (2E,)     — integration measure per edge  (e.g. E_rout)
    lap_pe      (N, k)    — Laplacian positional encodings
    y           (1,)      — integer class label
    bold        (N, T)    — raw BOLD (kept for optional CNN encoder)
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

    # ── 1. Proportional threshold → binary A + weighted Aw ────────────────
    # nan_to_num first: ABIDE has N > T so the pre-computed FC matrix can
    # contain NaN entries (zero-variance regions).  np.clip preserves NaN,
    # and nan * 0 = nan in numpy, so NaN would silently propagate into W.
    fc = np.nan_to_num(connectivity, nan=0.0, posinf=0.0, neginf=0.0)
    fc = np.clip(fc, 0.0, None)
    np.fill_diagonal(fc, 0.0)
    A, Aw = proportional_threshold(fc, cfg.precompute.threshold_pct)

    # ── 2. Weight matrix for measure computation ───────────────────────────
    mode = cfg.precompute.weight_mode
    if mode == "binary":
        W = A
    elif mode == "fc":
        W = Aw
    else:                                                # cost_penalised
        dist_mat = cdist(coords, coords, metric="euclidean")

        # λ: atlas-normalised decay constant.
        # If not set in config, use λ = 1/d̄ where d̄ is the mean distance
        # over connected edges — normalises the decay to the atlas scale so
        # that exp(-λ·d) ≈ 1/e at a typical connection length.
        lam = cfg.precompute.eco_lambda
        if lam is None:
            connected_dists = dist_mat[A > 0]
            d_bar = connected_dists.mean() if len(connected_dists) > 0 else 1.0
            lam = 1.0 / (d_bar + 1e-12)

        # W_ij = FC_ij · exp(-λ · dist_ij)
        decay = np.exp(-lam * dist_mat)
        W = Aw * decay
        np.fill_diagonal(W, 0.0)

    # ── 3. Topological measures (only what the config needs) ───────────────
    topo_metric_x, topo_metric_y = cfg.precompute.morphospace_pair
    topo_metric_x_attr = MEASURE_CODE_TO_ATTR[topo_metric_x]
    topo_metric_y_attr = MEASURE_CODE_TO_ATTR[topo_metric_y]

    measures: dict[str, np.ndarray] = {}
    for attr in (topo_metric_x_attr, topo_metric_y_attr):
        fn = _COMPUTE_FN[attr]
        m  = fn(W, eps)                                  # (N, N)
        # Guard against NaN/Inf from singular matrices (e.g. disconnected
        # nodes in ABIDE where N > T makes FC rank-deficient after thresholding)
        measures[attr] = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

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

    fc_edges   = _edge_vals(Aw)
    topo_metric_x_edges  = _edge_vals(measures[topo_metric_x_attr])
    topo_metric_y_edges  = _edge_vals(measures[topo_metric_y_attr])

    # Euclidean distances between region centroids for each edge.
    # Stored so the model can compute cost-penalised eta at forward time
    # (binary mode: eta = topological_blend / dist).
    dist_mat   = cdist(coords, coords, metric="euclidean")
    dist_edges = _edge_vals(dist_mat)

    # Phi: log-scale morphospace coordinates (E, 2)
    phi = np.stack(
        [np.log(topo_metric_x_edges.clip(eps, None)),
         np.log(topo_metric_y_edges.clip(eps, None))],
        axis=1,
    ).astype(np.float32)
    phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)

    # ── 6. Laplacian positional encodings ─────────────────────────────────
    lap_pe = _lap_pe(A, cfg.model.k_lap)                 # (N, k)

    # ── 7. Assemble PyG Data ───────────────────────────────────────────────
    data = Data(
        num_nodes  = N,
        edge_index = torch.from_numpy(edge_index).long(),
        phi        = torch.from_numpy(phi),
        FC         = torch.from_numpy(fc_edges),
        dist       = torch.from_numpy(dist_edges),
        lap_pe     = torch.from_numpy(lap_pe),
        y          = torch.tensor([label], dtype=torch.long),
        bold       = torch.from_numpy(bold.astype(np.float32)),
    )

    # Attach seg/int measure tensors under their canonical attribute names
    setattr(data, topo_metric_x_attr, torch.from_numpy(topo_metric_x_edges))
    setattr(data, topo_metric_y_attr, torch.from_numpy(topo_metric_y_edges))

    return data