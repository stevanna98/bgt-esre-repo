"""Edge betweenness centrality computation via Brandes' algorithm."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse.csgraph import shortest_path


def _compute_length_matrix(A_weighted: np.ndarray, eps: float) -> np.ndarray:
    """Convert weighted adjacency to path-length matrix.

    Uses the same inversion convention as routing.py: stronger FC weights map
    to shorter path lengths so that high-weight edges are preferred by Dijkstra.

    Args:
        A_weighted: Weighted adjacency matrix of shape (N, N).
        eps: Small constant added to weights before inversion.

    Returns:
        L: Path-length matrix of shape (N, N). Absent edges have infinite length.
    """
    L = np.where(A_weighted > 0, 1.0 / (A_weighted + eps), np.inf)
    np.fill_diagonal(L, 0.0)
    return L


def compute_edge_betweenness_centrality(
    A_weighted: np.ndarray, eps: float
) -> np.ndarray:
    """Compute edge betweenness centrality (EBC) for all edges in the graph.

    EBC[i, j] is the fraction of all shortest paths between node pairs (s, t)
    that traverse edge (i, j):

        EBC[i, j] = (1 / Z) * sum_{s != t} sigma(s, t | e_ij) / sigma(s, t)

    where Z = N * (N - 1) normalises by the total number of ordered node pairs.

    Implementation uses a Dijkstra-based Brandes backward-accumulation.
    FC-derived edge lengths are continuous floating-point values, so shortest-
    path ties are negligible and sigma(s, t) = 1 for all reachable pairs,
    which simplifies the dependency accumulation to integer counts.

    Algorithm outline (per source s):
      1. Run Dijkstra to obtain distances and predecessor tree from s.
      2. Traverse nodes in decreasing distance order (backward pass).
      3. For each node v with predecessor p: propagate pairwise dependency
         delta[p] += 1 + delta[v] and credit the directed edge (p -> v).
      4. After all sources, symmetrise and normalise.

    Args:
        A_weighted: Weighted adjacency matrix of shape (N, N).
        eps: Small constant for numerical stability in the length conversion.

    Returns:
        EBC: Edge betweenness centrality matrix of shape (N, N).
             Values lie in [0, 1]. Non-edges and diagonal are zero.

    Notes:
        Complexity is O(N * (N log N + E)) — effectively O(N^2 log N) for
        dense brain graphs.  For N > 300 a RuntimeWarning is emitted.
    """
    N = A_weighted.shape[0]
    if N > 300:
        warnings.warn(
            f"compute_edge_betweenness_centrality: N={N} > 300. "
            "EBC requires one Dijkstra pass per node (O(N^2 log N)); "
            "computation may be slow. Consider a coarser atlas.",
            RuntimeWarning,
            stacklevel=2,
        )

    L = _compute_length_matrix(A_weighted, eps)
    A_bin = (A_weighted > 0).astype(float)          # (N, N) binary adjacency
    EBC_directed = np.zeros((N, N))                  # accumulates directed counts

    for s in range(N):
        dist_s, pred_s = shortest_path(
            L,
            method="D",
            directed=False,
            indices=s,
            return_predecessors=True,
        )

        # Backward pass: process nodes in decreasing distance order so that
        # dependencies flow from leaves toward the source.
        delta = np.zeros(N)
        order = np.argsort(dist_s)[::-1]             # farthest first

        for v in order:
            if v == s or dist_s[v] == np.inf:
                continue
            p = int(pred_s[v])
            if p < 0:                                # unreachable / no predecessor
                continue
            c = 1.0 + delta[v]
            delta[p] += c
            EBC_directed[p, v] += c                  # directed edge p -> v from source s

    # Symmetrise: EBC_directed[p,v] holds paths going p->v (p is predecessor of v);
    # EBC_directed[v,p] holds paths going v->p (v is predecessor of p).
    # Together they cover all traversals of undirected edge {p, v} across all sources.
    EBC = EBC_directed + EBC_directed.T

    # Each unordered pair {s, t} appears twice in the all-source accumulation
    # (once as source s, once as source t) → divide by 2, then normalise by
    # the N*(N-1)/2 unordered pairs.  Net divisor = N*(N-1).
    norm = N * (N - 1)
    if norm > 0:
        EBC /= norm

    EBC = EBC * A_bin                                # zero out non-edges
    np.fill_diagonal(EBC, 0.0)
    return EBC
