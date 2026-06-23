"""Routing efficiency computation via shortest paths."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse.csgraph import shortest_path


def compute_length_matrix(A_weighted: np.ndarray, eps: float) -> np.ndarray:
    """Convert a weighted adjacency matrix to a path-length matrix.

    Edge length is the inverse of the FC weight, so stronger connections are
    shorter in path space.

    Args:
        A_weighted: Weighted adjacency matrix of shape (N, N).
        eps: Small constant added to weights before inversion.

    Returns:
        L: Path-length matrix of shape (N, N). Absent edges have infinite length.
    """
    N = A_weighted.shape[0]
    L = np.where(A_weighted > 0, 1.0 / (A_weighted + eps), np.inf)
    np.fill_diagonal(L, 0.0)
    return L


def compute_routing_efficiency(A_weighted: np.ndarray, eps: float) -> np.ndarray:
    """Compute pairwise routing efficiency via Floyd-Warshall shortest paths.

    Routing efficiency E_rout[i,j] = 1 / d_shortest(i,j), where d is the
    shortest geodesic path length on the graph.

    Args:
        A_weighted: Weighted adjacency matrix of shape (N, N).
        eps: Small constant for numerical stability.

    Returns:
        E_rout: Routing efficiency matrix of shape (N, N).

    Notes:
        Complexity is O(N^3). For N > 400 a warning is emitted.
    """
    N = A_weighted.shape[0]
    if N > 400:
        warnings.warn(
            f"compute_routing_efficiency: N={N} > 400. Floyd-Warshall is O(N^3); "
            "computation may be slow. Consider approximate methods.",
            RuntimeWarning,
            stacklevel=2,
        )

    L = compute_length_matrix(A_weighted, eps)
    D = shortest_path(L, method="FW", directed=False)   # (N, N)
    E_rout = np.where(D > 0, 1.0 / (D + eps), 0.0)
    np.fill_diagonal(E_rout, 0.0)
    return E_rout