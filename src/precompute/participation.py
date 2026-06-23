"""Participation coefficient per node and edge-level aggregation."""

from __future__ import annotations

import warnings

import numpy as np


def _louvain_communities(A_weighted: np.ndarray, seed: int = 0) -> np.ndarray:
    """Run Louvain community detection and return node community assignments.

    Uses the python-louvain (community) package if available, falling back to
    a greedy modularity implementation via networkx.

    Args:
        A_weighted: Weighted adjacency matrix of shape (N, N).
        seed: Random seed for reproducibility.

    Returns:
        communities: Integer community assignment array of shape (N,).
    """
    try:
        import community as community_louvain   # python-louvain
        import networkx as nx
        G = nx.from_numpy_array(A_weighted)
        partition = community_louvain.best_partition(G, random_state=seed)
        return np.array([partition[i] for i in range(len(partition))], dtype=int)
    except ImportError:
        pass

    try:
        import networkx as nx
        G = nx.from_numpy_array(A_weighted)
        communities = nx.algorithms.community.greedy_modularity_communities(G)
        assignment = np.zeros(A_weighted.shape[0], dtype=int)
        for c_idx, comm in enumerate(communities):
            for node in comm:
                assignment[node] = c_idx
        return assignment
    except Exception as e:
        warnings.warn(
            f"Community detection failed ({e}). "
            "Falling back to degree-based heuristic split."
        )
        # Fallback: split nodes into two groups by median degree
        degree = (A_weighted > 0).sum(axis=1)
        return (degree >= np.median(degree)).astype(int)


def compute_participation_coefficient(
    A_weighted: np.ndarray, eps: float, seed: int = 0
) -> np.ndarray:
    """Compute the participation coefficient for each node.

    The participation coefficient measures how evenly a node distributes its
    connections across network communities:

        PC_i = 1 − Σ_m  (k_{i,m} / k_i)²

    where k_{i,m} is the number of edges from node i to community m and k_i
    is the total degree of node i.  Values range in [0, 1]:
      - PC_i = 0 : all connections within a single community (pure segregation)
      - PC_i → 1 : connections evenly spread across all communities (integration)

    This is the weighted generalisation — k_{i,m} uses the sum of FC weights
    to community m (i.e. strength) rather than raw edge count.

    Args:
        A_weighted: Weighted adjacency matrix of shape (N, N).
        eps: Small constant for numerical stability.
        seed: Random seed for community detection.

    Returns:
        PC: Participation coefficient array of shape (N,), values in [0, 1].
    """
    N = A_weighted.shape[0]
    communities = _louvain_communities(A_weighted, seed=seed)
    unique_modules = np.unique(communities)

    strength = A_weighted.sum(axis=1)                          # (N,) total strength

    PC = np.zeros(N, dtype=np.float64)
    for m in unique_modules:
        mask = (communities == m)
        # Strength of node i to community m
        s_im = A_weighted[:, mask].sum(axis=1)                 # (N,)
        PC += (s_im / (strength + eps)) ** 2

    PC = 1.0 - PC
    PC = np.clip(PC, 0.0, 1.0)
    return PC


def compute_edge_participation(
    A_weighted: np.ndarray, eps: float, seed: int = 0
) -> np.ndarray:
    """Compute edge-level participation as the mean of endpoint PC values.

    For edge (i, j):
        EP[i, j] = (PC_i + PC_j) / 2

    This is a symmetric, continuous edge measure:
      - High EP : both endpoints connect across many modules → bridge edge
      - Low EP  : both endpoints are locally confined → intra-module edge

    Non-edges (A[i,j] = 0) are zeroed out.

    Args:
        A_weighted: Weighted adjacency matrix of shape (N, N).
        eps: Small constant for numerical stability in PC computation.
        seed: Random seed for community detection.

    Returns:
        EP: Edge participation matrix of shape (N, N), values in [0, 1].
            Non-edges are zero. Diagonal is zero.
    """
    PC = compute_participation_coefficient(A_weighted, eps, seed)   # (N,)

    # Symmetric mean of endpoint participation coefficients
    EP = 0.5 * (PC[:, None] + PC[None, :])                          # (N, N)

    # Zero out non-edges and diagonal
    A_bin = (A_weighted > 0).astype(float)
    EP = EP * A_bin
    np.fill_diagonal(EP, 0.0)
    return EP
