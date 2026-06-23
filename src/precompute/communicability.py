"""Communicability computation via matrix exponential."""

from __future__ import annotations

import numpy as np


def compute_communicability(A_weighted: np.ndarray, eps: float) -> np.ndarray:
    """Compute pairwise communicability via the weighted matrix exponential.

    Communicability G[i,j] = (exp(A_weighted))[i,j] sums contributions from
    walks of ALL lengths between nodes i and j, weighted by the product of
    edge weights along each walk divided by the walk factorial:

        G[i,j] = Σ_{k=0}^{∞}  (A^k)[i,j] / k!

    This contrasts with:
      - Routing efficiency: only uses the single shortest (cheapest) path
      - Diffusion efficiency: uses random-walk mean first-passage times

    Communicability captures redundant parallel pathways — two nodes are
    highly communicable if many independent routes connect them, even if no
    single route is particularly short.  This makes it a global integration
    measure that is orthogonal to EBC (shortest-path count) and E_rout.

    Implementation
    --------------
    We use the spectral decomposition A = V Λ Vᵀ (A is symmetric) so that
    exp(A) = V exp(Λ) Vᵀ, which is O(N³) — same cost as np.linalg.eigh.
    This is numerically stable for the FC-derived weight matrices used here.

    The result is normalised to [0, 1] per-graph by dividing by the maximum
    off-diagonal value, keeping the scale consistent across subjects.

    Non-edges (A[i,j] = 0) are NOT zeroed out — communicability is defined
    for all node pairs and includes indirect paths through intermediaries.
    The diagonal is set to zero to avoid self-loops in the morphospace.

    Args:
        A_weighted: Weighted adjacency matrix of shape (N, N). Must be
                    symmetric and non-negative.
        eps: Small constant for numerical stability in normalisation.

    Returns:
        G: Communicability matrix of shape (N, N), values in [0, 1].
           Diagonal is zero.  All off-diagonal entries are positive.
    """
    # Spectral decomposition — A is symmetric, so eigh is exact and stable
    eigvals, eigvecs = np.linalg.eigh(A_weighted)        # (N,), (N, N)

    # Matrix exponential via spectral formula: exp(A) = V diag(exp(λ)) Vᵀ
    exp_eigvals = np.exp(eigvals)                         # (N,)
    G = (eigvecs * exp_eigvals[None, :]) @ eigvecs.T     # (N, N)

    # Zero diagonal — communicability self-loops are not meaningful here
    np.fill_diagonal(G, 0.0)

    # Normalise to [0, 1] across the graph
    g_max = G.max()
    if g_max > eps:
        G = G / g_max

    return G.astype(np.float64)
