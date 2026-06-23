"""Diffusion efficiency computation via random walk mean first passage times."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve


def compute_transition_matrix(
    A_weighted: np.ndarray, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute row-normalised random walk transition matrix and stationary distribution.

    Args:
        A_weighted: Weighted adjacency matrix of shape (N, N).
        eps: Small constant for numerical stability.

    Returns:
        P: Row-normalised transition matrix of shape (N, N). P[i,j] = A[i,j] / s_i.
        pi: Stationary distribution of shape (N,). pi[i] = s_i / sum(s).
    """
    degree = A_weighted.sum(axis=1)                    # (N,)
    P = A_weighted / (degree[:, None] + eps)           # (N, N)
    pi = degree / (degree.sum() + eps)                 # (N,)
    return P, pi


def compute_fundamental_matrix(
    P: np.ndarray, pi: np.ndarray, eps: float
) -> np.ndarray:
    """Compute the fundamental matrix Z = (I - P + W)^{-1}.

    Args:
        P: Transition matrix of shape (N, N).
        pi: Stationary distribution of shape (N,).
        eps: Unused, retained for API consistency.

    Returns:
        Z: Fundamental matrix of shape (N, N).

    Notes:
        Uses scipy.linalg.solve for numerical stability instead of explicit inversion.
    """
    N = P.shape[0]
    W = np.tile(pi, (N, 1))       # (N, N); every row equals pi
    M = np.eye(N) - P + W         # (N, N)
    Z = solve(M, np.eye(N))       # (N, N)
    return Z


def compute_mfpt(Z: np.ndarray, pi: np.ndarray, eps: float) -> np.ndarray:
    """Compute mean first passage times from the fundamental matrix.

    MFPT[i, j] is the expected number of steps from node i to first reach node j.

    Args:
        Z: Fundamental matrix of shape (N, N).
        pi: Stationary distribution of shape (N,).
        eps: Small constant for numerical stability.

    Returns:
        MFPT: Mean first passage time matrix of shape (N, N). Diagonal is inf.
    """
    Z_diag = np.diag(Z)                                    # (N,)
    MFPT = (Z_diag[None, :] - Z) / (pi[None, :] + eps)    # (N, N)
    np.fill_diagonal(MFPT, np.inf)
    return MFPT


def compute_diffusion_efficiency(A_weighted: np.ndarray, eps: float) -> np.ndarray:
    """Compute pairwise diffusion efficiency.

    Diffusion efficiency E_diff[i,j] = 1 / MFPT[i,j].

    Args:
        A_weighted: Weighted adjacency matrix of shape (N, N).
        eps: Small constant for numerical stability.

    Returns:
        E_diff: Diffusion efficiency matrix of shape (N, N).
    """
    P, pi = compute_transition_matrix(A_weighted, eps)
    Z = compute_fundamental_matrix(P, pi, eps)
    MFPT = compute_mfpt(Z, pi, eps)
    E_diff = np.where(MFPT > 0, 1.0 / (MFPT + eps), 0.0)
    np.fill_diagonal(E_diff, 0.0)
    return E_diff
