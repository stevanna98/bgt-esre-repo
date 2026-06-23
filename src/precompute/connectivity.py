"""Functional connectivity matrix computation and thresholding."""

from __future__ import annotations

import numpy as np


def compute_fc_matrix(bold: np.ndarray, eps: float) -> np.ndarray:
    """Compute Pearson correlation functional connectivity matrix.

    Args:
        bold: BOLD signal array of shape (N, T).
        eps: Numerical stability constant (unused here; retained for API consistency).

    Returns:
        FC: Functional connectivity matrix of shape (N, N). Negative values are
            clipped to zero and the diagonal is set to zero.
    """
    FC = np.corrcoef(bold)   # (N, N)
    # np.corrcoef returns NaN for any region with zero variance (flat BOLD
    # signal).  This happens when N_regions > T (e.g. ABIDE: 316 regions,
    # 161 timepoints).  Replace NaN/Inf with 0 before clipping so they are
    # treated as absent connections rather than propagating through the pipeline.
    FC = np.nan_to_num(FC, nan=0.0, posinf=0.0, neginf=0.0)
    FC = np.clip(FC, 0, None)
    np.fill_diagonal(FC, 0.0)
    return FC


def proportional_threshold(
    FC: np.ndarray, threshold_pct: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply proportional threshold to FC matrix, retaining the top fraction of edges.

    Args:
        FC: Functional connectivity matrix of shape (N, N).
        threshold_pct: Fraction of edges to retain (e.g. 0.20 keeps top 20%).

    Returns:
        A: Binary adjacency matrix of shape (N, N).
        A_weighted: FC values on surviving edges, zeros elsewhere, shape (N, N).
    """
    N = FC.shape[0]
    upper_vals = FC[np.triu_indices(N, k=1)]
    positive_vals = upper_vals[upper_vals > 0]

    if len(positive_vals) == 0:
        A = np.zeros_like(FC)
        A_weighted = np.zeros_like(FC)
        return A, A_weighted

    q = np.quantile(positive_vals, 1.0 - threshold_pct)
    A = (FC >= q).astype(float)
    np.fill_diagonal(A, 0.0)
    A_weighted = FC * A
    return A, A_weighted
