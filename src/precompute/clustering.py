"""Edge clustering coefficient computation (weighted Onnela et al. variant)."""

from __future__ import annotations

import numpy as np


def compute_edge_clustering_coefficient(
    A_weighted: np.ndarray, eps: float
) -> np.ndarray:
    """Compute the weighted edge clustering coefficient for all graph edges.

    Uses the Onnela et al. (2005) weighted triangle intensity with a
    **continuous strength-based denominator** to avoid log-space discretisation.

    Weighted triangle intensity for edge (i, j):

        t_w[i, j] = A_cbrt[i, j] * (A_cbrt @ A_cbrt)[i, j]

    where A_cbrt = A_weighted^{1/3} element-wise.  Because (a*b)^{1/3} =
    a^{1/3} * b^{1/3}, the matrix product automatically sums over common
    neighbours without an explicit intersection mask.

    Denominator — continuous strength in cube-root space:

        denom[i, j] = min(s_cbrt_i - A_cbrt[i,j],  s_cbrt_j - A_cbrt[i,j])

    where s_cbrt_i = sum_k A_cbrt[i, k] is the cube-root node strength.
    Subtracting A_cbrt[i,j] removes the direct edge so only triangle
    partners contribute, exactly as binary degree - 1 did before — but
    continuously.

    Using continuous strength instead of integer binary degree makes ECC a
    fully continuous quantity: log(ECC) no longer shows discrete horizontal
    stripes caused by integer denominator values.

    Args:
        A_weighted: Weighted adjacency matrix of shape (N, N).
        eps: Small constant added to the denominator for numerical stability.

    Returns:
        ECC: Edge clustering coefficient matrix of shape (N, N), continuous
             values in [0, 1].  Non-edges and diagonal are zero.

    References:
        Onnela, J.-P. et al. (2005). Intensity and coherence of motifs in
        weighted complex networks. Physical Review E, 71(6), 065103.
    """
    A_bin  = (A_weighted > 0).astype(float)           # (N, N) binary adjacency

    # Cube-root weights — normalise magnitude, preserve sign structure
    A_cbrt = np.cbrt(A_weighted)                      # (N, N)

    # Weighted triangle intensity (vectorised)
    common_cbrt = A_cbrt @ A_cbrt                     # (N, N): Σ_k A_cbrt[i,k]*A_cbrt[j,k]
    t_w = A_cbrt * common_cbrt                        # (N, N): t_w[i,j]

    # Continuous denominator: cube-root node strength minus direct edge
    s_cbrt  = A_cbrt.sum(axis=1)                      # (N,)
    denom_i = s_cbrt[:, None] - A_cbrt               # (N, N): s_i - w_ij^{1/3}
    denom_j = s_cbrt[None, :] - A_cbrt               # (N, N): s_j - w_ij^{1/3}
    denom   = np.minimum(denom_i, denom_j)            # (N, N)

    ECC = np.where(denom > eps, t_w / denom, 0.0)
    ECC = ECC * A_bin                                 # zero out non-edges
    np.fill_diagonal(ECC, 0.0)
    return ECC
