"""Pure-PyTorch scatter utilities — no torch_scatter dependency required.

Implements the two primitives used across the model:
  - scatter_mean  : used by GraphReadout and the virtual-node update in model.py
  - scatter_softmax_stable : used by CCREAttention for normalised attention weights
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def scatter_mean(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> Tensor:
    """Compute the mean of ``src`` values that share the same ``index`` entry.

    Equivalent to ``torch_scatter.scatter_mean`` but implemented with pure
    PyTorch ops so no extra C++ extension is required.

    Args:
        src:      Source tensor.  Shape: ``(N, ...)`` when ``dim=0``.
        index:    1-D integer tensor of length N mapping each element to a
                  group.  Values must be in ``[0, dim_size)``.
        dim:      Dimension along which to scatter.  Currently only ``dim=0``
                  is fully tested (which covers all uses in this codebase).
        dim_size: Size of the output along ``dim``.  Defaults to
                  ``index.max() + 1``.

    Returns:
        Tensor of shape ``(dim_size, ...)`` with the group means.
        Groups with no elements contain zero.
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    # Build the broadcast index for multi-dimensional src
    idx = index
    for _ in range(src.dim() - 1):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)                          # (N, ...)

    # Allocate output and accumulate
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    out.scatter_add_(dim, idx, src)

    # Count elements per group (shape: (dim_size,))
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    count.scatter_add_(
        0,
        index,
        torch.ones(src.shape[0], dtype=src.dtype, device=src.device),
    )
    count = count.clamp(min=1.0)

    # Broadcast count to match out dimensions
    if src.dim() > 1:
        view_shape = [dim_size] + [1] * (src.dim() - 1)
        count = count.view(view_shape)

    return out / count


def scatter_max(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> Tensor:
    """Compute max of ``src`` values that share the same ``index`` entry."""
    if dim != 0:
        raise NotImplementedError("scatter_max currently supports dim=0 only")
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    idx = index
    for _ in range(src.dim() - 1):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)

    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.full(
        out_shape, -torch.inf, dtype=src.dtype, device=src.device
    )
    out.scatter_reduce_(dim, idx, src, reduce="amax", include_self=True)
    return torch.nan_to_num(out, neginf=0.0)


def scatter_softmax_stable(
    src: Tensor,
    index: Tensor,
    dim_size: int,
) -> Tensor:
    """Numerically stable scatter softmax grouped by ``index``.

    For each group ``g = index[e]``, computes:

        out[e] = exp(src[e] - max_{e': index[e']=g} src[e'])
               / sum_{e': index[e']=g} exp(src[e'] - max)

    This is the standard max-subtract trick applied per group, preventing
    overflow when logits are large.

    Args:
        src:      Logit tensor of shape ``(E, H)`` — one row per edge, one
                  column per attention head.
        index:    1-D integer tensor of length E.  ``index[e]`` is the
                  destination node for edge e (i.e. ``edge_index[1]``).
        dim_size: Total number of destination nodes N.

    Returns:
        Normalised attention weights of shape ``(E, H)``.  For each
        destination node and head, the values over its incoming edges sum to 1.
    """
    E = src.shape[0]
    extra = src.shape[1:]                             # (H,) for CCRE usage

    # Expand index to broadcast over extra dims
    idx = index
    for _ in extra:
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)                          # (E, H)

    # ── Step 1: max per group (for numerical stability) ────────────────────
    max_val = torch.full(
        (dim_size, *extra), float("-inf"), dtype=src.dtype, device=src.device
    )
    max_val.scatter_reduce_(0, idx, src, reduce="amax", include_self=True)
    # Nodes with no incoming edges remain -inf → replace with 0 so their
    # contribution to the denominator is 1 (harmless, they get no messages).
    max_val = torch.nan_to_num(max_val, neginf=0.0)

    # ── Step 2: subtract per-group max and exponentiate ────────────────────
    src_shifted = src - max_val[index]                # (E, H)
    exp_src = torch.exp(src_shifted)                  # (E, H)

    # ── Step 3: sum of exp per group ──────────────────────────────────────
    sum_exp = torch.zeros(
        dim_size, *extra, dtype=src.dtype, device=src.device
    )
    sum_exp.scatter_add_(0, idx, exp_src)

    # ── Step 4: normalise ─────────────────────────────────────────────────
    return exp_src / (sum_exp[index] + 1e-8)
