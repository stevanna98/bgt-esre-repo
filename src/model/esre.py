"""ESRE Attention: EcoSpace Rotary Encoding attention mechanism.

This is the core module of BGT-ESRE. It implements:
  - Asymmetric rotary position encoding using ecospace coordinates phi
  - Cost-conditioned value augmentation

The canonical scatter_softmax_stable is imported from utils.scatter.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing

from src.utils.scatter import scatter_softmax_stable


class ESREAttention(MessagePassing):
    """Eco-Space Rotary Encoding multi-head attention.

    Extends PyG MessagePassing with:
      1. Eco-derived rotation angles (psi) applied asymmetrically to Q and K.
      2. Value augmentation conditioned on morphospace coordinates phi.

    Args:
        hidden_dim: Total model dimension d. Must be divisible by num_heads.
        num_heads: Number of attention heads H.
        dropout: Attention dropout probability.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        # node_dim=0: PyG 2.7+ changed the default to -2; we keep 0 so that
        # 3-D Q/K/V tensors of shape (N, H, d_h) are gathered correctly along dim 0.
        super().__init__(aggr="add", node_dim=0)
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
        )
        assert (hidden_dim // num_heads) % 2 == 0, (
            f"d_h={hidden_dim // num_heads} must be even for block-diagonal rotation"
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_h = hidden_dim // num_heads

        # Standard QKV projections
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # ESRE parameters
        # W_psi: maps 2D morphospace phi to d_h/2 rotation angles per head.
        # Init std=0.1 so angles start at ~1 radian given |phi|~14 — large
        # enough for the CCRE signal to be visible to the optimiser from epoch 1.
        self.W_psi = nn.Parameter(
            torch.randn(num_heads, self.d_h // 2, 2) * 0.1
        )   # (H, d_h/2, 2)
        # V_phi: maps 2D morphospace phi to d_h value correction per head.
        # Followed by a learnable scale initialised to 0 so the correction
        # starts inactive and grows only as training demands it.
        self.V_phi = nn.Parameter(
            torch.randn(num_heads, self.d_h, 2) * 0.1
        )   # (H, d_h, 2)
        self.v_scale = nn.Parameter(torch.zeros(1))

        self.attn_drop = nn.Dropout(dropout)

        # Cache for last attention weights (used by loss module)
        self._last_alpha: Optional[Tensor] = None

    @staticmethod
    def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Block-diagonal rotation of x by angles encoded in (cos, sin).

        Applies the 2x2 rotation R = [[cos, -sin], [sin, cos]] to consecutive
        pairs of dimensions in the last axis of x.

        For queries, call with +sin_psi. For keys, call with -sin_psi.
        The asymmetric use ensures Q_rot · K_rot depends on phi_ij:
        R(+psi)^T R(-psi) = R(-2*psi).

        Args:
            x: Input tensor of shape (*, d_h).
            cos: Cosine values of shape (*, d_h//2).
            sin: Sine values of shape (*, d_h//2). Negate for keys.

        Returns:
            Rotated tensor of shape (*, d_h).
        """
        x1 = x[..., 0::2]   # even dimensions,  shape (*, d_h//2)
        x2 = x[..., 1::2]   # odd dimensions,   shape (*, d_h//2)
        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x1 * cos - x2 * sin
        x_rot[..., 1::2] = x1 * sin + x2 * cos
        return x_rot

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        phi_edges: Tensor
    ) -> Tensor:
        """Run ESRE attention over the graph.

        Args:
            x: Node feature matrix of shape (N, hidden_dim).
            edge_index: Edge connectivity of shape (2, E).
            phi_edges: Morphospace coordinates per edge of shape (E, 2).

        Returns:
            z: Updated node embeddings of shape (N, hidden_dim).
        """
        N = x.shape[0]
        H = self.num_heads
        d_h = self.d_h

        Q = self.W_Q(x).view(N, H, d_h)   # (N, H, d_h)
        K = self.W_K(x).view(N, H, d_h)   # (N, H, d_h)
        V = self.W_V(x).view(N, H, d_h)   # (N, H, d_h)

        out = self.propagate(
            edge_index,
            Q=Q,
            K=K,
            V=V,
            phi_edges=phi_edges,
            size=(N, N),
        )   # (N, H, d_h)

        out = out.reshape(N, H * d_h)   # (N, hidden_dim)
        return self.W_O(out)

    def message(
        self,
        Q_i: Tensor,
        K_j: Tensor,
        V_j: Tensor,
        phi_edges: Tensor,
        index: Tensor,
        size_i: int,
    ) -> Tensor:
        """Compute per-edge messages with ESRE attention.

        PyG calls this with gathered features; Q_i is the destination node's Q,
        K_j and V_j are the source node's K and V.

        Args:
            Q_i: Query vectors for destination nodes, shape (E, H, d_h).
            K_j: Key vectors for source nodes, shape (E, H, d_h).
            V_j: Value vectors for source nodes, shape (E, H, d_h).
            phi_edges: Morphospace coordinates for each edge, shape (E, 2).
            index: Destination node index for each edge, shape (E,).
            size_i: Total number of destination nodes N.

        Returns:
            msg: Weighted value messages, shape (E, H, d_h).
        """
        E = Q_i.shape[0]
        H = self.num_heads
        d_h = self.d_h

        # Step 2 — Angle projection: psi_ij = W_psi^h · phi_ij
        psi = torch.einsum("ec,hkc->ehk", phi_edges, self.W_psi)   # (E, H, d_h/2)
        cos_psi = torch.cos(psi)   # (E, H, d_h/2)
        sin_psi = torch.sin(psi)   # (E, H, d_h/2)

        # Step 3 — Asymmetric rotation
        # Query uses +sin (rotation by +psi), key uses -sin (rotation by -psi)
        Q_rot = self._rotate(Q_i, cos_psi, sin_psi)    # (E, H, d_h)
        K_rot = self._rotate(K_j, cos_psi, -sin_psi)   # (E, H, d_h)

        # Step 4 — Pre-softmax logit 
        scores = (Q_rot * K_rot).sum(dim=-1) / math.sqrt(d_h)   # (E, H)

        # Step 5 — Economy-weighted scatter softmax (grouped by destination)
        alpha = scatter_softmax_stable(scores, index, dim_size=size_i)   # (E, H)

        self._last_alpha = alpha.detach()
        alpha = self.attn_drop(alpha)       # (E, H)

        # Step 6 — Cost-conditioned value augmentation.
        # v_scale is initialised to 0 so the correction is inactive at init
        # and grows only as training finds it useful.
        v_correction = torch.einsum(
            "ec,hdc->ehd", phi_edges, self.V_phi
        )   # (E, H, d_h)
        V_aug = V_j + torch.tanh(self.v_scale) * v_correction   # (E, H, d_h)
        # V_aug = V_j + v_correction   # (E, H, d_h) --- IGNORE ---

        # Weighted message
        msg = alpha.unsqueeze(-1) * V_aug   # (E, H, d_h)
        return msg