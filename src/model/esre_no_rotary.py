"""ESREAttentionNoRotary: ablation variant of ESREAttention.

Identical to ESREAttention except the asymmetric rotary encoding is removed.
Q and K are used unrotated, so the attention score reduces to the standard
scaled dot-product:

    score_ij = (Q_i · K_j) / sqrt(d_h)

All other components are held exactly constant so the comparison isolates
the rotary component:

    Kept : W_Q, W_K, W_V, W_O (same QKV projections)
           V_phi, v_scale      (morphospace value augmentation)
           scatter_softmax_stable (same numerically stable softmax)
           attn_drop           (same dropout)
           _last_alpha         (same attention-weight cache)

    Removed : W_psi and the R(+psi) / R(-psi) rotation in message().

phi_edges is still accepted by forward() / message() so the layer signature
is identical to ESREAttention and existing training code requires no changes.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing

from src.utils.scatter import scatter_softmax_stable


class ESREAttentionNoRotary(MessagePassing):
    """Standard scaled dot-product multi-head attention over a graph.

    Drop-in replacement for ESREAttention with rotary encoding removed.
    Accepts the same arguments and returns the same shapes so it can be
    swapped into BGTESRELayer without any other changes.

    Args:
        hidden_dim: Total model dimension d. Must be divisible by num_heads.
        num_heads:  Number of attention heads H.
        dropout:    Attention dropout probability.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__(aggr="add", node_dim=0)
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
        )
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.d_h        = hidden_dim // num_heads

        # Identical QKV projections to ESREAttention
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Value augmentation — kept identical to ESREAttention
        self.V_phi   = nn.Parameter(torch.randn(num_heads, self.d_h, 2) * 0.1)
        self.v_scale = nn.Parameter(torch.zeros(1))

        self.attn_drop = nn.Dropout(dropout)

        self._last_alpha: Optional[Tensor] = None

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:          Tensor,
        edge_index: Tensor,
        phi_edges:  Tensor,
    ) -> Tensor:
        """Run standard (non-rotary) attention over the graph.

        Args:
            x:          Node feature matrix (N, hidden_dim).
            edge_index: Edge connectivity (2, E).
            phi_edges:  Morphospace coordinates per edge (E, 2).
                        Not used for scoring; forwarded to message() for
                        value augmentation only.

        Returns:
            Updated node embeddings (N, hidden_dim).
        """
        N   = x.shape[0]
        H   = self.num_heads
        d_h = self.d_h

        Q = self.W_Q(x).view(N, H, d_h)
        K = self.W_K(x).view(N, H, d_h)
        V = self.W_V(x).view(N, H, d_h)

        out = self.propagate(
            edge_index,
            Q=Q, K=K, V=V,
            phi_edges=phi_edges,
            size=(N, N),
        )   # (N, H, d_h)

        out = out.reshape(N, H * d_h)
        return self.W_O(out)

    def message(
        self,
        Q_i:       Tensor,
        K_j:       Tensor,
        V_j:       Tensor,
        phi_edges: Tensor,
        index:     Tensor,
        size_i:    int,
    ) -> Tensor:
        """Per-edge messages with standard dot-product attention (no rotation).

        Args:
            Q_i:       Query at destination node (E, H, d_h).
            K_j:       Key at source node (E, H, d_h).
            V_j:       Value at source node (E, H, d_h).
            phi_edges: Morphospace coordinates per edge (E, 2).
            index:     Destination node indices (E,).
            size_i:    Total number of destination nodes N.

        Returns:
            Weighted value messages (E, H, d_h).
        """
        # Standard scaled dot-product score — no rotation applied
        scores = (Q_i * K_j).sum(dim=-1) / math.sqrt(self.d_h)   # (E, H)

        alpha = scatter_softmax_stable(scores, index, dim_size=size_i)  # (E, H)

        self._last_alpha = alpha.detach()
        alpha = self.attn_drop(alpha)

        # Value augmentation: identical to ESREAttention
        v_correction = torch.einsum("ec,hdc->ehd", phi_edges, self.V_phi)  # (E, H, d_h)
        V_aug = V_j + torch.tanh(self.v_scale) * v_correction

        return alpha.unsqueeze(-1) * V_aug   # (E, H, d_h)
