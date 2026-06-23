"""BGTESRELayer: single transformer layer combining ESRE attention and FFN."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.model.esre import ESREAttention


class BGTESRELayer(nn.Module):
    """One BGT-ESRE transformer layer.

    Architecture (pre-norm):
        x = x + ESREAttention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Args:
        hidden_dim: Model dimension d.
        num_heads: Number of attention heads H.
        ffn_multiplier: FFN hidden dimension = ffn_multiplier * hidden_dim.
        dropout_attn: Attention dropout probability.
        dropout_ffn: FFN dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_multiplier: int,
        dropout_attn: float,
        dropout_ffn: float,
    ) -> None:
        super().__init__()
        self.attn = ESREAttention(hidden_dim, num_heads, dropout_attn)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_multiplier * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_ffn),
            nn.Linear(ffn_multiplier * hidden_dim, hidden_dim),
            nn.Dropout(dropout_ffn),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        phi_edges: Tensor,
    ) -> Tensor:
        """Forward pass through attention sub-layer and FFN sub-layer.

        Args:
            x: Node embeddings of shape (N, hidden_dim).
            edge_index: Edge connectivity of shape (2, E).
            phi_edges: Morphospace coordinates per edge of shape (E, 2).

        Returns:
            Updated node embeddings of shape (N, hidden_dim).
        """
        # Sub-layer 1: norm first, then CCRE attention + residual (pre-norm)
        x = x + self.attn(self.norm1(x), edge_index, phi_edges)
        # Sub-layer 2: norm first, then FFN + residual (pre-norm)
        x = x + self.ffn(self.norm2(x))
        return x

    def get_last_alpha(self) -> Optional[Tensor]:
        """Return cached attention weights from the last forward pass.

        Returns:
            Attention weights of shape (E, H), or None if no forward pass
            has been run yet.
        """
        return self.attn._last_alpha
