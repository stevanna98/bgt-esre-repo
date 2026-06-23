"""Graph-level readout modules."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.scatter import scatter_max, scatter_mean, scatter_softmax_stable


class GraphReadout(nn.Module):
    """Convert node embeddings into one subject embedding per graph."""

    def __init__(self, hidden_dim: int, mode: str = "mean") -> None:
        super().__init__()
        if mode not in {"mean", "max", "attention"}:
            raise ValueError(
                f"readout_pool must be one of mean, max, attention; got {mode!r}"
            )
        self.mode = mode
        self.score = nn.Linear(hidden_dim, 1) if mode == "attention" else None

    def forward(self, h: Tensor, batch: Tensor) -> Tensor:
        if self.mode == "mean":
            return scatter_mean(h, batch, dim=0)
        if self.mode == "max":
            return scatter_max(h, batch, dim=0)

        scores = self.score(h).squeeze(-1)
        weights = scatter_softmax_stable(
            scores.unsqueeze(-1),
            batch,
            dim_size=int(batch.max().item()) + 1,
        ).squeeze(-1)
        return scatter_mean(h * weights.unsqueeze(-1), batch, dim=0) * (
            torch.bincount(batch, minlength=int(batch.max().item()) + 1)
            .clamp(min=1)
            .to(h.dtype)
            .unsqueeze(-1)
        )
