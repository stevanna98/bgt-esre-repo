"""Graph-level readout modules."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.scatter import scatter_max, scatter_mean, scatter_softmax_stable


class GraphReadout(nn.Module):
    """Convert node embeddings into one subject embedding per graph."""

    def __init__(
        self,
        hidden_dim: int,
        mode: str = "mean",
        num_regions: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"mean", "max", "attention", "mean_std", "flatten"}:
            raise ValueError(
                "readout_pool must be one of mean, max, attention, mean_std, "
                "flatten; "
                f"got {mode!r}"
            )
        if mode == "flatten" and num_regions is None:
            raise ValueError("num_regions is required for readout_pool='flatten'")
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.num_regions = num_regions
        if mode == "mean_std":
            self.output_dim = 2 * hidden_dim
        elif mode == "flatten":
            self.output_dim = int(num_regions) * hidden_dim
        else:
            self.output_dim = hidden_dim
        self.score = nn.Linear(hidden_dim, 1) if mode == "attention" else None

    def forward(self, h: Tensor, batch: Tensor) -> Tensor:
        if self.mode == "mean":
            return scatter_mean(h, batch, dim=0)
        if self.mode == "mean_std":
            mean = scatter_mean(h, batch, dim=0)
            centered = h - mean[batch]
            var = scatter_mean(centered.square(), batch, dim=0)
            std = torch.sqrt(var.clamp_min(1e-12))
            return torch.cat([mean, std], dim=-1)
        if self.mode == "max":
            return scatter_max(h, batch, dim=0)
        if self.mode == "flatten":
            graph_count = int(batch.max().item()) + 1
            counts = torch.bincount(batch, minlength=graph_count)
            expected = int(self.num_regions)
            if not torch.all(counts == expected):
                raise RuntimeError(
                    "readout_pool='flatten' requires every graph in the batch to "
                    f"have {expected} nodes, got counts {counts.tolist()}"
                )
            return h.reshape(graph_count, expected * self.hidden_dim)

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
