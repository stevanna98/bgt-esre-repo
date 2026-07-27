"""Economy-free graph-transformer controls for locality experiments."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from src.loss import BGTCCRELoss
from src.model.bold_encoder import ParallelRegionEncoder
from src.model.readout import GraphReadout
from src.utils.config import BGTESREConfig
from src.utils.scatter import scatter_mean, scatter_softmax_stable


class StandardGraphAttention(MessagePassing):
    """Standard scaled dot-product multi-head attention on a sparse graph."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__(aggr="add", node_dim=0)
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_h = hidden_dim // num_heads
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self._last_alpha: Optional[Tensor] = None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_distance: Tensor | None = None,
    ) -> Tensor:
        del edge_distance
        n_nodes = x.shape[0]
        shape = (n_nodes, self.num_heads, self.d_h)
        q = self.W_Q(x).view(shape)
        k = self.W_K(x).view(shape)
        v = self.W_V(x).view(shape)
        out = self.propagate(
            edge_index, Q=q, K=k, V=v, size=(n_nodes, n_nodes)
        )
        return self.W_O(out.reshape(n_nodes, self.hidden_dim))

    def message(
        self,
        Q_i: Tensor,
        K_j: Tensor,
        V_j: Tensor,
        index: Tensor,
        size_i: int,
    ) -> Tensor:
        scores = (Q_i * K_j).sum(dim=-1) / math.sqrt(self.d_h)
        alpha = scatter_softmax_stable(scores, index, dim_size=size_i)
        self._last_alpha = alpha.detach()
        return self.attn_drop(alpha).unsqueeze(-1) * V_j


class DistanceBiasAttention(StandardGraphAttention):
    """Standard attention with a monotone non-positive Euclidean bias."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        mode: str = "learned_monotonic",
        initial_scale: float = 1.0,
    ) -> None:
        super().__init__(hidden_dim, num_heads, dropout)
        if mode not in {"learned_monotonic", "fixed"}:
            raise ValueError(
                "distance bias mode must be 'learned_monotonic' or 'fixed'"
            )
        if not math.isfinite(initial_scale) or initial_scale <= 0:
            raise ValueError("distance bias initial scale must be positive")
        self.distance_bias_mode = mode
        # Stable inverse-softplus, including for large user-supplied scales.
        raw = float(initial_scale) + math.log(-math.expm1(-float(initial_scale)))
        initial = torch.full((num_heads,), raw)
        if mode == "learned_monotonic":
            self.raw_distance_scale = nn.Parameter(initial)
        else:
            self.register_buffer("raw_distance_scale", initial)

    def distance_scale(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_distance_scale)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_distance: Tensor | None = None,
    ) -> Tensor:
        if edge_distance is None:
            raise ValueError("distance_bias requires data.edge_distance")
        if edge_distance.ndim != 1 or edge_distance.numel() != edge_index.shape[1]:
            raise ValueError("edge_distance must have shape (number_of_edges,)")
        if not torch.all(torch.isfinite(edge_distance)) or torch.any(edge_distance < 0):
            raise ValueError("edge_distance must be finite and non-negative")
        n_nodes = x.shape[0]
        shape = (n_nodes, self.num_heads, self.d_h)
        q = self.W_Q(x).view(shape)
        k = self.W_K(x).view(shape)
        v = self.W_V(x).view(shape)
        out = self.propagate(
            edge_index,
            Q=q,
            K=k,
            V=v,
            edge_distance=edge_distance,
            size=(n_nodes, n_nodes),
        )
        return self.W_O(out.reshape(n_nodes, self.hidden_dim))

    def message(
        self,
        Q_i: Tensor,
        K_j: Tensor,
        V_j: Tensor,
        edge_distance: Tensor,
        index: Tensor,
        size_i: int,
    ) -> Tensor:
        scores = (Q_i * K_j).sum(dim=-1) / math.sqrt(self.d_h)
        scores = scores - edge_distance.unsqueeze(-1) * self.distance_scale()
        alpha = scatter_softmax_stable(scores, index, dim_size=size_i)
        self._last_alpha = alpha.detach()
        return self.attn_drop(alpha).unsqueeze(-1) * V_j


class ControlTransformerLayer(nn.Module):
    """Pre-norm transformer layer with no morphospace or value augmentation."""

    def __init__(self, cfg: BGTESREConfig) -> None:
        super().__init__()
        model_cfg = cfg.model
        if model_cfg.variant == "distance_bias":
            self.attn = DistanceBiasAttention(
                model_cfg.hidden_dim,
                model_cfg.num_heads,
                model_cfg.dropout_attn,
                model_cfg.distance_bias_mode,
                model_cfg.distance_bias_init,
            )
        else:
            self.attn = StandardGraphAttention(
                model_cfg.hidden_dim,
                model_cfg.num_heads,
                model_cfg.dropout_attn,
            )
        self.norm1 = nn.LayerNorm(model_cfg.hidden_dim)
        self.norm2 = nn.LayerNorm(model_cfg.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(
                model_cfg.hidden_dim,
                model_cfg.ffn_multiplier * model_cfg.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(model_cfg.dropout_ffn),
            nn.Linear(
                model_cfg.ffn_multiplier * model_cfg.hidden_dim,
                model_cfg.hidden_dim,
            ),
            nn.Dropout(model_cfg.dropout_ffn),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_distance: Tensor | None = None,
    ) -> Tensor:
        x = x + self.attn(self.norm1(x), edge_index, edge_distance)
        return x + self.ffn(self.norm2(x))

    def get_last_alpha(self) -> Optional[Tensor]:
        return self.attn._last_alpha


class ControlTransformerModel(nn.Module):
    """Backbone-matched transformer for standard and distance controls."""

    VARIANTS = {"standard_transformer", "distance_bias", "distance_local_graph"}

    def __init__(self, cfg: BGTESREConfig) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model
        if model_cfg.variant not in self.VARIANTS:
            raise ValueError(f"unsupported control variant {model_cfg.variant!r}")
        expected_graph = (
            "distance_local"
            if model_cfg.variant == "distance_local_graph"
            else "fc"
        )
        if cfg.precompute.graph_construction != expected_graph:
            raise ValueError(
                f"{model_cfg.variant} requires {expected_graph!r} graph support"
            )
        if cfg.precompute.compute_economy or cfg.precompute.use_morphospace:
            raise ValueError("control models require economy computation disabled")
        if cfg.precompute.weight_mode == "cost_penalised":
            raise ValueError("control models cannot use cost-penalised weights")
        if model_cfg.value_gamma_mode != "zero":
            raise ValueError("control models require value_gamma_mode='zero'")

        self._use_bold_encoder = model_cfg.use_bold_encoder
        if self._use_bold_encoder:
            self.bold_encoder = ParallelRegionEncoder(
                num_regions=model_cfg.num_regions,
                d=model_cfg.hidden_dim,
                kernel_sizes=model_cfg.bold_cnn_kernel_sizes,
                dropout=model_cfg.bold_cnn_dropout,
            )
            self.bold_proj = None
        else:
            if model_cfg.bold_in_t is None:
                raise ValueError("bold_in_t is required without the BOLD encoder")
            self.bold_encoder = None
            self.bold_proj = nn.Linear(
                model_cfg.bold_in_t, model_cfg.hidden_dim, bias=False
            )

        self._use_lpe = model_cfg.use_lpe
        self.lap_proj = (
            nn.Linear(model_cfg.k_lap, model_cfg.hidden_dim, bias=False)
            if self._use_lpe
            else None
        )
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.norm = nn.LayerNorm(model_cfg.hidden_dim)
        self._use_virtual_node = model_cfg.use_virtual_node
        self.vn_emb = (
            nn.Embedding(1, model_cfg.hidden_dim)
            if self._use_virtual_node
            else None
        )
        self.layers = nn.ModuleList(
            [ControlTransformerLayer(cfg) for _ in range(model_cfg.num_layers)]
        )
        self.final_norm = nn.LayerNorm(model_cfg.hidden_dim)
        self.graph_readout = GraphReadout(
            model_cfg.hidden_dim,
            mode=model_cfg.readout_pool,
            num_regions=model_cfg.num_regions,
        )
        readout_dim = self.graph_readout.output_dim
        if self._use_virtual_node:
            readout_dim += model_cfg.hidden_dim
        self.readout_dropout = nn.Dropout(model_cfg.readout_dropout)
        self.readout = nn.Linear(readout_dim, model_cfg.num_classes)
        self.loss_fn = BGTCCRELoss(cfg)

    def _subject_readout(self, h: Tensor, batch: Tensor) -> Tensor:
        return self.graph_readout(h, batch)

    def forward(
        self,
        data: Data,
        epoch: int = 0,
        return_stage_embeddings: bool = False,
    ) -> dict:
        del epoch
        n_nodes = data.bold.shape[0]
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = data.bold.new_zeros(n_nodes, dtype=torch.long)
        n_graphs = int(batch.max().item()) + 1

        if self._use_bold_encoder:
            n_regions = self.bold_encoder.num_regions
            timepoints = data.bold.shape[-1]
            h = self.bold_encoder(
                data.bold.view(n_graphs, n_regions, timepoints)
            ).reshape(n_graphs * n_regions, self.cfg.model.hidden_dim)
        else:
            h = self.bold_proj(data.bold)
        if self._use_lpe:
            h = h + self.lap_proj(data.lap_pe)
        h = self.norm(self.dropout(h))
        if h.shape[0] != batch.shape[0]:
            raise RuntimeError("node features and graph nodes have different lengths")

        stages = {}
        if return_stage_embeddings:
            stages["encoder"] = self._subject_readout(h, batch)
        vn_h = (
            self.vn_emb.weight.expand(n_graphs, -1)
            if self._use_virtual_node
            else None
        )
        edge_distance = (
            data.edge_distance
            if self.cfg.model.variant == "distance_bias"
            else None
        )
        for layer_idx, layer in enumerate(self.layers, start=1):
            h = layer(h, data.edge_index, edge_distance)
            if self._use_virtual_node:
                vn_agg = scatter_mean(h, batch, dim=0)
                vn_h = torch.nn.functional.layer_norm(
                    vn_h + vn_agg, [h.shape[-1]]
                )
            if return_stage_embeddings:
                stages[f"layer_{layer_idx}"] = self._subject_readout(h, batch)

        h = self.final_norm(h)
        h_graph = self._subject_readout(h, batch)
        readout_input = (
            torch.cat([h_graph, vn_h], dim=-1)
            if self._use_virtual_node
            else h_graph
        )
        if return_stage_embeddings:
            stages["final"] = h_graph
            if self._use_virtual_node:
                stages["virtual_node"] = vn_h
            stages["readout_input"] = readout_input
        logits = self.readout(self.readout_dropout(readout_input))
        result = {
            "logits": logits,
            "h": h,
            "loss": self.loss_fn(logits=logits, y=data.y),
            "alpha": self.layers[-1].get_last_alpha(),
        }
        if return_stage_embeddings:
            result["stage_embeddings"] = stages
        return result
