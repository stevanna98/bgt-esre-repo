"""BGTESREModel: full Brain Graph Transformer with ESRE."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from src.utils.scatter import scatter_mean

from src.utils.config import BGTESREConfig
from src.model.bold_encoder import ParallelRegionEncoder
from src.model.layer import BGTESRELayer
from src.model.readout import GraphReadout


class BGTESREModel(nn.Module):
    """Full BGT-ESRE model.

    Architecture:
        1. InputEncoder  — project BOLD features (or raw BOLD via CNN) + LapPE,
                           initialise virtual node embeddings.
        2. L × BGTESRELayer — ESRE attention + FFN + virtual node update.
        3. BGTESRELoss   — task + economy alignment + head diversity.

    Args:
        cfg: Full BGT-ESRE configuration (``BGTESREConfig``).
    """

    def __init__(self, cfg: BGTESREConfig) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model
        pre_cfg   = cfg.precompute

        # ── Resolved measure attribute names (from config) ─────────────────
        self._topo_metric_x_attr: str = pre_cfg.topo_metric_x_attr # e.g. "E_diff"
        self._topo_metric_y_attr: str = pre_cfg.topo_metric_y_attr   # e.g. "E_rout"
        self._weight_mode: str = pre_cfg.weight_mode

        # ── BOLD encoder (optional CNN path) ──────────────────────────────
        self._use_bold_encoder = model_cfg.use_bold_encoder
        if model_cfg.use_bold_encoder:
            self.bold_encoder = ParallelRegionEncoder(
                num_regions=model_cfg.num_regions,
                d=model_cfg.hidden_dim,
                kernel_sizes=model_cfg.bold_cnn_kernel_sizes,
                dropout=model_cfg.bold_cnn_dropout,
            )
            self.bold_proj = None
        else:
            if model_cfg.bold_in_t is None:
                raise ValueError(
                    "ModelConfig.bold_in_t must be set when use_bold_encoder=False"
                )
            self.bold_encoder = None
            self.bold_proj = nn.Linear(
                model_cfg.bold_in_t,
                model_cfg.hidden_dim,
                bias=False,
            )
        self._use_lpe = model_cfg.use_lpe
        self.lap_proj = nn.Linear(model_cfg.k_lap, model_cfg.hidden_dim, bias=False) if model_cfg.use_lpe else None
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.norm = nn.LayerNorm(model_cfg.hidden_dim)
        # One learnable embedding shared across all virtual nodes in the batch.
        self.vn_emb = nn.Embedding(1, model_cfg.hidden_dim)

        # ── Transformer layers ────────────────────────────────────────────
        self.layers = nn.ModuleList(
            [
                BGTESRELayer(
                    model_cfg.hidden_dim,
                    model_cfg.num_heads,
                    model_cfg.ffn_multiplier,
                    model_cfg.dropout_attn,
                    model_cfg.dropout_ffn,
                )
                for _ in range(model_cfg.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(model_cfg.hidden_dim)
        self.graph_readout = GraphReadout(
            model_cfg.hidden_dim,
            mode=model_cfg.readout_pool,
            num_regions=model_cfg.num_regions,
        )

        # ── Readout ───────────────────────────────────────────────────────
        # Concatenates graph-pooled node embeddings with the virtual node
        # embedding so the classifier sees both local and global graph context.
        self.readout = nn.Linear(
            self.graph_readout.output_dim + model_cfg.hidden_dim,
            model_cfg.num_classes,
        )

        # ── Loss ──────────────────────────────────────────────────────────
        # Imported here to avoid a circular dependency at module level.
        from src.loss import BGTCCRELoss
        self.loss_fn = BGTCCRELoss(cfg)

    # ──────────────────────────────────────────────────────────────────────────

    def _subject_readout(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.graph_readout(h, batch)

    def forward(
        self,
        data: Data,
        epoch: int = 0,
        return_stage_embeddings: bool = False,
    ) -> dict:
        """Run the full forward pass.

        Args:
            data: Batched PyG Data object.  Required fields:

                * ``bold``       — ``(N, T)`` raw BOLD 
                * ``edge_index``— ``(2, E)``
                * ``phi``       — ``(E, 2)`` CCRE morphospace coordinates
                * ``lap_pe``    — ``(N, k_lap)`` Laplacian positional encodings
                * ``y``         — ``(G,)`` graph-level labels
                * ``batch``     — ``(N,)`` node-to-graph assignment
                * ``FC``        — ``(E,)`` functional connectivity weights
                * ``<topo_metric_x_attr>``— ``(E,)`` 
                * ``<topo_metric_y_attr>``— ``(E,)`` 

            epoch: Current training epoch (1-indexed), forwarded to the loss
                   for eco warmup scheduling.

        Returns:
            dict with keys:
                ``logits``      — ``(G, num_classes)``
                ``h``           — ``(N, d)`` final node embeddings
                ``loss``        — scalar total loss
                ``alpha``       — ``(E, H)`` attention weights from last layer (detached)
        """
        N     = data.bold.shape[0]
        # For un-batched single-graph Data objects (e.g. the single-subject
        # overfitting check), PyG does not set data.batch.  Fall back to an
        # all-zero assignment so every node belongs to graph 0.
        batch = data.batch
        if batch is None:
            batch = data.bold.new_zeros(N, dtype=torch.long)

        # ── 1. Bold encoding ─────────────────────────────────────────────
        bold = data.bold
        B = int(batch.max().item()) + 1

        if self._use_bold_encoder:
            R = self.bold_encoder.num_regions
            T = bold.shape[-1]
            bold_batched = bold.view(B, R, T)                        # (B, R, T)
            h_batched = self.bold_encoder(bold_batched)              # (B, R, d)
            h = h_batched.reshape(B * R, self.cfg.model.hidden_dim) # (N, d)
        else:
            h = self.bold_proj(bold)                                 # (N, d)

        if self._use_lpe:
            h = h + self.lap_proj(data.lap_pe)                     # (N, hidden_dim)
            h = self.norm(self.dropout(h))
        else:
            h = self.norm(self.dropout(h))

        if h.shape[0] != batch.shape[0]:
            raise RuntimeError(
                f"Encoded BOLD produced {h.shape[0]} node embeddings, but the "
                f"batched graph contains {batch.shape[0]} nodes. Check that "
                "BOLD, connectivity, and coords use the same region count and "
                "that the selected dataset loader matches the BOLD axis order."
            )

        stage_embeddings = {}
        if return_stage_embeddings:
            stage_embeddings["encoder"] = self._subject_readout(h, batch)

        vn_h = self.vn_emb.weight.expand(B, -1)              # (B, hidden_dim)

        # ── 2. Transformer layers with virtual node side channel ────────────
        d = h.shape[-1]
        for layer_idx, layer in enumerate(self.layers, start=1):
            h = layer(h, data.edge_index, data.phi)

            # Virtual node: aggregate from real nodes as a graph-level side
            # channel. Do not broadcast it back into every ROI embedding:
            # doing so injects a large repeated component that dominates
            # subject-level raw cosine similarity, especially with flatten
            # readout over a fixed atlas.
            vn_agg = scatter_mean(h, batch, dim=0)           # (G, d)
            vn_h   = torch.nn.functional.layer_norm(vn_h + vn_agg, [d])
            if return_stage_embeddings:
                stage_embeddings[f"layer_{layer_idx}"] = self._subject_readout(
                    h, batch
                )

        # ──3. Final normalisation ─────────────────────────────────────────
        h = self.final_norm(h)
        h_graph = self._subject_readout(h, batch)
        readout_input = torch.cat([h_graph, vn_h], dim=-1)
        if return_stage_embeddings:
            stage_embeddings["final"] = h_graph
            stage_embeddings["virtual_node"] = vn_h
            stage_embeddings["readout_input"] = readout_input

        # ── 4. Readout ─────────────────────────────────────────────────
        logits = self.readout(readout_input)                           # (G, num_classes)

        # ── 5. Collect cached attention weights from the last layer ────────
        alpha_last = self.layers[-1].get_last_alpha()

        # ── 7. Loss ───────────────────────────────────────────────────────
        loss = self.loss_fn(
            logits=logits,
            y=data.y,
        )

        result = dict(
            logits=logits,
            h=h,
            loss=loss,
            alpha=alpha_last,
        )
        if return_stage_embeddings:
            result["stage_embeddings"] = stage_embeddings
        return result
