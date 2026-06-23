"""BGTESREModel: full Brain Graph Transformer with ESRE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from src.utils.scatter import scatter_mean

from src.utils.config import BGTESREConfig
from src.model.bold_encoder import ParallelRegionEncoder
from src.model.layer import BGTESRELayer


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
            self.bold_proj = nn.Linear(model_cfg.bold_in_t, model_cfg.hidden_dim)
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

        # ── Readout ───────────────────────────────────────────────────────
        # Concatenates mean-pooled node embeddings with the virtual node
        # embedding so the classifier sees both local and global graph context.
        self.readout = nn.Linear(2 * model_cfg.hidden_dim, model_cfg.num_classes)

        # ── Loss ──────────────────────────────────────────────────────────
        # Imported here to avoid a circular dependency at module level.
        from src.loss import BGTCCRELoss
        self.loss_fn = BGTCCRELoss(cfg)

    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, data: Data, epoch: int = 0) -> dict:
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

        vn_h = self.vn_emb.weight.expand(B, -1)              # (B, hidden_dim)

        # ── 2. Transformer layers with virtual node update ─────────────────
        d = h.shape[-1]
        for layer in self.layers:
            h = layer(h, data.edge_index, data.phi)

            # Virtual node: aggregate from real nodes, normalise to prevent
            # unbounded accumulation, then broadcast back.
            vn_agg = scatter_mean(h, batch, dim=0)           # (G, d)
            vn_h   = F.layer_norm(vn_h + vn_agg, [d])       # (G, d)
            h      = h + vn_h[batch]                         # (N, d)

        # ──3. Final normalisation ─────────────────────────────────────────
        h = self.final_norm(h)

        # ── 4. Readout ─────────────────────────────────────────────────
        h_graph = scatter_mean(h, batch, dim=0)                        # (G, d)
        logits = self.readout(torch.cat([h_graph, vn_h], dim=-1))      # (G, num_classes)

        # ── 5. Collect cached attention weights from the last layer ────────
        alpha_last = self.layers[-1].get_last_alpha()

        # ── 7. Loss ───────────────────────────────────────────────────────
        loss = self.loss_fn(
            logits=logits,
            y=data.y,
        )

        return dict(
            logits=logits,
            h=h,
            loss=loss,
            alpha=alpha_last,
        )