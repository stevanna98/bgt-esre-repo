"""BGTCCREModelAblation: ablation variant of BGTCCREModel.

Removes the asymmetric rotary encoding (W_psi, R(±psi) rotation) and instead
injects morphospace information by projecting per-node mean morphospace
coordinates directly into the node embeddings before the transformer stack.

What changes vs BGTCCREModel
─────────────────────────────
  Removed : ESREAttention with rotary encoding
  Added   : ESREAttentionNoRotary (standard dot-product) in every layer
            phi_proj = Linear(2, hidden_dim) injected once before layer 1

Injection point
───────────────
After bold encoding + LapPE + norm (i.e. after the same pre-processing as the
full model), but before the first transformer layer:

    phi_node = scatter_mean(data.phi, data.edge_index[0], dim=0, dim_size=N)
    h = h + self.phi_proj(phi_node)

All other components are held exactly constant so the comparison isolates the
rotary vs. additive morphospace injection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.utils.scatter import scatter_mean
from src.utils.config import BGTESREConfig
from src.model.bold_encoder import ParallelRegionEncoder
from src.model.esre_no_rotary import ESREAttentionNoRotary
from src.model.readout import GraphReadout


# ── Ablation layer ────────────────────────────────────────────────────────────

class BGTESRELayerNoRotary(nn.Module):
    """Pre-norm transformer layer using standard dot-product attention.

    Drop-in replacement for BGTESRELayer. Identical architecture and forward
    signature; the only difference is that ESREAttentionNoRotary is used in
    place of ESREAttention.

    Args:
        hidden_dim:      Model dimension d.
        num_heads:       Number of attention heads H.
        ffn_multiplier:  FFN hidden dimension = ffn_multiplier * hidden_dim.
        dropout_attn:    Attention dropout probability.
        dropout_ffn:     FFN dropout probability.
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
        self.attn  = ESREAttentionNoRotary(hidden_dim, num_heads, dropout_attn)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, ffn_multiplier * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_ffn),
            nn.Linear(ffn_multiplier * hidden_dim, hidden_dim),
            nn.Dropout(dropout_ffn),
        )

    def forward(self, x, edge_index, phi_edges):
        x = x + self.attn(self.norm1(x), edge_index, phi_edges)
        x = x + self.ffn(self.norm2(x))
        return x

    def get_last_alpha(self):
        return self.attn._last_alpha


# ── Ablation model ────────────────────────────────────────────────────────────

class BGTESREModelAblation(nn.Module):
    """BGT-ESRE ablation: additive morphospace injection, no rotary encoding,
    no virtual node.

    Structurally identical to BGTESREModel with three changes:
      1. Each transformer layer uses BGTESRELayerNoRotary (standard
         dot-product attention).
      2. A single Linear(2, hidden_dim) projects per-node mean morphospace
         coordinates and adds them to the node embeddings before the first
         transformer layer.
      3. The virtual node is removed; readout uses mean pooling only.

    This isolates the contribution of asymmetric rotary encoding: any
    performance gap between this model and BGTESREModel is attributable
    solely to the rotary encoding mechanism.

    Args:
        cfg: Full BGT-ESRE configuration (``BGTESREConfig``).
    """

    def __init__(self, cfg: BGTESREConfig) -> None:
        super().__init__()
        self.cfg  = cfg
        model_cfg = cfg.model

        self._topo_metric_x_attr: str = cfg.precompute.topo_metric_x_attr
        self._topo_metric_y_attr: str = cfg.precompute.topo_metric_y_attr

        # ── BOLD encoder ──────────────────────────────────────────────────
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
                    "ModelConfig.bold_in_t must be set when "
                    "use_bold_encoder=False"
                )
            self.bold_encoder = None
            self.bold_proj = nn.Linear(
                model_cfg.bold_in_t,
                model_cfg.hidden_dim,
                bias=False,
            )

        self._use_lpe = model_cfg.use_lpe
        self.lap_proj = (
            nn.Linear(model_cfg.k_lap, model_cfg.hidden_dim, bias=False)
            if model_cfg.use_lpe else None
        )
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.norm    = nn.LayerNorm(model_cfg.hidden_dim)

        # ── Morphospace injection (replaces rotary encoding) ──────────────
        self.phi_proj = nn.Linear(2, model_cfg.hidden_dim, bias=False)

        # ── Transformer layers (no rotary, no virtual node) ───────────────
        self.layers = nn.ModuleList([
            BGTESRELayerNoRotary(
                model_cfg.hidden_dim,
                model_cfg.num_heads,
                model_cfg.ffn_multiplier,
                model_cfg.dropout_attn,
                model_cfg.dropout_ffn,
            )
            for _ in range(model_cfg.num_layers)
        ])

        self.final_norm = nn.LayerNorm(model_cfg.hidden_dim)
        self.graph_readout = GraphReadout(
            model_cfg.hidden_dim,
            mode=model_cfg.readout_pool,
            num_regions=model_cfg.num_regions,
        )

        # ── Readout (no virtual node concatenation) ───────────────────────
        self.readout = nn.Linear(
            self.graph_readout.output_dim,
            model_cfg.num_classes,
        )

        # ── Loss ──────────────────────────────────────────────────────────
        from src.loss import BGTCCRELoss
        self.loss_fn = BGTCCRELoss(cfg)

    # ─────────────────────────────────────────────────────────────────────────

    def _subject_readout(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.graph_readout(h, batch)

    def forward(
        self,
        data: Data,
        epoch: int = 0,
        return_stage_embeddings: bool = False,
    ) -> dict:
        """Run the ablation forward pass.

        Args:
            data:  Batched PyG Data with the same required fields as
                   BGTCCREModel.forward().
            epoch: Current training epoch (forwarded to loss).

        Returns:
            dict with ``logits`` (G, C), ``h`` (N, d), ``loss`` scalar,
            ``alpha`` (E, H) from the last layer.
        """
        N     = data.bold.shape[0]
        batch = data.batch
        if batch is None:
            batch = data.bold.new_zeros(N, dtype=torch.long)

        # ── 1. BOLD encoding ──────────────────────────────────────────────
        bold = data.bold
        B    = int(batch.max().item()) + 1

        if self._use_bold_encoder:
            R = self.bold_encoder.num_regions
            T = bold.shape[-1]
            h = self.bold_encoder(bold.view(B, R, T)).reshape(
                B * R, self.cfg.model.hidden_dim
            )
        else:
            h = self.bold_proj(bold)                     # (N, d)

        if self._use_lpe:
            h = h + self.lap_proj(data.lap_pe)
        h = self.norm(self.dropout(h))                   # (N, d)

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

        # ── 2. Morphospace injection ──────────────────────────────────────
        phi_node = scatter_mean(
            data.phi, data.edge_index[0], dim=0, dim_size=N
        )                                                # (N, 2)
        h = h + self.phi_proj(phi_node)                  # (N, d)
        if return_stage_embeddings:
            stage_embeddings["morphospace_injected"] = self._subject_readout(
                h, batch
            )

        # ── 3. Transformer layers (no virtual node update) ────────────────
        for layer_idx, layer in enumerate(self.layers, start=1):
            h = layer(h, data.edge_index, data.phi)      # (N, d)
            if return_stage_embeddings:
                stage_embeddings[f"layer_{layer_idx}"] = self._subject_readout(
                    h, batch
                )

        # ── 4. Final norm + readout (mean pooling only) ───────────────────
        h       = self.final_norm(h)                     # (N, d)
        h_graph = self._subject_readout(h, batch)
        if return_stage_embeddings:
            stage_embeddings["final"] = h_graph
            stage_embeddings["readout_input"] = h_graph
        logits  = self.readout(h_graph)                  # (G, C)

        # ── 5. Cached attention weights ───────────────────────────────────
        alpha_last = self.layers[-1].get_last_alpha()

        # ── 6. Loss ───────────────────────────────────────────────────────
        loss = self.loss_fn(logits=logits, y=data.y)

        result = dict(logits=logits, h=h, loss=loss, alpha=alpha_last)
        if return_stage_embeddings:
            result["stage_embeddings"] = stage_embeddings
        return result
