
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Measure registry ──────────────────────────────────────────────────────────

# Maps short config codes → PyG Data attribute names stored on each graph.
MEASURE_CODE_TO_ATTR: dict[str, str] = {
    "ediff": "E_diff",   # diffusion efficiency     (segregation proxy)
    "erout": "E_rout",   # routing efficiency        (integration proxy)
    "ebc":   "EBC",      # edge betweenness          (integration proxy)
    "ecc":   "ECC",      # edge clustering coeff.    (segregation proxy)
    "comm":  "G",        # communicability           (integration proxy)
    "ep":    "EP",       # edge participation        (integration proxy)
}


# ── Configuration dataclasses ─────────────────────────────────────────────────

@dataclass
class ModelConfig:
    num_regions: int
    hidden_dim: int = 128
    num_classes: int = 2
    num_layers: int = 4
    num_heads: int = 8
    ffn_multiplier: int = 2
    dropout: float = 0.1
    dropout_attn: float = 0.3
    dropout_ffn: float = 0.3
    bold_cnn_kernel_sizes: tuple = (7, 5, 3)
    bold_cnn_dropout: float = 0.1
    use_lpe: bool = False
    k_lap: int = 16
    use_bold_encoder: bool = False   # False → linear projection instead of CNN
    bold_in_t: Optional[int] = 35  # required when use_bold_encoder=False
    readout_pool: str = "mean"  # "mean" | "max" | "attention"


@dataclass
class LossConfig:
    label_smoothing: float = 0.0


@dataclass
class PrecomputeConfig:
    # Short codes from MEASURE_CODE_TO_ATTR; resolved to attr names at build time.
    morphospace_pair: tuple = ("ediff", "erout")
    # Resolved attribute names stored on PyG Data objects.
    topo_metric_x_attr: str = "E_diff"
    topo_metric_y_attr: str = "E_rout"
    weight_mode: str = "fc"       # "binary" | "fc" | "cost_penalised"
    threshold_pct: float = 1
    eps: float = 1e-8
    eco_lambda: Optional[float] = 0.1   # None → auto-computed from mean edge distance


@dataclass
class BGTESREConfig:
    model: ModelConfig
    loss: LossConfig = field(default_factory=LossConfig)
    precompute: PrecomputeConfig = field(default_factory=PrecomputeConfig)


# Alias used in build_graph.py
BGTCCREConfig = BGTESREConfig
