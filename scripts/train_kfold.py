#!/usr/bin/env python
"""K-fold cross-validation training for BGT-ESREE.

Strategy
--------
1. Load the full dataset once.
2. Call subject_to_data() ONCE per subject to precompute the graph structure
   (edge topology, phi, topological measures, Laplacian PE).  These are
   invariant across folds because they depend on the connectivity matrix, not
   the BOLD signal.
3. Per fold: fit a fresh scaler on training BOLD only, scale train+val BOLD,
   then swap data.bold in the cached Data objects — O(1) per subject, no
   repeated precompute.

Usage
-----
    python scripts/train_kfold.py --config configs/train.yaml
    python scripts/train_kfold.py --config configs/train.yaml --epochs 50
    python scripts/train_kfold.py --dataset hcp --epochs 50 --k 5
    python scripts/train_kfold.py --dataset abide --epochs 50 --k 5
    python scripts/train_kfold.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

# Make src.* importable when running from repo root or scripts/
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
os.environ.setdefault("XDG_CACHE_HOME", str(_REPO / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_REPO / ".cache" / "matplotlib"))

from src.data.build_graph import subject_to_data
from src.data.loaders import SubjectRecord, load_dataset
from src.model.model import BGTESREModel
from src.model.model_ablation import BGTESREModelAblation
from src.preprocess.combat import harmonize_subject_connectivity
from src.utils.config import (
    MEASURE_CODE_TO_ATTR,
    BGTESREConfig,
    LossConfig,
    ModelConfig,
    PrecomputeConfig,
)
from src.utils.metrics import compute_metrics
from src.utils.plotting import (
    plot_cosine_similarity_heatmap,
    plot_embedding_collapse_trends,
    plot_stage_cosine_architecture_grid,
    plot_attention_heatmap,
    plot_attention_per_label,
    plot_metric_train_val,
    plot_train_loss,
)

# ── Dataset roots ──────────────────────────────────────────────────────────────

DATASET_ROOTS: dict[str, str | None] = {
    "hcp": None,
    "abide": None,
    "ad_lmci": None,
    "nc_asd": None,
}

REQUIRED_DATASET_FILES: dict[str, tuple[str, ...]] = {
    "hcp": ("bold.npy", "connectivity.npy", "labels.npy", "coords.npy"),
    "abide": ("bold.npy", "connectivity.npy", "labels.npy", "coords.npy"),
    "ad_lmci": ("data_ad_lmci.npy", "labels_ad_lmci.npy", "desikan_coords_left.npy"),
    "nc_asd": ("data_nc_asd.npy", "labels_nc_asd.npy", "desikan_coords_left.npy"),
}

METRIC_NAMES = ["loss", "accuracy", "auc", "f1", "sensitivity", "specificity"]


# ── CLI ────────────────────────────────────────────────────────────────────────

def _flatten_yaml_config(config: dict, parser: argparse.ArgumentParser) -> dict:
    """Flatten grouped YAML sections into argparse destination names."""
    flat: dict = {}

    def _visit(mapping: dict, prefix: str = "") -> None:
        for raw_key, value in mapping.items():
            key = str(raw_key).replace("-", "_")
            location = f"{prefix}.{raw_key}" if prefix else str(raw_key)
            if isinstance(value, dict):
                _visit(value, location)
            else:
                if key in flat:
                    parser.error(f"duplicate YAML option '{key}' (at {location})")
                flat[key] = value

    _visit(config)
    return flat


def _coerce_option_value(
    key: str,
    value,
    action: argparse.Action,
    parser: argparse.ArgumentParser,
):
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if isinstance(value, bool):
            coerced = value
        elif isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                coerced = True
            elif lowered in {"0", "false", "no", "off"}:
                coerced = False
            else:
                parser.error(f"option '{key}' must be true or false")
        else:
            parser.error(f"option '{key}' must be true or false")
    elif action.type is not None:
        try:
            coerced = action.type(value)
        except (TypeError, ValueError) as exc:
            parser.error(f"invalid value for '{key}': {exc}")
    else:
        coerced = value

    if action.choices is not None and coerced not in action.choices:
        parser.error(
            f"invalid value for '{key}': {coerced!r}; "
            f"choose from {list(action.choices)}"
        )
    return coerced


def _load_yaml_defaults(path: str, parser: argparse.ArgumentParser) -> dict:
    try:
        import yaml
    except ImportError:
        parser.error("YAML configuration requires PyYAML (package 'pyyaml')")

    config_path = Path(path)
    if not config_path.is_file():
        parser.error(f"config file does not exist: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception as exc:
        parser.error(f"could not read YAML config {config_path}: {exc}")

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        parser.error("YAML config root must be a mapping")

    defaults = _flatten_yaml_config(raw, parser)
    actions = {action.dest: action for action in parser._actions}
    unknown = sorted(set(defaults) - set(actions))
    if unknown:
        parser.error("unknown YAML option(s): " + ", ".join(unknown))

    for key, value in defaults.items():
        action = actions[key]
        defaults[key] = _coerce_option_value(key, value, action, parser)
    return defaults


def _apply_set_overrides(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    actions = {action.dest: action for action in parser._actions}
    for raw in args.set_overrides:
        if "=" not in raw:
            parser.error(f"--set override must be KEY=VALUE, got {raw!r}")
        raw_key, raw_value = raw.split("=", 1)
        key = raw_key.strip().replace("-", "_").split(".")[-1]
        if not key:
            parser.error(f"--set override has an empty key: {raw!r}")
        if key not in actions or key in {"help", "set_overrides"}:
            parser.error(f"unknown --set option '{raw_key}'")
        setattr(
            args,
            key,
            _coerce_option_value(raw_key, raw_value.strip(), actions[key], parser),
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="K-fold cross-validation for BGT-ESRE / BGT-ESRE"
    )
    p.add_argument(
        "--config",
        default=None,
        help="YAML config file; explicit CLI options override YAML values",
    )
    p.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config/CLI option, e.g. --set model.no_bold_encoder=true",
    )
    # Dataset
    p.add_argument("--dataset", choices=["hcp", "abide", "ad_lmci", "nc_asd"],
                   help="Dataset to use (required unless --smoke)")
    p.add_argument("--data-dir", default=None,
                   help="Override the default data root for the chosen dataset")
    # Cross-validation
    p.add_argument("--k", type=int, default=5, help="Number of folds (default 5)")
    # Training
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight-decay",type=float, default=5e-2)
    p.add_argument("--scaler",      choices=["standard", "minmax", "none"], default="standard")
    p.add_argument("--selection-metric", choices=["auc", "loss"], default="loss",
                   help="Validation metric used for checkpointing and early stopping")
    p.add_argument("--patience",    type=int,   default=15,
                   help="Early-stopping patience in epochs (default 15)")
    p.add_argument("--lr-patience", type=int,   default=5,
                   help="ReduceLROnPlateau patience in epochs (default 5)")
    p.add_argument("--lr-factor",   type=float, default=0.5,
                   help="LR reduction factor (default 0.5)")
    p.add_argument("--min-lr",      type=float, default=1e-6,
                   help="Minimum LR floor (default 1e-6)")
    p.add_argument("--plot-every",  type=int,   default=5,
                   help="Refresh plots every N epochs (default 5)")
    p.add_argument("--embedding-monitor",
                   choices=["none", "train", "val", "all"],
                   default="val",
                   help="Split used for per-stage subject cosine-similarity plots")
    p.add_argument("--embedding-monitor-every", type=int, default=1,
                   help="Refresh embedding-collapse plots every N epochs")
    p.add_argument("--out-dir",     default=None,
                   help="Output directory (default: runs/{dataset}_{timestamp})")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      default="auto",
                   help="'cpu', 'cuda', 'cuda:0', etc., or 'auto'")
    # Model
    p.add_argument("--hidden-dim",  type=int,   default=64)
    p.add_argument("--num-layers",  type=int,   default=2)
    p.add_argument("--num-heads",   type=int,   default=4)
    p.add_argument("--k-lap",       type=int,   default=16)
    p.add_argument("--dropout",     type=float, default=0.5)
    p.add_argument("--readout-dropout", type=float, default=0.5,
                   help="Dropout applied to subject readout before classification")
    p.add_argument("--label-smoothing", type=float, default=0.05,
                   help="Cross-entropy label smoothing")
    p.add_argument("--readout-pool",
                   choices=["flatten", "mean", "mean_std", "max", "attention"],
                   default="flatten",
                   help="Graph-level subject pooling for the classifier")
    p.add_argument("--node-features", choices=["bold", "fc"], default="bold",
                   help="'bold' uses per-region BOLD time series as node "
                        "features; 'fc' uses each region's FC row as node "
                        "features")
    p.add_argument("--use-virtual-node", action="store_true",
                   help="Enable the optional graph-level virtual-node side channel")
    # Precompute / graph
    p.add_argument("--morphospace-x", default="comm",
                   choices=list(MEASURE_CODE_TO_ATTR),
                   help="Segregation axis measure (default: ediff)")
    p.add_argument("--morphospace-y", default="ebc",
                   choices=list(MEASURE_CODE_TO_ATTR),
                   help="Integration axis measure (default: erout)")
    p.add_argument("--weight-mode",   default="cost_penalised",
                   choices=["binary", "fc", "cost_penalised"])
    p.add_argument("--threshold-pct", type=float, default=1)
    # Harmonization
    p.add_argument("--combat-harmonize", action="store_true",
                   help="For ABIDE, apply fold-wise ComBat-style harmonization "
                        "to FC upper-triangle features before graph construction")
    p.add_argument("--combat-site-file", default=None,
                   help="Optional .npy/.txt/.csv site label file for ABIDE. "
                        "If omitted, the loader looks for sites.npy, site.npy, "
                        "site_ids.npy, site_labels.npy, batch.npy, or batches.npy")
    p.add_argument("--combat-preserve-label", action="store_true",
                   help="Include the class label as a covariate preserved by "
                        "ComBat. Disabled by default to avoid validation-label "
                        "use in preprocessing")
    p.add_argument("--combat-no-fisher-z", action="store_true",
                   help="Disable Fisher z transform before FC harmonization")
    # BOLD encoder
    p.add_argument("--no-bold-encoder", action="store_true",
                   help="Replace the CNN BOLD encoder with a single linear projection "
                        "(useful when no real BOLD is available, e.g. FC-only datasets)")
    # Model variant
    p.add_argument("--model", choices=["full", "ablation_no_rotary"], default="full",
                   help="'full' = BGTESREModel (rotary ESRE); "
                        "'ablation_no_rotary' = standard dot-product + additive phi injection")
    # Misc
    p.add_argument("--smoke", action="store_true",
                   help="Run a tiny synthetic end-to-end test and exit")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Parse --config first and install its values as defaults. The full second
    # parse makes explicitly supplied CLI arguments take precedence.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args(argv)

    parser = _build_parser()
    if config_args.config:
        parser.set_defaults(**_load_yaml_defaults(config_args.config, parser))
    args = parser.parse_args(argv)
    _apply_set_overrides(args, parser)
    return args


# ── Reproducibility ────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Config construction ────────────────────────────────────────────────────────

def make_config(
    args: argparse.Namespace,
    num_regions: int,
    n_classes: int,
    bold_in_t: int | None = None,
) -> BGTESREConfig:
    use_bold_encoder = not getattr(args, "no_bold_encoder", False)
    return BGTESREConfig(
        model=ModelConfig(
            num_regions=num_regions,
            hidden_dim=args.hidden_dim,
            num_classes=n_classes,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            dropout_attn=args.dropout,
            dropout_ffn=args.dropout,
            readout_dropout=args.readout_dropout,
            use_lpe=False,
            k_lap=args.k_lap,
            use_bold_encoder=use_bold_encoder,
            bold_in_t=bold_in_t if not use_bold_encoder else None,
            readout_pool=args.readout_pool,
            use_virtual_node=getattr(args, "use_virtual_node", False),
        ),
        loss=LossConfig(label_smoothing=args.label_smoothing),
        precompute=PrecomputeConfig(
            morphospace_pair=(args.morphospace_x, args.morphospace_y),
            topo_metric_x_attr=MEASURE_CODE_TO_ATTR[args.morphospace_x],
            topo_metric_y_attr=MEASURE_CODE_TO_ATTR[args.morphospace_y],
            weight_mode=args.weight_mode,
            threshold_pct=args.threshold_pct,
        ),
    )


# ── Data utilities ─────────────────────────────────────────────────────────────

def infer_normalised_bold_time_dim(subject: SubjectRecord) -> int:
    """Return BOLD T after applying the same axis handling as subject_to_data."""
    bold = subject.bold
    if subject.bold_axes == "TN":
        bold = bold.T

    if bold.ndim != 2:
        raise ValueError(
            f"{subject.subject_id}: expected 2-D BOLD, got shape {bold.shape}"
        )

    n_regions = subject.fc.shape[0]
    if bold.shape[0] != n_regions:
        if bold.shape[1] == n_regions:
            bold = bold.T
        else:
            raise ValueError(
                f"{subject.subject_id}: BOLD shape {bold.shape} is incompatible "
                f"with connectivity shape {subject.fc.shape}; one BOLD axis "
                f"must equal the region count {n_regions}"
            )
    return int(bold.shape[1])


def infer_node_feature_dim(subject: SubjectRecord, node_features: str) -> int:
    """Return feature length consumed by the linear node-feature projection."""
    if node_features == "bold":
        return infer_normalised_bold_time_dim(subject)
    if node_features == "fc":
        if subject.fc.ndim != 2 or subject.fc.shape[0] != subject.fc.shape[1]:
            raise ValueError(
                f"{subject.subject_id}: FC node features require a square "
                f"connectivity matrix, got {subject.fc.shape}"
            )
        return int(subject.fc.shape[1])
    raise ValueError(f"unknown node_features={node_features!r}")


def build_all_data(
    subjects: list[SubjectRecord],
    coords: np.ndarray,
    cfg: BGTESREConfig,
    node_features: str = "bold",
) -> list[Data]:
    """Precompute PyG Data objects for all subjects (expensive, called once).

    The stored data.bold is always (N, T) regardless of bold_axes.  When
    node_features="fc", data.bold stores FC rows as node features, so T=N.
    Topological measures (phi, E_diff, E_rout, …) are invariant across folds.
    """
    data_list = []
    for subj in tqdm(subjects, desc="Precomputing graphs", unit="subj"):
        if node_features == "bold":
            node_input = subj.bold
            node_input_axes = subj.bold_axes
        elif node_features == "fc":
            node_input = subj.fc
            node_input_axes = "NT"
        else:
            raise ValueError(f"unknown node_features={node_features!r}")
        data = subject_to_data(
            bold=node_input,
            connectivity=subj.fc,
            label=subj.label,
            coords=coords,
            cfg=cfg,
            bold_axes=node_input_axes,
        )
        data_list.append(data)
    return data_list


def _swap_bold(base: Data, new_bold_nt: np.ndarray) -> Data:
    """Return a new Data sharing all structure tensors but with a fresh bold.

    Args:
        base:        precomputed Data object (bold is (N, T))
        new_bold_nt: scaled bold as (N, T) numpy float32 array
    """
    new_data = Data()
    for key in base.keys():
        new_data[key] = base[key]
    new_data.bold = torch.from_numpy(new_bold_nt.astype(np.float32))
    return new_data


def scale_and_swap(
    scaler,
    base_data_list: list[Data],
    indices: list[int],
) -> list[Data]:
    """Transform bold for the given subjects and swap into new Data objects.

    scaler must already be fitted. base_data_list[i].bold is (N, T).
    """
    result = []
    for i in indices:
        raw_nt = base_data_list[i].bold.numpy()          # (N, T)
        scaled_nt = scaler.transform(raw_nt.T).T          # (T,N) → scale → (N,T)
        result.append(_swap_bold(base_data_list[i], scaled_nt.astype(np.float32)))
    return result


def load_site_file(path: str, n_subjects: int) -> np.ndarray:
    site_path = Path(path)
    if not site_path.is_file():
        raise FileNotFoundError(f"ComBat site file does not exist: {site_path}")
    if site_path.suffix == ".npy":
        sites = np.load(site_path, allow_pickle=True)
    else:
        sites = np.loadtxt(site_path, dtype=str, delimiter=",")
    sites = np.asarray(sites).reshape(-1)
    if sites.shape[0] != n_subjects:
        raise ValueError(
            f"ComBat site file {site_path} has {sites.shape[0]} labels, but "
            f"the dataset has {n_subjects} subjects"
        )
    return sites


def attach_sites(
    subjects: list[SubjectRecord],
    sites: np.ndarray,
) -> list[SubjectRecord]:
    return [
        subject._replace(site=sites[i].item() if hasattr(sites[i], "item") else sites[i])
        for i, subject in enumerate(subjects)
    ]


# ── Train / evaluate ──────────────────────────────────────────────────────────

def train_epoch(
    model: BGTESREModel,
    loader: PyGDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_classes: int,
) -> dict:
    model.train()
    all_logits, all_labels = [], []
    total_loss = 0.0
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        out["loss"].backward()
        optimizer.step()

        all_logits.append(out["logits"].detach().cpu())
        all_labels.append(batch.y.cpu().reshape(-1))
        total_loss += out["loss"].item() * batch.num_graphs
        n_graphs += batch.num_graphs

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    m = compute_metrics(logits, labels, n_classes)
    m["loss"] = total_loss / max(n_graphs, 1)
    return m


@torch.no_grad()
def evaluate(
    model: BGTESREModel,
    loader: PyGDataLoader,
    device: torch.device,
    n_classes: int,
) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        all_logits.append(out["logits"].cpu())
        all_labels.append(batch.y.cpu().reshape(-1))
        total_loss += out["loss"].item() * batch.num_graphs
        n_graphs += batch.num_graphs

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    m = compute_metrics(logits, labels, n_classes)
    m["loss"] = total_loss / max(n_graphs, 1)
    return m


# ── Attention collection ───────────────────────────────────────────────────────

@torch.no_grad()
def collect_attention(
    model: BGTESREModel,
    loader: PyGDataLoader,
    device: torch.device,
    num_regions: int,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Average edge-attention matrices over all subjects in *loader*.

    Uses the cached _last_alpha from the final transformer layer
    (shape: E × H per graph). Averages over heads, fills an N×N matrix
    per subject, then symmetrizes (A + Aᵀ) / 2 before accumulating.

    Returns:
        attn_overall:   (N, N) float32 — averaged over all subjects
        attn_per_label: {label: (N, N)} — averaged per class
    """
    model.eval()
    N = num_regions
    attn_sum = np.zeros((N, N), dtype=np.float64)
    attn_count = 0
    per_cls_sum: dict[int, np.ndarray] = defaultdict(lambda: np.zeros((N, N), dtype=np.float64))
    per_cls_cnt: dict[int, int] = defaultdict(int)

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        alpha = out.get("alpha")   # (E_batch, H) detached, or None
        if alpha is None:
            continue

        alpha_mean = alpha.float().mean(dim=-1).cpu().numpy()  # (E,)
        src_arr  = batch.edge_index[0].cpu().numpy()
        dst_arr  = batch.edge_index[1].cpu().numpy()
        batch_arr = batch.batch.cpu().numpy()
        labels_arr = batch.y.cpu().numpy().ravel()

        # ptr[g] = first node index of graph g in the concatenated batch
        ptr = batch.ptr.cpu().numpy()

        for g in range(batch.num_graphs):
            offset = int(ptr[g])
            edge_mask = batch_arr[src_arr] == g

            local_src   = src_arr[edge_mask]  - offset
            local_dst   = dst_arr[edge_mask]  - offset
            local_alpha = alpha_mean[edge_mask]

            A = np.zeros((N, N), dtype=np.float64)
            A[local_src, local_dst] = local_alpha
            A_sym = (A + A.T) / 2.0

            lbl = int(labels_arr[g])
            attn_sum += A_sym
            attn_count += 1
            per_cls_sum[lbl] += A_sym
            per_cls_cnt[lbl] += 1

    overall = (attn_sum / max(attn_count, 1)).astype(np.float32)
    per_label = {
        k: (v / max(per_cls_cnt[k], 1)).astype(np.float32)
        for k, v in per_cls_sum.items()
    }
    return overall, per_label


# ── Plot / save helpers ────────────────────────────────────────────────────────

def refresh_plots(history: dict, plots_dir: Path) -> None:
    """Overwrite all metric PNG files with the current history."""
    plot_train_loss(history, plots_dir / "train_loss.png")
    for metric in METRIC_NAMES:
        plot_metric_train_val(history, metric, plots_dir / f"{metric}.png")


def save_attention(
    attn_overall: np.ndarray,
    attn_per_label: dict[int, np.ndarray],
    plots_dir: Path,
    attn_dir: Path,
) -> None:
    """Save attention .npy files and render heatmaps (overwrite each time)."""
    np.save(attn_dir / "attn_overall.npy", attn_overall)
    plot_attention_heatmap(
        attn_overall,
        plots_dir / "attn_overall.png",
        title="Edge Attention — all classes",
    )
    for label, mat in attn_per_label.items():
        np.save(attn_dir / f"attn_label_{label}.npy", mat)
    plot_attention_per_label(attn_per_label, plots_dir)


def _cosine_similarity_matrix(emb: np.ndarray) -> np.ndarray:
    emb = emb.astype(np.float32, copy=False)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / np.clip(norm, 1e-12, None)
    sim = emb_norm @ emb_norm.T
    return np.clip(sim, -1.0, 1.0).astype(np.float32)


def _effective_rank(emb: np.ndarray) -> float:
    """Entropy-based rank of centered subject embeddings."""
    centered = emb - emb.mean(axis=0, keepdims=True)
    try:
        singular_vals = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    total = float(singular_vals.sum())
    if total <= 1e-12:
        return 0.0
    probs = singular_vals / total
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return float(np.exp(entropy))


def _embedding_dispersion_stats(emb: np.ndarray) -> dict:
    centered = emb - emb.mean(axis=0, keepdims=True)
    common_norm = float(np.linalg.norm(emb.mean(axis=0)))
    residual_norm = float(np.mean(np.linalg.norm(centered, axis=1)))
    if emb.shape[0] <= 1:
        mean_pairwise_l2 = float("nan")
    else:
        diffs = emb[:, None, :] - emb[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        mask = ~np.eye(emb.shape[0], dtype=bool)
        mean_pairwise_l2 = float(np.mean(dists[mask]))
    return dict(
        mean_feature_std=float(np.mean(np.std(emb, axis=0))),
        centered_fro_norm=float(np.linalg.norm(centered)),
        mean_pairwise_l2=mean_pairwise_l2,
        common_norm=common_norm,
        mean_residual_norm=residual_norm,
        common_to_residual_norm=(
            common_norm / residual_norm if residual_norm > 1e-12 else float("inf")
        ),
        effective_rank=_effective_rank(emb),
    )


def _embedding_similarity_stats(sim: np.ndarray) -> dict:
    n = sim.shape[0]
    if n <= 1:
        return dict(
            n_subjects=n,
            mean_offdiag=float("nan"),
            median_offdiag=float("nan"),
            max_offdiag=float("nan"),
            min_offdiag=float("nan"),
            std_offdiag=float("nan"),
        )
    mask = ~np.eye(n, dtype=bool)
    vals = sim[mask]
    return dict(
        n_subjects=n,
        mean_offdiag=float(np.mean(vals)),
        median_offdiag=float(np.median(vals)),
        max_offdiag=float(np.max(vals)),
        min_offdiag=float(np.min(vals)),
        std_offdiag=float(np.std(vals)),
    )


@torch.no_grad()
def collect_stage_embeddings(
    model: BGTESREModel,
    loader: PyGDataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Collect graph-level subject embeddings from each model stage."""
    model.eval()
    by_stage: dict[str, list[np.ndarray]] = defaultdict(list)
    for batch in loader:
        batch = batch.to(device)
        out = model(batch, return_stage_embeddings=True)
        for stage, emb in out["stage_embeddings"].items():
            by_stage[stage].append(emb.detach().cpu().numpy())
    return {
        stage: np.concatenate(parts, axis=0)
        for stage, parts in by_stage.items()
        if parts
    }


def save_embedding_similarity_monitor(
    model: BGTESREModel,
    loader: PyGDataLoader,
    device: torch.device,
    epoch: int,
    split: str,
    plots_dir: Path,
    embed_dir: Path,
    trend_history: dict[str, list[float]],
) -> None:
    """Save per-stage subject cosine similarities and collapse metrics."""
    stage_embeddings = collect_stage_embeddings(model, loader, device)
    epoch_plot_dir = plots_dir / "embedding_similarity" / f"epoch_{epoch:03d}"
    epoch_data_dir = embed_dir / f"epoch_{epoch:03d}"
    epoch_plot_dir.mkdir(parents=True, exist_ok=True)
    epoch_data_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = embed_dir / "collapse_metrics.jsonl"
    raw_sims: dict[str, np.ndarray] = {}
    centered_sims: dict[str, np.ndarray] = {}
    with metrics_path.open("a", encoding="utf-8") as f:
        for stage, emb in stage_embeddings.items():
            sim = _cosine_similarity_matrix(emb)
            centered_sim = _cosine_similarity_matrix(
                emb - emb.mean(axis=0, keepdims=True)
            )
            raw_sims[stage] = sim
            centered_sims[stage] = centered_sim
            stats = _embedding_similarity_stats(sim)
            centered_stats = _embedding_similarity_stats(centered_sim)
            dispersion_stats = _embedding_dispersion_stats(emb)
            np.save(epoch_data_dir / f"{stage}_embeddings.npy", emb)
            np.save(epoch_data_dir / f"{stage}_cosine.npy", sim)
            np.save(epoch_data_dir / f"{stage}_centered_cosine.npy", centered_sim)
            plot_cosine_similarity_heatmap(
                sim,
                epoch_plot_dir / f"{stage}.png",
                title=f"{split} cosine similarity - {stage} - epoch {epoch}",
            )
            plot_cosine_similarity_heatmap(
                centered_sim,
                epoch_plot_dir / f"{stage}_centered.png",
                title=f"{split} centered cosine - {stage} - epoch {epoch}",
            )

            trend_history[f"{stage}/raw"].append(stats["mean_offdiag"])
            trend_history[f"{stage}/centered"].append(
                centered_stats["mean_offdiag"]
            )
            record = dict(
                epoch=epoch,
                split=split,
                stage=stage,
                **stats,
                centered_mean_offdiag=centered_stats["mean_offdiag"],
                centered_median_offdiag=centered_stats["median_offdiag"],
                centered_max_offdiag=centered_stats["max_offdiag"],
                centered_min_offdiag=centered_stats["min_offdiag"],
                centered_std_offdiag=centered_stats["std_offdiag"],
                **dispersion_stats,
            )
            f.write(json.dumps(record) + "\n")

    plot_stage_cosine_architecture_grid(
        raw_sims,
        centered_sims,
        stage_order=list(stage_embeddings.keys()),
        out_path=epoch_plot_dir / "architecture_cosine_grid.png",
        title=f"{split} subject cosine similarities through architecture - epoch {epoch}",
    )

    plot_embedding_collapse_trends(
        trend_history,
        plots_dir / "embedding_collapse_mean_cosine.png",
    )


def _print_epoch(
    fold: int,
    epoch: int,
    total: int,
    train_m: dict,
    val_m: dict,
    lr: float,
) -> None:
    def _fmt(m: dict) -> str:
        auc = m.get("auc", float("nan"))
        auc_str = f"{auc:.3f}" if not (auc != auc) else " nan "
        return (
            f"loss={m.get('loss', float('nan')):.4f}  "
            f"acc={m.get('accuracy', float('nan')):.3f}  "
            f"auc={auc_str}  "
            f"f1={m.get('f1', float('nan')):.3f}  "
            f"sens={m.get('sensitivity', float('nan')):.3f}  "
            f"spec={m.get('specificity', float('nan')):.3f}"
        )
    tag = f"[Fold {fold} | Epoch {epoch:>3}/{total}]"
    print(f"{tag} train  {_fmt(train_m)}  lr={lr:.2e}")
    print(f"{tag} val    {_fmt(val_m)}")


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


# ── Per-fold training ──────────────────────────────────────────────────────────

def train_fold(
    fold_idx: int,
    train_data: list[Data],
    val_data: list[Data],
    all_fold_data: list[Data],
    cfg: BGTESREConfig,
    args: argparse.Namespace,
    fold_out: Path,
    n_classes: int,
) -> dict:
    """Train one fold and save artefacts.  Returns per-fold summary dict."""
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    plots_dir = fold_out / "plots"
    attn_dir  = fold_out / "attn"
    embed_dir = fold_out / "embeddings"
    plots_dir.mkdir(parents=True, exist_ok=True)
    attn_dir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)

    train_loader = PyGDataLoader(train_data, batch_size=args.batch_size,
                                 shuffle=True,  drop_last=False)
    train_monitor_loader = PyGDataLoader(train_data, batch_size=args.batch_size,
                                         shuffle=False, drop_last=False)
    val_loader   = PyGDataLoader(val_data,   batch_size=args.batch_size,
                                 shuffle=False, drop_last=False)
    all_loader   = PyGDataLoader(all_fold_data, batch_size=args.batch_size,
                                 shuffle=False, drop_last=False)
    monitor_loaders = {
        "train": train_monitor_loader,
        "val": val_loader,
        "all": all_loader,
    }

    ModelCls = BGTESREModelAblation if getattr(args, "model", "full") == "ablation_no_rotary" else BGTESREModel
    model = ModelCls(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    selection_metric = getattr(args, "selection_metric", "loss")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if selection_metric == "loss" else "max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    history: dict[str, list[float]] = {
        f"{split}_{m}": [] for split in ("train", "val") for m in METRIC_NAMES
    }
    history["lr"] = []
    embedding_trend_history: dict[str, list[float]] = defaultdict(list)

    best_val_auc     = -float("inf")
    best_val_loss    =  float("inf")
    best_val_metrics: dict = {}
    best_epoch       = -1
    epochs_no_improve = 0
    stopped_early = False

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    tracemalloc.start()
    peak_rss = 0
    t_start  = time.time()

    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(model, train_loader, optimizer, device, n_classes)
        val_m   = evaluate(model, val_loader, device, n_classes)

        current_lr = optimizer.param_groups[0]["lr"]
        for m in METRIC_NAMES:
            history[f"train_{m}"].append(train_m.get(m, float("nan")))
            history[f"val_{m}"].append(val_m.get(m, float("nan")))
        history["lr"].append(current_lr)

        _print_epoch(fold_idx, epoch, args.epochs, train_m, val_m, current_lr)

        # Track best checkpoint by the configured validation metric.
        val_auc  = val_m.get("auc", float("nan"))
        val_loss = val_m.get("loss", float("inf"))
        improved = False
        if selection_metric == "loss":
            metric_valid = val_loss == val_loss
            metric_value = val_loss
            improved = metric_valid and (
                val_loss < best_val_loss
                or (val_loss == best_val_loss and val_auc > best_val_auc)
            )
        else:
            metric_valid = val_auc == val_auc
            metric_value = val_auc
            improved = metric_valid and (
                val_auc > best_val_auc
                or (val_auc == best_val_auc and val_loss < best_val_loss)
            )

        if metric_valid:
            if improved:
                best_val_auc     = val_auc
                best_val_loss    = val_loss
                best_val_metrics = val_m
                best_epoch       = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            scheduler.step(metric_value)

        if improved:
            torch.save(
                dict(config=asdict(cfg), epoch=epoch,
                     model_variant=getattr(args, "model", "full"),
                     model_state=model.state_dict(), metrics=val_m),
                fold_out / "best_model.pt",
            )

        # RSS snapshot
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)

        monitor_split = getattr(args, "embedding_monitor", "val")
        monitor_every = max(1, int(getattr(args, "embedding_monitor_every", 1)))
        if monitor_split != "none" and epoch % monitor_every == 0:
            save_embedding_similarity_monitor(
                model=model,
                loader=monitor_loaders[monitor_split],
                device=device,
                epoch=epoch,
                split=monitor_split,
                plots_dir=plots_dir,
                embed_dir=embed_dir,
                trend_history=embedding_trend_history,
            )

        # Periodic plot + attention refresh
        if epoch % args.plot_every == 0 or epoch == args.epochs:
            refresh_plots(history, plots_dir)
            attn_overall, attn_per_label = collect_attention(
                model, val_loader, device, cfg.model.num_regions
            )
            save_attention(attn_overall, attn_per_label, plots_dir, attn_dir)

        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(no val {selection_metric} improvement for "
                  f"{args.patience} epochs)")
            stopped_early = True
            break

    # Final attention on the complete fold dataset (train + val)
    attn_overall_all, attn_per_label_all = collect_attention(
        model, all_loader, device, cfg.model.num_regions
    )
    save_attention(attn_overall_all, attn_per_label_all, plots_dir, attn_dir)

    # Memory bookkeeping
    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.time() - t_start

    summary: dict = dict(
        fold=fold_idx,
        best_epoch=best_epoch,
        best_val_auc=best_val_auc,
        best_val_loss=best_val_loss,
        best_val_accuracy=best_val_metrics.get("accuracy", float("nan")),
        best_val_sensitivity=best_val_metrics.get("sensitivity", float("nan")),
        best_val_specificity=best_val_metrics.get("specificity", float("nan")),
        best_val_f1=best_val_metrics.get("f1", float("nan")),
        stopped_early=stopped_early,
        stopped_epoch=len(history["lr"]),
        final_lr=history["lr"][-1] if history["lr"] else args.lr,
        train_time_seconds=round(elapsed, 1),
        train_time_human=_format_time(elapsed),
        peak_python_ram_bytes=peak_python,
        peak_rss_bytes=peak_rss,
    )
    if device.type == "cuda":
        summary["peak_gpu_bytes"] = torch.cuda.max_memory_allocated(device)

    with open(fold_out / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\nFold {fold_idx} done — best val AUC {best_val_auc:.4f} "
        f"at epoch {best_epoch}  ({_format_time(elapsed)})\n"
    )
    return summary


# ── Cross-validation loop ──────────────────────────────────────────────────────

def run_cv(
    subjects: list[SubjectRecord],
    base_data_list: list[Data] | None,
    cfg: BGTESREConfig,
    args: argparse.Namespace,
    out_dir: Path,
    n_classes: int,
    coords: np.ndarray,
) -> None:
    labels_arr = np.array([s.label for s in subjects])
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    fold_summaries = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(subjects)), labels_arr)
    ):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx}  —  train {len(train_idx)}  val {len(val_idx)}")
        print(f"{'='*60}\n")

        if getattr(args, "combat_harmonize", False):
            if args.dataset != "abide":
                raise ValueError("--combat-harmonize is currently supported for ABIDE only")
            fold_indices = list(train_idx) + list(val_idx)
            print("  Applying fold-wise ComBat harmonization to ABIDE FC ...")
            fold_subjects, combat_summary = harmonize_subject_connectivity(
                subjects,
                list(train_idx),
                fold_indices,
                preserve_label=getattr(args, "combat_preserve_label", False),
                fisher_z=not getattr(args, "combat_no_fisher_z", False),
            )
            if combat_summary["unseen_target_sites"]:
                print(
                    "  Warning: validation contains site(s) absent from the "
                    "training fold; no site-specific ComBat parameters were "
                    "available for "
                    + ", ".join(combat_summary["unseen_target_sites"])
                )
            print(
                "  ComBat train sites: "
                + ", ".join(combat_summary["train_sites"])
                + "\n"
            )
            fold_base_data = build_all_data(
                fold_subjects,
                coords=coords,
                cfg=cfg,
                node_features=getattr(args, "node_features", "bold"),
            )
            train_base = fold_base_data[: len(train_idx)]
            val_base = fold_base_data[len(train_idx) :]
        else:
            if base_data_list is None:
                raise RuntimeError("base_data_list is required when ComBat is disabled")
            train_base = [base_data_list[i] for i in train_idx]
            val_base = [base_data_list[i] for i in val_idx]

        # ── Scale BOLD (fit on train only, apply to both) ─────────────────
        if args.scaler == "none":
            train_data = train_base
            val_data = val_base
        else:
            scaler = StandardScaler() if args.scaler == "standard" else MinMaxScaler()
            train_flat = np.vstack(
                [data.bold.numpy().T for data in train_base]
            )  # (S_train * T, N)
            scaler.fit(train_flat)
            train_data = scale_and_swap(
                scaler,
                train_base,
                list(range(len(train_base))),
            )
            val_data = scale_and_swap(
                scaler,
                val_base,
                list(range(len(val_base))),
            )
        all_fold_data = train_data + val_data

        fold_out = out_dir / f"fold_{fold_idx}"
        fold_out.mkdir(parents=True, exist_ok=True)

        summary = train_fold(
            fold_idx, train_data, val_data, all_fold_data,
            cfg, args, fold_out, n_classes,
        )
        fold_summaries.append(summary)

    _save_cv_summary(fold_summaries, out_dir)


def _save_cv_summary(summaries: list[dict], out_dir: Path) -> None:
    metrics_of_interest = [
        "best_val_auc", "best_val_loss",
        "best_val_accuracy", "best_val_sensitivity",
        "best_val_specificity", "best_val_f1",
        "train_time_seconds",
    ]
    agg = {}
    for key in metrics_of_interest:
        vals = [s[key] for s in summaries if key in s]
        if vals:
            agg[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)))

    cv_summary = dict(
        n_folds=len(summaries),
        fold_summaries=summaries,
        aggregated=agg,
    )
    path = out_dir / "cv_summary.json"
    with open(path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"\nCV summary saved to {path}")
    for key, stats in agg.items():
        print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")


# ── Smoke test ─────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    """End-to-end test on tiny synthetic data (no real fMRI needed)."""
    import tempfile

    print("\n" + "=" * 60)
    print("  SMOKE TEST — synthetic data, no real fMRI required")
    print("=" * 60 + "\n")

    N_SUBJ, N_REG, T = 20, 10, 60
    N_CLASSES = 2
    rng = np.random.default_rng(0)

    subjects = []
    for i in range(N_SUBJ):
        bold = rng.standard_normal((N_REG, T)).astype(np.float32)
        fc   = np.corrcoef(bold)
        subjects.append(
            SubjectRecord(
                bold=bold, fc=fc,
                label=i % N_CLASSES,
                subject_id=f"smoke_{i:03d}",
                bold_axes="NT",
            )
        )
    coords = rng.standard_normal((N_REG, 3)).astype(np.float32)

    # Minimal config
    cfg = BGTESREConfig(
        model=ModelConfig(
            num_regions=N_REG,
            hidden_dim=16,
            num_classes=N_CLASSES,
            num_layers=2,
            num_heads=2,
            ffn_multiplier=2,
            dropout=0.0,
            dropout_attn=0.0,
            dropout_ffn=0.0,
            bold_cnn_kernel_sizes=(3,),
            bold_cnn_dropout=0.0,
            use_bold_encoder=True,
            use_lpe=True,
            k_lap=4,
        ),
        precompute=PrecomputeConfig(
            morphospace_pair=("ediff", "erout"),
            topo_metric_x_attr="E_diff",
            topo_metric_y_attr="E_rout",
            weight_mode="fc",
            threshold_pct=0.5,
        ),
    )

    # Fake args
    class _Args:
        k            = 2
        epochs       = 4
        batch_size   = 4
        lr           = 1e-3
        weight_decay = 0.0
        scaler       = "standard"
        patience     = 10
        lr_patience  = 2
        lr_factor    = 0.5
        min_lr       = 1e-6
        plot_every   = 2
        embedding_monitor = "val"
        embedding_monitor_every = 1
        seed         = 0
        device       = "cpu"
    args = _Args()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        seed_everything(args.seed)
        print("Precomputing graphs …")
        base_data = build_all_data(
            subjects,
            coords,
            cfg,
            node_features=getattr(args, "node_features", "bold"),
        )
        print("Running cross-validation …")
        run_cv(subjects, base_data, cfg, args, out_dir, N_CLASSES, coords)
        assert (out_dir / "cv_summary.json").exists(), "cv_summary.json not created"
        for fold in range(args.k):
            assert (out_dir / f"fold_{fold}" / "run_summary.json").exists()

    print("\n" + "=" * 60)
    print("  SMOKE TEST PASSED")
    print("=" * 60 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def _print_file_tree(root: str, limit: int = 20) -> None:
    entries = sorted(os.listdir(root))[:limit]
    print(f"\nFile tree: {root}  (first {limit} entries)")
    for e in entries:
        full = os.path.join(root, e)
        tag  = "[dir]" if os.path.isdir(full) else "     "
        print(f"  {tag}  {e}")
    print()


def _validate_dataset_files(dataset: str, root: str) -> None:
    missing = [
        name for name in REQUIRED_DATASET_FILES[dataset]
        if not os.path.isfile(os.path.join(root, name))
    ]
    if missing:
        print(
            f"error: dataset '{dataset}' is missing required file(s) in {root}: "
            + ", ".join(missing)
        )
        sys.exit(1)


def main() -> None:
    args = parse_args()

    if args.smoke:
        smoke_test()
        return

    if args.dataset is None:
        print("error: --dataset is required (hcp | abide) unless --smoke is set.")
        sys.exit(1)

    seed_everything(args.seed)

    # Resolve data root
    root = args.data_dir or DATASET_ROOTS[args.dataset]
    if root is None:
        print(
            f"error: no data directory configured for dataset '{args.dataset}'. "
            "Set data.data_dir in configs/train.yaml or pass --data-dir."
        )
        sys.exit(1)
    if not os.path.isdir(root):
        print(
            f"error: data directory does not exist or is not a directory: {root}"
        )
        sys.exit(1)
    _validate_dataset_files(args.dataset, root)

    # Print file tree so the user can verify the layout
    _print_file_tree(root, limit=20)

    # Load dataset
    print(f"Loading {args.dataset.upper()} from {root} …")
    subjects, coords = load_dataset(args.dataset, root)
    if args.combat_site_file is not None:
        subjects = attach_sites(subjects, load_site_file(args.combat_site_file, len(subjects)))
    print(f"  {len(subjects)} subjects loaded  |  {coords.shape[0]} regions  |  "
          f"coords shape {coords.shape}")
    if getattr(args, "combat_harmonize", False):
        if args.dataset != "abide":
            print("error: --combat-harmonize is currently supported for dataset 'abide' only")
            sys.exit(1)
        if any(subject.site is None for subject in subjects):
            print(
                "error: --combat-harmonize requires ABIDE site labels. Add one "
                "of sites.npy, site.npy, site_ids.npy, site_labels.npy, "
                "batch.npy, or batches.npy to the data directory, or pass "
                "--combat-site-file."
            )
            sys.exit(1)
        site_counts = defaultdict(int)
        for subject in subjects:
            site_counts[str(subject.site)] += 1
        print(
            "  ComBat harmonization: enabled for FC upper-triangle features | "
            f"{len(site_counts)} sites"
        )

    labels_arr = np.array([s.label for s in subjects])
    n_classes  = int(labels_arr.max()) + 1
    unique, counts = np.unique(labels_arr, return_counts=True)
    print(f"  Label distribution: " +
          "  ".join(f"class {c}={n}" for c, n in zip(unique, counts)))

    node_features = getattr(args, "node_features", "bold")
    if node_features == "fc" and not getattr(args, "no_bold_encoder", False):
        print("error: --node-features fc requires --no-bold-encoder")
        sys.exit(1)
    print(f"  Node features: {node_features}\n")

    # Out dir
    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = args.model if args.model != "full" else "esre"
        args.out_dir = str(_REPO / "runs" / f"{args.dataset}_{model_tag}_{ts}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"  Output directory: {out_dir}\n")

    # Config — infer node-feature length when the CNN is disabled.
    bold_in_t = (
        infer_node_feature_dim(subjects[0], node_features)
        if getattr(args, "no_bold_encoder", False)
        else None
    )
    if bold_in_t is not None:
        print(f"  Linear node-feature projection input length: {bold_in_t}\n")
    cfg = make_config(args, coords.shape[0], n_classes, bold_in_t=bold_in_t)

    # Precompute graphs once unless fold-wise ComBat changes the connectivity.
    if getattr(args, "combat_harmonize", False):
        print(
            "Graph structures will be built inside each fold after ComBat "
            "harmonization.\n"
        )
        base_data_list = None
    else:
        print(f"Precomputing graph structures (morphospace: "
              f"{args.morphospace_x} × {args.morphospace_y}) …")
        if args.dataset == "hcp":
            print("  (HCP N=379: expect ~2–8 min depending on hardware)")
        base_data_list = build_all_data(
            subjects,
            coords,
            cfg,
            node_features=node_features,
        )
        print(f"  {len(base_data_list)} graphs ready.\n")

    run_cv(subjects, base_data_list, cfg, args, out_dir, n_classes, coords)


if __name__ == "__main__":
    main()
