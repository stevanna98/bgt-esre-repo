#!/usr/bin/env python
"""Plot group-difference attention against ABIDE morphospace coordinates.

The script consumes one or more completed ``train_kfold.py`` run directories.
For each run it reloads every fold's ``best_model.pt``, reconstructs the same
fold split and preprocessing, collects out-of-fold edge attention, and plots:

    x-axis: log morphospace coordinate 1 from the trained run
    y-axis: log morphospace coordinate 2 from the trained run
    color:  mean attention(label-positive) - mean attention(label-negative)

Example:
    python scripts/plot_abide_attention_delta.py \
        /path/to/abide_ediff_comm_run \
        /path/to/abide_ebc_comm_run \
        --data-dir /path/to/abide \
        --out-file /path/to/group_delta_attention.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
os.environ.setdefault("XDG_CACHE_HOME", str(_REPO / ".cache"))
os.environ.setdefault("XDG_CONFIG_HOME", str(_REPO / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_REPO / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from scripts.train_kfold import (  # noqa: E402
    attach_sites,
    build_all_data,
    infer_node_feature_dim,
    load_site_file,
    make_config,
    scale_and_swap,
)
from src.data.loaders import load_dataset  # noqa: E402
from src.model.model import BGTESREModel  # noqa: E402
from src.model.model_ablation import BGTESREModelAblation  # noqa: E402
from src.preprocess.combat import harmonize_subject_connectivity  # noqa: E402
from src.utils.config import (  # noqa: E402
    BGTESREConfig,
    LossConfig,
    MEASURE_CODE_TO_ATTR,
    ModelConfig,
    PrecomputeConfig,
)


MEASURE_LABELS = {
    "ediff": "log Diffusion Eff.",
    "erout": "log Routing Eff.",
    "ebc": "log Edge Betweenness",
    "ecc": "log Edge Clustering",
    "comm": "log Communicability",
    "ep": "log Edge Participation",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create ABIDE group delta attention morphospace figure."
    )
    p.add_argument(
        "run_dirs",
        nargs="+",
        help="Completed train_kfold.py run directories containing fold_*/best_model.pt",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Override data_dir stored in each run's resolved_args.json.",
    )
    p.add_argument(
        "--out-file",
        default=None,
        help="Figure path. Defaults inside the run directory for one panel.",
    )
    p.add_argument(
        "--split",
        choices=["val", "train", "all"],
        default="val",
        help="Subjects used per fold. Default val gives out-of-fold attention.",
    )
    p.add_argument(
        "--prevalence",
        type=float,
        default=0.50,
        help="Minimum fraction of evaluated subjects containing an edge.",
    )
    p.add_argument(
        "--label-positive",
        type=int,
        default=1,
        help="Class label used as the positive group in delta attention.",
    )
    p.add_argument(
        "--label-negative",
        type=int,
        default=0,
        help="Class label subtracted from the positive group.",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output resolution. Default matches the reference figure scale.",
    )
    p.add_argument(
        "--folds",
        default=None,
        help="Optional comma-separated fold ids, e.g. 0,1,2. Default: all folds.",
    )
    return p.parse_args()


def _namespace_from_json(path: Path) -> SimpleNamespace:
    with path.open("r", encoding="utf-8") as f:
        values = json.load(f)
    return SimpleNamespace(**values)


def _load_torch(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _config_from_dict(raw: dict) -> BGTESREConfig:
    return BGTESREConfig(
        model=ModelConfig(**raw["model"]),
        loss=LossConfig(**raw.get("loss", {})),
        precompute=PrecomputeConfig(**raw.get("precompute", {})),
    )


def _load_checkpoint_config(
    checkpoint: dict,
    args: SimpleNamespace,
    num_regions: int,
    n_classes: int,
    subjects,
) -> BGTESREConfig:
    if "config" in checkpoint:
        return _config_from_dict(checkpoint["config"])

    bold_in_t = (
        infer_node_feature_dim(subjects[0], getattr(args, "node_features", "bold"))
        if getattr(args, "no_bold_encoder", False)
        else None
    )
    return make_config(args, num_regions, n_classes, bold_in_t=bold_in_t)


def _load_model(checkpoint_path: Path, cfg: BGTESREConfig, device: torch.device):
    checkpoint = _load_torch(checkpoint_path, device)
    variant = checkpoint.get("model_variant", "full")
    model_cls = BGTESREModelAblation if variant == "ablation_no_rotary" else BGTESREModel
    model = model_cls(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def _resolve_out_file(args: argparse.Namespace) -> Path:
    if args.out_file is not None:
        return Path(args.out_file)
    if len(args.run_dirs) == 1:
        return Path(args.run_dirs[0]) / "plots" / f"group_delta_attention_{args.split}.png"
    return Path(args.run_dirs[0]).parent / f"group_delta_attention_{args.split}.png"


def _selected_folds(args: argparse.Namespace, k: int) -> set[int]:
    if args.folds is None:
        return set(range(k))
    return {int(x.strip()) for x in args.folds.split(",") if x.strip()}


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class EdgeAccumulator:
    def __init__(self, num_regions: int, label_positive: int, label_negative: int):
        self.num_regions = num_regions
        self.label_positive = label_positive
        self.label_negative = label_negative
        self.n_subjects = 0
        self.present = np.zeros((num_regions, num_regions), dtype=np.int32)
        self.x_sum = np.zeros((num_regions, num_regions), dtype=np.float64)
        self.y_sum = np.zeros((num_regions, num_regions), dtype=np.float64)
        self.attn_sum = {
            label_positive: np.zeros((num_regions, num_regions), dtype=np.float64),
            label_negative: np.zeros((num_regions, num_regions), dtype=np.float64),
        }
        self.attn_count = {
            label_positive: np.zeros((num_regions, num_regions), dtype=np.int32),
            label_negative: np.zeros((num_regions, num_regions), dtype=np.int32),
        }
        self.label_subjects = {label_positive: 0, label_negative: 0}

    def add_subject(
        self,
        label: int,
        local_src: np.ndarray,
        local_dst: np.ndarray,
        alpha: np.ndarray,
        phi: np.ndarray,
    ) -> None:
        self.n_subjects += 1
        if label in self.label_subjects:
            self.label_subjects[label] += 1

        by_pair: dict[tuple[int, int], list[float]] = {}
        for src, dst, attn, coords in zip(local_src, local_dst, alpha, phi):
            src_i = int(src)
            dst_i = int(dst)
            if src_i == dst_i:
                continue
            i, j = (src_i, dst_i) if src_i < dst_i else (dst_i, src_i)
            entry = by_pair.setdefault((i, j), [0.0, 0.0, 0.0, 0.0])
            entry[0] += float(attn)
            entry[1] += 1.0
            entry[2] += float(coords[0])
            entry[3] += float(coords[1])

        for (i, j), (attn_sum, count, x_sum, y_sum) in by_pair.items():
            self.present[i, j] += 1
            self.x_sum[i, j] += x_sum / count
            self.y_sum[i, j] += y_sum / count
            if label in self.attn_sum:
                self.attn_sum[label][i, j] += attn_sum / count
                self.attn_count[label][i, j] += 1

    def table(self, prevalence: float) -> list[dict[str, float | int]]:
        min_count = int(math.ceil(prevalence * max(self.n_subjects, 1)))
        rows = []
        pos = self.label_positive
        neg = self.label_negative
        for i in range(self.num_regions):
            for j in range(i + 1, self.num_regions):
                count = int(self.present[i, j])
                if count < min_count:
                    continue
                pos_count = int(self.attn_count[pos][i, j])
                neg_count = int(self.attn_count[neg][i, j])
                if pos_count == 0 or neg_count == 0:
                    continue
                x = self.x_sum[i, j] / count
                y = self.y_sum[i, j] / count
                pos_mean = self.attn_sum[pos][i, j] / pos_count
                neg_mean = self.attn_sum[neg][i, j] / neg_count
                rows.append(
                    {
                        "roi_i": i,
                        "roi_j": j,
                        "log_x": x,
                        "log_y": y,
                        "delta_attention": pos_mean - neg_mean,
                        "positive_attention": pos_mean,
                        "negative_attention": neg_mean,
                        "prevalence": count / max(self.n_subjects, 1),
                        "present_count": count,
                        "positive_count": pos_count,
                        "negative_count": neg_count,
                    }
                )
        return rows


@torch.no_grad()
def _collect_loader_attention(
    model,
    loader: PyGDataLoader,
    device: torch.device,
    acc: EdgeAccumulator,
) -> None:
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        alpha = out.get("alpha")
        if alpha is None:
            continue

        alpha_mean = alpha.float().mean(dim=-1).detach().cpu().numpy()
        phi = batch.phi.detach().cpu().numpy()
        src_arr = batch.edge_index[0].detach().cpu().numpy()
        dst_arr = batch.edge_index[1].detach().cpu().numpy()
        batch_arr = batch.batch.detach().cpu().numpy()
        labels_arr = batch.y.detach().cpu().numpy().reshape(-1)
        ptr = batch.ptr.detach().cpu().numpy()

        for graph_idx in range(batch.num_graphs):
            offset = int(ptr[graph_idx])
            edge_mask = batch_arr[src_arr] == graph_idx
            acc.add_subject(
                label=int(labels_arr[graph_idx]),
                local_src=src_arr[edge_mask] - offset,
                local_dst=dst_arr[edge_mask] - offset,
                alpha=alpha_mean[edge_mask],
                phi=phi[edge_mask],
            )


def _prepare_subjects(run_args: SimpleNamespace, data_dir: str | None):
    root = data_dir or run_args.data_dir
    if root is None:
        raise ValueError("No data_dir found. Pass --data-dir.")
    subjects, coords = load_dataset(run_args.dataset, root)
    if getattr(run_args, "combat_site_file", None) is not None:
        subjects = attach_sites(
            subjects,
            load_site_file(run_args.combat_site_file, len(subjects)),
        )
    return subjects, coords


def _build_fold_data(
    subjects,
    coords: np.ndarray,
    cfg: BGTESREConfig,
    run_args: SimpleNamespace,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    base_data_list,
):
    if getattr(run_args, "combat_harmonize", False):
        fold_indices = list(train_idx) + list(val_idx)
        fold_subjects, _ = harmonize_subject_connectivity(
            subjects,
            list(train_idx),
            fold_indices,
            preserve_label=getattr(run_args, "combat_preserve_label", False),
            fisher_z=not getattr(run_args, "combat_no_fisher_z", False),
        )
        fold_base_data = build_all_data(
            fold_subjects,
            coords=coords,
            cfg=cfg,
            node_features=getattr(run_args, "node_features", "bold"),
        )
        train_base = fold_base_data[: len(train_idx)]
        val_base = fold_base_data[len(train_idx) :]
    else:
        train_base = [base_data_list[i] for i in train_idx]
        val_base = [base_data_list[i] for i in val_idx]

    scaler_name = getattr(run_args, "scaler", "standard")
    if scaler_name == "none":
        return train_base, val_base

    scaler = StandardScaler() if scaler_name == "standard" else MinMaxScaler()
    train_flat = np.vstack([data.bold.numpy().T for data in train_base])
    scaler.fit(train_flat)
    train_data = scale_and_swap(scaler, train_base, list(range(len(train_base))))
    val_data = scale_and_swap(scaler, val_base, list(range(len(val_base))))
    return train_data, val_data


def collect_run_panel(
    run_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    resolved_args_path = run_dir / "resolved_args.json"
    if not resolved_args_path.is_file():
        raise FileNotFoundError(f"Missing {resolved_args_path}")

    run_args = _namespace_from_json(resolved_args_path)
    if run_args.dataset != "abide":
        raise ValueError(f"{run_dir} is dataset={run_args.dataset!r}, expected 'abide'")

    subjects, coords = _prepare_subjects(run_args, args.data_dir)
    labels_arr = np.array([s.label for s in subjects])
    n_classes = int(labels_arr.max()) + 1
    folds = _selected_folds(args, int(run_args.k))
    skf = StratifiedKFold(
        n_splits=int(run_args.k),
        shuffle=True,
        random_state=int(run_args.seed),
    )

    first_ckpt_path = run_dir / f"fold_{min(folds)}" / "best_model.pt"
    first_ckpt = _load_torch(first_ckpt_path, device)
    cfg = _load_checkpoint_config(
        first_ckpt,
        run_args,
        num_regions=coords.shape[0],
        n_classes=n_classes,
        subjects=subjects,
    )

    base_data_list = None
    if not getattr(run_args, "combat_harmonize", False):
        base_data_list = build_all_data(
            subjects,
            coords=coords,
            cfg=cfg,
            node_features=getattr(run_args, "node_features", "bold"),
        )

    acc = EdgeAccumulator(
        num_regions=cfg.model.num_regions,
        label_positive=args.label_positive,
        label_negative=args.label_negative,
    )
    batch_size = args.batch_size or int(run_args.batch_size)

    split_iter = list(skf.split(np.zeros(len(subjects)), labels_arr))
    for fold_idx, (train_idx, val_idx) in tqdm(
        list(enumerate(split_iter)),
        desc=f"{run_dir.name}: folds",
        unit="fold",
    ):
        if fold_idx not in folds:
            continue
        ckpt_path = run_dir / f"fold_{fold_idx}" / "best_model.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing {ckpt_path}")
        model, _ = _load_model(ckpt_path, cfg, device)
        train_data, val_data = _build_fold_data(
            subjects,
            coords,
            cfg,
            run_args,
            train_idx,
            val_idx,
            base_data_list,
        )
        if args.split == "train":
            eval_data = train_data
        elif args.split == "all":
            eval_data = train_data + val_data
        else:
            eval_data = val_data
        loader = PyGDataLoader(eval_data, batch_size=batch_size, shuffle=False)
        _collect_loader_attention(model, loader, device, acc)

    x_code, y_code = cfg.precompute.morphospace_pair
    rows = acc.table(args.prevalence)
    if not rows:
        raise RuntimeError(
            f"No edges survived prevalence={args.prevalence} for {run_dir}"
        )
    return {
        "run_dir": str(run_dir),
        "rows": rows,
        "x_code": x_code,
        "y_code": y_code,
        "x_attr": MEASURE_CODE_TO_ATTR.get(x_code, x_code),
        "y_attr": MEASURE_CODE_TO_ATTR.get(y_code, y_code),
        "config": asdict(cfg),
        "n_subjects": acc.n_subjects,
        "label_subjects": acc.label_subjects,
    }


def _kde_fill(ax, values: np.ndarray, orientation: str, color: str = "0.82") -> None:
    finite = values[np.isfinite(values)]
    if finite.size < 3 or np.allclose(finite.min(), finite.max()):
        return
    grid = np.linspace(float(finite.min()), float(finite.max()), 256)
    try:
        density = gaussian_kde(finite)(grid)
    except Exception:
        hist, edges = np.histogram(finite, bins=40, density=True)
        grid = 0.5 * (edges[:-1] + edges[1:])
        density = hist
    max_density = float(np.nanmax(density))
    if not np.isfinite(max_density) or max_density <= 0.0:
        return
    density = density / max_density
    if orientation == "x":
        ax.fill_between(grid, 0.0, density, color=color, edgecolor="0.55", linewidth=1.0)
        ax.plot(grid, density, color="0.55", linewidth=1.0)
        ax.set_ylim(0.0, 1.05)
    else:
        ax.fill_betweenx(grid, 0.0, density, color=color, edgecolor="0.55", linewidth=1.0)
        ax.plot(density, grid, color="0.55", linewidth=1.0)
        ax.set_xlim(0.0, 1.05)


def _panel_arrays(panel: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = panel["rows"]
    x = np.array([row["log_x"] for row in rows], dtype=np.float64)
    y = np.array([row["log_y"] for row in rows], dtype=np.float64)
    delta = np.array([row["delta_attention"] for row in rows], dtype=np.float64)
    return x, y, delta


def plot_panels(panels: list[dict], out_file: Path, prevalence: float, dpi: int) -> None:
    n_panels = len(panels)
    fig = plt.figure(figsize=(5.16 * n_panels, 5.93), constrained_layout=False)
    fig.subplots_adjust(
        left=0.12 / max(n_panels, 1),
        right=0.91,
        bottom=0.16,
        top=0.91,
    )
    outer = fig.add_gridspec(1, n_panels, wspace=0.42)

    for idx, panel in enumerate(panels):
        sub = outer[idx].subgridspec(
            2,
            3,
            width_ratios=[1.0, 0.055, 0.06],
            height_ratios=[0.21, 1.0],
            wspace=0.03,
            hspace=0.04,
        )
        ax_top = fig.add_subplot(sub[0, 0])
        ax = fig.add_subplot(sub[1, 0], sharex=ax_top)
        ax_right = fig.add_subplot(sub[1, 1], sharey=ax)
        cax = fig.add_subplot(sub[1, 2])

        x, y, delta = _panel_arrays(panel)
        vmax = float(np.nanpercentile(np.abs(delta), 99))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = float(np.nanmax(np.abs(delta))) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        scatter = ax.scatter(
            x,
            y,
            c=delta,
            s=5,
            cmap="RdBu_r",
            norm=norm,
            linewidths=0,
            alpha=0.9,
        )
        x_span = float(np.nanmax(x) - np.nanmin(x))
        y_span = float(np.nanmax(y) - np.nanmin(y))
        x_span = x_span if x_span > 0.0 else 1.0
        y_span = y_span if y_span > 0.0 else 1.0
        ax.set_xlim(float(np.nanmin(x) - 0.05 * x_span), float(np.nanmax(x) + 0.05 * x_span))
        ax.set_ylim(float(np.nanmin(y) - 0.05 * y_span), float(np.nanmax(y) + 0.05 * y_span))

        cb = fig.colorbar(scatter, cax=cax)
        cb.set_label(r"$\Delta$ (norm)", rotation=90, fontsize=8, labelpad=8)
        cb.ax.yaxis.set_label_position("left")
        cb.ax.yaxis.set_ticks_position("right")
        cb.ax.tick_params(labelsize=7)

        _kde_fill(ax_top, x, "x")
        _kde_fill(ax_right, y, "y")
        ax_top.axis("off")
        ax_right.axis("off")

        x_code = panel["x_code"]
        y_code = panel["y_code"]
        ax.set_xlabel(MEASURE_LABELS.get(x_code, f"log {x_code}"), fontsize=9)
        ax.set_ylabel(MEASURE_LABELS.get(y_code, f"log {y_code}"), fontsize=9)
        ax_top.set_title(
            "Group $\\Delta$ Attention\n"
            f"n={len(panel['rows'])} edges ($\\geq${prevalence:.0%} prevalence)",
            fontsize=7,
            pad=2,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        ax.text(
            0.5,
            -0.19,
            f"({chr(ord('A') + idx)})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=16,
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)


def _write_panel_outputs(panels: list[dict], out_file: Path, args: argparse.Namespace) -> None:
    for idx, panel in enumerate(panels):
        letter = chr(ord("A") + idx)
        csv_path = out_file.with_name(f"{out_file.stem}_panel_{letter}_edges.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(panel["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(panel["rows"])

    meta = {
        "figure": str(out_file),
        "split": args.split,
        "prevalence": args.prevalence,
        "label_positive": args.label_positive,
        "label_negative": args.label_negative,
        "panels": [
            {
                "run_dir": panel["run_dir"],
                "x_code": panel["x_code"],
                "y_code": panel["y_code"],
                "n_edges": len(panel["rows"]),
                "n_subjects": panel["n_subjects"],
                "label_subjects": panel["label_subjects"],
            }
            for panel in panels
        ],
    }
    with out_file.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.prevalence <= 1.0:
        raise ValueError("--prevalence must be in (0, 1]")

    device = _device(args.device)
    out_file = _resolve_out_file(args)
    panels = [
        collect_run_panel(Path(run_dir), args, device)
        for run_dir in args.run_dirs
    ]
    plot_panels(panels, out_file, args.prevalence, args.dpi)
    _write_panel_outputs(panels, out_file, args)
    print(f"Saved figure: {out_file}")
    print(f"Saved metadata: {out_file.with_suffix('.json')}")


if __name__ == "__main__":
    main()
