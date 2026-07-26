#!/usr/bin/env python
"""Generate ABIDE pairwise ecospace figures.

This script reproduces the two HCP-style ecospace summaries for ABIDE:

1. Edge-averaged ecospace:
   one point per edge per class, averaged over subjects where that edge exists,
   keeping only edges whose prevalence is at least the requested threshold.

2. Subject-level ecospace:
   one point per subject, using that subject's mean log-measure value across
   its thresholded edges, with class covariance ellipses and centroids.

Example:
    python scripts/plot_abide_ecospace.py \
        --data-dir "/path/to/ABIDE/combat_ready" \
        --run-dir "/path/to/outputs_ediff_comm" \
        --mode both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
os.environ.setdefault("XDG_CACHE_HOME", str(_REPO / ".cache"))
os.environ.setdefault("XDG_CONFIG_HOME", str(_REPO / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_REPO / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.spatial.distance import cdist
from tqdm import tqdm

from src.data.loaders import load_dataset
from src.precompute.betweenness import compute_edge_betweenness_centrality
from src.precompute.clustering import compute_edge_clustering_coefficient
from src.precompute.communicability import compute_communicability
from src.precompute.connectivity import proportional_threshold
from src.precompute.diffusion import compute_diffusion_efficiency
from src.precompute.routing import compute_routing_efficiency


MEASURES = [
    ("ediff", "Log Diffusion Eff.", compute_diffusion_efficiency),
    ("erout", "Log Routing Eff.", compute_routing_efficiency),
    ("ebc", "Log Edge Betweenness", compute_edge_betweenness_centrality),
    ("ecc", "Log Edge Clustering", compute_edge_clustering_coefficient),
    ("comm", "Log Communicability", compute_communicability),
]
MEASURE_LABEL = {code: label for code, label, _ in MEASURES}
MEASURE_CODE_TO_INDEX = {code: idx for idx, (code, _, _) in enumerate(MEASURES)}

DEFAULT_LABEL_NAMES = {
    0: "Control",
    1: "ASD",
}
DEFAULT_LABEL_COLORS = {
    0: "tab:blue",
    1: "tab:red",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ABIDE all-pairwise ecospace.")
    p.add_argument("--data-dir", required=True, help="ABIDE directory with .npy arrays.")
    p.add_argument(
        "--run-dir",
        default=None,
        help="Optional train run directory. If present, graph settings are read "
             "from resolved_args.json.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to run_dir/plots/ecospace or data_dir/ecospace_plots.",
    )
    p.add_argument(
        "--mode",
        choices=["edge", "subject", "both"],
        default="both",
        help="Which attached-style figure(s) to generate.",
    )
    p.add_argument("--prevalence", type=float, default=0.50)
    p.add_argument("--threshold-pct", type=float, default=None)
    p.add_argument(
        "--weight-mode",
        choices=["binary", "fc", "cost_penalised"],
        default=None,
    )
    p.add_argument(
        "--eco-lambda",
        default=None,
        help="Numeric lambda, 'auto', or omitted to inherit/default.",
    )
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--log-eps", type=float, default=1e-8)
    p.add_argument(
        "--cache-file",
        default=None,
        help="Optional .npz cache. Existing cache is reused unless --recompute is set.",
    )
    p.add_argument("--recompute", action="store_true")
    p.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Debug option: compute only the first N subjects.",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def _parse_eco_lambda(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"auto", "none", "null"}:
        return None
    return float(text)


def _load_run_args(run_dir: str | None) -> dict:
    if run_dir is None:
        return {}
    path = Path(run_dir) / "resolved_args.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing run config: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_graph_settings(args: argparse.Namespace) -> dict:
    run_args = _load_run_args(args.run_dir)
    threshold_pct = (
        args.threshold_pct
        if args.threshold_pct is not None
        else float(run_args.get("threshold_pct", 0.15))
    )
    weight_mode = (
        args.weight_mode
        if args.weight_mode is not None
        else str(run_args.get("weight_mode", "cost_penalised"))
    )
    eco_lambda = (
        _parse_eco_lambda(args.eco_lambda)
        if args.eco_lambda is not None
        else _parse_eco_lambda(run_args.get("eco_lambda", 0.1))
    )
    return {
        "threshold_pct": threshold_pct,
        "weight_mode": weight_mode,
        "eco_lambda": eco_lambda,
    }


def _resolve_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return Path(args.out_dir)
    if args.run_dir is not None:
        return Path(args.run_dir) / "plots" / "ecospace"
    return Path(args.data_dir) / "ecospace_plots"


def _resolve_cache_file(args: argparse.Namespace, out_dir: Path) -> Path:
    if args.cache_file is not None:
        return Path(args.cache_file)
    return out_dir / "abide_ecospace_cache.npz"


def _weight_matrix(
    fc: np.ndarray,
    coords: np.ndarray,
    threshold_pct: float,
    weight_mode: str,
    eco_lambda: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    fc = np.clip(fc, 0.0, None)
    np.fill_diagonal(fc, 0.0)
    A, Aw = proportional_threshold(fc, threshold_pct)
    if weight_mode == "binary":
        return A, A
    if weight_mode == "fc":
        return A, Aw

    dist_mat = cdist(coords, coords, metric="euclidean")
    lam = eco_lambda
    if lam is None:
        connected_dists = dist_mat[A > 0]
        d_bar = connected_dists.mean() if len(connected_dists) else 1.0
        lam = 1.0 / (d_bar + 1e-12)
    W = Aw * np.exp(-lam * dist_mat)
    np.fill_diagonal(W, 0.0)
    return A, W


def _compute_log_measures(
    W: np.ndarray,
    log_eps: float,
    eps: float,
) -> np.ndarray:
    values = []
    for _, _, fn in MEASURES:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            mat = fn(W, eps)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        values.append(np.log(np.clip(mat, log_eps, None)).astype(np.float32))
    return np.stack(values, axis=0)


def compute_ecospace_cache(
    data_dir: str,
    graph_settings: dict,
    *,
    eps: float,
    log_eps: float,
    max_subjects: int | None = None,
) -> dict:
    subjects, coords = load_dataset("abide", data_dir)
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    labels = np.array([subject.label for subject in subjects], dtype=np.int64)
    unique_labels = np.array(sorted(set(labels.tolist())), dtype=np.int64)
    label_to_pos = {int(label): pos for pos, label in enumerate(unique_labels)}

    n_subjects = len(subjects)
    n_regions = coords.shape[0]
    n_measures = len(MEASURES)
    triu = np.triu_indices(n_regions, k=1)
    n_edges = len(triu[0])
    n_labels = len(unique_labels)

    present_count = np.zeros(n_edges, dtype=np.int32)
    edge_sum_by_label = np.zeros((n_labels, n_measures, n_edges), dtype=np.float64)
    edge_count_by_label = np.zeros((n_labels, n_edges), dtype=np.int32)
    subject_values = np.full((n_subjects, n_measures), np.nan, dtype=np.float32)

    for subj_idx, subject in enumerate(tqdm(subjects, desc="Computing ABIDE ecospace", unit="subj")):
        A, W = _weight_matrix(
            subject.fc.copy(),
            coords,
            graph_settings["threshold_pct"],
            graph_settings["weight_mode"],
            graph_settings["eco_lambda"],
        )
        log_measures = _compute_log_measures(W, log_eps=log_eps, eps=eps)
        edge_mask = A[triu] > 0
        if not np.any(edge_mask):
            continue

        edge_indices = np.flatnonzero(edge_mask)
        edge_values = log_measures[:, triu[0][edge_indices], triu[1][edge_indices]]
        present_count[edge_mask] += 1
        label_pos = label_to_pos[int(subject.label)]
        edge_count_by_label[label_pos, edge_mask] += 1
        edge_sum_by_label[label_pos][:, edge_indices] += edge_values
        subject_values[subj_idx] = edge_values.mean(axis=1)

    return {
        "labels": labels,
        "unique_labels": unique_labels,
        "subject_values": subject_values,
        "present_count": present_count,
        "edge_sum_by_label": edge_sum_by_label,
        "edge_count_by_label": edge_count_by_label,
        "triu_rows": triu[0].astype(np.int32),
        "triu_cols": triu[1].astype(np.int32),
        "measure_codes": np.array([code for code, _, _ in MEASURES]),
        "n_subjects": np.array([n_subjects], dtype=np.int64),
        "n_regions": np.array([n_regions], dtype=np.int64),
        "threshold_pct": np.array([graph_settings["threshold_pct"]], dtype=np.float64),
        "weight_mode": np.array([graph_settings["weight_mode"]]),
        "eco_lambda": np.array([
            np.nan if graph_settings["eco_lambda"] is None else graph_settings["eco_lambda"]
        ], dtype=np.float64),
    }


def _save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cache)


def _load_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _label_name(label: int) -> str:
    return DEFAULT_LABEL_NAMES.get(label, f"Class {label}")


def _label_color(label: int) -> str:
    return DEFAULT_LABEL_COLORS.get(label, f"C{label}")


def _pair_axes() -> list[tuple[str, str]]:
    codes = [code for code, _, _ in MEASURES]
    return list(combinations(codes, 2))


def _style_grid_axis(ax) -> None:
    ax.grid(True, color="0.82", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("0.75")
    ax.tick_params(labelsize=7)


def plot_edge_ecospace(cache: dict, out_path: Path, prevalence: float, dpi: int) -> None:
    labels = cache["labels"]
    unique_labels = [int(x) for x in cache["unique_labels"]]
    n_subjects = int(cache["n_subjects"][0])
    present_count = cache["present_count"]
    edge_sum_by_label = cache["edge_sum_by_label"]
    edge_count_by_label = cache["edge_count_by_label"]
    keep = present_count >= int(np.ceil(prevalence * n_subjects))
    n_kept = int(keep.sum())

    pairs = _pair_axes()
    fig, axes = plt.subplots(2, 5, figsize=(30, 10), dpi=dpi)
    fig.suptitle(
        "ABIDE -- All Pairwise Ecospace\n"
        "(one point per edge, averaged over subjects where edge exists)\n"
        f"(prevalence >= {prevalence:.0%})",
        fontsize=17,
        y=0.985,
    )

    for ax, (x_code, y_code) in zip(axes.ravel(), pairs):
        x_idx = MEASURE_CODE_TO_INDEX[x_code]
        y_idx = MEASURE_CODE_TO_INDEX[y_code]
        for label_pos, label in enumerate(unique_labels):
            count = edge_count_by_label[label_pos]
            valid = keep & (count > 0)
            x = edge_sum_by_label[label_pos, x_idx, valid] / count[valid]
            y = edge_sum_by_label[label_pos, y_idx, valid] / count[valid]
            ax.scatter(
                x,
                y,
                s=4,
                alpha=0.35,
                c=_label_color(label),
                edgecolors="none",
                label=_label_name(label),
            )
        ax.set_title(
            f"{MEASURE_LABEL[x_code]}\nvs {MEASURE_LABEL[y_code]}",
            fontsize=8,
        )
        ax.set_xlabel(MEASURE_LABEL[x_code], fontsize=8)
        ax.set_ylabel(MEASURE_LABEL[y_code], fontsize=8)
        _style_grid_axis(ax)

    axes.ravel()[0].legend(fontsize=8, loc="upper left", markerscale=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    meta_path = out_path.with_suffix(".json")
    meta = {
        "figure": str(out_path),
        "mode": "edge",
        "n_subjects": n_subjects,
        "n_edges_prevalence_filtered": n_kept,
        "prevalence": prevalence,
        "label_counts": {
            _label_name(int(label)): int((labels == int(label)).sum())
            for label in unique_labels
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _covariance_ellipse(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    n_std: float = 1.8,
) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    vals = np.clip(vals, 0.0, None)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(vals)
    center = (float(np.mean(x)), float(np.mean(y)))
    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.20,
        linewidth=2.0,
    )
    ax.add_patch(ellipse)
    ax.scatter(
        [center[0]],
        [center[1]],
        marker="D",
        s=70,
        facecolor=color,
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
    )


def plot_subject_ecospace(cache: dict, out_path: Path, dpi: int) -> None:
    labels = cache["labels"]
    unique_labels = [int(x) for x in cache["unique_labels"]]
    subject_values = cache["subject_values"]
    pairs = _pair_axes()
    n_subjects = int(cache["n_subjects"][0])

    fig, axes = plt.subplots(2, 5, figsize=(30, 10), dpi=dpi)
    fig.suptitle(
        "ABIDE -- All Pairwise Ecospace\n"
        "(one point per subject)\n"
        f"(n={n_subjects} subjects)",
        fontsize=17,
        y=0.985,
    )

    for ax, (x_code, y_code) in zip(axes.ravel(), pairs):
        x_idx = MEASURE_CODE_TO_INDEX[x_code]
        y_idx = MEASURE_CODE_TO_INDEX[y_code]
        for label in unique_labels:
            mask = labels == label
            x = subject_values[mask, x_idx]
            y = subject_values[mask, y_idx]
            ax.scatter(
                x,
                y,
                s=12,
                alpha=0.35,
                c=_label_color(label),
                label=_label_name(label),
            )
            _covariance_ellipse(ax, x, y, color=_label_color(label))
        ax.set_xlabel(MEASURE_LABEL[x_code], fontsize=8)
        ax.set_ylabel(MEASURE_LABEL[y_code], fontsize=8)
        _style_grid_axis(ax)

    axes.ravel()[0].legend(fontsize=9, loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    meta_path = out_path.with_suffix(".json")
    meta = {
        "figure": str(out_path),
        "mode": "subject",
        "n_subjects": n_subjects,
        "label_counts": {
            _label_name(int(label)): int((labels == int(label)).sum())
            for label in unique_labels
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.prevalence <= 1.0:
        raise ValueError("--prevalence must be in (0, 1]")

    out_dir = _resolve_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _resolve_cache_file(args, out_dir)

    if cache_file.is_file() and not args.recompute:
        print(f"Loading ecospace cache: {cache_file}")
        cache = _load_cache(cache_file)
    else:
        graph_settings = _resolve_graph_settings(args)
        print(
            "Computing ecospace with "
            f"threshold_pct={graph_settings['threshold_pct']}, "
            f"weight_mode={graph_settings['weight_mode']}, "
            f"eco_lambda={graph_settings['eco_lambda']}"
        )
        cache = compute_ecospace_cache(
            args.data_dir,
            graph_settings,
            eps=args.eps,
            log_eps=args.log_eps,
            max_subjects=args.max_subjects,
        )
        _save_cache(cache, cache_file)
        print(f"Saved ecospace cache: {cache_file}")

    if args.mode in {"edge", "both"}:
        path = out_dir / "abide_all_pairwise_ecospace_edges.png"
        plot_edge_ecospace(cache, path, args.prevalence, args.dpi)
        print(f"Saved edge-level figure: {path}")
    if args.mode in {"subject", "both"}:
        path = out_dir / "abide_all_pairwise_ecospace_subjects.png"
        plot_subject_ecospace(cache, path, args.dpi)
        print(f"Saved subject-level figure: {path}")


if __name__ == "__main__":
    main()
