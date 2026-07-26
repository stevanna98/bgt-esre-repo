#!/usr/bin/env python
"""Plot ABIDE functional-connectivity matrices.

Creates heatmaps like the attached examples from the ABIDE ``connectivity.npy``:

  - mean FC across all subjects
  - mean FC per label
  - optional label-difference FC

By default the script plots signed FC values with a red/blue color scale.
Use ``--positive-only`` to clip negative correlations to zero, matching the
graph-building pipeline used by training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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


DEFAULT_LABEL_NAMES = {
    0: "Control",
    1: "ASD",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ABIDE mean FC heatmaps.")
    p.add_argument("--data-dir", required=True, help="ABIDE directory with connectivity.npy and labels.npy.")
    p.add_argument(
        "--run-dir",
        default=None,
        help="Optional run directory. Output defaults to run_dir/plots/fc_matrix.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to run_dir/plots/fc_matrix or data_dir/fc_matrix_plots.",
    )
    p.add_argument(
        "--positive-only",
        action="store_true",
        help="Clip negative FC values to zero and use a red-only color scale.",
    )
    p.add_argument(
        "--signed",
        action="store_true",
        help="Deprecated compatibility flag. Signed FC is now the default.",
    )
    p.add_argument(
        "--include-diagonal",
        action="store_true",
        help="Keep diagonal values. Default sets diagonal to zero before averaging.",
    )
    p.add_argument(
        "--robust",
        action="store_true",
        help="Use percentile color limits instead of fixed min/max.",
    )
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--figsize", default=None, help="Figure size as WIDTH,HEIGHT inches.")
    p.add_argument("--prefix", default="abide_fc")
    return p.parse_args()


def _resolve_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return Path(args.out_dir)
    if args.run_dir is not None:
        return Path(args.run_dir) / "plots" / "fc_matrix"
    return Path(args.data_dir) / "fc_matrix_plots"


def _parse_figsize(value: str | None, n_regions: int) -> tuple[float, float]:
    if value is not None:
        width, height = value.split(",", 1)
        return float(width), float(height)
    if n_regions <= 80:
        return 6.0, 5.2
    return 7.2, 6.2


def _prepare_fc(fc: np.ndarray, *, signed: bool, include_diagonal: bool) -> np.ndarray:
    fc = np.nan_to_num(fc.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    if not signed:
        fc = np.clip(fc, 0.0, None)
    if not include_diagonal:
        diag = np.arange(fc.shape[-1])
        fc[:, diag, diag] = 0.0
    return fc


def _mean_by_label(fc: np.ndarray, labels: np.ndarray) -> dict[int | str, np.ndarray]:
    means: dict[int | str, np.ndarray] = {"all": fc.mean(axis=0)}
    for label in sorted(set(labels.tolist())):
        means[int(label)] = fc[labels == label].mean(axis=0)
    return means


def _limits(matrix: np.ndarray, *, signed: bool, robust: bool, vmin: float | None, vmax: float | None) -> tuple[float, float]:
    if vmin is not None and vmax is not None:
        return vmin, vmax
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return (-1.0, 1.0) if signed else (0.0, 1.0)

    if signed:
        if robust:
            lim = float(np.nanpercentile(np.abs(finite), 99))
        else:
            lim = float(np.nanmax(np.abs(finite)))
        lim = lim if lim > 0.0 else 1.0
        return (vmin if vmin is not None else -lim, vmax if vmax is not None else lim)

    upper = (
        float(np.nanpercentile(finite, 99))
        if robust
        else float(np.nanmax(finite))
    )
    upper = upper if upper > 0.0 else 1.0
    return (vmin if vmin is not None else 0.0, vmax if vmax is not None else upper)


def _plot_matrix(
    matrix: np.ndarray,
    out_path: Path,
    *,
    title: str,
    signed: bool,
    robust: bool,
    vmin: float | None,
    vmax: float | None,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = _limits(matrix, signed=signed, robust=robust, vmin=vmin, vmax=vmax)
    cmap = "RdBu_r" if signed else "Reds"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(matrix, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Region", fontsize=11)
    ax.set_ylabel("Region", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_grid(
    means: dict[int | str, np.ndarray],
    labels: np.ndarray,
    out_path: Path,
    *,
    signed: bool,
    robust: bool,
    vmin: float | None,
    vmax: float | None,
    dpi: int,
) -> None:
    keys = ["all"] + [int(label) for label in sorted(set(labels.tolist()))]
    n_cols = len(keys)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.4 * n_cols, 4.8), dpi=dpi)
    if n_cols == 1:
        axes = [axes]

    all_values = np.concatenate([means[key].ravel() for key in keys])
    lo, hi = _limits(all_values, signed=signed, robust=robust, vmin=vmin, vmax=vmax)
    cmap = "RdBu_r" if signed else "Reds"
    im = None
    for ax, key in zip(axes, keys):
        matrix = means[key]
        im = ax.imshow(matrix, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        if key == "all":
            title = f"All subjects (n={len(labels)})"
        else:
            title = f"{DEFAULT_LABEL_NAMES.get(key, f'Class {key}')} (n={int((labels == key).sum())})"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Region", fontsize=10)
        ax.set_ylabel("Region", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle("ABIDE Mean Functional Connectivity", fontsize=14)
    fig.subplots_adjust(left=0.04, right=0.94, bottom=0.10, top=0.86, wspace=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    out_dir = _resolve_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    fc_path = data_dir / "connectivity.npy"
    labels_path = data_dir / "labels.npy"
    if not fc_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"Expected connectivity.npy and labels.npy in {data_dir}"
        )

    fc = np.load(fc_path)
    labels = np.load(labels_path).astype(int).reshape(-1)
    if fc.shape[0] != labels.shape[0]:
        raise ValueError(
            f"connectivity has {fc.shape[0]} subjects but labels has {labels.shape[0]}"
        )

    signed = not args.positive_only
    fc = _prepare_fc(fc, signed=signed, include_diagonal=args.include_diagonal)
    means = _mean_by_label(fc, labels)
    n_regions = int(fc.shape[-1])
    figsize = _parse_figsize(args.figsize, n_regions)
    mode_tag = "signed" if signed else "positive"

    saved = []
    for key, matrix in means.items():
        if key == "all":
            title = f"ABIDE Mean FC - All Subjects (n={len(labels)})"
            stem = f"{args.prefix}_mean_all_{mode_tag}"
        else:
            name = DEFAULT_LABEL_NAMES.get(int(key), f"class_{key}")
            title = f"ABIDE Mean FC - {name} (n={int((labels == key).sum())})"
            stem = f"{args.prefix}_mean_label_{key}_{mode_tag}"
        np.save(out_dir / f"{stem}.npy", matrix.astype(np.float32))
        png_path = out_dir / f"{stem}.png"
        _plot_matrix(
            matrix,
            png_path,
            title=title,
            signed=signed,
            robust=args.robust,
            vmin=args.vmin,
            vmax=args.vmax,
            dpi=args.dpi,
            figsize=figsize,
        )
        saved.append(str(png_path))

    unique_labels = sorted(set(labels.tolist()))
    if len(unique_labels) == 2:
        low, high = unique_labels
        diff = means[high] - means[low]
        stem = f"{args.prefix}_mean_difference_label_{high}_minus_{low}_{mode_tag}"
        np.save(out_dir / f"{stem}.npy", diff.astype(np.float32))
        png_path = out_dir / f"{stem}.png"
        _plot_matrix(
            diff,
            png_path,
            title=(
                "ABIDE Mean FC Difference - "
                f"{DEFAULT_LABEL_NAMES.get(high, high)} minus {DEFAULT_LABEL_NAMES.get(low, low)}"
            ),
            signed=True,
            robust=args.robust,
            vmin=args.vmin,
            vmax=args.vmax,
            dpi=args.dpi,
            figsize=figsize,
        )
        saved.append(str(png_path))

    grid_path = out_dir / f"{args.prefix}_mean_grid_{mode_tag}.png"
    _save_grid(
        means,
        labels,
        grid_path,
            signed=signed,
        robust=args.robust,
        vmin=args.vmin,
        vmax=args.vmax,
        dpi=args.dpi,
    )
    saved.append(str(grid_path))

    meta = {
        "data_dir": str(data_dir),
        "n_subjects": int(fc.shape[0]),
        "n_regions": n_regions,
        "signed": bool(signed),
        "positive_only": bool(args.positive_only),
        "include_diagonal": bool(args.include_diagonal),
        "label_counts": {
            DEFAULT_LABEL_NAMES.get(int(label), f"Class {label}"): int((labels == label).sum())
            for label in unique_labels
        },
        "saved": saved,
    }
    meta_path = out_dir / f"{args.prefix}_metadata_{mode_tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved FC matrix plots to: {out_dir}")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
