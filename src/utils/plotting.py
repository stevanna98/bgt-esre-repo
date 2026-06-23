"""Pure matplotlib plotting utilities for training diagnostics.

Every function:
  - Writes a PNG to *out_path* (or one PNG per label into *out_dir*).
  - Overwrites any existing file — no history accumulates.
  - Calls plt.close(fig) after saving.
  - Creates parent directories automatically.

No seaborn dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_train_loss(
    history: Dict[str, List[float]],
    out_path: str | Path,
) -> None:
    """Training loss curve (train split only)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vals = history.get("train_loss", [])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(vals) + 1), vals, color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_metric_train_val(
    history: Dict[str, List[float]],
    metric_name: str,
    out_path: str | Path,
) -> None:
    """Train + validation curve for a single metric."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_vals = history.get(f"train_{metric_name}", [])
    val_vals   = history.get(f"val_{metric_name}", [])
    n = max(len(train_vals), len(val_vals))
    epochs = list(range(1, n + 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    if train_vals:
        ax.plot(epochs[: len(train_vals)], train_vals,
                color="tab:blue", label="train")
    if val_vals:
        ax.plot(epochs[: len(val_vals)], val_vals,
                color="tab:orange", linestyle="--", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f"{metric_name.capitalize()} — train vs val")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_attention_heatmap(
    matrix: np.ndarray,
    out_path: str | Path,
    title: str = "Edge Attention",
) -> None:
    """Symmetric (N, N) attention heatmap with colorbar."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vmax = float(np.nanpercentile(matrix, 99)) or 1.0
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=0.0, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Region (destination)")
    ax.set_ylabel("Region (source)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_attention_per_label(
    matrices_by_label: Dict[int, np.ndarray],
    out_dir: str | Path,
) -> None:
    """One heatmap per class label → attn_label_{k}.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, matrix in sorted(matrices_by_label.items()):
        plot_attention_heatmap(
            matrix,
            out_dir / f"attn_label_{label}.png",
            title=f"Edge Attention — class {label}",
        )
