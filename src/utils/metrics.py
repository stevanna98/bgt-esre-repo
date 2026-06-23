"""Classification metric helpers.

All functions accept raw numpy logits + integer labels and return a flat dict.
AUC is nan (not a crash) when only one class is present in the batch.
Sensitivity / specificity are macro-averaged for multiclass.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 2,
) -> dict:
    """Compute accuracy, AUC, weighted-F1, sensitivity, specificity.

    Args:
        logits:      (N, C) unnormalized scores.
        labels:      (N,) integer ground-truth.
        num_classes: number of classes C.

    Returns:
        dict with float values; AUC is nan when undefined.
    """
    probs = _softmax(logits)          # (N, C)
    preds = logits.argmax(axis=1)     # (N,)

    acc  = float(accuracy_score(labels, preds))
    f1   = float(f1_score(labels, preds, average="weighted", zero_division=0))
    auc  = _safe_auc(labels, probs, num_classes)
    sens, spec = _sens_spec(labels, preds, num_classes)

    return dict(accuracy=acc, auc=auc, f1=f1, sensitivity=sens, specificity=spec)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def _safe_auc(
    labels: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        if num_classes == 2:
            return float(roc_auc_score(labels, probs[:, 1]))
        return float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        return float("nan")


def _sens_spec(
    labels: np.ndarray,
    preds: np.ndarray,
    num_classes: int,
) -> tuple[float, float]:
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    if num_classes == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        return float(sens), float(spec)

    # Multiclass: one-vs-rest sensitivity/specificity, macro average
    total = cm.sum()
    sens_list, spec_list = [], []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else float("nan"))
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else float("nan"))

    return float(np.nanmean(sens_list)), float(np.nanmean(spec_list))
