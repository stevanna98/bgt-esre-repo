"""Fold-wise ComBat-style harmonization for connectivity matrices.

This module implements the location/scale part of ComBat for dense subject x
feature matrices. It is deliberately fit only on training subjects and then
applied to validation subjects to avoid cross-validation leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.data.loaders import SubjectRecord


@dataclass
class CombatModel:
    """Parameters fitted on a training fold."""

    beta: np.ndarray
    pooled_mean: np.ndarray
    pooled_std: np.ndarray
    site_mean: dict[str, np.ndarray]
    site_std: dict[str, np.ndarray]
    feature_count: int
    n_regions: int
    triu_idx: tuple[np.ndarray, np.ndarray]
    fisher_z: bool
    preserve_label: bool


def harmonize_subject_connectivity(
    subjects: Sequence[SubjectRecord],
    train_indices: Sequence[int],
    target_indices: Sequence[int],
    *,
    preserve_label: bool = False,
    fisher_z: bool = True,
    eps: float = 1e-6,
) -> tuple[list[SubjectRecord], dict]:
    """Return target subjects with FC matrices harmonized by training sites.

    Args:
        subjects: Full subject list.
        train_indices: Indices used to fit site effects.
        target_indices: Indices to transform, usually train or validation.
        preserve_label: Include the class label as a biological covariate.
            This is disabled by default because using validation labels in a
            preprocessing transform is usually not appropriate for model
            selection or deployment.
        fisher_z: Apply Fisher z before harmonizing correlations and invert
            back to correlation scale afterward.
        eps: Numerical floor for standard deviations.

    Returns:
        ``(harmonized_subjects, summary)``.
    """
    sites = _subject_sites(subjects)
    train_sites = sites[np.asarray(train_indices)]
    unique_train_sites = sorted(set(train_sites.tolist()))
    if len(unique_train_sites) < 2:
        raise ValueError(
            "ComBat harmonization needs at least two acquisition sites in the "
            f"training fold, got {unique_train_sites}"
        )

    fc_stack = np.stack([subjects[i].fc for i in train_indices], axis=0)
    model = fit_combat_fc(
        fc_stack,
        train_sites,
        labels=np.array([subjects[i].label for i in train_indices]),
        preserve_label=preserve_label,
        fisher_z=fisher_z,
        eps=eps,
    )

    target_fc = np.stack([subjects[i].fc for i in target_indices], axis=0)
    target_sites = sites[np.asarray(target_indices)]
    target_labels = np.array([subjects[i].label for i in target_indices])
    harmonized_fc = transform_combat_fc(
        target_fc,
        target_sites,
        model,
        labels=target_labels,
        eps=eps,
    )

    harmonized_subjects = [
        subjects[idx]._replace(fc=harmonized_fc[pos].astype(np.float32))
        for pos, idx in enumerate(target_indices)
    ]
    unseen_sites = sorted(set(target_sites.tolist()) - set(unique_train_sites))
    summary = {
        "train_sites": unique_train_sites,
        "target_sites": sorted(set(target_sites.tolist())),
        "unseen_target_sites": unseen_sites,
        "preserve_label": preserve_label,
        "fisher_z": fisher_z,
    }
    return harmonized_subjects, summary


def fit_combat_fc(
    fc_stack: np.ndarray,
    sites: Sequence[str],
    *,
    labels: np.ndarray | None = None,
    preserve_label: bool = False,
    fisher_z: bool = True,
    eps: float = 1e-6,
) -> CombatModel:
    """Fit ComBat-style parameters on FC upper-triangle features."""
    if fc_stack.ndim != 3 or fc_stack.shape[1] != fc_stack.shape[2]:
        raise ValueError(f"fc_stack must have shape (S, N, N), got {fc_stack.shape}")
    n_subjects, n_regions, _ = fc_stack.shape
    if len(sites) != n_subjects:
        raise ValueError("sites length must match number of subjects")

    triu_idx = np.triu_indices(n_regions, k=1)
    x = fc_stack[:, triu_idx[0], triu_idx[1]].astype(np.float64, copy=False)
    if fisher_z:
        x = np.arctanh(np.clip(x, -1.0 + eps, 1.0 - eps))

    design = _design_matrix(
        n_subjects,
        labels=labels,
        preserve_label=preserve_label,
    )
    beta = np.linalg.pinv(design) @ x
    residual = x - design @ beta

    pooled_mean = residual.mean(axis=0)
    pooled_std = residual.std(axis=0)
    pooled_std = np.maximum(pooled_std, eps)

    site_arr = _normalise_sites(sites)
    site_mean: dict[str, np.ndarray] = {}
    site_std: dict[str, np.ndarray] = {}
    for site in sorted(set(site_arr.tolist())):
        mask = site_arr == site
        site_residual = residual[mask]
        site_mean[site] = site_residual.mean(axis=0)
        site_std[site] = np.maximum(site_residual.std(axis=0), eps)

    return CombatModel(
        beta=beta,
        pooled_mean=pooled_mean,
        pooled_std=pooled_std,
        site_mean=site_mean,
        site_std=site_std,
        feature_count=x.shape[1],
        n_regions=n_regions,
        triu_idx=triu_idx,
        fisher_z=fisher_z,
        preserve_label=preserve_label,
    )


def transform_combat_fc(
    fc_stack: np.ndarray,
    sites: Sequence[str],
    model: CombatModel,
    *,
    labels: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply fitted ComBat-style parameters to FC matrices."""
    if fc_stack.ndim != 3 or fc_stack.shape[1] != model.n_regions:
        raise ValueError(
            f"fc_stack must have shape (S, {model.n_regions}, {model.n_regions}), "
            f"got {fc_stack.shape}"
        )

    n_subjects = fc_stack.shape[0]
    x = fc_stack[:, model.triu_idx[0], model.triu_idx[1]].astype(
        np.float64,
        copy=False,
    )
    if model.fisher_z:
        x = np.arctanh(np.clip(x, -1.0 + eps, 1.0 - eps))

    design = _design_matrix(
        n_subjects,
        labels=labels,
        preserve_label=model.preserve_label,
    )
    covariate_part = design @ model.beta
    residual = x - covariate_part

    site_arr = _normalise_sites(sites)
    adjusted = np.empty_like(residual)
    for site in sorted(set(site_arr.tolist())):
        mask = site_arr == site
        if site in model.site_mean:
            adjusted[mask] = (
                (residual[mask] - model.site_mean[site])
                / model.site_std[site]
                * model.pooled_std
                + model.pooled_mean
            )
        else:
            # A site absent from the training fold has no estimable ComBat
            # parameters. Leave its residuals on the model's pooled scale.
            adjusted[mask] = residual[mask]

    harmonized_features = covariate_part + adjusted
    if model.fisher_z:
        harmonized_features = np.tanh(harmonized_features)

    out = np.zeros((n_subjects, model.n_regions, model.n_regions), dtype=np.float64)
    out[:, model.triu_idx[0], model.triu_idx[1]] = harmonized_features
    out[:, model.triu_idx[1], model.triu_idx[0]] = harmonized_features
    diag = np.arange(model.n_regions)
    out[:, diag, diag] = 1.0
    return out


def _subject_sites(subjects: Sequence[SubjectRecord]) -> np.ndarray:
    sites = [subject.site for subject in subjects]
    if any(site is None for site in sites):
        raise ValueError(
            "ComBat harmonization requires ABIDE site labels. Add one of "
            "sites.npy, site.npy, site_ids.npy, site_labels.npy, batch.npy, "
            "or batches.npy to the ABIDE data directory, or pass "
            "--combat-site-file."
        )
    return _normalise_sites(sites)


def _normalise_sites(sites: Iterable[object]) -> np.ndarray:
    return np.asarray([str(site) for site in sites])


def _design_matrix(
    n_subjects: int,
    *,
    labels: np.ndarray | None,
    preserve_label: bool,
) -> np.ndarray:
    cols = [np.ones((n_subjects, 1), dtype=np.float64)]
    if preserve_label:
        if labels is None:
            raise ValueError("labels are required when preserve_label=True")
        label_col = np.asarray(labels, dtype=np.float64).reshape(n_subjects, 1)
        label_col = label_col - label_col.mean(axis=0, keepdims=True)
        cols.append(label_col)
    return np.concatenate(cols, axis=1)
