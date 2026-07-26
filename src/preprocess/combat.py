"""Fold-wise parametric empirical-Bayes ComBat for connectivity matrices.

The implementation follows the location/scale ComBat model of Johnson,
Li, and Rabinovic (2007), as adapted by neuroCombat. Functional-connectivity
edges are treated as features. All regression coefficients, hyperpriors, and
posterior site effects are estimated from a training fold only and can then be
applied unchanged to validation subjects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.data.loaders import SubjectRecord


@dataclass
class CombatModel:
    """Parametric empirical-Bayes ComBat parameters fitted on a training fold.

    ``site_mean`` and ``site_std`` contain the posterior additive effect
    ``gamma_star`` and posterior multiplicative standard deviation
    ``sqrt(delta_star)`` on standardized data. The longer explicit names retain
    both the raw and posterior estimates for diagnostics.
    """

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
    label_mean: float | None
    gamma_hat: dict[str, np.ndarray]
    delta_hat: dict[str, np.ndarray]
    gamma_star: dict[str, np.ndarray]
    delta_star: dict[str, np.ndarray]
    gamma_bar: dict[str, float]
    t2: dict[str, float]
    a_prior: dict[str, float]
    b_prior: dict[str, float]
    iterations: dict[str, int]
    converged: dict[str, bool]


def harmonize_subject_connectivity(
    subjects: Sequence[SubjectRecord],
    train_indices: Sequence[int],
    target_indices: Sequence[int],
    *,
    preserve_label: bool = False,
    fisher_z: bool = True,
    eps: float = 1e-6,
) -> tuple[list[SubjectRecord], dict]:
    """Return target subjects harmonized using training-fold estimates only."""
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
    target_labels = (
        np.array([subjects[i].label for i in target_indices])
        if preserve_label
        else None
    )
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
        "method": "parametric_empirical_bayes",
        "train_sites": unique_train_sites,
        "target_sites": sorted(set(target_sites.tolist())),
        "unseen_target_sites": unseen_sites,
        "preserve_label": preserve_label,
        "fisher_z": fisher_z,
        "converged": model.converged,
        "iterations": model.iterations,
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
    convergence_tol: float = 1e-4,
    max_iter: int = 1000,
) -> CombatModel:
    """Fit standard parametric empirical-Bayes ComBat to FC edge features.

    ComBat's additive site effects use a Normal prior across edges and its
    multiplicative variance effects use an inverse-gamma prior. Posterior
    estimates are obtained with the standard alternating updates.
    """
    if fc_stack.ndim != 3 or fc_stack.shape[1] != fc_stack.shape[2]:
        raise ValueError(f"fc_stack must have shape (S, N, N), got {fc_stack.shape}")
    n_subjects, n_regions, _ = fc_stack.shape
    if len(sites) != n_subjects:
        raise ValueError("sites length must match number of subjects")
    if n_subjects < 2:
        raise ValueError("ComBat needs at least two training subjects")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if convergence_tol <= 0:
        raise ValueError("convergence_tol must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    site_arr = _normalise_sites(sites)
    unique_sites = sorted(set(site_arr.tolist()))
    if len(unique_sites) < 2:
        raise ValueError(
            "ComBat needs at least two acquisition sites, "
            f"got {unique_sites}"
        )
    site_counts = {site: int(np.sum(site_arr == site)) for site in unique_sites}
    singleton_sites = [site for site, count in site_counts.items() if count < 2]
    if singleton_sites:
        raise ValueError(
            "ComBat scale effects require at least two training subjects per "
            "site; singleton site(s): " + ", ".join(singleton_sites)
        )

    triu_idx = np.triu_indices(n_regions, k=1)
    x = fc_stack[:, triu_idx[0], triu_idx[1]].astype(np.float64, copy=False)
    if x.shape[1] < 2:
        raise ValueError(
            "Empirical-Bayes ComBat needs at least two FC edge features"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("fc_stack contains non-finite upper-triangle values")
    if fisher_z:
        x = np.arctanh(np.clip(x, -1.0 + eps, 1.0 - eps))

    label_col, label_mean = _label_column(
        n_subjects,
        labels=labels,
        preserve_label=preserve_label,
    )
    batch_design = np.column_stack([site_arr == site for site in unique_sites]).astype(
        np.float64
    )
    design = (
        batch_design
        if label_col is None
        else np.column_stack((batch_design, label_col))
    )

    coefficients = np.linalg.pinv(design) @ x
    n_sites = len(unique_sites)
    site_coefficients = coefficients[:n_sites]
    site_weights = np.array(
        [site_counts[site] / n_subjects for site in unique_sites],
        dtype=np.float64,
    )
    grand_mean = site_weights @ site_coefficients
    covariate_beta = coefficients[n_sites:]
    beta = np.vstack((grand_mean, covariate_beta))

    fitted = design @ coefficients
    pooled_variance = np.mean(np.square(x - fitted), axis=0)
    pooled_std = np.sqrt(np.maximum(pooled_variance, eps**2))

    biological_mean = _biological_mean(
        n_subjects,
        beta,
        labels=labels,
        preserve_label=preserve_label,
        label_mean=label_mean,
    )
    standardized = (x - biological_mean) / pooled_std

    gamma_hat: dict[str, np.ndarray] = {}
    delta_hat: dict[str, np.ndarray] = {}
    gamma_star: dict[str, np.ndarray] = {}
    delta_star: dict[str, np.ndarray] = {}
    gamma_bar: dict[str, float] = {}
    t2: dict[str, float] = {}
    a_prior: dict[str, float] = {}
    b_prior: dict[str, float] = {}
    iterations: dict[str, int] = {}
    converged: dict[str, bool] = {}

    for site in unique_sites:
        site_data = standardized[site_arr == site]
        raw_gamma = site_data.mean(axis=0)
        raw_delta = np.maximum(site_data.var(axis=0, ddof=1), eps**2)

        gamma_location = float(raw_gamma.mean())
        gamma_variance = max(float(raw_gamma.var(ddof=1)), eps**2)
        prior_a, prior_b = _inverse_gamma_moments(raw_delta, eps=eps)
        post_gamma, post_delta, n_iter, did_converge = _parametric_posterior(
            site_data,
            raw_gamma,
            raw_delta,
            gamma_location,
            gamma_variance,
            prior_a,
            prior_b,
            eps=eps,
            convergence_tol=convergence_tol,
            max_iter=max_iter,
        )

        gamma_hat[site] = raw_gamma
        delta_hat[site] = raw_delta
        gamma_star[site] = post_gamma
        delta_star[site] = post_delta
        gamma_bar[site] = gamma_location
        t2[site] = gamma_variance
        a_prior[site] = prior_a
        b_prior[site] = prior_b
        iterations[site] = n_iter
        converged[site] = did_converge

    return CombatModel(
        beta=beta,
        pooled_mean=grand_mean,
        pooled_std=pooled_std,
        site_mean=gamma_star,
        site_std={
            site: np.sqrt(np.maximum(delta, eps**2))
            for site, delta in delta_star.items()
        },
        feature_count=x.shape[1],
        n_regions=n_regions,
        triu_idx=triu_idx,
        fisher_z=fisher_z,
        preserve_label=preserve_label,
        label_mean=label_mean,
        gamma_hat=gamma_hat,
        delta_hat=delta_hat,
        gamma_star=gamma_star,
        delta_star=delta_star,
        gamma_bar=gamma_bar,
        t2=t2,
        a_prior=a_prior,
        b_prior=b_prior,
        iterations=iterations,
        converged=converged,
    )


def transform_combat_fc(
    fc_stack: np.ndarray,
    sites: Sequence[str],
    model: CombatModel,
    *,
    labels: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply fixed training-fold empirical-Bayes ComBat estimates."""
    if (
        fc_stack.ndim != 3
        or fc_stack.shape[1] != model.n_regions
        or fc_stack.shape[2] != model.n_regions
    ):
        raise ValueError(
            f"fc_stack must have shape (S, {model.n_regions}, {model.n_regions}), "
            f"got {fc_stack.shape}"
        )
    if len(sites) != fc_stack.shape[0]:
        raise ValueError("sites length must match number of subjects")

    n_subjects = fc_stack.shape[0]
    x = fc_stack[:, model.triu_idx[0], model.triu_idx[1]].astype(
        np.float64,
        copy=False,
    )
    if not np.all(np.isfinite(x)):
        raise ValueError("fc_stack contains non-finite upper-triangle values")
    if model.fisher_z:
        x = np.arctanh(np.clip(x, -1.0 + eps, 1.0 - eps))

    biological_mean = _biological_mean(
        n_subjects,
        model.beta,
        labels=labels,
        preserve_label=model.preserve_label,
        label_mean=model.label_mean,
    )
    standardized = (x - biological_mean) / model.pooled_std

    site_arr = _normalise_sites(sites)
    adjusted = standardized.copy()
    for site in sorted(set(site_arr.tolist())):
        mask = site_arr == site
        if site in model.gamma_star:
            adjusted[mask] = (
                standardized[mask] - model.gamma_star[site]
            ) / np.sqrt(np.maximum(model.delta_star[site], eps**2))
        # An unseen site has no training-derived effect estimate. Its values
        # remain unchanged instead of estimating anything from target data.

    harmonized_features = adjusted * model.pooled_std + biological_mean
    if model.fisher_z:
        harmonized_features = np.tanh(harmonized_features)

    out = np.zeros((n_subjects, model.n_regions, model.n_regions), dtype=np.float64)
    out[:, model.triu_idx[0], model.triu_idx[1]] = harmonized_features
    out[:, model.triu_idx[1], model.triu_idx[0]] = harmonized_features
    diag = np.arange(model.n_regions)
    out[:, diag, diag] = 1.0
    return out


def _inverse_gamma_moments(
    delta_hat: np.ndarray,
    *,
    eps: float,
) -> tuple[float, float]:
    """Method-of-moments inverse-gamma hyperparameters used by ComBat."""
    mean = max(float(np.mean(delta_hat)), eps**2)
    variance = max(float(np.var(delta_hat, ddof=1)), eps**4)
    a_prior = (2.0 * variance + mean**2) / variance
    b_prior = (mean * variance + mean**3) / variance
    return float(a_prior), float(b_prior)


def _parametric_posterior(
    site_data: np.ndarray,
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    gamma_bar: float,
    t2: float,
    a_prior: float,
    b_prior: float,
    *,
    eps: float,
    convergence_tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Iteratively solve the standard parametric ComBat posterior updates."""
    n_subjects = site_data.shape[0]
    gamma_old = gamma_hat.copy()
    delta_old = delta_hat.copy()

    for iteration in range(1, max_iter + 1):
        denominator = t2 * n_subjects + delta_old
        gamma_new = (
            t2 * n_subjects * gamma_hat + delta_old * gamma_bar
        ) / np.maximum(denominator, eps**2)
        sum_squared = np.square(site_data - gamma_new).sum(axis=0)
        delta_new = (
            0.5 * sum_squared + b_prior
        ) / (n_subjects / 2.0 + a_prior - 1.0)
        delta_new = np.maximum(delta_new, eps**2)

        gamma_change = np.max(
            np.abs(gamma_new - gamma_old) / np.maximum(np.abs(gamma_old), eps)
        )
        delta_change = np.max(
            np.abs(delta_new - delta_old) / np.maximum(np.abs(delta_old), eps**2)
        )
        if max(float(gamma_change), float(delta_change)) < convergence_tol:
            return gamma_new, delta_new, iteration, True
        gamma_old = gamma_new
        delta_old = delta_new

    return gamma_old, delta_old, max_iter, False


def _label_column(
    n_subjects: int,
    *,
    labels: np.ndarray | None,
    preserve_label: bool,
) -> tuple[np.ndarray | None, float | None]:
    if not preserve_label:
        return None, None
    if labels is None:
        raise ValueError("labels are required when preserve_label=True")
    label_values = np.asarray(labels, dtype=np.float64).reshape(-1)
    if label_values.shape[0] != n_subjects:
        raise ValueError("labels length must match number of subjects")
    label_mean = float(label_values.mean())
    return label_values - label_mean, label_mean


def _biological_mean(
    n_subjects: int,
    beta: np.ndarray,
    *,
    labels: np.ndarray | None,
    preserve_label: bool,
    label_mean: float | None,
) -> np.ndarray:
    design = np.ones((n_subjects, 1), dtype=np.float64)
    if preserve_label:
        if labels is None:
            raise ValueError("labels are required when preserve_label=True")
        label_values = np.asarray(labels, dtype=np.float64).reshape(-1)
        if label_values.shape[0] != n_subjects:
            raise ValueError("labels length must match number of subjects")
        if label_mean is None:
            raise ValueError("fitted label mean is missing")
        design = np.column_stack((design, label_values - label_mean))
    return design @ beta


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
