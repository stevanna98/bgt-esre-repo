import unittest

import numpy as np

from src.data.loaders import SubjectRecord
from src.preprocess.combat import (
    fit_combat_fc,
    harmonize_subject_connectivity,
    transform_combat_fc,
)


def _fc_from_fisher_features(
    features: np.ndarray,
    n_regions: int,
) -> np.ndarray:
    triu = np.triu_indices(n_regions, k=1)
    fc = np.zeros((features.shape[0], n_regions, n_regions), dtype=np.float64)
    correlations = np.tanh(features)
    fc[:, triu[0], triu[1]] = correlations
    fc[:, triu[1], triu[0]] = correlations
    diag = np.arange(n_regions)
    fc[:, diag, diag] = 1.0
    return fc


def _fisher_features(fc: np.ndarray) -> np.ndarray:
    triu = np.triu_indices(fc.shape[1], k=1)
    return np.arctanh(np.clip(fc[:, triu[0], triu[1]], -0.999999, 0.999999))


class CombatTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.n_regions = 10
        self.n_per_site = 40
        self.sites = np.repeat(["A", "B", "C"], self.n_per_site)
        n_subjects = self.sites.size
        n_features = self.n_regions * (self.n_regions - 1) // 2
        self.labels = np.tile([0, 1], n_subjects // 2)

        baseline = rng.normal(0.0, 0.15, size=n_features)
        label_effect = rng.normal(0.18, 0.025, size=n_features)
        site_shift = {"A": -0.55, "B": 0.45, "C": 0.05}
        site_scale = {"A": 0.65, "B": 1.45, "C": 1.0}
        features = np.empty((n_subjects, n_features), dtype=np.float64)
        for i, site in enumerate(self.sites):
            features[i] = (
                baseline
                + label_effect * self.labels[i]
                + site_shift[site]
                + rng.normal(0.0, site_scale[site] * 0.2, size=n_features)
            )
        self.fc = _fc_from_fisher_features(features, self.n_regions)

    def test_empirical_bayes_reduces_site_location_and_scale_effects(self) -> None:
        model = fit_combat_fc(self.fc, self.sites)
        adjusted = transform_combat_fc(self.fc, self.sites, model)

        before = _fisher_features(self.fc)
        after = _fisher_features(adjusted)

        before_means = np.stack(
            [before[self.sites == site].mean(axis=0) for site in ("A", "B", "C")]
        )
        after_means = np.stack(
            [after[self.sites == site].mean(axis=0) for site in ("A", "B", "C")]
        )
        before_stds = np.stack(
            [before[self.sites == site].std(axis=0) for site in ("A", "B", "C")]
        )
        after_stds = np.stack(
            [after[self.sites == site].std(axis=0) for site in ("A", "B", "C")]
        )

        self.assertLess(
            np.mean(np.ptp(after_means, axis=0)),
            0.25 * np.mean(np.ptp(before_means, axis=0)),
        )
        self.assertLess(
            np.mean(np.ptp(np.log(after_stds), axis=0)),
            0.4 * np.mean(np.ptp(np.log(before_stds), axis=0)),
        )
        self.assertTrue(all(model.converged.values()))
        self.assertTrue(
            any(
                not np.allclose(model.gamma_hat[site], model.gamma_star[site])
                for site in model.gamma_hat
            )
        )
        self.assertTrue(
            any(
                not np.allclose(model.delta_hat[site], model.delta_star[site])
                for site in model.delta_hat
            )
        )
        self.assertTrue(np.allclose(adjusted, adjusted.transpose(0, 2, 1)))
        self.assertTrue(
            np.allclose(np.diagonal(adjusted, axis1=1, axis2=2), 1.0)
        )

    def test_unseen_site_is_not_estimated_from_target_data(self) -> None:
        train_mask = self.sites != "C"
        model = fit_combat_fc(self.fc[train_mask], self.sites[train_mask])
        unseen_fc = self.fc[~train_mask]
        adjusted = transform_combat_fc(
            unseen_fc,
            np.repeat("unseen", unseen_fc.shape[0]),
            model,
        )
        self.assertTrue(np.allclose(adjusted, unseen_fc, atol=1e-12))

    def test_preserved_label_uses_training_center_for_every_transform(self) -> None:
        model = fit_combat_fc(
            self.fc,
            self.sites,
            labels=self.labels,
            preserve_label=True,
        )
        together = transform_combat_fc(
            self.fc[:6],
            self.sites[:6],
            model,
            labels=self.labels[:6],
        )
        separately = np.concatenate(
            [
                transform_combat_fc(
                    self.fc[i : i + 1],
                    self.sites[i : i + 1],
                    model,
                    labels=self.labels[i : i + 1],
                )
                for i in range(6)
            ],
            axis=0,
        )
        self.assertTrue(np.allclose(together, separately, atol=1e-12))
        self.assertAlmostEqual(model.label_mean, float(self.labels.mean()))

    def test_scale_model_rejects_singleton_training_site(self) -> None:
        sites = self.sites.copy()
        sites[0] = "singleton"
        with self.assertRaisesRegex(ValueError, "singleton"):
            fit_combat_fc(self.fc, sites)

    def test_fold_wrapper_does_not_use_target_subjects_when_fitting(self) -> None:
        subjects = [
            SubjectRecord(
                bold=np.zeros((self.n_regions, 2), dtype=np.float32),
                fc=self.fc[i],
                label=int(self.labels[i]),
                subject_id=f"subject_{i}",
                bold_axes="NT",
                site=self.sites[i],
            )
            for i in range(len(self.sites))
        ]
        train_indices = (
            list(range(10))
            + list(range(self.n_per_site, self.n_per_site + 10))
            + list(range(2 * self.n_per_site, 2 * self.n_per_site + 10))
        )
        train_only, _ = harmonize_subject_connectivity(
            subjects,
            train_indices,
            train_indices,
        )
        train_and_target, summary = harmonize_subject_connectivity(
            subjects,
            train_indices,
            train_indices + list(range(10, 20)),
        )
        for expected, actual in zip(train_only, train_and_target):
            self.assertTrue(np.allclose(expected.fc, actual.fc))
        self.assertEqual(summary["method"], "parametric_empirical_bayes")


if __name__ == "__main__":
    unittest.main()
