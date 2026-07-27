from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.build_graph import (
    atlas_distance_matrix,
    distance_local_adjacency,
    normalise_atlas_distances,
    subject_to_data,
)
from src.model.controls import (
    ControlTransformerModel,
    DistanceBiasAttention,
)
from src.model.model import BGTESREModel
from src.precompute.connectivity import proportional_threshold
from src.utils.config import BGTESREConfig, ModelConfig, PrecomputeConfig


def _arrays(n: int = 6, t: int = 12):
    rng = np.random.default_rng(8)
    bold = rng.normal(size=(n, t)).astype(np.float32)
    fc = np.corrcoef(bold)
    coords = rng.normal(size=(n, 3)).astype(np.float32)
    return bold, fc, coords


def _control_cfg(variant: str, n: int = 6, t: int = 12):
    return BGTESREConfig(
        model=ModelConfig(
            num_regions=n,
            variant=variant,
            hidden_dim=8,
            num_classes=2,
            num_layers=2,
            num_heads=2,
            dropout=0.0,
            dropout_attn=0.0,
            dropout_ffn=0.0,
            use_bold_encoder=False,
            bold_in_t=t,
            use_lpe=True,
            k_lap=3,
            readout_pool="mean",
            value_gamma_mode="zero",
        ),
        precompute=PrecomputeConfig(
            threshold_pct=0.5,
            weight_mode="fc",
            graph_construction=(
                "distance_local" if variant == "distance_local_graph" else "fc"
            ),
            local_graph_density=0.4,
            compute_economy=False,
            use_morphospace=False,
        ),
    )


class DistanceGraphTests(unittest.TestCase):
    def test_distance_matrix_and_normalisation(self):
        coords = np.array([[0, 0, 0], [3, 4, 0], [0, 0, 12]], dtype=float)
        distance = atlas_distance_matrix(coords)
        self.assertAlmostEqual(distance[0, 1], 5.0)
        self.assertAlmostEqual(distance[0, 2], 12.0)
        self.assertTrue(np.allclose(distance, distance.T))
        normalised = normalise_atlas_distances(distance)
        upper = normalised[np.triu_indices(3, 1)]
        self.assertAlmostEqual(float(upper.mean()), 1.0)

    def test_local_graph_exact_symmetric_deterministic_and_fc_independent(self):
        coords = np.array(
            [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
            dtype=float,
        )
        first = distance_local_adjacency(coords, 0.3)
        second = distance_local_adjacency(coords, 0.3)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.array_equal(first, first.T))
        self.assertTrue(np.all(np.diag(first) == 0))
        self.assertEqual(int(np.triu(first, 1).sum()), round(0.3 * 10))

        bold, fc, _ = _arrays(n=5)
        cfg = _control_cfg("distance_local_graph", n=5)
        graph_a = subject_to_data(bold, fc, 0, coords, cfg)
        graph_b = subject_to_data(bold, -fc, 0, coords, cfg)
        graph_c = subject_to_data(bold, fc, 1, coords, cfg)
        self.assertTrue(torch.equal(graph_a.edge_index, graph_b.edge_index))
        self.assertTrue(torch.equal(graph_a.edge_index, graph_c.edge_index))

    def test_standard_support_matches_fc_threshold(self):
        bold, fc, coords = _arrays()
        cfg = _control_cfg("standard_transformer")
        graph = subject_to_data(bold, fc, 0, coords, cfg)
        clean_fc = np.clip(np.nan_to_num(fc), 0.0, None)
        np.fill_diagonal(clean_fc, 0.0)
        adjacency, _ = proportional_threshold(clean_fc, 0.5)
        expected_undirected = int(np.triu(adjacency, 1).sum())
        self.assertEqual(graph.edge_index.shape[1], 2 * expected_undirected)
        self.assertFalse(hasattr(graph, "edge_distance"))

    def test_economy_computation_is_bypassed(self):
        bold, fc, coords = _arrays()
        cfg = _control_cfg("standard_transformer")
        with patch.dict(
            "src.data.build_graph._COMPUTE_FN",
            {key: lambda *_: (_ for _ in ()).throw(AssertionError("called"))
             for key in ["E_diff", "E_rout", "EBC", "ECC", "G", "EP"]},
        ):
            graph = subject_to_data(bold, fc, 0, coords, cfg)
        self.assertFalse(hasattr(graph, "phi"))
        self.assertFalse(hasattr(graph, "FC"))

    def test_distance_bias_values_depend_only_on_atlas_coordinates(self):
        bold, fc, coords = _arrays()
        cfg = _control_cfg("distance_bias")
        graph_a = subject_to_data(bold, fc, 0, coords, cfg)
        graph_b = subject_to_data(bold, 2.0 * fc, 1, coords, cfg)
        self.assertTrue(torch.equal(graph_a.edge_index, graph_b.edge_index))
        self.assertTrue(torch.equal(graph_a.edge_distance, graph_b.edge_distance))


class DistanceAttentionTests(unittest.TestCase):
    def test_bias_is_monotone_and_shapes_are_correct(self):
        attention = DistanceBiasAttention(4, 2, 0.0, initial_scale=1.0)
        with torch.no_grad():
            attention.W_Q.weight.zero_()
            attention.W_K.weight.zero_()
        x = torch.randn(3, 4)
        edge_index = torch.tensor([[0, 1], [2, 2]])
        output = attention(x, edge_index, torch.tensor([0.5, 1.5]))
        self.assertEqual(tuple(output.shape), (3, 4))
        self.assertEqual(tuple(attention._last_alpha.shape), (2, 2))
        self.assertTrue(torch.all(attention._last_alpha[0] > attention._last_alpha[1]))
        self.assertTrue(torch.allclose(attention.distance_scale(), torch.ones(2)))

    def test_fixed_mode_has_no_trainable_distance_scale(self):
        attention = DistanceBiasAttention(4, 2, 0.0, mode="fixed")
        names = dict(attention.named_parameters())
        self.assertNotIn("raw_distance_scale", names)


class ControlModelTests(unittest.TestCase):
    def test_all_controls_forward_backward_and_checkpoint(self):
        bold, fc, coords = _arrays()
        for variant in ControlTransformerModel.VARIANTS:
            with self.subTest(variant=variant):
                cfg = _control_cfg(variant)
                graph = subject_to_data(bold, fc, 1, coords, cfg)
                batch = next(iter(DataLoader([graph], batch_size=1)))
                model = ControlTransformerModel(cfg)
                output = model(batch)
                output["loss"].backward()
                self.assertEqual(tuple(output["logits"].shape), (1, 2))
                self.assertTrue(any(p.grad is not None for p in model.parameters()))
                parameter_names = " ".join(name.lower() for name, _ in model.named_parameters())
                for forbidden in ("phi", "psi", "rotary", "v_scale", "vphi"):
                    self.assertNotIn(forbidden, parameter_names)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "model.pt"
                    torch.save(model.state_dict(), path)
                    restored = ControlTransformerModel(cfg)
                    restored.load_state_dict(torch.load(path, map_location="cpu"))

    def test_distance_is_only_attached_to_distance_bias(self):
        bold, fc, coords = _arrays()
        for variant in ("standard_transformer", "distance_local_graph"):
            graph = subject_to_data(bold, fc, 0, coords, _control_cfg(variant))
            self.assertFalse(hasattr(graph, "edge_distance"))
        bias_graph = subject_to_data(
            bold, fc, 0, coords, _control_cfg("distance_bias")
        )
        self.assertTrue(hasattr(bias_graph, "edge_distance"))

    def test_full_model_path_still_has_original_economy_fields(self):
        bold, fc, coords = _arrays()
        cfg = BGTESREConfig(
            model=ModelConfig(
                num_regions=6,
                variant="full",
                hidden_dim=8,
                num_classes=2,
                num_layers=1,
                num_heads=2,
                use_bold_encoder=False,
                bold_in_t=12,
                use_lpe=True,
                k_lap=3,
                readout_pool="mean",
            ),
            precompute=PrecomputeConfig(
                morphospace_pair=("ecc", "ebc"),
                topo_metric_x_attr="ECC",
                topo_metric_y_attr="EBC",
                threshold_pct=0.5,
                weight_mode="fc",
            ),
        )
        graph = subject_to_data(bold, fc, 0, coords, cfg)
        for field in ("phi", "FC", "dist", "ECC", "EBC"):
            self.assertTrue(hasattr(graph, field))
        batch = next(iter(DataLoader([graph], batch_size=1)))
        result = BGTESREModel(cfg)(batch)
        self.assertEqual(tuple(result["logits"].shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
