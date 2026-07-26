import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scripts.train_kfold import (
    append_cost_value_gamma,
    collect_cost_value_gamma,
    parse_args,
)
from src.model.esre import ESREAttention
from src.model.esre_no_rotary import ESREAttentionNoRotary
from src.model.model import BGTESREModel
from src.model.model_ablation import BGTESREModelAblation
from src.utils.config import BGTESREConfig, ModelConfig
from src.utils.plotting import plot_cost_value_gamma


class _Attention(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.v_scale = nn.Parameter(torch.tensor([value], dtype=torch.float32))


class _Layer(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.attn = _Attention(value)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(0.25), _Layer(-0.5)])


class GammaTrackingTests(unittest.TestCase):
    def test_collects_raw_and_effective_gamma_per_layer(self) -> None:
        values = collect_cost_value_gamma(_Model())
        self.assertAlmostEqual(values["layer_1"]["raw_v_scale"], 0.25)
        self.assertAlmostEqual(values["layer_1"]["gamma"], float(np.tanh(0.25)))
        self.assertAlmostEqual(values["layer_2"]["raw_v_scale"], -0.5)
        self.assertAlmostEqual(values["layer_2"]["gamma"], float(np.tanh(-0.5)))

    def test_appends_jsonl_history_and_renders_plot(self) -> None:
        model = _Model()
        history: dict[str, list[float]] = {}
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "gamma" / "value_gamma.jsonl"
            jsonl_path.parent.mkdir(parents=True)
            append_cost_value_gamma(model, 1, history, jsonl_path)
            append_cost_value_gamma(model, 2, history, jsonl_path)

            records = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["epoch"] for record in records], [1, 2])
            self.assertEqual(len(history["gamma_layer_1"]), 2)
            self.assertEqual(len(history["v_scale_layer_2"]), 2)

            plot_path = Path(tmp) / "plots" / "cost_value_gamma.png"
            plot_cost_value_gamma(history, plot_path)
            self.assertTrue(plot_path.is_file())
            self.assertGreater(plot_path.stat().st_size, 0)

    def test_gamma_modes_are_exact_and_fixed_modes_have_no_parameter(self) -> None:
        for attention_cls in (ESREAttention, ESREAttentionNoRotary):
            learned = attention_cls(8, 2, 0.0, value_gamma_mode="learned")
            fixed_one = attention_cls(8, 2, 0.0, value_gamma_mode="one")
            fixed_zero = attention_cls(8, 2, 0.0, value_gamma_mode="zero")

            self.assertIsInstance(learned.v_scale, nn.Parameter)
            self.assertAlmostEqual(float(learned.value_gamma()), 0.01, places=7)
            self.assertIsNone(fixed_one.v_scale)
            self.assertEqual(float(fixed_one.value_gamma()), 1.0)
            self.assertIsNone(fixed_zero.v_scale)
            self.assertEqual(float(fixed_zero.value_gamma()), 0.0)

    def test_cli_accepts_fixed_one_mode(self) -> None:
        args = parse_args(
            ["--value-gamma-mode", "one", "--value-gamma-init", "0.05"]
        )
        self.assertEqual(args.value_gamma_mode, "one")
        self.assertEqual(args.value_gamma_init, 0.05)

    def test_tracking_fixed_mode_records_gamma_without_raw_scale(self) -> None:
        model = _Model()
        model.layers[0].attn = ESREAttention(
            8,
            2,
            0.0,
            value_gamma_mode="one",
        )
        values = collect_cost_value_gamma(model)
        self.assertIsNone(values["layer_1"]["raw_v_scale"])
        self.assertEqual(values["layer_1"]["gamma"], 1.0)

    def test_fixed_mode_propagates_into_both_model_variants(self) -> None:
        cfg = BGTESREConfig(
            model=ModelConfig(
                num_regions=4,
                hidden_dim=8,
                num_layers=2,
                num_heads=2,
                use_bold_encoder=False,
                bold_in_t=4,
                value_gamma_mode="one",
            )
        )
        for model_cls in (BGTESREModel, BGTESREModelAblation):
            model = model_cls(cfg)
            self.assertTrue(
                all(layer.attn.value_gamma_mode == "one" for layer in model.layers)
            )
            self.assertTrue(
                all(layer.attn.v_scale is None for layer in model.layers)
            )

    def test_learned_gamma_and_value_projection_train_from_first_batch(self) -> None:
        torch.manual_seed(4)
        attention = ESREAttention(
            8,
            2,
            0.0,
            value_gamma_mode="learned",
            value_gamma_init=0.01,
        )
        x = torch.randn(4, 8)
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
            dtype=torch.long,
        )
        phi = torch.randn(edge_index.shape[1], 2)
        attention(x, edge_index, phi).square().mean().backward()

        self.assertIsNotNone(attention.v_scale.grad)
        self.assertGreater(float(attention.v_scale.grad.abs().sum()), 0.0)
        self.assertIsNotNone(attention.V_phi.grad)
        self.assertGreater(float(attention.V_phi.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
