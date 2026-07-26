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
)
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


if __name__ == "__main__":
    unittest.main()
