#!/usr/bin/env python
"""Combine completed control runs into rebuttal-ready numerical tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


MODEL_LABELS = {
    "standard_transformer": ("Standard Graph Transformer", "None"),
    "distance_bias": ("Distance-bias Transformer", "Euclidean distance bias"),
    "distance_local_graph": (
        "Distance-local Graph Transformer", "Shortest-distance graph"
    ),
    "full": ("BGT-ESRE", "Economy-aligned topology + ESRE"),
}
DATASET_LABELS = {"hcp": "HCP-YA", "abide": "ABIDE I"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate completed BGT-ESRE distance-control runs"
    )
    parser.add_argument("run_dirs", nargs="+", help="Completed CV run directories")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _load_complete_run(run_dir: Path) -> tuple[dict, list[dict], dict]:
    summary_path = run_dir / "cv_summary.json"
    results_path = run_dir / "control_results.csv"
    splits_path = run_dir / "cv_splits.json"
    for path in (summary_path, results_path, splits_path):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete run {run_dir}: missing {path.name}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = int(splits["n_splits"])
    if int(summary["n_folds"]) != expected or len(rows) != expected:
        raise ValueError(
            f"incomplete run {run_dir}: expected {expected} folds, "
            f"found summary={summary['n_folds']} rows={len(rows)}"
        )
    return summary, rows, splits


def _mean_std(rows: list[dict], metric: str) -> tuple[float, float]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    return float(np.mean(values)), float(np.std(values))


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[tuple[str, str], tuple[dict, list[dict]]] = {}
    split_signatures: dict[str, str] = {}
    all_rows: list[dict] = []

    for raw_dir in args.run_dirs:
        run_dir = Path(raw_dir)
        summary, rows, splits = _load_complete_run(run_dir)
        dataset = str(summary["dataset"])
        model = str(summary["model_variant"])
        key = (dataset, model)
        if key in runs:
            raise ValueError(f"duplicate completed run for {dataset}/{model}")
        signature = json.dumps(
            {
                "subject_ids": splits["subject_ids"],
                "folds": splits["folds"],
            },
            sort_keys=True,
        )
        if dataset in split_signatures and split_signatures[dataset] != signature:
            raise ValueError(f"{dataset} runs do not use identical CV splits")
        split_signatures[dataset] = signature
        runs[key] = (summary, rows)
        all_rows.extend(rows)

    result_fields = [
        "dataset", "model", "fold", "seed", "accuracy", "auc", "f1",
        "sensitivity", "specificity", "balanced_accuracy", "parameter_count",
        "checkpoint_path", "split_file",
    ]
    with (out_dir / "control_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    aggregate_rows = []
    for (dataset, model), (summary, rows) in runs.items():
        row = {
            "dataset": dataset,
            "model": model,
            "n_folds": len(rows),
            "seed": summary["seed"],
            "parameter_count": rows[0]["parameter_count"],
        }
        for metric in (
            "accuracy", "auc", "f1", "sensitivity", "specificity",
            "balanced_accuracy",
        ):
            mean, std = _mean_std(rows, metric)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        full = runs.get((dataset, "full"))
        row["delta_auc_full_minus_control"] = (
            _mean_std(full[1], "auc")[0] - row["auc_mean"]
            if full is not None and model != "full"
            else 0.0 if model == "full" else float("nan")
        )
        aggregate_rows.append(row)

    aggregate_fields = list(aggregate_rows[0]) if aggregate_rows else []
    with (out_dir / "control_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    def formatted(dataset: str, model: str, metric: str) -> str:
        run = runs.get((dataset, model))
        if run is None:
            return "—"
        mean, std = _mean_std(run[1], metric)
        return f"{mean:.3f} ± {std:.3f}"

    lines = [
        "# Distance/locality control results",
        "",
        "| Model | Edge information | HCP-YA ACC | HCP-YA AUC | ABIDE I ACC | ABIDE I AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in (
        "standard_transformer", "distance_bias", "distance_local_graph", "full"
    ):
        label, edge_info = MODEL_LABELS[model]
        lines.append(
            f"| {label} | {edge_info} | {formatted('hcp', model, 'accuracy')} | "
            f"{formatted('hcp', model, 'auc')} | "
            f"{formatted('abide', model, 'accuracy')} | "
            f"{formatted('abide', model, 'auc')} |"
        )
    lines.extend(
        [
            "",
            "## AUC differences (BGT-ESRE − control)",
            "",
            "| Control | HCP-YA ΔAUC | ABIDE I ΔAUC |",
            "|---|---:|---:|",
        ]
    )
    for model in (
        "standard_transformer", "distance_bias", "distance_local_graph"
    ):
        values = []
        for dataset in ("hcp", "abide"):
            full = runs.get((dataset, "full"))
            control = runs.get((dataset, model))
            if full is None or control is None:
                values.append("—")
            else:
                delta = _mean_std(full[1], "auc")[0] - _mean_std(
                    control[1], "auc"
                )[0]
                values.append(f"{delta:.3f}")
        lines.append(f"| {MODEL_LABELS[model][0]} | {values[0]} | {values[1]} |")
    lines.extend(
        [
            "",
            "All entries are validation-fold mean ± population standard deviation. "
            "The held-out validation fold is also used for checkpoint selection; "
            "it is not an independent test set.",
        ]
    )
    (out_dir / "control_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote completed-run tables to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
