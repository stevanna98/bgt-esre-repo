# BGT-ESRE Training

This repository contains the BGT-ESRE graph training code and a K-fold training entry point.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric may require platform-specific wheels for CUDA or Apple Silicon. If the generic install is not suitable for your machine, install those two packages using their official selectors first, then install the remaining requirements.

## Smoke Test

Run the synthetic end-to-end check before launching a real training job:

```bash
python scripts/train_kfold.py --smoke
```

This does not need real fMRI data. It checks graph construction, batching, model forward/backward, metrics, plots, attention export, and summary output.

## Training

Set `data.data_dir` in `configs/train.yaml`, or pass `--data-dir` on the command line.

Expected dataset layouts are documented in `configs/train.yaml`.

```bash
python scripts/train_kfold.py --config configs/train.yaml
```

For FC-only datasets such as `ad_lmci` or `nc_asd`, use the linear BOLD projection:

```bash
python scripts/train_kfold.py --config configs/train.yaml --dataset ad_lmci --no-bold-encoder
```

Equivalent config-style overrides are also supported:

```bash
python scripts/train_kfold.py --config configs/train.yaml --set model.no_bold_encoder=true
```

## ABIDE ComBat harmonization

When `combat_harmonize: true`, each fold fits parametric empirical-Bayes
ComBat using only its training subjects. The fitted additive and multiplicative
site effects are then applied unchanged to the fold's training and validation
FC matrices before graph construction. This requires site labels and at least
two training subjects from each site in every fold.

By default FC values are Fisher-z transformed before harmonization and class
labels are not used as preprocessing covariates. Site labels can be placed in
the data directory as `sites.npy` (or another supported site filename listed
in `configs/train.yaml`) or supplied with `--combat-site-file`.

Run the focused harmonization tests with:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Embedding Collapse Monitor

Training saves per-stage subject cosine-similarity heatmaps by default for the validation split every epoch:

```text
runs/<run>/fold_<k>/plots/embedding_similarity/epoch_001/
runs/<run>/fold_<k>/plots/embedding_collapse_mean_cosine.png
runs/<run>/fold_<k>/embeddings/epoch_001/
runs/<run>/fold_<k>/embeddings/collapse_metrics.jsonl
```

The stages include `encoder`, each transformer `layer_<n>`, `final`, and `readout_input`. If `use_virtual_node` is enabled, the monitor also includes `virtual_node`.

Control this in `configs/train.yaml`:

```yaml
embedding_monitor: val        # none | train | val | all
embedding_monitor_every: 1
```

## Subject Readout Pooling

The default subject embedding preserves atlas order by flattening the per-region node embeddings. This avoids washing out ROI-specific signal with graph-level averaging. You can replace it with mean+std, plain mean, max, or learned attention pooling:

```bash
python scripts/train_kfold.py --config configs/train.yaml --readout-pool attention
```

or in YAML:

```yaml
model:
  readout_pool: flatten   # flatten | mean_std | mean | max | attention
  use_virtual_node: false
```
