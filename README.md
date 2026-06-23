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

## Embedding Collapse Monitor

Training saves per-stage subject cosine-similarity heatmaps by default for the validation split every epoch:

```text
runs/<run>/fold_<k>/plots/embedding_similarity/epoch_001/
runs/<run>/fold_<k>/plots/embedding_collapse_mean_cosine.png
runs/<run>/fold_<k>/embeddings/epoch_001/
runs/<run>/fold_<k>/embeddings/collapse_metrics.jsonl
```

The stages include `encoder`, each transformer `layer_<n>`, `final`, and `readout_input`.

Control this in `configs/train.yaml`:

```yaml
embedding_monitor: val        # none | train | val | all
embedding_monitor_every: 1
```
