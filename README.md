# BGT-ESRE Training

BGT-ESRE is a graph-transformer training pipeline for subject-level
classification from functional MRI. It supports HCP, ABIDE, and two FC-only
datasets, with stratified K-fold cross-validation, fold-wise ABIDE ComBat
harmonization, configurable graph construction, ESRE attention, diagnostic
plots, and checkpoint selection.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric may require platform-specific wheels for CUDA or
Apple Silicon. If the generic installation is unsuitable, install those two
packages using their official selectors first and then install the remaining
requirements.

## Verify the installation

Run the synthetic end-to-end smoke test:

```bash
python scripts/train_kfold.py --smoke
```

It requires no real fMRI data and checks graph construction, batching, model
forward/backward passes, metrics, checkpoints, plots, attention export, gamma
tracking, and cross-validation summaries.

Run the focused unit tests with:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Dataset layouts

Pass the directory containing the files—not an individual `.npy` file—to
`--data-dir`.

```text
HCP
  bold.npy                  (subjects, regions, time)
  connectivity.npy          (subjects, regions, regions)
  labels.npy                (subjects,)
  coords.npy                (regions, 3)

ABIDE
  bold.npy                  (subjects, time, regions)
  connectivity.npy          (subjects, regions, regions)
  labels.npy                (subjects,)
  coords.npy                (regions, 3)
  sites.npy                 (subjects,)  required when ComBat is enabled

AD/LMCI
  data_ad_lmci.npy          (subjects, regions, regions, features)
  labels_ad_lmci.npy        (subjects,)
  desikan_coords_left.npy   (regions, 3)

NC/ASD
  data_nc_asd.npy           (subjects, regions, regions, features)
  labels_nc_asd.npy         (subjects,)
  desikan_coords_left.npy   (regions, 3)
```

For ABIDE, the site file may instead be named `site.npy`, `site_ids.npy`,
`site_labels.npy`, `batch.npy`, or `batches.npy`. A `.npy`, `.txt`, or `.csv`
file can also be supplied explicitly with `--combat-site-file`.

## Training

Every leaf in `configs/train.yaml` maps to a command-line option. Explicit CLI
arguments override YAML values. The checked-in YAML contains Colab-specific
input and output paths, so override them when running elsewhere:

```bash
python scripts/train_kfold.py \
  --config configs/train.yaml \
  --data-dir /path/to/abide \
  --out-dir runs/abide_kfold
```

The current checked-in experiment configuration uses:

```text
dataset                 ABIDE
cross-validation        5 folds
epochs / batch size     100 / 8
node features           FC rows with linear projection
model                   full ESRE, 2 layers, hidden size 32, 4 heads
readout                 atlas-order flattening
ComBat                  enabled; diagnosis excluded
morphospace             log(ediff), log(communicability)
graph weights           cost_penalised, lambda=0.1, top 15% positive FC
value gamma             fixed to zero
```

For the FC-only datasets, disable the ABIDE-only ComBat setting inherited from
the YAML and provide the matching data directory:

```bash
python scripts/train_kfold.py \
  --config configs/train.yaml \
  --dataset ad_lmci \
  --data-dir /path/to/ad_lmci \
  --out-dir runs/ad_lmci_kfold \
  --set harmonization.combat_harmonize=false \
  --node-features fc \
  --no-bold-encoder
```

Nested config-style overrides are also supported; only the final key is used:

```bash
python scripts/train_kfold.py \
  --config configs/train.yaml \
  --set training.epochs=50 \
  --set model.readout_pool=attention
```

Use `python scripts/train_kfold.py --help` for the complete option list.

## Graph and morphospace construction

FC matrices are cleaned, restricted to non-negative connections, and
proportionally thresholded. The measure-computation matrix is selected with
`weight_mode`:

```text
binary          W = thresholded adjacency
fc              W = retained FC weights
cost_penalised  W_ij = FC_ij * exp(-lambda * distance_ij)
```

When `eco_lambda: auto` is used, lambda is the inverse mean physical distance
over retained edges. The two configured topological measures are computed from
`W`, sampled on retained edges, and log-transformed into the two-dimensional
edge coordinate `phi`. Available measures are `ediff`, `erout`, `ebc`, `ecc`,
`comm`, and `ep`.

The full model uses `phi` in both asymmetric rotary Q/K attention and the
optional value correction. `--no-morphospace` zeros `phi` while preserving the
FC-derived graph topology.

## ABIDE empirical-Bayes ComBat

With `combat_harmonize: true`, every fold fits parametric empirical-Bayes
ComBat using only its training subjects. It treats Fisher-z-transformed FC
upper-triangle edges as features, estimates posterior additive and
multiplicative site effects, and applies those fixed estimates to the fold's
training and validation FC matrices before graph construction.

Each training fold must contain at least two acquisition sites and two
subjects per site. A validation site absent from the training fold cannot be
estimated and is left unchanged with a warning.

The checked-in configuration uses:

```yaml
harmonization:
  combat_harmonize: true
  combat_preserve_label: false
  combat_no_fisher_z: false
```

Consequently, diagnosis is excluded from ComBat. Keeping
`combat_preserve_label: false` also avoids requiring validation diagnoses in
preprocessing.

## Cost-value gamma

For edge \(j \rightarrow i\), the ESRE value correction is:

```text
V_aug = V_j + gamma * (V_phi * phi_ij)
```

Configure gamma with:

```yaml
model:
  value_gamma_mode: zero     # learned | one | zero
  value_gamma_init: 0.01     # used only in learned mode
```

The modes are:

```text
learned  gamma = tanh(v_scale), optimized by AdamW
one      gamma = 1, full value correction with no trainable gate
zero     gamma = 0, value correction disabled
```

In learned mode, `v_scale` is initialized to `atanh(value_gamma_init)`, so the
effective initial gamma is exactly `0.01` by default. Both `v_scale` and
`V_phi` receive gradients from the first training batch.

Run a learned-gamma experiment with:

```bash
python scripts/train_kfold.py \
  --config configs/train.yaml \
  --value-gamma-mode learned \
  --value-gamma-init 0.01
```

Every epoch records and plots gamma for each transformer layer:

```text
<out_dir>/fold_<k>/plots/cost_value_gamma.png
<out_dir>/fold_<k>/gamma/value_gamma.jsonl
```

The JSONL records effective gamma and, in learned mode, raw `v_scale`.

## Model variants and readout

The available model variants are:

```text
full                asymmetric morphospace rotary Q/K encoding
ablation_no_rotary  standard Q/K attention plus additive phi injection
```

Both variants retain the configurable cost-value branch. Select the ablation
with `--model ablation_no_rotary`.

The default subject readout preserves atlas order by flattening the per-region
embeddings. Alternatives are `mean_std`, `mean`, `max`, and learned
`attention`:

```bash
python scripts/train_kfold.py \
  --config configs/train.yaml \
  --readout-pool attention
```

An optional graph-level virtual-node side channel can be enabled with
`--use-virtual-node` for the full model.

## Training outputs

A completed output directory has the following main artifacts:

```text
<out_dir>/
  resolved_args.json
  cv_summary.json
  fold_<k>/
    best_model.pt
    run_summary.json
    attn/
      attn_overall.npy
      attn_label_<label>.npy
    gamma/
      value_gamma.jsonl
    embeddings/
      collapse_metrics.jsonl
      epoch_<n>/
    plots/
      train_loss.png
      loss.png
      accuracy.png
      auc.png
      f1.png
      sensitivity.png
      specificity.png
      attn_overall.png
      attn_label_<label>.png
      cost_value_gamma.png
      embedding_collapse_mean_cosine.png
      embedding_similarity/epoch_<n>/
```

`best_model.pt` is selected using `selection_metric` (`auc` or `loss`).
Early stopping and `ReduceLROnPlateau` use the same metric.

## Embedding-collapse monitor

The monitor can collect subject embeddings and cosine-similarity matrices from
the encoder, every transformer layer, the final representation, and the
readout input. When enabled, it runs at `embedding_monitor_every` intervals:

```yaml
training:
  embedding_monitor: val       # none | train | val | all
  embedding_monitor_every: 1
```

When the full model uses a virtual node, its embedding is included as an
additional monitored stage.
