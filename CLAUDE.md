# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geometric interpretability of the **ESP-AVES2 EAT-family** of audio encoders
released by the Earth Species Project. We collect residual-stream activations
from 4 EAT checkpoints over a frozen slice of NatureLM-audio-training and
study how the geometry of those activations is shaped by training. Plus a
random-init EAT baseline (architecture only, no learned weights) that anchors
absolute magnitudes.

Additionally, Sid's **probe pipeline** runs LORO logistic-regression probes
across species pairs drawn from xeno-canto + NatureLM to measure how
phylogenetic distance shapes layer-by-layer separability.

Read `RESULTS.md` for the running narrative (CLAIM / RETRACTED / OPEN sections).

## Models in scope

Four trained EAT checkpoints (all 13 layers — `model.pos_drop` + 12
transformer blocks — at hidden dim 768) plus one random-init baseline:

| key                       | description                                    |
|---------------------------|------------------------------------------------|
| `eat_all`                 | EAT pretrained on bio + non-bio audio          |
| `eat_bio`                 | EAT pretrained on bio-only audio               |
| `sl_eat_all_ssl_all`      | `eat_all` + SSL fine-tune on bio + non-bio     |
| `sl_eat_bio_ssl_all`      | `eat_bio` + SSL fine-tune on bio + non-bio     |
| `random_init_eat_seed42`  | EAT-base architecture, random reinit at seed 42|

Two more random-init seeds (7 and 13) were extracted to validate init
variability — shards have been deleted to save disk; per-seed CSVs persist
under `artifacts/.../random_init_variability/`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio transformers huggingface_hub safetensors \
            pyarrow matplotlib scikit-learn scipy timm
# For probe pipeline:
pip install avex datasets esp-aves soundfile
```

The EAT base architecture is fetched on first use via
`AutoModel.from_pretrained("worstchan/EAT-base_epoch30_pretrain", trust_remote_code=True)`.
Per-checkpoint safetensors are pulled from the corresponding
`EarthSpeciesProject/esp-aves2-*` HF repos. Audio comes from the
`EarthSpeciesProject/NatureLM-audio-training` parquet shards via
`hf_hub_download` — these get cached under `~/.cache/huggingface/hub/`
(~14G; required for any new extraction or audio-mixing experiment).

## Running scripts

All scripts are standalone, run from the project root:

```bash
source venv/bin/activate
python -W ignore <script>.py                      # geometry pipeline
python -W ignore experiments/<experiment>.py      # probe experiments
```

## Geometry pipeline (Evan)

### Step 1 — extraction

```
collect_esp_aves2_activations.py --manifest <jsonl> --models <key,key,...>
```

Hooks the 13 layers (`model.pos_drop` + `model.blocks.0..11`), forwards each
manifest item through the model, and writes shards (one .pt per ~25 samples,
shape `(B, 13, 513, 768)` in float16) to
`artifacts/roadmap_part1/<manifest>/<model>/shards/`. Resumable via the
existing-shards scan. The `--random-init-seed N` path (used for the baseline)
loads EAT-base, then walks `init_weights` + `reset_parameters` + a
`normal(0, 0.02)` fallback.

### Step 2 — geometry analysis

Each script reads from shards and writes to
`artifacts/comparisons/<manifest>/nway_eat_all4/<subdir>/`:

| script                              | purpose                                     |
|-------------------------------------|---------------------------------------------|
| `nway_compare_eat_models.py`        | Consolidated pooled embeddings + cross-model CKA |
| `step2_spectral_dim_eat.py`         | Singular values, eff_rank, PR, TwoNN per (model, layer) |
| `step2_subspace_angles_eat.py`      | L2-norm histograms + subspace overlap |
| `step2_pooled_vs_frame_eat.py`      | Pooled-vs-frame distortion check |
| `step2_tier1_frame_level.py`        | Frame-level eff_rank/PR/TwoNN/MLE-ID across all 4 trained models |
| `step2_random_init_compare.py`      | 5-way comparison: trained models vs random-init baseline |
| `step2_random_init_variability.py`  | Init variability across random-init seeds 7/13/42 |

## Probe pipeline (Sid)

### Module structure

- **`data/loader.py`** — centralized model loading and activation extraction. `build_dataset` (local files) and `build_naturelm_dataset` (HuggingFace streaming). Both return `{layer_index: (X, y)}`.
- **`probes/train.py`** — `train_all_layers` runs LORO cross-validation across all 13 layers.
- **`probes/evaluate.py`** — `run_evaluation` saves accuracy curve PNG + LDA projection PNG to `results/`.
- **`experiments/`** — entry points wiring `data/loader`, `probes/train`, `probes/evaluate`. Runnable: `animals_vs_music.py`, `species.py`, `naturelm_probe_all_pairs.py`.
- **`scripts/batch_extract_naturelm.py`** — resume-safe batch extractor for 18 species × N recordings.
- **`scripts/plot_phylogenetic_gradient.py`** — plots peak accuracy vs MYA distance across species pairs.

### Layer indexing convention

13 layers total:
- Index 0 = CNN `feature_projection` output (embedding layer, labeled `"emb"`)
- Indices 1–12 = transformer layers 0–11 (labeled `"T0"`–`"T11"`)

### Running probe experiments

```bash
# Extract activations for NatureLM scaling (GPU recommended):
python -W ignore scripts/batch_extract_naturelm.py --rows 1000 --device cuda

# Run all 10 species-pair probe experiments:
python -W ignore experiments/naturelm_probe_all_pairs.py
```

## Frame-level subsampling

Where frame-level analyses subsample, we draw 50 frames per item uniformly
from the valid-token range with seed 42 (600 × 50 = 30,000 rows per
(model, layer)). TwoNN and MLE-ID further subsample to 10,000 rows.

## Geometry primitives

Defined in `step2_tier1_frame_level.py`; reused via import by later scripts.

- **Effective rank** = `exp(-Σ p_i log p_i)` over normalized eigenvalues of
  the centered covariance.
- **Participation ratio** = `(Σλ)² / Σλ²`.
- **Intrinsic dimension** — TwoNN(k=2) and MLE-ID(k=20). MLE-ID is the
  preferred estimator; TwoNN has a known L4 failure mode (see `RESULTS.md` §7).
- **Subspace overlap** — `mean(cos(principal_angles))` between top-k=10 PCA
  bases via `scipy.linalg.subspace_angles`. 1.0 = identical, 0.0 = orthogonal.

## Pooling convention

The pooled comparison expects mean over `tokens[1:valid_token_count]`,
**skipping token 0** (EAT's CLS-like token). See
`compare_esp_aves2_models.pooled_layer_vectors`. Frame-level analyses
include token 0; that's the existing convention everywhere.

## Data

- `artifacts/manifests/naturelm_by_source_100each_20260418T171459Z.jsonl` —
  the frozen 600-sample manifest (100 × 7 source datasets), tracked.
- `artifacts/roadmap_part1/<manifest>/<model>/shards/` — per-model
  activation shards (~5.8G/model, gitignored). 5 models present:
  the 4 trained + `random_init_eat_seed42`.
- `artifacts/comparisons/<manifest>/nway_eat_all4/{step2_*,random_init_*}/`
  — committed CSVs, plots, and per-seed stats.
- `audio/` — local xeno-canto recordings for probe experiments (gitignored).
- `results/probe-output/` — probe accuracy PNGs and LDA plots per species pair.

## Key findings (current state — see `RESULTS.md` for full claims + retractions)

1. **Pooling distorts geometry** universally. Frame-level eff_rank > pooled
   eff_rank everywhere; the ratio varies ×2–×15 across (model, layer).
2. **Bio fine-tuning produces a learned directional separation.**
   `sl_eat_bio_ssl_all` drops bio-vs-non-bio top-10 cos to 0.57 at L9 vs a
   random-init baseline of 0.91 at the same layer.
3. **Late-layer collapse splits the family by `_bio` vs not.**
   `sl_eat_all_ssl_all` L12 eff_rank (11.2) ≈ random-init baseline (9.8).
4. **Architecture sets manifold dim, training expands the linear envelope.**
   Random-init MLE-ID = 11–15; trained MLE-ID = 7–14. Training expands the
   eff_rank/MLE-ID ratio from ~1 (random) to 17–43 (trained).
5. **Probe accuracy scales with phylogenetic distance.** Same-genus pairs peak
   at T5–T6 (~85–93%); cross-order pairs separate by T0–T1 (~97–99%). This is
   the probe-level signature of the geometric compression Evan documents.

## Conventions

- Random seed: 42 throughout for data subsampling and random-init.
- Plots: 150 dpi, `bbox_inches="tight"`, PNG.
- PCA to 50 dims before logistic regression probes (768-dim is too slow on CPU).
- All artifacts under `artifacts/comparisons/` are committed; shards under
  `artifacts/roadmap_part1/` are gitignored. Audio files and `activations/` gitignored.
- Suppress sklearn convergence warnings with `python -W ignore <script>.py`.
- When extending the pipeline, write the metric definition once in
  `step2_tier1_frame_level.py` and import elsewhere — do not duplicate primitives.
