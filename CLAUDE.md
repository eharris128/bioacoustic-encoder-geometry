# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geometric interpretability of the **ESP-AVES2 EAT-family** of audio encoders
released by the Earth Species Project. We collect residual-stream activations
from 4 EAT checkpoints over a frozen slice of NatureLM-audio-training and
study how the geometry of those activations is shaped by training. Plus a
random-init EAT baseline (architecture only, no learned weights) that anchors
absolute magnitudes.

The project pivoted from an earlier exploratory phase on the legacy AVES
torchaudio model + Bullfinch recordings; that phase is **out of scope**. See
`memory/MEMORY.md` and the project memory at
`~/.claude-heron/projects/-home-evan-projects-sentient-futures/memory/`.

Read `RESULTS.md` for the running narrative (CLAIM / RETRACTED / OPEN
sections) and `TODO.md` for what's next.

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
```

The EAT base architecture is fetched on first use via
`AutoModel.from_pretrained("worstchan/EAT-base_epoch30_pretrain", trust_remote_code=True)`.
Per-checkpoint safetensors are pulled from the corresponding
`EarthSpeciesProject/esp-aves2-*` HF repos. Audio comes from the
`EarthSpeciesProject/NatureLM-audio-training` parquet shards via
`hf_hub_download` — these get cached under `~/.cache/huggingface/hub/`
(~14G; required for any new extraction or audio-mixing experiment).

## Running scripts

All scripts are standalone, run from the project root. No test suite — each
script writes CSVs/PNGs to `artifacts/comparisons/...` and prints results to
stdout.

```bash
source venv/bin/activate
python -W ignore <script>.py
```

## Pipeline

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
`normal(0, 0.02)` fallback for the 2/150 parameters those paths miss.

### Step 2 — geometry analysis

Each script reads from shards and writes to
`artifacts/comparisons/<manifest>/nway_eat_all4/<subdir>/`:

| script                              | purpose                                     |
|-------------------------------------|---------------------------------------------|
| `nway_compare_eat_models.py`        | Consolidated pooled embeddings + cross-model CKA |
| `step2_spectral_dim_eat.py`         | Singular values, eff_rank, PR, TwoNN per (model, layer) |
| `step2_subspace_angles_eat.py`      | L2-norm histograms + across-layer/model/bio-vs-non-bio top-10 subspace overlap |
| `step2_pooled_vs_frame_eat.py`      | Pooled-vs-frame distortion check on `sl_eat_bio_ssl_all` |
| `step2_tier1_frame_level.py`        | Frame-level eff_rank/PR/TwoNN/MLE-ID across all 4 trained models + bio-vs-non-bio at frame level |
| `step2_random_init_compare.py`      | 5-way comparison: trained models vs random-init baseline |
| `step2_random_init_variability.py`  | Init variability across random-init seeds 7/13/42 |

### Frame-level subsampling

Where frame-level analyses subsample, we draw 50 frames per item uniformly
from the valid-token range with seed 42 (600 × 50 = 30,000 rows per
(model, layer)). TwoNN and MLE-ID further subsample to 10,000 rows.

## Geometry primitives

Defined in `step2_tier1_frame_level.py`; reused via import by later scripts.

- **Effective rank** = `exp(-Σ p_i log p_i)` over normalized eigenvalues of
  the centered covariance.
- **Participation ratio** = `(Σλ)² / Σλ²`.
- **Intrinsic dimension** — TwoNN(k=2) and MLE-ID(k=20). MLE-ID is the
  preferred estimator; TwoNN has a known L4 failure mode (see
  `RESULTS.md` §7).
- **Subspace overlap** — `mean(cos(principal_angles))` between top-k=10 PCA
  bases via `scipy.linalg.subspace_angles`. 1.0 = identical, 0.0 =
  orthogonal.

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
  the 4 trained + `random_init_eat_seed42`. Seeds 7 and 13 were extracted,
  stats computed, shards deleted.
- `artifacts/comparisons/<manifest>/nway_eat_all4/{step2_*,random_init_*}/`
  — committed CSVs, plots, and per-seed stats.

Raw audio waveforms live in the HF parquet cache
(`~/.cache/huggingface/hub/datasets--EarthSpeciesProject--NatureLM-audio-training/`,
~14G). **Do not delete** while audio-mixing follow-ups are open.

## Key findings (current state — see `RESULTS.md` for full claims + retractions)

1. **Pooling distorts geometry** universally. Frame-level eff_rank > pooled
   eff_rank everywhere; the ratio varies ×2–×15 across (model, layer).
2. **Bio fine-tuning produces a learned directional separation.**
   `sl_eat_bio_ssl_all` drops bio-vs-non-bio top-10 cos to 0.57 at L9 vs a
   random-init baseline of 0.91 at the same layer.
3. **Late-layer collapse splits the family by `_bio` vs not.**
   `sl_eat_all_ssl_all` L12 eff_rank (11.2) is essentially identical to the
   random-init baseline (9.8); the bio fine-tunes retain 180+ at L12.
4. **Architecture sets manifold dim, training expands the linear envelope.**
   Random-init MLE-ID = 11–15; trained MLE-ID = 7–14. Training does not
   widen the manifold — it expands the eff_rank/MLE-ID *ratio* from ~1
   (random) to 17–43 (trained).
5. **Init variability is tight.** Seeds 7/13/42 random-init eff_rank spreads
   ≤1.3 across all layers vs trained-vs-random gaps of ~200–350.

Retracted: the L4 TwoNN dip (estimator artifact) and the "pooled L0 ≈ 3
across all four models = shared tokenizer" story (pooling artifact). See
`RESULTS.md` §7–§8.

## Scope and ownership

- **In scope (us):** Step 1 + Step 2 + Step 3a (audio mixing) + Step 3b
  (species barycenters) + Step 3c (Veitch hierarchy test). Plus per-Class
  and per-Order taxonomic resolution at frame level (the geometric
  complement to the teammate's probes). See `TODO.md`.
- **Owned by teammate:** linear probes, attribution, noise dynamics. Do not
  duplicate. Coordinate on manifest enrichment with Class/Order/Species
  labels (the teammate already has them via probe training).
- **Out of scope:** Section 3 (SAEs, dictionary learning), legacy AVES
  exploration, cross-species call-type transfer, RSA with CRCNS zebra-finch.

## Conventions

- Random seed: 42 throughout for data subsampling and random-init.
- Plots: 150 dpi, `bbox_inches="tight"`, PNG.
- All artifacts under `artifacts/comparisons/` are committed; shards under
  `artifacts/roadmap_part1/` are gitignored.
- Suppress sklearn convergence warnings with `python -W ignore <script>.py`.
- When extending the pipeline, write the metric definition once in
  `step2_tier1_frame_level.py` and import elsewhere — do not duplicate
  primitives across scripts.
