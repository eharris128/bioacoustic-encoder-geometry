# Sentient Futures — Geometry of ESP-AVES2 EAT audio encoders

Geometric interpretability of the **ESP-AVES2 EAT-family** of audio encoders
released by [Earth Species Project](https://www.earthspecies.org/). We
collect residual-stream activations from four EAT checkpoints over a frozen
slice of `EarthSpeciesProject/NatureLM-audio-training` and study how the
geometry of those activations is shaped by training. A random-init EAT
baseline (architecture only, no learned weights) anchors absolute
magnitudes.

See [`RESULTS.md`](RESULTS.md) for the running narrative (CLAIM / RETRACTED
/ OPEN sections).

## Models

| key                       | description                                    |
|---------------------------|------------------------------------------------|
| `eat_all`                 | EAT pretrained on bio + non-bio audio          |
| `eat_bio`                 | EAT pretrained on bio-only audio               |
| `sl_eat_all_ssl_all`      | `eat_all` + SSL fine-tune on bio + non-bio     |
| `sl_eat_bio_ssl_all`      | `eat_bio` + SSL fine-tune on bio + non-bio     |
| `random_init_eat_seed42`  | EAT-base architecture, random reinit at seed 42|

All five expose 13 layers (`model.pos_drop` + 12 transformer blocks) at
hidden dim 768. Two extra random-init seeds (7 and 13) are extracted to
validate init variability; only their per-seed CSVs persist on disk.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio transformers huggingface_hub safetensors \
            pyarrow matplotlib scikit-learn scipy timm
```

The EAT base architecture is fetched on first use via
`AutoModel.from_pretrained("worstchan/EAT-base_epoch30_pretrain", trust_remote_code=True)`.
Per-checkpoint safetensors pull from `EarthSpeciesProject/esp-aves2-*`.
Audio comes from `EarthSpeciesProject/NatureLM-audio-training` parquet
shards via `hf_hub_download` (cached under `~/.cache/huggingface/hub/`,
~14 GB). Network access to Hugging Face is required for any extraction.

## Pipeline

### Step 1 — extraction

```bash
python collect_esp_aves2_activations.py \
  --manifest artifacts/manifests/naturelm_by_source_100each_20260418T171459Z.jsonl \
  --models eat_all,eat_bio,sl_eat_all_ssl_all,sl_eat_bio_ssl_all
```

Hooks the 13 layers, forwards each manifest item, and writes shards
(`(B, 13, 513, 768)` in float16, ~25 samples per shard) to
`artifacts/roadmap_part1/<manifest>/<model>/shards/`. Resumable. The
`--random-init-seed N` path loads EAT-base, then walks `init_weights` +
`reset_parameters` + a `normal(0, 0.02)` fallback for the 2/150 parameters
those paths miss.

### Step 2 — geometry analysis

Each script reads from shards and writes to
`artifacts/comparisons/<manifest>/nway_eat_all4/<subdir>/`:

| script                              | purpose                                     |
|-------------------------------------|---------------------------------------------|
| `nway_compare_eat_models.py`        | Pooled embeddings + cross-model CKA        |
| `step2_spectral_dim_eat.py`         | Singular values, `eff_rank`, PR, TwoNN per (model, layer) |
| `step2_subspace_angles_eat.py`      | L2-norm histograms + top-10 subspace overlap (across-layer / across-model / bio-vs-non-bio) |
| `step2_pooled_vs_frame_eat.py`      | Pooled-vs-frame distortion check on `sl_eat_bio_ssl_all` |
| `step2_tier1_frame_level.py`        | Frame-level `eff_rank` / PR / TwoNN / MLE-ID across all four trained models |
| `step2_random_init_compare.py`      | 5-way comparison: trained vs random-init   |
| `step2_random_init_variability.py`  | Init variability across seeds 7 / 13 / 42  |
| `step2_taxonomic_frame_level.py`    | Per-Class and per-Order frame-level metrics |
| `step3a_audio_mixing_pilot.py`      | Mixing-ratio sweep on bio axis             |
| `step3b_species_barycenters.py`     | Within-class species barycenters           |
| `step3c_veitch_hierarchy.py`        | Veitch orthogonality test for Class / Order|

Frame-level analyses subsample 50 frames per item uniformly from the
valid-token range with seed 42 (600 × 50 = 30,000 rows per (model, layer));
TwoNN and MLE-ID further subsample to 10,000 rows.

## Geometry primitives

Defined in `step2_tier1_frame_level.py` and imported elsewhere:

- **Effective rank** = `exp(-Σ p_i log p_i)` over normalized eigenvalues of the centered covariance.
- **Participation ratio** = `(Σλ)² / Σλ²`.
- **Intrinsic dimension** — TwoNN (k=2) and MLE-ID (k=20). MLE-ID is the preferred estimator; TwoNN has a known L4 failure mode (see `RESULTS.md` §7).
- **Subspace overlap** — `mean(cos(principal_angles))` between top-k=10 PCA bases via `scipy.linalg.subspace_angles`. 1.0 = identical, 0.0 = orthogonal.

## Pooling convention

Pooled comparisons take the mean over `tokens[1:valid_token_count]`,
**skipping token 0** (EAT's CLS-like token). See
`compare_esp_aves2_models.pooled_layer_vectors`. Frame-level analyses
include token 0.

## Data

- `artifacts/manifests/naturelm_by_source_100each_20260418T171459Z.jsonl` — frozen 600-sample manifest (100 × 7 source datasets), tracked.
- `artifacts/roadmap_part1/<manifest>/<model>/shards/` — per-model activation shards (~5.8 GB / model, gitignored).
- `artifacts/comparisons/<manifest>/nway_eat_all4/...` — committed CSVs and plots.

## Headline findings

1. **Pooling distorts geometry.** Frame-level `eff_rank` > pooled `eff_rank` everywhere; the ratio varies ×2–×15 across (model, layer).
2. **Bio fine-tuning produces a learned directional separation.** `sl_eat_bio_ssl_all` drops bio-vs-non-bio top-10 cos to 0.57 at L9 vs a random-init baseline of 0.91 at the same layer.
3. **Late-layer collapse splits the family by `_bio` vs not.** `sl_eat_all_ssl_all` L12 `eff_rank` (11.2) is essentially identical to the random-init baseline (9.8); the bio fine-tunes retain 180+ at L12.
4. **Architecture sets manifold dim, training expands the linear envelope.** Random-init MLE-ID = 11–15; trained MLE-ID = 7–14. Training does not widen the manifold — it expands the `eff_rank` / MLE-ID *ratio* from ~1 (random) to 17–43 (trained).
5. **Init variability is tight.** Seeds 7 / 13 / 42 random-init `eff_rank` spreads ≤ 1.3 across all layers vs trained-vs-random gaps of ~200–350.

See `RESULTS.md` for the full claim list with retractions.

## Conventions

- Random seed: 42 throughout for data subsampling and random-init.
- Plots: 150 dpi, `bbox_inches="tight"`, PNG.
- Suppress sklearn convergence warnings with `python -W ignore <script>.py`.
- New metric primitives go in `step2_tier1_frame_level.py` and are imported elsewhere — do not duplicate across scripts.

## Reference

- Roadmap PDF: [`references/roadmaps/aves2_interp_roadmap.pdf`](references/roadmaps/aves2_interp_roadmap.pdf)
