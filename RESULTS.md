# Results: geometry of the ESP-AVES2 EAT-family on a NatureLM-audio sample

**Status: working draft, not preprint-ready.** Sections marked **CLAIM** are
findings we believe; **RETRACTED** are earlier claims that did not survive
later analysis; **OPEN** are questions raised but not resolved here. Numbers
trace back to committed CSVs and scripts named in each section. Single-seed
point estimates throughout — no bootstrap CIs yet.

**Last update: 2026-04-27** — added random-init EAT baseline (§2), revised
§3–§6 to anchor against it, and confirmed §2 numbers are stable across init
(seeds 7 / 13 / 42). The baseline substantially strengthens §3
(bio-vs-nonbio) and §5 (late-layer collapse), and *re-frames* §6 (intrinsic
dim) — the original "conserved at 7–14" claim missed that the random-init
baseline reads in the same range.

---

## 1. Setup

- **Trained models (n=4).** Four ESP-AVES2 checkpoints from the Earth Species
  Project HuggingFace org, all of the EAT family:
  - `eat_all` — EAT pretrained on the union of bio + non-bio audio.
  - `eat_bio` — EAT pretrained on bio-only audio.
  - `sl_eat_all_ssl_all` — `eat_all` with an SSL fine-tune on the bio + non-bio
    union.
  - `sl_eat_bio_ssl_all` — `eat_bio` with an SSL fine-tune on the bio + non-bio
    union.
  All four are 13-layer transformers (L0 = post-projection, L1–L12 = transformer
  blocks) with hidden dim 768.
- **Baseline (added 2026-04-27).** `random_init_eat_seed42` — same EAT-base
  architecture instantiated via `AutoModel.from_pretrained` then completely
  reinitialized (HF `init_weights` + per-module `reset_parameters`, with a
  `normal(0, 0.02)` fallback for the 2/150 parameters those paths miss). Same
  inputs, same extraction protocol, same metrics. Anchors absolute numbers in
  §3–§6.
- **Manifest.** Frozen sample manifest
  `naturelm_by_source_100each_20260418T171459Z`: 100 samples × 7 source datasets
  = 600 samples shared across all five models. Sources are NatureLM-audio
  training sources; 4 of 7 are "bio" (Xeno-canto, iNaturalist, Animal Sound
  Archive, Watkins) and 3 are non-bio.
- **Activation extraction.** Per-model per-sample, all 13 layers' activations
  stored at `(513, 768)` per layer (513 patch tokens including a CLS-like
  token at index 0) in `artifacts/roadmap_part1/<manifest>/<model>/shards/`.
- **Frame subsampling.** Where frame-level analyses subsample, we draw 50
  frames per item uniformly from the valid-token range (seed 42), giving
  600 × 50 = 30,000 rows per (model, layer).
- **Pooled extraction.** Mean over `tokens[1:valid_token_count]` (skipping the
  CLS-like token at index 0) gives one 768-dim vector per (model, layer,
  sample).
- **Geometry primitives.**
  - *Effective rank:* `exp(-Σ p_i log p_i)` over normalized eigenvalues of the
    centered covariance.
  - *Participation ratio:* `(Σλ)² / Σλ²`.
  - *Intrinsic dimension:* TwoNN (Facco et al. 2017, k=2) and MLE-ID
    (Levina-Bickel 2005, k=20). Both subsample 10,000 rows.
  - *Subspace overlap:* mean(cos(principal angles)) between top-k=10 PCA bases
    via `scipy.linalg.subspace_angles`. 1.0 = identical, 0.0 = orthogonal.

---

## 2. The random-init EAT baseline

Same architecture, same inputs, no learned weights. Frame-level results
(seed 42, 50 frames × 600 items = 30,000 rows per layer) — flat across the
network:

| layer | eff_rank | TwoNN | MLE-ID(k=20) | bio-vs-non-bio cos (top-10) |
|------:|---------:|------:|-------------:|----------------------------:|
| 0     |    10.14 |  5.14 |         9.45 |                       0.987 |
| 4     |    11.75 |  8.20 |        13.23 |                       0.991 |
| 7     |    11.28 |  8.17 |        12.15 |                       0.989 |
| 9     |    10.77 |  8.02 |        11.64 |                       0.909 |
| 12    |     9.76 |  7.67 |        10.95 |                       0.957 |

Eff_rank stays in 10–12 across all 13 layers. MLE-ID stays in 11–15. Bio
and non-bio frames live in essentially the same top-10 subspace (cos ≥ 0.91
everywhere, ≥ 0.98 in early layers). Pooled eff_rank is even more compressed
— **1.3–2.1 across every layer**, vs trained models' 3–148 pooled eff_rank.

Source: `artifacts/.../nway_eat_all4/random_init_baseline/`. Plots:
`frame_effective_rank_all5.png`, `frame_mle_id_all5.png`,
`frame_bio_vs_nonbio_all5.png`.

This baseline anchors absolute numbers in every claim that follows. The key
facts it pins down:

- **No layer-wise eff_rank growth without learning.** The architecture +
  random weights produces a flat eff_rank curve. Every climb in §3 is
  attributable to learned weights, not to depth or to LayerNorm dynamics.
- **No directional class separation without learning.** Bio and non-bio frames
  are essentially indistinguishable in the random-init top-10 subspace.
  Every drop in §4 below ~0.95 is attributable to learning.
- **Manifold intrinsic dim is set near 11 by the architecture alone.**
  Trained models' MLE-ID range (7–14) is *at or below* baseline. Training
  does not increase manifold dim — it slightly compresses it while massively
  expanding the linear envelope.

**Init variability (validated 2026-04-27, seeds 7 / 13 / 42).** The headline
numbers above are not seed-specific. Across the three seeds, per-layer
spreads are:

| metric (frame-level)             | max(max-min) across layers | max std across layers |
|----------------------------------|---------------------------:|----------------------:|
| Effective rank                   |                       1.29 |                  0.65 |
| MLE-ID(k=20)                     |                       0.51 |                  0.27 |
| Bio-vs-non-bio cos (top-10)      |                      0.069 |                     — |

These spreads are ~30–500× smaller than the trained-vs-random gaps they
anchor (eff_rank trained vs random ≈ 200–340; cos trained vs random ≈ 0.34
at the L9 minimum). Source:
`random_init_variability/seed_spread_frame_stats.csv`,
`seed_spread_bio_vs_nonbio.csv`, plots in `init_variability_*.png`. Shards
for seeds 7 and 13 are deleted after stat extraction; per-seed CSVs persist.

---

## 3. CLAIM — Mean-pooling materially distorts the linear geometry, in the
same direction for every model in the family

**What we found.** Frame-level effective rank is strictly greater than pooled
effective rank for every (model, layer). Now anchored against random-init,
which sits at frame-level eff_rank 10–12 and pooled eff_rank 1.3–2.1:

| layer | random_init (frame / pooled) | eat_all (f / p) | sl_eat_bio_ssl_all (f / p) |
|------:|-----------------------------:|----------------:|---------------------------:|
| 0     | 10.14 / 1.33 (×7.6)          | 11.0 / 3.0 (×3.7)| 17.5 / 2.8 (×6.2)         |
| 9     | 10.77 / 2.12 (×5.1)          | 192 / 70 (×2.7) | 315 / 148 (×2.1)           |
| 12    |  9.76 / 2.09 (×4.7)          |  63 / 17 (×3.8) | 189 / 108 (×1.8)           |

Pooling compresses every model's linear subspace, baseline included. Source:
`pooled_per_layer_stats_all5.csv`, `frame_per_layer_stats_all5.csv`.

**Why it matters.** Mean-pooling a sequence into a single 768-dim vector folds
within-clip variance (silence, vocalization, noise, distinct call elements)
into the across-clip variance the spectrum measures. Pooling therefore
*understates* the linear-subspace width the model traverses. The baseline
makes the effect cleanest: random init has nothing to pool *toward*, and the
pooled rank collapses to ~2 — almost all variance lives in within-clip
fluctuations the architecture imposes, not in across-clip differences.

**Caveat for paper-grade.** The "frame-level" representation here uses a
50-frames-per-item uniform subsample. We have not tested sensitivity to the
number of frames (10/100/all) or to the sampling rule (uniform vs stratified
by activity).

---

## 4. CLAIM — Bio fine-tuning separates "bio" from "non-bio" inputs in
subspace direction; the effect is unambiguously *learned*

**What we found.** Top-10 subspace overlap (mean cos principal angles)
between bio-only and non-bio-only frames, per (model, layer). Random-init
provides a clean upper-bound (cos ≈ 0.99 = "identical subspaces"):

| layer | random_init | eat_all | eat_bio | sl_eat_all_ssl_all | sl_eat_bio_ssl_all |
|------:|------------:|--------:|--------:|-------------------:|-------------------:|
| 0     |       0.987 |   0.899 |   0.819 |              0.769 |              0.899 |
| 6     |       0.991 |   0.785 |   0.744 |              0.733 |          **0.648** |
| 7     |       0.989 |   0.752 |   0.755 |              0.727 |          **0.601** |
| 9     |       0.909 |   0.810 |   0.800 |              0.684 |          **0.570** |

Source: `frame_bio_vs_nonbio_all5.csv`.

**Two facts the baseline pins down:**

1. **All four trained models drop substantially below the baseline.** Even
   at L0, where trained models look "barely separated" at 0.77–0.90, the
   baseline reads 0.987 — so a 0.10–0.22 drop from baseline is real
   directional learning, not just architectural noise.
2. **`sl_eat_bio_ssl_all` is uniquely separated.** It minimizes at 0.570 (L9)
   vs the baseline's 0.909 at the same layer — a gap of **0.34** between
   "bio fine-tune + SSL" and "no learning." The other three trained models
   drop only 0.10–0.30 from baseline.

The pooled comparison is even more dramatic:

| layer | random_init | eat_all | sl_eat_bio_ssl_all |
|------:|------------:|--------:|-------------------:|
| 9     |       0.907 |   0.545 |          **0.332** |

Source: `pooled_bio_vs_nonbio_all5.csv`. So the originally headlined "0.33
at L9" pooled number IS real — it's just that pooling inflates the *gap*
relative to baseline by squeezing both bio and non-bio into a few directions.

**Caveat for paper-grade.** No statistical test on the gap. Bootstrap over
items (resample with replacement, recompute basis + angles) would give a CI
on each cos value. Until that is done, the "consistently lowest" framing is
a visual claim.

---

## 5. CLAIM — Late-layer collapse splits the family by `_bio` vs not, and
`sl_eat_all_ssl_all` collapses *back to the random-init baseline*

**What we found.** Frame-level L11–L12 effective rank, against the baseline:

| layer | random_init | eat_all | eat_bio | sl_eat_all_ssl_all | sl_eat_bio_ssl_all |
|------:|------------:|--------:|--------:|-------------------:|-------------------:|
| 10    |       10.31 |   184.6 |   165.6 |              160.5 |              283.3 |
| 11    |       10.18 |   224.4 |   180.5 |               62.4 |              253.9 |
| 12    |        9.76 |    62.6 |   180.4 |          **11.17** |              188.7 |

Source: `frame_per_layer_stats_all5.csv`.

**Key facts the baseline pins down:**

1. **`sl_eat_all_ssl_all` L12 eff_rank (11.17) is essentially identical to
   random-init L12 (9.76).** SSL fine-tuning on top of `eat_all` drives the
   final-layer linear subspace down to baseline width — i.e., the L12
   representation is no wider than what the architecture would produce with
   no training at all. This is a striking result.
2. **`eat_bio` and `sl_eat_bio_ssl_all` L12 eff_rank (180+) is 18× the
   baseline.** The two bio-pretrained models retain dramatic linear-subspace
   width through the output layer.
3. **`eat_all` L12 eff_rank (62.6) sits between baseline and the bio models
   — about 6× the baseline.** Partial collapse.

**Why it matters.** "Late-layer collapse" is no longer a vague qualitative
claim — it has a clean reference point. The bio pretrain genuinely preserves
output-layer variance; SSL on top of non-bio pretrain undoes it down to
baseline.

**Caveat for paper-grade.** Two models per condition (bio-pretrained vs not)
is too small a factorial to attribute the effect cleanly to bio data
specifically. It could equally be attributed to checkpoint-specific
properties of `eat_bio`. Mixed-data pretraining controls would be needed to
disentangle.

---

## 6. CLAIM — Training expands the linear envelope by ~30×, while keeping
the local manifold dim at or below the random-init baseline

**Previous framing (now wrong):** "Intrinsic dimension is conserved across
the family at 7–14; the linear envelope is what differs." This implied the
trained 7–14 range was a learned property.

**Corrected framing.** Random-init MLE-ID(k=20) frame-level reads **11–15**
across all layers. Trained models read **7–14**. The trained 7–14 range is
*at or slightly below* the architecture-only baseline. So:

- *Manifold dimension is set near 11 by the architecture, not by training.*
  No claim about training "compressing the manifold" can survive — most of
  the manifold-dim signal exists at random init.
- *The interesting learned property is the ratio* eff_rank / MLE-ID, which
  separates random-init from the trained models cleanly:

| layer | random_init | eat_all | eat_bio | sl_eat_all_ssl_all | sl_eat_bio_ssl_all |
|------:|------------:|--------:|--------:|-------------------:|-------------------:|
| 7     |  11.3 / 12.2 = **0.93** | 260 / 11.4 = **22.8** | 200 / 11.5 = **17.4** | 210 / 13.1 = **16.0** | 348 / 9.1 = **38.2** |
| 9     |  10.8 / 11.6 = **0.93** | 192 / 9.7 = **19.8**  | 159 / 9.3 = **17.1**  | 185 / 8.8 = **21.1**  | 315 / 7.2 = **43.5** |

Source: `frame_per_layer_stats_all5.csv`.

The random-init ratio sits near 1.0 — its data lives in a manifold roughly
as wide as its linear envelope (no useful low-dim curvature). Trained models
push this ratio to 17–43 — they wrap a low-dim manifold inside a *much*
wider linear subspace. **That ratio expansion is the learned property**, not
the absolute value of either dimension.

**Caveat for paper-grade.** MLE-ID with k=20 averages over 19 distance ratios
per point and is much more stable than TwoNN(k=2). But we have not
cross-checked against a third intrinsic-dim estimator (correlation dimension,
Hidalgo). Single-seed.

---

## 7. RETRACTED — The L4 TwoNN dip

**Original claim** (from `step2_pooled_vs_frame_eat.py` on `sl_eat_bio_ssl_all`,
2026-04-26): TwoNN intrinsic dim drops to ~2.6 at L4 sandwiched between ~10
and ~7. Reported as a possible model phenomenon worth investigating.

**Why retracted.** Running TwoNN(k=2) and MLE-ID(k=20) side-by-side across all
four trained models shows three of four exhibit the L4 TwoNN crash:

| model              | L3 TwoNN | L4 TwoNN | L5 TwoNN | L4 MLE-ID(k=20) |
|--------------------|---------:|---------:|---------:|----------------:|
| eat_all            | 10.06    | **0.95** | 8.64     | 13.27           |
| eat_bio            | 9.05     | 9.42     | 9.33     | 11.53           |
| sl_eat_all_ssl_all | 8.67     | **1.67** | 8.71     | 13.69           |
| sl_eat_bio_ssl_all | 10.16    | **2.63** | 7.36     | 14.05           |

MLE-ID with k=20 reads stable values at L4 in all four models. The L4 dip is
a TwoNN(k=2) failure mode triggered by a subset of these models, not a
geometric phenomenon. The random-init baseline does not show an L4 dip
(TwoNN 8.20 at L4) — but does show an isolated L11 TwoNN crash (2.53)
unrelated to training. Source: `frame_twonn_vs_mle_id_all4.png`,
`frame_per_layer_stats_all5.csv`.

---

## 8. RETRACTED — "Effective rank ≈ 3 at L0 across all four models means a
shared tokenizer subspace"

**Original claim** (from `step2_spectral_dim_eat.py`, 2026-04-26): all four
models had pooled L0 eff_rank ~3, suggesting a convergent low-dim L0 subspace.

**Why retracted.** Pooled L0 eff_rank ≈ 3 was a pooling artifact. Frame-level
L0 eff_rank spreads to **11 / 26 / 44 / 17** across the trained models, with
a random-init baseline of **10.14**. The "convergent ~3" was mean-pooling
collapsing within-clip variance into a few across-clip directions across all
five models — including the random-init one (pooled L0 = 1.33).

The cross-model L0 subspace cos analysis from `step2_subspace_angles_eat.py`
(2026-04-26) had already partially weakened this: three trained models
agree at cos 0.91–0.98 but `eat_bio` is ~orthogonal at cos 0.28–0.32. Not
convergent in any meaningful sense.

---

## 9. OPEN — Questions that have not been answered

1. **Bootstrap CIs on every claim.** Single-seed, single-subsample throughout.
   Bootstrap over items (resample with replacement, recompute basis +
   eigendecomposition + angles) would give CIs on every number in §3–§6.
   Most important now that the baseline gives the trained-vs-random gap a
   clean magnitude — we need to know its uncertainty.
2. **Sensitivity to frame count.** All frame-level numbers use 50 frames/item.
   Robustness to 10 / 100 / all-valid is untested.
3. **Sensitivity to top-k for subspace overlap.** Bio-vs-non-bio overlap
   uses k=10. The effect could strengthen or weaken at k=5 or k=50.
4. **Mechanism for late-layer `_bio` vs not collapse split (§5).** Why does
   `sl_eat_all_ssl_all` collapse to baseline at L12 while `sl_eat_bio_ssl_all`
   stays at 18× baseline? Candidate explanations: (a) bio pretraining produces
   an output-side representation used for finer-grained downstream tasks;
   (b) SSL collapse interacts with pretraining data in a specific way;
   (c) checkpoint-specific quirks of `eat_bio`. None are tested.
5. **Per-source frame-level structure.** All frame-level analyses pool across
   the 7 source datasets. The pooled per-source eff_rank slicing in
   `step2_spectral_dim/effective_rank_by_source.csv` should be redone at
   frame level.
6. **Within-clip frame structure.** Frames within a single clip almost
   certainly have very different geometry (silence vs vocalization). The
   subsampling treats all frames as exchangeable. Whether this matters for
   any of the claims is untested.
7. **Audio mixing along the bio↔non-bio direction (§4 follow-up).** Take a
   bio clip A and a non-bio clip B, generate audio mixtures
   `M(α) = (1-α)·A + α·B` for α ∈ {0, 0.25, 0.5, 0.75, 1}, run each through
   `sl_eat_bio_ssl_all`, project onto the top-10 bio-only and non-bio-only
   subspaces from §4. Three diagnostic outcomes: (a) **smooth linear
   interpolation** of cos angles in α — bio-vs-non-bio is implemented as a
   linear feature; (b) **sharp threshold** — gating/attention mechanism, not
   a continuous feature; (c) **off-manifold excursion** at intermediate α
   (MLE-ID jumps) — model treats mixtures as out-of-distribution. Converts
   §4 from a descriptive observation to a mechanistic claim. Requires the
   HF NatureLM-audio-training parquet cache (raw audio waveforms) — *do not
   delete* until this is run. Compute is small (~50-100 mixed clips × 1
   model × 13 layers, ~10 minutes once scripted).

---

## 10. What this looks like as a preprint

The random-init baseline strengthens the case for any of three framings.
None has been picked.

- **"Bio-vs-non-bio fine-tuning leaves a directional signature in
  mid-network."** §4 + §5, anchored by §2's baseline. The story is now:
  random init gives essentially no class separation (cos 0.95+); SSL +
  bio-pretrained pushes mid-network cos to 0.57; SSL + non-bio-pretrained
  collapses output-layer width back to the random baseline. Two clean
  learned effects that move in opposite directions.
- **"Mean-pooling distorts the linear geometry of audio-encoder
  representations."** §3 + §7 + §8, anchored by §2's pooled-vs-frame
  baseline. Method-paper flavor; cleanest if we add a non-EAT control.
- **"Training expands the linear envelope while preserving the local
  manifold dim set by architecture."** §6, anchored by §2. Most novel framing
  and arguably most interesting, but also the one most exposed to objections
  about MLE-ID estimator robustness.

---

## 11. Provenance

- Active scripts: `step2_spectral_dim_eat.py`, `step2_subspace_angles_eat.py`,
  `step2_pooled_vs_frame_eat.py`, `step2_tier1_frame_level.py`,
  `nway_compare_eat_models.py`, `collect_esp_aves2_activations.py`,
  `step2_random_init_compare.py`, `step2_random_init_variability.py`.
- Active artifacts:
  `artifacts/comparisons/<manifest>/nway_eat_all4/{step2_spectral_dim,step2_subspace_angles,step2_pooled_vs_frame,step2_tier1_frame_level,random_init_baseline,random_init_variability}/`.
- Random-init shards: only `random_init_eat_seed42` is retained on disk.
  Seeds 7 and 13 were extracted, stats computed, and shards deleted to fit
  the 234G partition; per-seed stat CSVs persist under
  `random_init_variability/`.
- Manifest: `naturelm_by_source_100each_20260418T171459Z`.
- Random seeds: 42 throughout for data subsampling. Random-init seeds
  evaluated: 7, 13, 42.
- Last update: 2026-04-27.
