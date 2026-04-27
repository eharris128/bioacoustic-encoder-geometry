# Results: geometry of the ESP-AVES2 EAT-family on a NatureLM-audio sample

**Status: working draft, not preprint-ready.** Sections marked **CLAIM** are
findings we believe; **RETRACTED** are earlier claims that did not survive
later analysis; **OPEN** are questions raised but not resolved here. Numbers
trace back to committed CSVs and scripts named in each section. Single-seed
point estimates throughout — no bootstrap CIs yet.

---

## 1. Setup

- **Models (n=4).** Four ESP-AVES2 checkpoints from the Earth Species Project
  HuggingFace org, all of the EAT family:
  - `eat_all` — EAT pretrained on the union of bio + non-bio audio.
  - `eat_bio` — EAT pretrained on bio-only audio.
  - `sl_eat_all_ssl_all` — `eat_all` with an SSL fine-tune on the bio + non-bio
    union.
  - `sl_eat_bio_ssl_all` — `eat_bio` with an SSL fine-tune on the bio + non-bio
    union.
  All four are 13-layer transformers (L0 = post-projection, L1–L12 = transformer
  blocks) with hidden dim 768.
- **Manifest.** Frozen sample manifest
  `naturelm_by_source_100each_20260418T171459Z`: 100 samples × 7 source datasets
  = 600 samples shared across all four models. Sources are NatureLM-audio
  training sources; 4 of 7 are "bio" (Xeno-canto, iNaturalist, Animal Sound
  Archive, Watkins) and 3 are non-bio (FSD50K, FreeSound, etc., grouped under
  "non-nature").
- **Activation extraction.** Per-model per-sample, all 13 layers' frame-level
  activations stored in `artifacts/roadmap_part1/<manifest>/<model>/shards/`.
- **Frame subsampling.** Where frame-level analyses subsample, we draw 50
  frames per item uniformly from the valid-token range (seed 42), giving
  600 × 50 = 30,000 rows per (model, layer).
- **Pooled extraction.** Mean-pooling over the valid-token range gives one
  768-dim vector per (model, layer, sample). Pooled tensors are consolidated
  in `artifacts/comparisons/<manifest>/nway_eat_all4/pooled_embeddings_all4.npz`.
- **Geometry primitives.**
  - *Effective rank:* `exp(-Σ p_i log p_i)` over normalized eigenvalues of the
    centered covariance.
  - *Participation ratio:* `(Σλ)² / Σλ²`.
  - *Intrinsic dimension:* TwoNN (Facco et al. 2017, k=2) and MLE-ID
    (Levina-Bickel 2005, k=20). Both subsample 10,000 rows.
  - *Subspace overlap:* mean(cos(principal angles)) between top-k=10 PCA bases
    via `scipy.linalg.subspace_angles`. 1.0 = identical, 0.0 = orthogonal.

---

## 2. CLAIM — Mean-pooling materially distorts the linear geometry, in the
same direction for every model in the family

**What we found.** For every (model, layer), frame-level effective rank is
strictly greater than pooled effective rank. The ratio varies dramatically:

| layer | eat_all | eat_bio | sl_eat_all_ssl_all | sl_eat_bio_ssl_all |
|------:|--------:|--------:|-------------------:|-------------------:|
| 0     | 11.0 / 3.0 (×3.7) | 26.0 / 2.9 (×8.9) | 43.6 / 3.0 (×14.5) | 17.5 / 2.8 (×6.2) |
| 9     | 192 / 70 (×2.7)   | 159 / 82 (×1.9)   | 185 / 84 (×2.2)    | 315 / 148 (×2.1)  |
| 12    | 63 / 17 (×3.8)    | 180 / 70 (×2.6)   | 11 / 6 (×1.9)      | 189 / 108 (×1.8)  |

Frame-level vs pooled, ratio in parentheses. Source:
`artifacts/.../step2_tier1_frame_level/pooled_vs_frame_summary_all4.csv`.

**Why it matters.** Mean-pooling a sequence into a single 768-dim vector folds
within-clip variance (silence, vocalization, noise, distinct call elements)
into the across-clip variance the spectrum measures. Pooling therefore
*understates* the linear-subspace width the model actually traverses. Plot:
`pooled_vs_frame_effective_rank_all4.png`.

**Caveat for paper-grade.** The "frame-level" representation here uses a
50-frames-per-item uniform subsample. We have not tested sensitivity to the
number of frames (10/100/all) or to the sampling rule (uniform vs stratified
by activity).

---

## 3. CLAIM — Bio fine-tuning separates "bio" from "non-bio" inputs in
subspace direction, but the effect is smaller than mean-pooling suggested

**What we found.** Top-10 subspace overlap (mean cos principal angles)
between bio-only and non-bio-only frames, per (model, layer):

| layer | eat_all | eat_bio | sl_eat_all_ssl_all | sl_eat_bio_ssl_all |
|------:|--------:|--------:|-------------------:|-------------------:|
| 6     | 0.785   | 0.744   | 0.733              | **0.648**          |
| 7     | 0.752   | 0.755   | 0.727              | **0.601**          |
| 9     | 0.810   | 0.800   | 0.684              | **0.570**          |

`sl_eat_bio_ssl_all` (the bio fine-tune that also got an SSL pass) consistently
shows the lowest cos — i.e., the most directional separation between bio and
non-bio top-10 subspaces — across L6–L11. The minimum is 0.570 at L9. The
other three models stay in 0.68–0.81. Source:
`step2_tier1_frame_level/frame_bio_vs_nonbio_all4.csv`.

**Relation to last night's pooled claim.** The pooled-level analysis from
`10601b9` reported `sl_eat_bio_ssl_all` reaching cos = 0.33 at L9 vs 0.55–0.70
elsewhere. At frame level the minimum is 0.57 vs 0.68–0.81 — same
ordering, same model, same layer of strongest separation, but the absolute
gap between this model and the rest is roughly halved. Pooling inflated the
headline number.

**Caveat for paper-grade.** No statistical test on the gap. Bootstrap over
items (resample with replacement, recompute basis + angles) would give a CI
on each cos value and a p-value on the cross-model gap. Until that is done,
the "consistently lowest" framing is a visual claim, not a tested one.

---

## 4. CLAIM — Late-layer collapse splits the family by `_bio` vs not

**What we found.** Frame-level L11–L12 effective rank diverges sharply:

| layer | eat_all | eat_bio | sl_eat_all_ssl_all | sl_eat_bio_ssl_all |
|------:|--------:|--------:|-------------------:|-------------------:|
| 10    | 184.6   | 165.6   | 160.5              | 283.3              |
| 11    | 224.4   | 180.5   |  62.4              | 253.9              |
| 12    |  62.6   | 180.4   |  11.2              | 188.7              |

The two models trained without bio data in pretraining (`eat_all`,
`sl_eat_all_ssl_all`) collapse hard at the output: `sl_eat_all_ssl_all`
drops from eff_rank 160 at L10 to 11 at L12 (×14.4 compression). The two
models with bio pretraining (`eat_bio`, `sl_eat_bio_ssl_all`) retain
eff_rank ~180–190 through L12. SSL on top of `eat_all` collapses *more*
than `eat_all` alone (62 → 11 at L12); SSL on top of `eat_bio` collapses
*less* (180 → 189 at L12). Source: same CSV as §2.

**Why it matters.** This is a candidate "fine-tuning shapes terminal
representation" story specific to this family. The bio pretraining seems to
leave the output-side representation usable for downstream variance; the
non-bio pretraining + SSL combo over-compresses. We do not currently have a
mechanistic explanation.

**Caveat for paper-grade.** Two models per condition is too small a factorial
to attribute the effect cleanly to "bio pretraining." It could equally be
attributed to specific properties of the `eat_bio` checkpoint that don't
generalize. A control with bio + non-bio mixed pretraining at known mixing
ratios would be needed.

---

## 5. CLAIM — Intrinsic dimension is conserved across the family at 7–14;
the linear envelope is what differs

**What we found.** Frame-level MLE-ID(k=20) sits in the range 7–14 for every
(model, layer). Effective rank, by contrast, swings 11–348 across the same
(model, layer) cells. Source: `frame_per_layer_stats_all4.csv`.

**Interpretation.** The data lives on a curved low-dimensional manifold
(intrinsic dim ~10) embedded in a much wider linear subspace (eff_rank up to
~350). Both pretraining and fine-tuning move the linear envelope but leave
the local manifold dimension roughly invariant.

**Caveat.** MLE-ID is the standard cross-check for TwoNN; we have not
validated it against a third estimator (e.g., correlation dimension, Hidalgo).
The "~conserved" framing should be replaced with a tested claim before
publication.

---

## 6. RETRACTED — The L4 TwoNN dip

**Original claim** (from `step2_pooled_vs_frame_eat.py` on `sl_eat_bio_ssl_all`,
2026-04-26): TwoNN intrinsic dim drops to ~2.6 at L4 sandwiched between ~10
and ~7. Reported as a possible model phenomenon worth investigating.

**Why retracted.** Running TwoNN(k=2) and MLE-ID(k=20) side-by-side across all
four models shows three of four exhibit the L4 TwoNN crash:

| model              | L3 TwoNN | L4 TwoNN | L5 TwoNN | L4 MLE-ID(k=20) |
|--------------------|---------:|---------:|---------:|----------------:|
| eat_all            | 10.06    | **0.95** | 8.64     | 13.27           |
| eat_bio            | 9.05     | 9.42     | 9.33     | 11.53           |
| sl_eat_all_ssl_all | 8.67     | **1.67** | 8.71     | 13.69           |
| sl_eat_bio_ssl_all | 10.16    | **2.63** | 7.36     | 14.05           |

MLE-ID with k=20 reads stable values at L4 in all four models. The L4 dip is
a TwoNN(k=2) failure mode triggered by a subset of these models, not a
geometric phenomenon. Source: `frame_twonn_vs_mle_id_all4.png`,
`frame_per_layer_stats_all4.csv`.

---

## 7. RETRACTED — "Effective rank ≈ 3 at L0 across all four models means a
shared tokenizer subspace"

**Original claim** (from `step2_spectral_dim_eat.py`, 2026-04-26): all four
models had pooled L0 eff_rank ~3, suggesting a convergent low-dim L0 subspace.

**Why retracted.** This was already weakened in `step2_subspace_angles_eat.py`
(2026-04-26) when L0 across-model subspace cos was 0.91–0.98 for three models
but 0.28–0.32 for `eat_bio` — not actually convergent. The frame-level Tier 1
analysis closes the door entirely: pooled L0 eff_rank ≈ 3 was a pooling
artifact. Frame-level L0 eff_rank spreads to 11 / 26 / 44 / 17 across the
family (§2). The "convergent L0 dimensionality" claim was an artifact of
mean-pooling collapsing within-clip variance into a few across-clip directions.

---

## 8. OPEN — Questions raised by Tier 1 that have not been answered

1. **Bootstrap CIs on every claim.** None of the numbers above have
   uncertainty bars yet. Single-seed, single-subsample. Required before any
   paper claim.
2. **Random-init / shuffled-data control.** No baseline anchors any of the
   eff_rank or subspace-overlap numbers. Reviewers will demand at least one.
3. **Sensitivity to frame count.** All frame-level numbers use 50 frames/item.
   Robustness to 10 / 100 / all-valid is untested.
4. **Sensitivity to top-k for subspace overlap.** Bio-vs-non-bio overlap
   uses k=10. The effect could strengthen or weaken at k=5 or k=50.
5. **Mechanism for late-layer `_bio` vs not collapse split.** Why do the bio
   models retain L12 variance while non-bio models collapse? Candidate
   explanations: (a) bio pretraining produces an output-side representation
   used for finer-grained downstream tasks; (b) SSL collapse interacts with
   pretraining data in a specific way; (c) checkpoint-specific quirks of
   `eat_bio`. None of these are tested.
6. **Per-source frame-level structure.** All frame-level analyses pool across
   the 7 source datasets. The pooled per-source eff_rank slicing in
   `step2_spectral_dim/effective_rank_by_source.csv` should be redone at
   frame level.
7. **Within-clip frame structure.** Frames within a single clip almost
   certainly have very different geometry (silence vs vocalization). The
   subsampling treats all frames as exchangeable. Whether this matters for
   any of the claims above is untested.

---

## 9. What this looks like as a preprint

There are at least three plausible framings for the same body of evidence.
Picking among them changes which experiments are worth doing next.

- **"Mean-pooling distorts the linear geometry of audio-encoder
  representations."** §2 + §6 + §7 + a control on a non-EAT model. Method-
  paper flavor; clean, narrow, and we already have the data for the EAT half.
- **"Bio-vs-non-bio fine-tuning leaves a directional signature in
  mid-network."** §3 + §4 + a mechanistic follow-up. Bio-audio-research
  flavor; needs the bootstrap CIs and ideally a third bio-pretrained model.
- **"The geometry of the ESP-AVES2 EAT family."** §2–§7 as a descriptive
  paper on these four checkpoints specifically. Easier to write but less
  generalizable.

We have not picked one.

---

## 10. Provenance

- Active scripts: `step2_spectral_dim_eat.py`, `step2_subspace_angles_eat.py`,
  `step2_pooled_vs_frame_eat.py`, `step2_tier1_frame_level.py`,
  `nway_compare_eat_models.py`.
- Active artifacts:
  `artifacts/comparisons/<manifest>/nway_eat_all4/{step2_spectral_dim,step2_subspace_angles,step2_pooled_vs_frame,step2_tier1_frame_level}/`.
- Manifest: `naturelm_by_source_100each_20260418T171459Z`.
- Random seed: 42 throughout.
- Last update: 2026-04-27.
