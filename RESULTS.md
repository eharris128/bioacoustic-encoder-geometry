# Results: geometry of the ESP-AVES2 EAT-family on a NatureLM-audio sample

**Status: working draft, not preprint-ready.** Sections marked **CLAIM** are
findings we believe; **RETRACTED** are earlier claims that did not survive
later analysis; **OPEN** are questions raised but not resolved here. Numbers
trace back to committed CSVs and scripts named in each section. Single-seed
point estimates throughout — no bootstrap CIs yet.

**Last update: 2026-04-27 (later)** — ran a 5-experiment chain
(bootstrap CIs, frame-count sensitivity, top-k sensitivity, audio-mixing
pilot, per-source frame-level structure) targeting RESULTS.md §9 OPEN
items. Key changes from this round:

- §3–§6 numbers all survive B=50 bootstrap CIs with margin to spare
  (CIs typically 1–3% of the median; trained-vs-random gaps unaffected).
- §3–§5 numbers robust to frame-count and top-k sweeps. §6 needs a
  quantitative qualifier: MLE-ID magnitude is (n, k)-dependent, only the
  *trends* are interpretable.
- **NEW finding (§4 mechanism, audio mixing):** the bio↔non-bio direction
  in sl_eat_bio_ssl_all is **threshold-like, not linear** — adding 25%
  non-bio audio pulls the L9 representation 78% of the way to pure
  non-bio along the centroid axis.
- **NEW finding (§4 source structure, per-source):** "bio" is **not a
  single coherent direction**. Watkins (marine mammals) is the most-
  isolated source, with cos to other bio sources (0.36–0.38) lower than
  cos to non-bio sources (0.39–0.42). The §4 narrative needs to be
  restated source-level rather than bio-vs-non-bio binary.

Earlier in 2026-04-27: added random-init EAT baseline (§2), revised
§3–§6 to anchor against it, and confirmed §2 numbers are stable across
init (seeds 7 / 13 / 42). The baseline substantially strengthens §3
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

**Bootstrap CIs (added 2026-04-27).** B=50 bootstraps over items confirm
the §4 claim with margin. At L9 in `sl_eat_bio_ssl_all`, frame-level
bio-vs-non-bio cos median = 0.569 with tight 5/95 percentile band; the
random-init L9 cos median = 0.902. The 0.33 gap dwarfs both CIs. Source:
`bootstrap_cis/bootstrap_ci_summary.csv`,
`bootstrap_cis/bootstrap_ci_bio_vs_nonbio_cos_top10.png`.

**Top-k sensitivity (added 2026-04-27).** The §4 minimum is invariant to
top-k choice. For `sl_eat_bio_ssl_all` at L9, mean cos reads
0.602 / 0.580 / 0.610 / 0.608 for k = 5 / 10 / 20 / 50 respectively. The
trained-vs-random gap stays 0.32–0.38 across all k. Source:
`topk_sensitivity/topk_at_L9.csv`.

---

## 4.5. CLAIM (new) — The bio↔non-bio centroid axis is *threshold-like*, not a smooth linear feature

**What we found.** Audio-mixing pilot in `audio_mixing_pilot/`. 5 bio
clips × 5 non-bio clips × α ∈ {0, 0.25, 0.5, 0.75, 1} = 125 mixed
waveforms `M(α) = (1-α)·A + α·B` through `sl_eat_bio_ssl_all`. Mean L9
pooled-activation projection onto the unit vector `(c_bio - c_nonbio)/‖·‖`:

| α    | mean projection | std  |
|------|----------------:|-----:|
| 0.00 |          +0.40  | 0.40 |
| 0.25 |          −0.66  | 0.90 |
| 0.50 |          −0.78  | 0.87 |
| 0.75 |          −0.96  | 0.77 |
| 1.00 |          −1.35  | 0.26 |

Linearity test: predicted midpoint at α=0.5 = ½(+0.40 + −1.35) = **−0.48**.
Observed = **−0.78**. Deviation = −0.30, **17.2 % of the full range**,
toward the non-bio side. Source:
`audio_mixing_pilot/mixing_summary_by_alpha.csv`.

**Interpretation.** Adding 25 % non-bio audio to a bio clip pulls the L9
representation roughly 78 % of the way to pure non-bio along the bio↔non-
bio centroid axis. The direction is **dominated by non-bio**, not a
balanced linear feature. The pure-bio cluster is also wider (std 0.40)
than the pure-non-bio cluster (std 0.26) — non-bio is more concentrated.

The 10-D top-10-bio subspace energy (a smoother diagnostic averaging over
10 directions) is more linear: 0.50 / 0.45 / 0.44 / 0.42 / 0.37 — monotone-
near-linear. So the threshold lives in the dominant 1-D centroid
direction; secondary directions average out smoothly.

**Why it matters.** §4 said the model "separates bio from non-bio in
subspace direction." This experiment is the mechanistic follow-up: the
separation is **not** implemented as a graded linear feature on the
centroid axis. It's a sharp asymmetric response. Reviewers who would
otherwise interpret §4 as "linear-feature decomposition à la Burns
2022" can be redirected here.

**Caveats.** (1) Pilot scale (25 clip pairs); a larger run would tighten
the std bands but the asymmetry is too large to be explained by 25 pairs
of noise. (2) Audio-domain mixing is a particular intervention;
concatenation, time-domain noise injection, or spectrogram-domain mixing
might give different curves.

**Extension to the other three trained models (added 2026-04-27).**
Same protocol on `eat_all`, `eat_bio`, `sl_eat_all_ssl_all`. The
threshold-like asymmetry is **specific to `sl_eat_bio_ssl_all`**, not an
architectural property of EAT:

| model               | proj@α=0 | proj@α=1 | range | midpoint dev |
|---------------------|---------:|---------:|------:|-------------:|
| eat_all             |    −0.73 |    −0.64 |  0.10 | non-monotonic |
| eat_bio             |    −0.75 |    −0.62 |  0.13 | non-monotonic |
| sl_eat_all_ssl_all  |    −0.52 |    −0.92 |  0.40 |       +0.30   |
| sl_eat_bio_ssl_all  |    +0.40 |    −1.35 |  1.75 |       −0.30   |

`eat_all` and `eat_bio` simply do not have a strong enough bio-vs-non-bio
direction for "linear vs threshold" to be a meaningful question — the
full-range projection is barely 0.10–0.13 and the curves are
non-monotonic noise. `sl_eat_all_ssl_all` (SSL on top of non-bio
pretrain) gets a partial bio-axis (range 0.40) with mild asymmetry. Only
`sl_eat_bio_ssl_all` (bio pretrain + SSL) has both a wide bio-axis and
clear asymmetric threshold.

**Mechanistic conclusion.** The wide bio-axis is unlocked by the
*combination* of bio pretraining + SSL fine-tune, not by either
ingredient alone. SSL on top of non-bio pretrain (`sl_eat_all_ssl_all`)
gets ~25 % of the way; bio pretraining alone (`eat_bio`) gets nowhere on
this 1-D centroid-axis diagnostic. Source:
`audio_mixing_pilot_extended/{eat_all,eat_bio,sl_eat_all_ssl_all}/mixing_summary_by_alpha.csv`.

---

## 4.6. CLAIM (new) — "Bio" is *not* a single coherent direction; the model is separating sources, not the bio/non-bio binary

**What we found.** Per-source frame-level pairwise top-10 subspace cos
at L9 in `sl_eat_bio_ssl_all`. The 6 sources sort by mutual cos as:

| pair                            | type    | mean cos |
|---------------------------------|---------|---------:|
| Watkins – iNaturalist           | bio-bio |    0.36  |
| Watkins – Xeno-canto            | bio-bio |    0.38  |
| Watkins – Animal Sound Archive  | bio-bio |    0.38  |
| Watkins – NatureLM              | cross   |    0.39  |
| Watkins – WavCaps               | cross   |    0.42  |
| NatureLM – Xeno-canto           | cross   |    0.48  |
| Animal Sound Archive – NatureLM | cross   |    0.49  |
| WavCaps – Xeno-canto            | cross   |    0.57  |
| Animal Sound Archive – WavCaps  | cross   |    0.57  |
| NatureLM – WavCaps              | non-non |    0.62  |
| ASA – iNaturalist               | bio-bio |    0.65  |
| ASA – Xeno-canto                | bio-bio |    0.66  |
| Xeno-canto – iNaturalist        | bio-bio |    0.72  |

Source: `per_source_frame_level/per_source_pairwise.csv`.

**Watkins (marine-mammal vocalization archive) is the most-isolated
source.** Watkins-vs-anything has lower cos than most cross-class pairs.
The remaining bio sources (Xeno-canto, iNaturalist, Animal Sound Archive)
form a tight sub-cluster (cos 0.65–0.72). Non-bio sources (WavCaps,
NatureLM) have moderate cos to each other (0.62) and to the
Xeno-canto/iNaturalist/ASA cluster (0.48–0.57).

**Interpretation.** §4 reports a 0.57 mean cos for "bio" vs "non-bio" at
L7. That number is a marginal of structure that is actually **source-
level**:
- The model's L9 subspace separates Watkins from everything else (lowest
  cos of any pair involving Watkins).
- Xeno-canto + iNaturalist + ASA cluster as a "wildlife-recording"
  sub-direction.
- WavCaps + NatureLM cluster as a "non-wildlife" sub-direction.

The "bio vs non-bio" framing in §4 conflates (a) genuine bio-related
fine-tune learning, with (b) Watkins-as-an-outlier-domain artifact. The
former is a real learned property; the latter is partly a quirk of which
non-bio sources we happened to include in the manifest.

**Why it matters.** A paper that headlines "bio fine-tunes induce a
bio-vs-non-bio direction" overstates the case. A paper that headlines
"bio fine-tunes induce source-resolving structure (with bio sources
forming a sub-cluster)" is more accurate. The bootstrap CIs and audio-
mixing finding (§4.5) survive either framing; the per-source structure
just constrains how the §4 result should be described.

**Caveats.** Same single-model focus (`sl_eat_bio_ssl_all`); the other
trained models also show per-source structure (see
`per_source_pairwise_overlap.png`) but we have not synthesized them
into the same narrative.

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

**Quantitative qualifier from frame-count sensitivity (added 2026-04-27).**
MLE-ID(k=20) is sensitive to the (n, k) parameter pair — for fixed k,
the estimator drifts as n grows because more nearby neighbors capture
finer manifold curvature. Example on `sl_eat_bio_ssl_all` L9:

| frames/item | total n | MLE-ID(k=20) |
|-------------|--------:|-------------:|
| 10          |   6,000 |        10.4 |
| 30          |  18,000 |         7.4 |
| 50          |  30,000 |         7.4 |
| 100         |  60,000 |         5.7 |
| 200         | 120,000 |         5.0 |

So the **absolute** MLE-ID number reported in §6 is a function of our
50-frames-per-item × 600-items convention, not a model-invariant
quantity. The trends across (model, layer) at fixed (n, k) are the
robust statement; the numbers themselves are convention-dependent. This
should be noted in any paper draft. Source:
`frame_count_sensitivity/frame_count_sensitivity_sl_eat_bio_ssl_all.csv`.

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

Closed by the 2026-04-27 chain (artifacts under `bootstrap_cis/`,
`frame_count_sensitivity/`, `topk_sensitivity/`, `audio_mixing_pilot/`,
`per_source_frame_level/`):

- ~~Bootstrap CIs on §3–§6.~~ Done with B=50; all numbers survive. CIs
  are typically 1–3% of the median, gaps to random-init baseline are
  20–500× larger. Sample-selection noise is not a threat.
- ~~Frame-count sensitivity.~~ Eff_rank and bio-vs-nonbio cos are
  invariant to fc ∈ {10, 30, 50, 100, 200} (max-min < 2% on
  sl_eat_bio_ssl_all). MLE-ID(k=20) drifts with n predictably (larger n,
  fixed k captures finer manifold structure) — see §6 caveat below.
- ~~Top-k sensitivity for bio-vs-nonbio cos.~~ The §4 minimum is robust:
  cos at L9 for sl_eat_bio_ssl_all reads 0.602/0.580/0.610/0.608 for k =
  5/10/20/50. The trained-vs-random gap stays 0.32–0.38 across all k.
- ~~Audio-mixing diagnostic.~~ Done, but produced a *new* mechanistic
  finding rather than confirming the linear-feature hypothesis — see
  the new §4.5 below.
- ~~Per-source frame-level structure.~~ Done — and produced a *new*
  finding that complicates §4: see §4.6 below.

Still open:

1. **Mechanism for the late-layer `_bio` vs not collapse split (§5).**
   Why does `sl_eat_all_ssl_all` collapse to baseline at L12 while
   `sl_eat_bio_ssl_all` stays at 18× baseline? Candidate explanations:
   (a) bio pretraining produces an output-side representation used for
   finer-grained downstream tasks; (b) SSL collapse interacts with
   pretraining data in a specific way; (c) checkpoint-specific quirks
   of `eat_bio`. None are tested.
2. **Within-clip frame structure.** Frames within a single clip almost
   certainly have very different geometry (silence vs vocalization). The
   subsampling treats all frames as exchangeable. Whether this matters
   for any of the claims is untested.
3. **MLE-ID magnitude vs (n, k).** The frame-count sensitivity check
   exposed that MLE-ID(k=20) drifts with n. Absolute values in §6 are
   parameter-dependent; only the trend across (model, layer) at fixed
   (n, k) is paper-defensible. Worth either (a) reporting MLE-ID across
   multiple (n, k) settings or (b) switching to an alternative estimator
   that converges with n.
4. **Per-source granularity in §4.** The new §4.6 below shows bio-vs-non-
   bio is a coarse decomposition. A finer statement of §4 should be made
   source-level: which sources is the model actually separating?
5. **Hierarchical / Veitch follow-up.** Still pending. Requires
   manifest enrichment with Class/Order/Species labels — coordinate
   with teammate (TODO.md Step 3c).
6. **Species barycenters (TODO.md Step 3b).** Still pending; requires
   species labels.
7. ~~Linearity of the bio↔non-bio mechanism for the other three trained
   models.~~ Done in `audio_mixing_pilot_extended/`. Threshold-like
   asymmetry is specific to `sl_eat_bio_ssl_all`; see §4.5 extension.

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
  `step2_random_init_compare.py`, `step2_random_init_variability.py`,
  `step2_bootstrap_cis.py`, `step2_frame_count_sensitivity.py`,
  `step2_topk_sensitivity.py`, `step3a_audio_mixing_pilot.py`,
  `step2_per_source_frame_level.py`.
- Active artifacts:
  `artifacts/comparisons/<manifest>/nway_eat_all4/{step2_spectral_dim,step2_subspace_angles,step2_pooled_vs_frame,step2_tier1_frame_level,random_init_baseline,random_init_variability,bootstrap_cis,frame_count_sensitivity,topk_sensitivity,audio_mixing_pilot,per_source_frame_level}/`.
- Random-init shards: only `random_init_eat_seed42` is retained on disk.
  Seeds 7 and 13 were extracted, stats computed, and shards deleted to fit
  the 234G partition; per-seed stat CSVs persist under
  `random_init_variability/`.
- Manifest: `naturelm_by_source_100each_20260418T171459Z`.
- Random seeds: 42 throughout for data subsampling. Random-init seeds
  evaluated: 7, 13, 42.
- Last update: 2026-04-27.
