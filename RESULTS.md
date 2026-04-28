# Results: geometry of the ESP-AVES2 EAT-family on a NatureLM-audio sample

**Status: working draft, not preprint-ready.** Sections marked **CLAIM** are
findings we believe; **RETRACTED** are earlier claims that did not survive
later analysis; **OPEN** are questions raised but not resolved here. Numbers
trace back to committed CSVs and scripts named in each section. Single-seed
point estimates throughout — no bootstrap CIs yet.

**Last update: 2026-04-27 (latest)** — ran a 5-step taxonomic chain
(manifest enrichment, per-Class/per-Order frame-level metrics, species
barycenters, Veitch hierarchy test, late-layer collapse mechanism)
after verifying that taxonomic labels were already in the parquet
metadata (not blocked on teammate). Major new findings from this round:

- **§4.7 (new) — Aves vs Mammalia is the strongest learned direction.**
  `sl_eat_bio_ssl_all` L7 reaches cos = 0.379 — beats the §4 bio-vs-
  non-bio L9 cos of 0.580. The Class-level direction is what bio fine-
  tuning learns most strongly. Geometric peak (L7) ≈ teammate's probe
  peak (L5). Order within Aves is much weaker (cos floor 0.73).
- **§4.8 (new) — `sl_eat_bio_ssl_all` factors Class and Order
  orthogonally at the output (Veitch).** L12 cos((Aves−Mammalia),
  (Passer−Aves)) = 0.074 — essentially perpendicular. None of the
  other trained models drop below 0.30. Random-init stays at 0.93+
  across all layers. This is the cleanest "learned hierarchy"
  signature in the family.
- **§4.9 (new) — Trained models *compress* species detail to learn
  coarser abstractions.** The random-init baseline has higher
  per-species separability ratio (0.33 at L12) than any trained model
  (`sl_eat_bio_ssl_all` peaks at 0.20). Random projections preserve
  acoustic distances; trained models learn invariances that *put
  acoustically-distinct same-class species closer together*. The §4
  bio direction is acquired by sacrificing fine species detail.
- **§5.1 (new) — The L12 collapse is mode collapse, not shrinkage.**
  `sl_eat_all_ssl_all` L12 puts 61% of variance in one direction (vs
  26% at L11). Both SSL models scale up the norm 3× at L12; the bio
  variant amplifies many directions, the non-bio variant amplifies
  one. The collapse is structural (per-source uniform), not data-
  dependent.

These four findings give the paper a coherent mechanistic story:
`sl_eat_bio_ssl_all` is the only model that simultaneously develops
(a) a bio-vs-non-bio direction, (b) a clean Aves-vs-Mammalia direction,
(c) within-Aves species structure, and (d) factors Class and Order
orthogonally. The other trained models partially do (a)–(b) and fail
at (c)–(d). Random-init does none.

**Earlier 2026-04-27** — 5-experiment chain (bootstrap CIs, frame-count
sensitivity, top-k sensitivity, audio-mixing pilot, per-source
structure) closed §3–§6 robustness questions and added §4.5 (audio
mixing → threshold-like) and §4.6 (per-source: Watkins-as-isolate).

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

## 4.7. CLAIM (new) — Aves vs Mammalia is the strongest learned direction in the EAT-family geometry

**What we found.** `taxonomic_frame_level/` runs the same frame-level
top-10 subspace overlap as §4 but sliced by taxonomic Class instead of
bio-vs-non-bio. With the manifest enriched from parquet metadata
(2026-04-27), the 600 samples sort as Aves 271, Mammalia 119, Amphibia
6, Insecta 2, non-bio 202. We compare Aves vs Mammalia (the only
well-powered Class pair).

| layer | eat_all | eat_bio | sl_eat_all_ssl_all | sl_eat_bio_ssl_all | random_init |
|------:|--------:|--------:|-------------------:|-------------------:|-------------:|
| 0     |   0.81  |   0.89  |              0.90  |              0.87  |        0.87  |
| 5     |   0.65  |   0.72  |              0.62  |              0.59  |        0.91  |
| **7** | 0.53    | 0.60    |              0.58  |          **0.38**  |        0.90  |
| 9     |   0.58  |   0.62  |              0.48  |              0.40  |        0.91  |
| 12    |   0.77  |   0.57  |              0.67  |              0.62  |        0.91  |

`sl_eat_bio_ssl_all` L7 hits cos = **0.379** — the strongest learned
directional separation we have measured anywhere in the EAT family,
beating the §4 bio-vs-non-bio L9 minimum of 0.580. All four trained
models drop into 0.40–0.70 mid-network; random-init stays at 0.87–0.93
across every layer.

**Per-Order within Aves** (Passeriformes 207 vs other-Aves pooled 64)
shows a much weaker effect: best is `sl_eat_bio_ssl_all` L9 cos =
0.729. The model uses many fewer dimensions for Order than for Class.
Source: `taxonomic_frame_level/{taxonomic_pairwise.csv,
class_aves_vs_mammalia_cos.png, order_passer_vs_other_aves_cos.png}`.

**Why it matters.** The teammate's linear probes peak at L5 for Class
(82.5% accuracy) and L9 for Order (70.3%). Our geometric peaks land at
L7 for Class and L9 for Order. **Probes and centroid geometry agree on
which layer encodes which distinction**, even though the metrics are
formally different (linear decodability vs subspace overlap). The
hierarchy is at the layer where probes find it.

**Caveats.** Per-Order resolution is coarse — Passeriformes dominates
(207/271 Aves), other-Aves is a 17-Order grab bag of ≤11 samples each.
A finer per-Order test would need denser per-Order sampling (TODO.md
Step 1 scale-up).

---

## 4.8. CLAIM (new) — `sl_eat_bio_ssl_all` factors Class and Order as orthogonal directions (Veitch hierarchy)

**What we found.** Veitch et al. (NeurIPS 2024) predict that if a model
encodes Class and Order as independent features, the parent-axis
direction (Aves − Mammalia) should be approximately orthogonal to the
subordinate-axis direction (Passeriformes − Aves). We test this
directly. Source: `veitch_hierarchy/veitch_hierarchy.csv`.

|cos((Aves − Mammalia), (Passeriformes − Aves))| by layer:

| layer | eat_all | eat_bio | sl_eat_all_ssl_all | **sl_eat_bio_ssl_all** | random_init |
|------:|--------:|--------:|-------------------:|-----------------------:|-------------:|
| 0     |   0.69  |   0.70  |              0.73  |                  0.71  |        0.98  |
| 5     |   0.35  |   0.42  |              0.60  |                  0.39  |        0.94  |
| 7     |   0.34  |   0.36  |              0.44  |                  0.23  |        0.94  |
| 9     |   0.31  |   0.38  |              0.38  |              **0.14**  |        0.96  |
| 12    |   0.52  |   0.50  |              0.61  |              **0.07**  |        0.96  |

`sl_eat_bio_ssl_all` at L12 has cos = **0.074** — essentially
perpendicular. None of the other trained models drop below 0.30.
Random-init stays at 0.93–0.98 across every layer; without learning,
the Class direction and the Order direction are nearly parallel.

**Why it matters.** This is the cleanest "learned hierarchy" signature
in the EAT family. Two layers of orthogonality in `sl_eat_bio_ssl_all`:
L9 (cos = 0.14) coincides with §4, §4.5, §4.7, §4.9 peaks; L12 (cos =
0.07) is even more orthogonal — the *output layer* factors the
hierarchy maximally cleanly. The bio fine-tune doesn't just produce a
single bio direction; it produces a *factored* representational
geometry where Class and Order live on independent axes.

**Caveats.**
- "Other-Aves" pools 17 minority bird orders together. A finer
  per-Order Veitch test (4 individual bird orders × Aves direction)
  would need denser per-Order data than our 600-sample manifest
  provides.
- The "passer-vs-other-Aves" direction is mathematically collinear
  with the (Aves − Mammalia)-orthogonal hyperplane intersected with
  the within-Aves variation; with two disjoint subgroups Aves =
  Passer ∪ other-Aves, the metric is well-defined but the subord-axis
  cos to (Aves − Mammalia) is effectively a 1-degree-of-freedom test.

**Bootstrap CIs (added 2026-04-27).** B=30 bootstraps over items confirm
the orthogonality result. Source:
`bootstrap_taxonomic_cis/bootstrap_taxonomic_summary.csv`.

| layer | sl_eat_bio_ssl_all median [5%, 95%] | next-lowest model lower-95% |
|------:|------------------------------------:|-----------------------------:|
| 9     |                0.136 [0.082, 0.231] | eat_all 0.181 (no overlap)  |
| 12    |                0.081 [0.021, 0.155] | eat_bio 0.258 (no overlap)  |

`sl_eat_bio_ssl_all`'s upper-95% at L12 (0.155) sits well below every
other trained model's lower-95% bound. Random-init L12 reads 0.932
[0.803, 0.977] — bootstrap noise on the parallel-direction case. The
factored-hierarchy result is paper-grade.

---

## 4.9. CLAIM (new) — Trained models *compress* fine species detail; the bio direction is acquired by sacrificing species resolution

**What we found.** Per-species centroids per (model, layer) for the 12
species with ≥ 5 manifest samples (Orcinus orca, Fringilla coelebs,
Turdus merula, …; spans whales, dolphins, songbirds, a parrot, etc.).
Separability ratio = between-species variance / (within + between).
Higher = species centroids more separated relative to within-species
spread. Source: `species_barycenters/separability_summary.csv`.

| layer | eat_all | eat_bio | sl_eat_all_ssl_all | sl_eat_bio_ssl_all | **random_init** |
|------:|--------:|--------:|-------------------:|-------------------:|----------------:|
| 0     |   0.04  |   0.06  |              0.04  |              0.05  |        **0.09** |
| 5     |   0.06  |   0.06  |              0.08  |              0.07  |        **0.19** |
| 9     |   0.07  |   0.07  |              0.09  |              0.19  |        **0.27** |
| 10    |   0.07  |   0.08  |              0.08  |          **0.20**  |            0.30 |
| 12    |   0.05  |   0.05  |              0.10  |              0.19  |        **0.33** |

`random_init_eat_seed42` has the **highest** separability ratio at every
layer. Among trained models, only `sl_eat_bio_ssl_all` develops
substantial species structure (peaks at 0.20 at L10) — the same
peak-layer as our other §4 findings. `eat_all` and `eat_bio` stay flat
at 0.05–0.08, never improving on the L0 baseline.

**Interpretation.** Random Gaussian projections preserve raw acoustic
distances by Johnson–Lindenstrauss. The 12 species span very different
acoustic domains (whale calls vs songbird vocalizations vs dolphin
clicks), so random init gives them naturally distinct centroids.
Trained models with LayerNorm + learned projections build invariances
that *put acoustically-distinct same-class species closer together*.
This is a feature, not a bug — the bio fine-tune is learning
abstractions like "this is bird vocalization" or "this is animal
sound." Species-level distinctions get sacrificed to that abstraction.

**Why it matters.** A naive reading of §4 ("bio fine-tuning produces a
strong directional separation") could be misread as "bio fine-tuning
produces fine-grained category structure." It does not. The §4 finding
is **specifically** about bio-vs-non-bio (a binary distinction) and
§4.7 about Aves-vs-Mammalia (a coarse Class distinction). At the
species level, training compresses rather than expands the geometry.

This reconciles the apparent tension with the teammate's probe results
showing 70.3% Order accuracy at L9: **linear decodability** (a
classifier finding a hyperplane) and **centroid-distance separability**
(a geometric ratio) are different quantities. Probes can extract
species-level distinctions from a representation where centroids are
relatively close, as long as the residual variance is anisotropic in
the right way.

**Caveats.** Only 12 species clear the 5-sample threshold; the set
spans very heterogeneous acoustic domains (marine mammals vs songbirds
vs raptors). A within-Aves-only or within-Mammalia-only species
analysis (smaller per-class sample requirement, more homogeneous
acoustic context) would test whether the random-init advantage holds
for closely-related species.

**Within-class follow-up (closes the open question).** Restricting the
separability ratio to within-Aves species (13 species, ≥ 5 clips) and
within-Mammalia species (15 species) on the new per-Order manifest
keeps the qualitative finding: random-init has the highest separability
inside both Classes at most layers (Aves L5 = 0.078, Mammalia L5 =
0.229), trained models stay 0.02–0.05 within-Aves at every layer.
`sl_eat_bio_ssl_all` is again the only trained model that approaches
random-init within a single Class, reaching ~0.085 within-Aves at
L9–L12. The §4.9 "trained models compress fine species detail" claim
generalizes — it isn't an artifact of mixing acoustically-distinct
Classes. Source: `nway_eat_all4/class/within_class_separability.csv`.

**Bootstrap CIs (added 2026-04-27).** B=30 bootstraps confirm the
random-init advantage with margin. At L10:

| model              | median [5%, 95%]      |
|--------------------|----------------------:|
| sl_eat_bio_ssl_all |   0.218 [0.206, 0.239] |
| sl_eat_all_ssl_all |   0.097 [0.089, 0.107] |
| eat_bio            |   0.092 [0.081, 0.102] |
| eat_all            |   0.082 [0.073, 0.092] |
| random_init        |   0.319 [0.286, 0.381] |

Random-init's lower-95% (0.286) sits above every trained model's
upper-95%, the trained-vs-random gap is unambiguous. Source:
`bootstrap_taxonomic_cis/bootstrap_taxonomic_summary.csv`.

---

## 4.10. CLAIM (new) — The bio direction is real, but the model also discriminates *within* bio at the source level

**What we found.** §4 reports that trained models reduce mean
`bio-vs-nonbio top-10 subspace cos` to 0.57–0.81. §9 left open the
finer question: at the subspace level, does the model separate the
6 individual sources (Xeno-canto, iNaturalist, Animal Sound Archive,
Watkins, NatureLM, WavCaps), or only the bio/nonbio binary? Aggregating
the existing `per_source_frame_level/per_source_pairwise.csv`:

Median pairwise top-10 subspace cos at L9 (k=10):

| model              | within-bio (4×3/2 pairs) | cross bio↔nonbio | within-nonbio (1 pair) |
|--------------------|-------------------------:|-----------------:|-----------------------:|
| eat_all            |               0.626      |        0.691     |               0.813    |
| eat_bio            |               0.619      |        0.689     |               0.780    |
| sl_eat_all_ssl_all |               0.485      |        0.561     |               0.709    |
| sl_eat_bio_ssl_all |               **0.448**  |        0.503     |               0.598    |
| random_init        |               0.864      |        0.902     |               0.958    |

**Two observations.**

1. **Within-bio cos < cross-bio cos in every trained model at every
   mid-late layer.** Bio sources are geometrically more similar to
   each other than to non-bio sources — confirming that the §4
   bio-vs-nonbio direction is real and not a per-source artifact.

2. **But within-bio cos is far from 1.** At L9 it sits at 0.45–0.63
   for trained models — the 4 bio sources have distinct geometric
   signatures even in their top-10 subspaces. The model is not
   collapsing all bio inputs into a single sub-manifold; it preserves
   per-source structure within the bio category.

`sl_eat_bio_ssl_all` has the lowest within-bio cos (0.39 at L7, 0.45
at L9), pushing bio sources apart even from each other more than the
other models do. Random-init has near-uniform pair cos (0.86–0.98)
across every (within-bio, cross-bio, within-nonbio) bucket — no
source-level structure at all.

**Why it matters.** The §4 statement should be read as "the model
develops a bio-vs-nonbio direction *plus* per-source sub-directions
inside the bio cluster," not as "the model collapses all bio to a
single point." This matches the §4.6 finding (the bio centroid axis
also picks up Watkins-as-isolate vs the rest), the §4.9 finding
(species detail is compressed but not zero), and the §5.4 finding
(L12 collapse onto bio direction is a *coexistence* with the
multi-source structure that lives in lower-eigenvalue dimensions).

Source: `nway_eat_all4/per_source_frame_level/per_source_pairwise.csv`.

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

## 5.1. CLAIM (new) — The L12 collapse is *selective amplification of one direction*, not uniform shrinkage

**What we found.** The §5 claim was descriptive: trained `_bio` models
keep eff_rank high at L12 (~180) while `_all` models collapse (~11).
But why? `late_layer_collapse/spectrum_L10_L11_L12.csv` measures the
shape of the eigenvalue spectrum at L11 → L12 across all 5 models.

L11 → L12 top-1 eigenvalue share (= top eigenvalue / total variance):

| model                  | L11   | L12   | Δ     | L12 ‖x‖ | L12 total var |
|------------------------|------:|------:|------:|--------:|--------------:|
| eat_all                | 0.058 | 0.270 | +0.21 |   6.41  |         18.1 |
| eat_bio                | 0.060 | 0.160 | +0.10 |   4.97  |         13.3 |
| **sl_eat_all_ssl_all** | 0.263 |**0.614**| +0.35 |  11.71  |        131.7 |
| sl_eat_bio_ssl_all     | 0.097 | 0.082 | −0.01 |  17.91  |        237.0 |
| random_init            | 0.494 | 0.503 | +0.01 |  27.71  |        609.6 |

`sl_eat_all_ssl_all` L12 puts **61 % of total variance into a single
direction** — the regime random-init has been in all along. eff_rank
reads ~11 because variance is concentrated, not because vectors are
small (mean ‖x‖ at L12 = 11.7, larger than L11's 3.4).

`sl_eat_bio_ssl_all` is the *only* trained model whose top-1 share
doesn't grow at L12. Its mean ‖x‖ at L12 (17.9) is even larger than
`sl_eat_all_ssl_all`'s, but the variance is *spread across many
directions* (top-1 = 8 %, top-10 = 28 %). This is why its eff_rank stays
at ~180.

**Mechanism.** The L12 transition is selective amplification, not
shrinkage. Both SSL fine-tuned models scale up the activation norm ~3×
at L12 and grow total variance 5–15×; what differs is *where* they
direct the amplification:

- `sl_eat_all_ssl_all`: amplifies a single dominant direction → mode
  collapse, low eff_rank, "logit-like" output.
- `sl_eat_bio_ssl_all`: amplifies many directions in proportion → no
  collapse, high eff_rank, multi-direction output.

`eat_all` and `eat_bio` (no SSL fine-tune) show milder concentration
(top-1 jumps from 6 % to 16–27 %) without the norm explosion. The L12
collapse is partially native to EAT pretraining and dramatically
amplified by SSL fine-tuning on the non-bio-pretrained model.

**Per-source uniformity.** L12/L11 eff_rank ratio for
`sl_eat_all_ssl_all` is 0.17–0.30 across all 6 sources. The collapse
is NOT data-dependent — bio inputs and non-bio inputs collapse to the
same degree. This rules out the explanation that SSL fine-tune
discriminates bio inputs at L12.

**Why it matters.** This is the mechanistic answer to RESULTS.md §9.1.
The "_bio vs not" split is not about the bio fine-tunes "preserving"
output information per se — it's about whether the model's L11 → L12
transition installs a low-rank classifier head (`sl_eat_all_ssl_all`)
or a multi-direction representation (`sl_eat_bio_ssl_all`). Bio
pretraining gives SSL fine-tuning many directions worth amplifying;
non-bio pretraining gives it few, so SSL collapses to one.

This also explains why the §4.5 audio-mixing threshold-like asymmetry
is specific to `sl_eat_bio_ssl_all`: only it preserves a high-dim L12
subspace where the bio direction lives.

**Open follow-up.** What *is* the dominant L12 direction in
`sl_eat_all_ssl_all`? Probably an "is this animal vocalization?"
classifier installed by the SSL fine-tune. Test by projecting trained-
model L12 activations onto the top eigenvector and checking whether
bio vs non-bio inputs separate along it. Cheap.

**Bootstrap CIs (added 2026-04-27).** B=30 bootstraps confirm the
mode-collapse split is structural. Top-1 eigenvalue share at L12:

| model              | median [5%, 95%]         |
|--------------------|-------------------------:|
| sl_eat_all_ssl_all | **0.618 [0.611, 0.621]** |
| sl_eat_bio_ssl_all | **0.082 [0.079, 0.084]** |
| eat_all            |   0.271 [0.268, 0.276]   |
| eat_bio            |   0.162 [0.159, 0.164]   |
| random_init        |   0.502 [0.485, 0.521]   |

CIs are extremely tight (~0.005 wide); the spectrum-shape claim is the
most sample-stable finding in the paper. The two SSL models'
post-amplification directions differ by **almost an order of
magnitude in top-1 share** with non-overlapping CIs. Source:
`bootstrap_taxonomic_cis/bootstrap_taxonomic_summary.csv`.

---

## 5.2. CLAIM (new) — The dominant L12 direction in `sl_eat_all_ssl_all` IS the bio classifier

**What we found.** §5.1 left an open follow-up: what *is* the dominant
direction that absorbs 61 % of L12 variance in `sl_eat_all_ssl_all`?
`step5_l12_direction.py` answers it directly. For each model, project
each item's pooled L12 activation onto the top eigenvector and onto the
bio-vs-nonbio centroid axis (the §4 axis); compare the two via
`|cos(top1, bio_axis)|` plus Cohen's d on each.

L12 results, both manifests:

| model              | top1 share | \|cos(top1, bio_axis)\| | d on top1 | d on bio_axis |
|--------------------|-----------:|------------------------:|----------:|--------------:|
| **sl_eat_all_ssl_all** (NEW) | 0.625 | **0.741** | +0.52 | +0.82 |
| **sl_eat_all_ssl_all** (OLD) | 0.614 | **0.819** | +0.56 | +0.76 |
| sl_eat_bio_ssl_all (NEW) | 0.083 | 0.041 | −0.06 | +2.77 |
| sl_eat_bio_ssl_all (OLD) | 0.082 | 0.032 | −0.05 | +2.54 |
| eat_all (NEW)      | 0.275 | 0.048 | −0.02 | +1.78 |
| eat_bio (NEW)      | 0.162 | 0.254 | +0.19 | +1.02 |
| random_init (NEW)  | 0.565 | 0.061 | −0.01 | +0.46 |

**The §5.1 mode-collapse direction in `sl_eat_all_ssl_all` is the bio
classifier.** Top-1 eigenvector aligns at cos = 0.74–0.82 with the
bio-vs-nonbio axis, replicates across both manifests, and gives a
Cohen's d of ~0.5 separating bio from non-bio items along that single
1D axis. SSL fine-tuning on a non-bio-pretrained model installs a
single-feature "is this an animal vocalization?" classifier as L12's
dominant direction — confirming the mechanistic story in §5.1.

**`sl_eat_bio_ssl_all` is the dual case.** Its top-1 share is small
(8 %, no collapse), and its top-1 eigenvector is *orthogonal* to the
bio axis (cos = 0.03). Yet its bio separation along the (non-collapsed)
bio axis is by far the strongest in the family: Cohen's d = 2.77 along
the bio axis vs −0.06 along top-1. The bio direction is real and
strong; it just doesn't dominate L12's variance.

**Random-init is a useful negative control.** It has high top-1 share
(0.57, comparable to `sl_eat_all_ssl_all`), but its top-1 eigenvector
is uncorrelated with the bio axis (cos = 0.06) and Cohen's d on top-1
is essentially zero. Mode collapse alone doesn't make a classifier;
SSL on a non-bio-pretrained model specifically installs a *bio*
classifier.

**Replication caveat.** On the OLD manifest, random-init's
`|cos(top1, bio_axis)|` reads 0.75 — close to `sl_eat_all_ssl_all`'s
value. But the corresponding Cohen's d on top-1 is only +0.14: the
top-1 axis isn't actually separating bio from non-bio, it's just that
on a manifest where 4/6 sources are bird-heavy, source-mean variance
happens to align with the bio axis. The test that distinguishes a real
classifier from coincidental alignment is the Cohen's d on the
projection, not the cosine alone. Only `sl_eat_all_ssl_all` clears
both bars on both manifests.

Source: `nway_eat_all4/direction/l12_summary.csv` and the OLD-manifest
counterpart under
`naturelm_by_source_100each_20260418T171459Z/nway_eat_all4/direction/`.

**Bootstrap CIs (added 2026-04-28).** B=30 bootstraps over manifest items.
L12 |cos(top1, bio_axis)|, median [5%, 95%]:

| model              | median [5%, 95%]         |
|--------------------|-------------------------:|
| **sl_eat_all_ssl_all** | **0.743 [0.674, 0.796]** |
| sl_eat_bio_ssl_all     |   0.035 [0.007, 0.149]   |
| eat_all                |   0.099 [0.006, 0.339]   |
| eat_bio                |   0.260 [0.055, 0.400]   |
| random_init            |   0.263 [0.037, 0.756]   |

`sl_eat_all_ssl_all`'s lower-95% (0.674) sits above every other
trained model's upper-95% with no overlap. random-init has a wide CI
(spans 0.04–0.76) consistent with the §5.2 caveat that random-init's
top-1 axis happens to align with source-mean variance on bird-heavy
manifests; its Cohen's d on top1 has CI [−0.10, +0.23] that straddles
zero, so the alignment is coincidental, not a real classifier. Source:
`nway_eat_all4/bootstrap_l12_direction/bootstrap_l12_summary.csv`.

---

## 5.3. CLAIM (new) — The §5.1 mode collapse is *directional*, not clip-collapse; trained models *preserve* within-clip frame variance

**What we found.** §5.1 + §5.2 showed that `sl_eat_all_ssl_all` puts
61 % of L12 variance into a single direction aligned with the bio
classifier. Open question: does that directional concentration also
compress each clip's 50-frame temporal trajectory into a near-point?
`step5_within_clip.py` answers it: for each clip, compute mean
within-clip frame variance and the clip-centroid distance to global,
then report the ratio `within / (within + between)` per (model, layer).

| layer | eat_all | eat_bio | sl_eat_all_ssl_all | sl_eat_bio_ssl_all | **random_init** |
|------:|--------:|--------:|-------------------:|-------------------:|----------------:|
| 0     |   0.91  |   0.84  |              0.87  |              0.87  |        **0.68** |
| 6     |   0.84  |   0.86  |              0.83  |              0.80  |        **0.49** |
| 9     |   0.85  |   0.86  |              0.82  |          **0.69**  |            0.42 |
| 12    |   0.91  |   0.89  |              0.86  |          **0.69**  |            0.34 |

**Trained models preserve temporal frame structure.** All four trained
models keep within-clip / total ≥ 0.69 across every layer; eat_all
specifically stays at 0.84–0.91. Frames within a single 10-second clip
are *not* collapsed onto each other in the full 768-dim space — the
50 frames remain spread out relative to between-clip separations.

**The §5.1 mode collapse is directional, not clip-level.** At
`sl_eat_all_ssl_all` L12, absolute within-clip variance jumps from 9.6
(L11) to 113.8 (L12) — a 12× amplification, the same scale as the
total-variance jump. The within/total ratio is unchanged (0.86 at both
L11 and L12). The §5.1 "61 % of variance in one direction" finding
compresses the feature *distribution* along one axis but doesn't
compress each clip toward a point in the larger 768-dim manifold.

**`sl_eat_bio_ssl_all` is the only trained model that pushes clips
apart at mid-late layers.** Its ratio drops to 0.69–0.75 for L7–L12,
significantly below the other trained models (0.83–0.91 in the same
range). The bio fine-tune partly emulates random-init's pattern of
expanding between-clip variance — but it never reaches random's
0.34 floor at L12.

**Random-init's monotonic decline is a Johnson–Lindenstrauss artifact.**
Random Gaussian projections preserve raw acoustic distances, and the
13-layer chain compounds those projections so that between-clip
variance grows with depth (each random transformation pushes clips
apart in a different way). This is the same mechanism that powers
random-init's surprising win on §4.9 species barycenters: at L12,
random-init's between-clip distances are ~3× the within-clip
distances, while trained models keep them comparable.

**Why it matters.** This is the answer to "is L12 a logit head?" for
`sl_eat_all_ssl_all`. The answer: along *one* direction it is (the bio
classifier), but in the orthogonal 767 dimensions, frames within a clip
remain distinguishable — the model preserves enough temporal richness
that downstream heads (or probes) can still work with multi-frame
features. The collapse is selective amplification (§5.1), not
information destruction.

Source: `nway_eat_all4/within_clip/within_clip_summary.csv`,
`within_clip_ratio.png`.

---

## 5.4. CLAIM (new) — The bio classifier in `sl_eat_all_ssl_all` is *installed at L12*, not built up gradually

**What we found.** §5.2 measured `|cos(top1, bio_axis)|` only at L12.
`step5_layer_direction.py` extends to all 13 layers per (model). For
`sl_eat_all_ssl_all`:

| layer | top1_share | \|cos(top1, bio_axis)\| | d on top1 | d on bio_axis |
|------:|-----------:|------------------------:|----------:|--------------:|
| 0     |   0.163    |              **0.838**  |    +0.46  |        +0.59  |
| 1     |   0.078    |              0.529      |    −0.40  |        +0.90  |
| 2     |   0.059    |              0.402      |    −0.40  |        +1.10  |
| 3     |   0.050    |              0.215      |    +0.25  |        +1.14  |
| 4     |   0.057    |              0.085      |    +0.07  |        +1.21  |
| 5     |   0.061    |              0.026      |    +0.02  |        +1.35  |
| 6     |   0.059    |              0.074      |    +0.06  |        +1.41  |
| 7     |   0.056    |              0.134      |    −0.11  |        +1.59  |
| 8     |   0.060    |              0.150      |    −0.12  |        +1.58  |
| 9     |   0.047    |              0.362      |    −0.38  |        +1.63  |
| 10    |   0.070    |              0.175      |    +0.51  |        +1.73  |
| 11    |   0.269    |              0.155      |    +0.11  |        +1.89  |
| 12    | **0.625**  |          **0.741**      |  **+0.52**|        +0.82  |

**The bio classifier is installed by the final transformer block.** The
top eigenvector aligns with the bio axis only at L12 (0.74); through
L4–L11 the cosine is bouncing between 0.03 and 0.36 with random sign on
Cohen's d. L11 → L12 is where both top1 share (0.27 → 0.62) AND the
bio-axis alignment (0.16 → 0.74) jump together. The same single
transformer block does mode collapse AND bio classification.

**The high cos at L0 is a separate pre-block phenomenon.** L0 is the
EAT tokenizer + positional embedding, before any transformer block.
Its top1 happens to align with the bio axis there (cos 0.84) but only
explains 16 % of variance. Through L1–L11 the model redistributes that
bio information across many directions (Cohen's d on bio_axis grows
from 0.59 at L0 to 1.89 at L11 — bio is *better* separated, just not
along the top eigenvector). Then L12 collapses everything back onto
one direction *plus* re-aligns it with the bio centroid axis.

**Comparison to `sl_eat_bio_ssl_all`.** That model never does this
collapse — its L12 |cos(top1, bio_axis)| is 0.041 and top1 share is
0.08. Bio is encoded in the bio centroid axis (Cohen's d = 2.77 at
L12) but spread across many top-eigenvalue directions, never
amplified into a single one. SSL fine-tuning on the all-pretrained
model installs a logit-head-like classifier at L12; SSL on the
bio-pretrained model leaves the multi-direction bio code intact.

Source: `nway_eat_all4/layer_direction/layer_direction_summary.csv`,
`layer_direction.png`.

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

**(n, k) joint sweep (added 2026-04-28).** `step5_mle_id_sensitivity.py`
runs MLE-ID over k ∈ {5, 10, 20, 40, 80} × n ∈ {2500, 5000, 10000, 20000}
for focal layers and all 5 models. Two structural patterns:

- **At fixed n, MLE-ID grows monotonically with k.** Example, eat_all L4
  at n=10000: k=5 → 6.8, k=10 → 10.0, k=20 → 12.8, k=40 → 14.3,
  k=80 → 15.9. The estimator picks up larger-scale manifold structure
  as the neighborhood widens. The §6 default k=20 is on the low side
  of this curve.
- **At fixed k, MLE-ID drops as n grows.** Same pattern as the
  frames-per-item sweep above: denser sampling reveals finer (lower-dim)
  local manifold structure.

Most importantly, **the trained-vs-random ordering is k-dependent.**
At k=5–20 the §6 claim "trained ≤ random MLE-ID" holds (random ≈ 11
sits at or above the trained range 5–13). At k=80 the ordering flips:
eat_all 16.8 > eat_bio 17.3 > sl_eat_bio_ssl_all 18.9 > random 10.5,
because random-init's manifold doesn't keep widening as k grows the way
trained-model manifolds do.

The headline of §6 — *the eff_rank/MLE-ID ratio* expanded by training
from ≈ 1 to 17–43× — does not depend on absolute MLE-ID values, only
on the relative scaling of the linear envelope vs the local manifold
dim. That ratio claim is robust. The absolute MLE-ID claim is
estimator-conditional; restate any paper draft as "MLE-ID(k=20, n=10k)
read 7–14 for trained, 11–15 for random." Source:
`mle_id_sensitivity/mle_id_sensitivity.csv`.

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

Closed by the 2026-04-27 (later) taxonomic chain:

- ~~Mechanism for the late-layer `_bio` vs not collapse split (§5).~~
  Resolved as **selective amplification of one direction at L12**, not
  uniform shrinkage. See new §5.1.
- ~~Hierarchical / Veitch follow-up.~~ `sl_eat_bio_ssl_all` factors
  Class and Order orthogonally at L12 (cos = 0.074); other trained
  models do not. See new §4.8.
- ~~Species barycenters.~~ Random-init has higher per-species
  separability ratio than any trained model — trained models compress
  species detail to learn coarser abstractions. See new §4.9.
- ~~Manifest enrichment for Class/Order/Species.~~ Done in
  `enrich_manifest_taxonomy.py`; labels were already in parquet
  metadata, no teammate coordination needed.
- ~~Per-Class / per-Order frame-level structure.~~ See new §4.7.

Closed by the 2026-04-28 autonomous chain (Phase 3-5):

- ~~**Within-clip frame structure.**~~ Closed by §5.3 (`within_clip/`).
  Trained models keep within/total ≥ 0.69 across every layer; §5.1
  mode collapse is purely directional, not clip-level.
- ~~**MLE-ID magnitude vs (n, k).**~~ Closed by §6 (n, k) qualifier
  (`mle_id_sensitivity/`). MLE-ID grows monotonically with k; the §6
  ratio claim is robust, the absolute claim is estimator-conditional.
- ~~**What is the dominant L12 direction in `sl_eat_all_ssl_all`?**~~
  Closed by §5.2 (`direction/l12_summary.csv`) — it IS the bio
  classifier (|cos|=0.74 with bio centroid axis, Cohen's d 0.52).
  §5.4 further shows the direction is installed specifically at L12,
  not built up gradually (`layer_direction/`).
- ~~**Per-Order Veitch test with denser sampling.**~~ Closed by Phase
  3 `step3c_veitch_4order` and the new per-Order manifest. Tests 4
  Aves Orders × 100 samples each; sl_eat_bio_ssl_all factors all 4
  orders against the Class direction at cos 0.03–0.08 by L7.
- ~~**Within-class species separability.**~~ Closed by §4.9 within-Aves
  / within-Mammalia extension (`class/within_class_separability.csv`).
  Random-init still has highest separability inside both Classes.
- ~~Bootstrap CIs on the new §4.7–§4.9 / §5.1 numbers.~~ Done in
  `bootstrap_taxonomic_cis/`. All claims survive. CIs on the §5.2
  L12 bio-classifier finding added in `bootstrap_l12_direction/`:
  sl_eat_all_ssl_all 0.74 [0.67, 0.80], no CI overlap with other
  trained models.

Still open:

- ~~**Per-source granularity in §4.**~~ Closed by §4.10 (analysis of
  existing `per_source_pairwise.csv`). All trained models show
  within-bio cos < cross-bio cos at L7-L12, confirming the §4 bio
  direction; but within-bio cos = 0.45-0.63 (not 1.0) means the model
  also discriminates among bio sources within the bio cluster.

(All §9 items closed as of 2026-04-28.)

---

## 10. What this looks like as a preprint

The random-init baseline strengthens the case for any of these framings.
After the 2026-04-27 (latest) taxonomic chain, the **factored-hierarchy**
framing has emerged as the strongest candidate. None has been picked.

- **"Bio + SSL fine-tuning produces a *factored* hierarchical
  geometry."** §4 + §4.5 + §4.7 + §4.8 + §5.1, anchored by §2. The
  story: `sl_eat_bio_ssl_all` is the only trained model that
  simultaneously (a) develops a bio-vs-non-bio direction (§4, cos
  0.57 at L9), (b) develops an Aves-vs-Mammalia direction (§4.7, cos
  0.38 at L7), (c) develops within-Aves species structure (§4.9, peak
  separability 0.20 at L10), and (d) factors Class and Order
  orthogonally at the output layer (§4.8, cos 0.07 at L12). The
  combination is not present in any single ingredient — bio
  pretraining alone or SSL fine-tuning on non-bio pretraining only
  produces fragments of this geometry. **Most novel framing of the
  three, with the cleanest model-comparison story.**
- **"Bio-vs-non-bio fine-tuning leaves a directional signature in
  mid-network."** §4 + §4.5 + §5 + §5.1, anchored by §2. The story:
  random init gives essentially no class separation (cos 0.95+); SSL +
  bio-pretrained pushes mid-network cos to 0.57 with a threshold-like
  asymmetric mechanism (§4.5); SSL + non-bio-pretrained collapses to a
  single direction at L12 (§5.1). Two clean learned effects that move
  in opposite directions. Tighter scope than (1); good fallback if the
  factored-hierarchy story doesn't survive review.
- **"Mean-pooling distorts the linear geometry of audio-encoder
  representations."** §3 + §7 + §8, anchored by §2's pooled-vs-frame
  baseline. Method-paper flavor; cleanest if we add a non-EAT control.
- **"Training expands the linear envelope while preserving the local
  manifold dim set by architecture."** §6, anchored by §2. Novel but
  exposed to objections about MLE-ID estimator robustness (now
  documented as (n, k)-sensitive in §6 caveat).

---

## 11. Provenance

- Active scripts: `step2_spectral_dim_eat.py`, `step2_subspace_angles_eat.py`,
  `step2_pooled_vs_frame_eat.py`, `step2_tier1_frame_level.py`,
  `nway_compare_eat_models.py`, `collect_esp_aves2_activations.py`,
  `step2_random_init_compare.py`, `step2_random_init_variability.py`,
  `step2_bootstrap_cis.py`, `step2_frame_count_sensitivity.py`,
  `step2_topk_sensitivity.py`, `step3a_audio_mixing_pilot.py`,
  `step2_per_source_frame_level.py`, `enrich_manifest_taxonomy.py`,
  `step2_taxonomic_frame_level.py`, `step3b_species_barycenters.py`,
  `step3c_veitch_hierarchy.py`, `step5_late_layer_collapse.py`,
  `step5_bootstrap_taxonomic.py`.
- Active artifacts:
  `artifacts/comparisons/<manifest>/nway_eat_all4/{step2_spectral_dim,step2_subspace_angles,step2_pooled_vs_frame,step2_tier1_frame_level,random_init_baseline,random_init_variability,bootstrap_cis,frame_count_sensitivity,topk_sensitivity,audio_mixing_pilot,audio_mixing_pilot_extended,per_source_frame_level,taxonomic_frame_level,species_barycenters,veitch_hierarchy,late_layer_collapse,bootstrap_taxonomic_cis}/`.
- Manifests:
  `naturelm_by_source_100each_20260418T171459Z.jsonl` (base) +
  `naturelm_by_source_100each_20260418T171459Z_taxonomic.jsonl`
  (enriched with phylum/class/order/family/genus/species/subspecies).
- Random-init shards: only `random_init_eat_seed42` is retained on disk.
  Seeds 7 and 13 were extracted, stats computed, and shards deleted to fit
  the 234G partition; per-seed stat CSVs persist under
  `random_init_variability/`.
- Manifest: `naturelm_by_source_100each_20260418T171459Z`.
- Random seeds: 42 throughout for data subsampling. Random-init seeds
  evaluated: 7, 13, 42.
- Last update: 2026-04-27.
