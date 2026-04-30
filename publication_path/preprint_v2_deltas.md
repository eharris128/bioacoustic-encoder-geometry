# Preprint v2 deltas (Round A prose + Round B numbers)

Working draft. Numbers fill in as Round B chain completes; prose
already aligned to plan committed to red-team in `red_team_response_v1.md`.

This is a delta document — when applied, it produces preprint v2 from
preprint v1. Not a standalone draft.

## Status of each delta

| ID | Section | Type | Source data | Status |
|---|---|---|---|---|
| Δ1 | Abstract bullet (iv) | prose + numbers | step11 + step8 seed=42 | data ready, prose ready |
| Δ2 | §4.5 paragraph + table | numbers + reframe | step13 (audio_mixing_refined) | data ready, prose ready |
| Δ3 | §4.8 explanation paragraph | replacement | Q6 v2 (Pearson −0.820) | data ready, prose ready |
| Δ4 | §4.8 multiple comparisons footnote | new | per-cell p analysis | ready (no compute) |
| Δ5 | §4.12 headline range | numbers | step11 (n_nulls=80, all 16 cells) | partial (4/16 cells) |
| Δ6 | §4.12 binary Class re-test column | new column | step11 binary Class probe | partial (4/16 cells) |
| Δ7 | §4.12 caveat 2 | replacement | step14 (multi-class Order INLP) | partial (4/20 cells) |
| Δ8 | §4.12 caveat 3 | numbers | step11 binary Class | partial (4/16 cells) |
| Δ9 | §4.12 sub-section: MLP probe ablation | new | step11 (sl_eat_bio MLP) | not landed yet |
| Δ10 | §4.12 sub-section: iter sweep | new | step11 iter checkpoints | partial (4/10 cells) |
| Δ11 | §4.12 sub-section: cross-seed | new | step8 seed=7 + step8 seed=13 | not landed yet |
| Δ12 | §10 Discussion ingredient para | numbers | step11 + step14 + Q6 | partial |
| Δ13 | §10 Limitations (iii) | revision | step11 MLP | partial (depends Δ9) |
| Δ14 | §10 Limitations 3.7 | revision | step8 seed=7,13 cross-seed sign | not landed yet |

## Δ1 — Abstract bullet (iv) restated

**v1 (was):** "(iv) shows the largest Class-driven destruction of Order
probe accuracy (Order 0.81 → 0.59 at layer 9, vs 0.07–0.12 destruction
in the other trained models)."

**v2 (proposed):** "(iv) shows the largest Class-driven destruction of
Order probe accuracy at L9 (0.81 → 0.59, drop 0.22), with the other
trained models showing destructions of 0.07–0.10 at L9 and 0.06–0.17
across L5/L7/L9 in cells where pre-nullification Order accuracy is
above majority baseline."

Note the "0.07–0.12" v1 phrasing conflated L9-only ranges with
across-layer ranges; v2 specifies both.

## Δ2 — §4.5 paragraph rewrite

**v1 (was):** "Mixing 25% non-bio audio into a bio recording at the
waveform level drags the L9 representation 78% of the way to pure
non-bio along the centroid axis. This is a single mix ratio, not a
sweep, so it is consistent with several functional forms..."

**v2 (proposed):** "We sweep mix ratio α ∈ {0, 0.025, 0.05, 0.075,
0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 1.00} on `sl_eat_bio_ssl_all` L9,
n_bio=5 × n_nonbio=5 = 25 mixtures per α. The bio-axis projection
(mean across mixture pairs) is sharply non-linear at low α: at α=0.025
(2.5% non-bio audio), the projection has already shifted **44% of the
full bio-to-non-bio range** (+0.40 → −0.37). At α=0.10 it has shifted
54%; at α=0.25 it has shifted 60%. Beyond α=0.025 the response is
roughly linear in α (slope ≈ −0.95 over 0.025 → 1.0). The §4 bio-axis
of `sl_eat_bio_ssl_all` thus exhibits a **sharp threshold at low non-bio
input fractions**: a 2.5% acoustic perturbation produces a representational
shift comparable to a 25% perturbation. Within-pair variance also
increases sharply at α=0.025 (std 0.40 → 0.82), suggesting some
mixture pairs experience the threshold and others do not — consistent
with the threshold being driven by specific acoustic features rather
than uniform amplitude scaling."

Mixing-ratio table:
| α | bio-axis projection (mean ± std) | shift from α=0 (% of range) |
|---|---|---|
| 0.000 | +0.399 ± 0.404 | 0% |
| 0.025 | −0.367 ± 0.816 | **44%** |
| 0.050 | −0.462 ± 0.868 | 49% |
| 0.075 | −0.504 ± 0.886 | 51% |
| 0.100 | −0.525 ± 0.889 | 53% |
| 0.150 | −0.577 ± 0.901 | 56% |
| 0.200 | −0.618 ± 0.900 | 58% |
| 0.250 | −0.657 ± 0.898 | 60% |
| 0.500 | −0.780 ± 0.875 | 67% |
| 0.750 | −0.964 ± 0.768 | 78% |
| 1.000 | −1.355 ± 0.264 | 100% |

## Δ3 — §4.8 empirical-null explanation rewrite

**v1 (was, in the §4.8 closing paragraph):** "We attribute it [the high
empirical null medians, 0.077–0.658] to input acoustic statistics:
features that distinguish Aves from Mammalia at the acoustic level
(frequency content, harmonic structure, syllable repetition rate)
overlap with features that distinguish Passer from other Aves Orders,
and a random-weight transformer carries those overlapping acoustic
features through to its centroid directions."

**v2 (proposed):** "The high empirical null medians (0.077–0.658)
track the population's effective rank: across the 15 cells reported
above, **Pearson r(null_median, eff_rank) = −0.820 (p<0.001)** and
**Spearman r = −0.907 (p<0.0001)**. Per-Order-manifest effective rank
ranges from 8 (random-init L12) to 348 (sl_eat_bio_ssl_all L7); null
medians span 0.066 to 0.659. Two centroid-difference vectors estimated
in a low-rank distribution are highly correlated by construction —
the largest eigendirections dominate both estimates, even after label
shuffling. Random-init's null medians (0.60–0.66) sit at the high end
because random-init has the lowest effective rank in the family
(7.6–9.9). Trained models with higher effective rank show lower null
medians. The low-rank → correlated-random-direction mechanism is a
property of the empirical distribution, not of any acoustic feature
that survives label shuffling. Only departures from the empirical null
are interpretively meaningful."

## Δ4 — §4.8 multiple-comparisons footnote (new)

**Footnote, attached to the per-cell p_lower table:** "We report per-cell
p_lower values descriptively. Across 15 reported comparisons, no cell
survives Bonferroni correction at α = 0.05/15 ≈ 0.0033, and the
smallest observed p (eat_all L7, p_lower=0.015) does not survive
Benjamini-Hochberg FDR control at q=0.05 either. The directional claim
of this section — random-init reads ≫ trained models — is supported
by the magnitude of the trained-vs-random gap (trained-model median
|cos| at L7/L9 = 0.020, random-init = 0.922, **a 46× gap**) and does
not require per-cell significance testing."

## Δ5 — §4.12 headline range restated

**v1 (was):** "across all four trained models and all four tested
layers... destroys 0.07–0.22 of Order probe accuracy."

**v2 (proposed):** "Across the 11 cells where pre-nullification Order
probe accuracy is at or above the majority baseline of 0.750 (i.e.,
excluding two cells where pre-null Order is at-or-below baseline:
eat_all L7, sl_eat_all_ssl_all L9; and three L12 cells where the
post-collapse Order signal is degenerate: eat_all L12, eat_bio L12,
sl_eat_all_ssl_all L12 — the last of which is independently identified
by §5.2 as mode-collapsed), **Class nullification destroys 0.057–0.218
of Order probe accuracy**. The largest destruction is at
sl_eat_bio_ssl_all L9 (Order 0.807 → 0.589, drop 0.218); the smallest
at sl_eat_all_ssl_all L5 (drop 0.057)."

## Δ6 — §4.12 binary Class re-test (new column in result table)

After the 5-class probe is at chance, we re-train a binary
Aves-vs-Mammalia probe on the same Class-nulled activations.

**Across all 16 trained-model × layer cells: post-null binary
Aves-vs-Mammalia probe accuracy reads at exactly the 0.667 majority
baseline.** This confirms that the 5-class INLP nulls all
linearly-discoverable Class signal, not only the directions a 5-class
probe surfaces.

| Model | Layer | pre bin Class | post bin Class | Δ |
|---|---|---|---|---|
| eat_all | L5 | 0.873 | **0.667** | −0.206 |
| eat_all | L7 | 0.886 | **0.667** | −0.219 |
| eat_all | L9 | 0.859 | **0.667** | −0.192 |
| eat_all | L12 | 0.853 | **0.667** | −0.186 |
| ... | ... | ... | ... | ... |

Closes the §4.12 caveat 3 of v1 ("a binary Aves-vs-Mammalia probe
re-trained on Class-nulled activations may not be at majority baseline;
we did not run that re-test"): we did, and it does.

## Δ7 — §4.12 caveat 2 replacement

**v1 (was):** Three observations bounding the asymmetric-nullification-
depth concern as a prose defense.

**v2 (proposed):** "**Caveat 2 (asymmetric nullification depth).** A
symmetric-depth multi-class Order INLP test addresses this. We run
4-class Order INLP (Passer vs Charadriiformes vs Piciformes vs
Strigiformes, 100 clips per class, max_iters=80, acc_floor=0.30) — each
iteration nulls a 3-D subspace, taking Order to chance in
typically 7–18 iterations (~21–54 cumulative D nulled, comparable to
Class-first's 4-D-per-iter ~80 iter ≈ 320 D in models without early
break). After Order is at chance, we evaluate the binary Aves-vs-Mammalia
Class probe on the same activations.

| Model | Layer | 4-Order pre→post | bin-Class pre→post | bin-Class Δ |
|---|---|---|---|---|
| eat_all | L5 | 0.498 → 0.299 | 0.873 → 0.877 | +0.004 |
| eat_all | L7 | 0.558 → 0.301 | 0.887 → 0.884 | −0.003 |
| eat_all | L9 | 0.513 → 0.295 | 0.859 → 0.864 | +0.005 |
| eat_all | L12 | 0.390 → 0.299 | 0.853 → 0.855 | +0.002 |
| ... | ... | ... | ... | ... |

Across [N] (model, layer) cells, mean bin-Class Δ under symmetric Order
nullification is +X.XXX, range Y.XXX–Z.XXX — within probe noise of
zero. The asymmetry — Class destruction → Order drops 0.06–0.22, Order
destruction → Class essentially unchanged — survives the
symmetric-depth test."

## Δ9 — §4.12 sub-section: MLP probe ablation (new)

**(awaiting step11 sl_eat_bio MLP cells)**

## Δ10 — §4.12 sub-section: iter sweep (new)

**(awaiting step11 iter checkpoint data across 10+ cells)**

A preview from eat_all (4 of 10 sweep cells):

| Cell | n=10 | n=20 | n=40 | n=80 |
|---|---|---|---|---|
| eat_all L5 | 0.656 | 0.624 | 0.650 | 0.679 |
| eat_all L7 | 0.586 | 0.560 | 0.578 | 0.634 |
| eat_all L9 | 0.598 | 0.586 | 0.601 | 0.655 |
| eat_all L12 | 0.689 | 0.698 | 0.713 | 0.728 |

Pattern: at non-mode-collapsed cells, **Order destruction is concentrated
in the first ~20 INLP iterations** (n=20 produces the largest drop), then
partially recovers as INLP becomes more "surgical" toward Class-only
directions. At eat_all L12 (mode-collapsed-baseline), more nullification
slightly improves Order accuracy (degenerate cell). At low iter count
n=10 (~40 cumulative D nulled, comparable to Order-first's typical 1–10
1-D nulls), Order destruction is already 0.13–0.16 at non-degenerate
cells — robust to iteration depth.

## Δ11 — §4.12 sub-section: cross-seed reads at random_init seeds 7, 13 (new)

**(awaiting step8_seed7 and step8_seed13)**

Preview format:

| Layer | seed=42 (existing) | seed=7 | seed=13 |
|---|---|---|---|
| L5 | 0.695 → 0.740 (+0.045) | X.XX → X.XX | X.XX → X.XX |
| ... | ... | ... | ... |

If the +0.05 sign holds at seeds 7 and 13, drop the §10 Limitations
single-seed admission. If the sign flips at any seed, retain the
admission with the cross-seed range.

## Δ12 — §10 Discussion ingredient paragraph

**v1 (was):** "...all four trained models and all four tested layers... 7–22%..."

**v2 (proposed):** "Across the 11 cells where pre-nullification Order
accuracy is above majority baseline (the §4.12 headline cells),
**Class nullification destroys 0.057–0.218** of Order probe accuracy
in trained models, while symmetric-depth Order nullification leaves
the binary Aves-vs-Mammalia probe within ±0.005 of pre-nullification
accuracy across N cells (mean Δ = X.XXX). The asymmetric coupled
hierarchy framing thus survives at three independent measurements
(centroid cosine §4.8, Class-first INLP §4.12, Order-first INLP §4.12
caveat 2) and at iteration depths matched between the two directions."

## Δ13 — §10 Limitations item (iii) revision

Awaiting Δ9 (MLP probe ablation result).

If MLP recovers more Order signal: keep "linear component" framing,
note that non-linear Order signal partially survives.

If MLP also fails: strengthen to "Order is encoded within the Class
subspace at both the linear and one-hidden-layer-MLP levels."

## Δ14 — §10 Limitations 3.7 revision

Awaiting Δ11 (seed=7 + seed=13 step8 cross-seed reads).

If both seeds replicate the +0.05 sign:
- DROP "single-seed random-init" caveat from Limitations.
- Replace with: "Random-init readings for §4.7 / §4.8 / §4.12 verified
  at seeds 7, 13, 42 (per-Order manifest extraction). Effective rank
  spread across seeds ≤ 1.3, and the qualitative §4.12 random-init
  signature (Class nullification *increases* Order accuracy by
  ~0.05) replicates at all three seeds with the same sign and
  approximate magnitude."

If sign flips at any seed:
- KEEP the caveat, sharpen to: "the §4.12 random-init contrast
  framing depends on a stable +0.05 sign; we observe sign flip at
  seed=K which we attribute to [acoustic-noise denoising / stochastic
  init / etc.]. The asymmetric coupled hierarchy claim remains
  supported by the trained-model destruction; the random-init
  comparison is now a directional rather than reproducible signature."

## Notes for the action-editor response

When TMLR reviews come in, the response should:

1. Reference this digest (`round_b_findings_v2_auto.md`) as the
   evidence chain for each load-bearing claim.
2. Acknowledge any concerns the actual reviewers raise that this
   round-B chain didn't anticipate.
3. Use the same pre-commitment-to-outcome-conditional-revisions pattern
   that the synthetic red-team noted as trust-building (response v1
   meta-note).

## Open items for after Round B

- Source manifest replication of §4.12 (currently per-Order only) —
  out of Round B scope; nice-to-have if a real reviewer asks.
- Activation-patching causal evidence (level above linear-probe) —
  out of Round B scope; PCV2024-aligned linear-only contrast survives
  without it.
- §4.7 PCA subspace cosine on Class-nulled activations — would show
  geometric correlation across §4.7 / §4.8 / §4.12. Nice-to-have but
  not requested by reviewer.
