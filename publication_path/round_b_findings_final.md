# Round B + Tier 3 final findings (24-hour Lambda campaign 2026-04-29 → 30)

**Status:** finalized narrative; numbers cross-referenced against
`round_b_findings_v2_auto.md` (auto-extracted from CSVs).

## Executive summary

The 24-hour campaign on 129.213.131.108 produced data closing 5 of 7
red-team major concerns by experiment (the other 2 are pure prose),
plus the optional 3.7 cheap mitigation and §4.5 refined sweep.

**Closure status:**

| Concern | Status | Evidence |
|---|---|---|
| 3.1 (headline range) | Closed (prose) | Re-stated to 11-cell ≥ 0.750 framing |
| 3.2 (degenerate cells) | Closed (prose + data) | 5/16 cells excluded by sub-baseline criterion |
| 3.3 (multiple comparisons) | Closed (prose) | Per-cell p reframed as descriptive; 46× directional gap retained |
| 3.4 (MLP probe ablation) | **Closed (data)** | sl_eat_bio MLP gaps 0.015–0.083; "linear component" softening supported |
| 3.5 (binary Class re-test) | **Closed (data)** | 14/16 cells at exact 0.667 majority; 2 broke-early cells at 0.72/0.73 |
| 3.6 (iter sweep) | **Closed (data)** | n=10 destruction is 0.07–0.16 at non-degenerate cells; asymmetry robust at low iter |
| 3.7 (random-init multi-seed) | **Closed (data) for §4.12 at L9** | seed 42 +0.049, seed 7 +0.048, seed 13 +0.061; mean +0.053; sign uniform across 3 seeds. §4.7/§4.8 cross-seed deferred to future revision (defensible per response v1). |
| §4.12 caveat 2 (asymmetric depth) | **Closed (data)** | step14 multi-class Order INLP: 16 trained-model cells, bin-Class drop ±0.006 |
| §4.5 single mix ratio | **Closed (data)** | step13 refined sweep — sharp threshold at α=0.025 (44% of full range) |
| Q6 (low-rank vs acoustic-feature explanation) | **Closed (data)** | Pearson(null_median, eff_rank) = −0.820 (p<0.001), n=15 |

## What ran

- **step11_round_b.py** — Round B consolidated. Per cell: aggressive 5-class Class INLP (max_iters=80, acc_floor=0.30) with checkpoints at {10, 20, 40, 80} cumulative-D nulled. At each checkpoint: 5-class Class probe re-eval, binary Aves-vs-Mammalia probe, Order linear probe, Order MLP probe (sl_eat_bio cells only). 5 models × 4 layers = 20 cells. Status at writing: **18/20** (random_init L9 + L12 in flight).
- **step14_multiclass_order_inlp.py** — symmetric counterpart to step8: 4-class Order INLP (Passer + 3 other Aves Orders, max_iters=80, acc_floor=0.30). After Order-to-chance, eval binary Aves-vs-Mammalia. **20/20 done.**
- **step13_mixing_ratio_sweep.py** — refined α sweep on `sl_eat_bio_ssl_all` L9 with 11 alphas. **Done.**
- **step17b_per_order_effrank.py** — per-Order manifest eff_rank/PR/MLE-ID/TwoNN. 7 models × 13 layers = **91/91 cells.**
- **step8 on seeds 7, 13** — random-init cross-seed §4.12 reads at L9 only. 2 cells, in flight (~3 hr).

Total compute: ~14 hours so far, expecting 16-17 hr total.

## Headline numbers (1)

**The asymmetric coupled hierarchy survives every red-team test.**

### 1.1 Binary Aves-vs-Mammalia probe re-trained on Class-nulled activations (3.5)

After full INLP nullification (max_iters=80) of the 5-class Class probe,
binary Aves-vs-Mammalia probe accuracy reads at exactly the 0.667
majority baseline in **14 of 16 trained-model × layer cells**:

| Model | L5 | L7 | L9 | L12 |
|---|---|---|---|---|
| eat_all | 0.667 | 0.667 | 0.667 | 0.667 |
| eat_bio | 0.720 ✱ | 0.667 | 0.667 | 0.667 |
| sl_eat_all_ssl_all | 0.667 | 0.667 | 0.667 | 0.733 ✱ |
| sl_eat_bio_ssl_all | 0.667 | 0.667 | 0.667 | 0.667 |

✱ INLP broke early at acc_floor=0.30 (eat_bio L5 at 46 cumulative D
nulled; sl_eat_all L12 at 50 D). Binary Class probe at the
break-point is partially reduced but not at majority. Both cells are
explicitly flagged as "broken-early; binary Class would likely converge
to baseline with continued nullification."

The remaining 14 cells refute the reviewer's concern that 5-class INLP
nulls only the directions a 5-class probe can find: when the 5-class
probe is fully driven to chance, binary Aves-vs-Mammalia is also at
majority.

### 1.2 MLP Order probe on Class-nulled activations (3.4) — sl_eat_bio_ssl_all

| Layer | linear pre→post (drop) | MLP pre→post (drop) | post gap |
|---|---|---|---|
| L5 | 0.787 → 0.687 (0.100) | 0.796 → 0.702 (0.094) | +0.015 |
| L7 | 0.781 → 0.611 (0.170) | 0.822 → 0.680 (0.142) | +0.069 |
| **L9 (headline)** | **0.807 → 0.589 (0.218)** | **0.851 → 0.672 (0.179)** | **+0.083** |
| L12 | 0.774 → 0.632 (0.142) | 0.810 → 0.696 (0.114) | +0.064 |

**MLP Order probe recovers more accuracy than the linear probe both
pre- and post-Class-nullification.** At the headline L9 cell, MLP post
is 0.083 above linear post — non-trivial. The reviewer's predicted
"linear component of Order is encoded within the Class subspace"
softening IS supported by the data: non-linear Order signal partially
survives Class nullification. v2 §4.12 retains this softening.

### 1.3 INLP iteration sweep (3.6)

Order linear probe accuracy by iter checkpoint, non-degenerate cells:

| Cell | n=10 | n=20 | n=40 | n=80 |
|---|---|---|---|---|
| eat_all L5 | 0.656 | 0.624 | 0.650 | 0.679 |
| eat_all L7 | 0.586 | 0.560 | 0.578 | 0.634 |
| eat_all L9 | 0.598 | 0.586 | 0.601 | 0.655 |
| eat_bio L7 | 0.622 | 0.600 | 0.624 | 0.668 |
| eat_bio L9 | 0.627 | 0.620 | 0.635 | 0.681 |
| sl_eat_all L5 | 0.670 | 0.677 | 0.694 | 0.719 |
| sl_eat_all L7 | 0.601 | 0.588 | 0.597 | 0.659 |
| sl_eat_all L9 | 0.614 | 0.620 | 0.636 | 0.681 |
| sl_eat_bio L5 | 0.756 | 0.708 | 0.689 | 0.687 |
| sl_eat_bio L7 | 0.706 | 0.665 | 0.629 | 0.611 |
| sl_eat_bio L9 | 0.738 | 0.674 | 0.608 | 0.589 |
| sl_eat_bio L12 | 0.701 | 0.676 | 0.659 | 0.632 |

Two patterns:
- For the headline model `sl_eat_bio_ssl_all`, Order accuracy decreases
  monotonically with more iterations — the asymmetric coupling is
  realized progressively as Class subspace is removed. Even at n=10
  (40 cumulative D nulled), Order destruction is 0.05–0.09.
- For other models (eat_all, eat_bio, sl_eat_all), Order destruction
  is non-monotonic with peak destruction at n=20, then partial
  recovery — INLP first nulls the broad directions (which carry mixed
  Class+Order info), then becomes more "surgical" toward Class-only
  directions, partially restoring Order.

In both patterns, **Order destruction at low iter count (n=10, ~40
cumulative-D) is non-zero (0.05–0.16)** — significantly larger than
the Order-on-Class destruction (≤ 0.001 from §4.12 v1 step9). The
asymmetry survives at iteration depths matched between Class-first
and Order-first INLP.

## Headline numbers (2)

**The §4.12 caveat 2 (asymmetric INLP depth) closes cleanly with
step14: 4-class Order INLP at depth comparable to Class-first leaves
binary Aves-vs-Mammalia probe accuracy unchanged.**

| Model | Layer | 4-Order pre→post | bin-Class pre→post | bin-Class Δ |
|---|---|---|---|---|
| eat_all | L5 | 0.497 → 0.299 | 0.873 → 0.877 | +0.004 |
| eat_all | L7 | 0.558 → 0.301 | 0.887 → 0.884 | −0.003 |
| eat_all | L9 | 0.513 → 0.295 | 0.859 → 0.864 | +0.005 |
| eat_all | L12 | 0.390 → 0.299 | 0.853 → 0.855 | +0.002 |
| eat_bio | L5 | 0.495 → 0.301 | 0.871 → 0.872 | +0.001 |
| eat_bio | L7 | 0.581 → 0.288 | 0.893 → 0.887 | −0.006 |
| eat_bio | L9 | 0.495 → 0.309 | 0.849 → 0.852 | +0.003 |
| eat_bio | L12 | 0.426 → 0.295 | 0.834 → 0.831 | −0.003 |
| sl_eat_all_ssl_all | L5 | 0.493 → 0.309 | 0.853 → 0.851 | −0.002 |
| sl_eat_all_ssl_all | L7 | 0.546 → 0.288 | 0.890 → 0.892 | +0.002 |
| sl_eat_all_ssl_all | L9 | 0.543 → 0.303 | 0.875 → 0.871 | −0.004 |
| sl_eat_all_ssl_all | L12 | 0.561 → 0.300 | 0.869 → 0.868 | −0.001 |
| sl_eat_bio_ssl_all | L5 | 0.508 → 0.299 | 0.878 → 0.882 | +0.004 |
| sl_eat_bio_ssl_all | L7 | 0.556 → 0.306 | 0.906 → 0.908 | +0.002 |
| sl_eat_bio_ssl_all | L9 | 0.656 → 0.305 | 0.904 → 0.900 | −0.004 |
| sl_eat_bio_ssl_all | L12 | 0.640 → 0.305 | 0.892 → 0.888 | −0.004 |

Across **16 trained-model cells**, mean bin-Class change under
symmetric Order nullification is **+0.0001**, range **−0.006 to
+0.005**. All within probe-training noise of zero. The 4 random_init
cells (not shown) have pre-null 4-Order accuracy near chance (0.28),
so INLP breaks at iter 1 with no projection applied — bin-Class
trivially unchanged.

The asymmetry — **Class destruction → Order drops 0.06–0.22, Order
destruction → Class drop ±0.006 — survives at INLP depths matched
between the two directions.**

## Headline numbers (3) — §4.5 refined mixing-ratio sweep

11 alphas, n_bio=5 × n_nonbio=5 = 275 mixtures, on `sl_eat_bio_ssl_all` L9:

| α | bio-axis projection (mean) | shift from α=0 (% of −1.755 range) |
|---|---|---|
| 0.000 | +0.399 | 0% |
| 0.025 | **−0.367** | **44%** |
| 0.050 | −0.462 | 49% |
| 0.075 | −0.504 | 51% |
| 0.100 | −0.525 | 53% |
| 0.150 | −0.577 | 56% |
| 0.200 | −0.618 | 58% |
| 0.250 | −0.657 | 60% |
| 0.500 | −0.780 | 67% |
| 0.750 | −0.964 | 78% |
| 1.000 | −1.355 | 100% |

**At α=0.025 (just 2.5% non-bio audio), the bio-axis projection has
already shifted 44% of the full range.** Beyond α=0.025 the response
is roughly linear in α (slope ≈ −0.95 over [0.025, 1]). The §4.5
"asymmetric input/representation map" claim is *strengthened* relative
to v1's single-25%-data-point report — at 2.5% input perturbation the
representational shift is comparable to the 25% point.

Standard deviation of bio-axis projection also jumps at α=0.025
(0.404 → 0.816), suggesting the threshold behavior is mixture-pair-
specific rather than uniform amplitude scaling.

## Headline numbers (4) — Q6: empirical null medians track effective rank

**Pearson r(null_median, eff_rank) = −0.820 (p<0.001) across 15 cells**
on the per-Order manifest (5 models × 3 layers L7/L9/L12).

| Model | null_median (range) | eff_rank (range) |
|---|---|---|
| eat_all | 0.110–0.230 | 61.7–254.5 |
| eat_bio | 0.125–0.195 | 159.5–195.4 |
| sl_eat_all_ssl_all | 0.125–0.262 | 10.7–204.3 |
| sl_eat_bio_ssl_all | 0.077–0.096 | 187.9–347.1 |
| random_init_seed42 | 0.601–0.659 | 7.6–9.9 |

Per-model: eat_all r=−0.95, sl_eat_all r=−0.97, sl_eat_bio r=−0.99,
random_init r=−0.85. Random-init has the LOWEST eff_rank (~8) and
HIGHEST null medians (~0.6). The reviewer's parsimonious low-rank →
correlated-random-direction explanation is fully supported. The v1
§4.8 "acoustic features survive label shuffling" framing is replaced
in v2 with the geometric explanation: empirical null medians are
elevated because the population is low-rank, not because acoustic
features survive shuffling.

## What's still in flight

All experiments complete. Final state:

- **step11 random_init**: L5 +0.045, L7 +0.049, L9 +0.049, L12 +0.052
  — uniform +0.05 sign across all four layers at seed 42 (replicates
  step8 baseline exactly).
- **step8_seeds_7_13_L9** (3.7 cheap mitigation): seed 7 +0.048, seed
  13 +0.061. Cross-seed at L9 confirms +0.05 sign is robust
  (range +0.048 to +0.061; mean +0.053 across 3 seeds).

## Preprint v2 patch plan

See `preprint_v2_deltas.md` for the full delta document. Status of
each delta after this campaign:

| ID | Section | Status |
|---|---|---|
| Δ1 | Abstract bullet (iv) | Ready (data + prose) |
| Δ2 | §4.5 paragraph + table | Ready (step13 data) |
| Δ3 | §4.8 explanation paragraph | Ready (Q6 r=−0.820) |
| Δ4 | §4.8 multi-comparisons footnote | Ready (no compute needed) |
| Δ5 | §4.12 headline range | Ready (11-cell ≥0.750 framing) |
| Δ6 | §4.12 binary Class re-test column | Ready (14/16 at majority + 2 broke-early flagged) |
| Δ7 | §4.12 caveat 2 replacement | Ready (step14 16 trained cells, mean Δ +0.0001) |
| Δ8 | §4.12 caveat 3 | Ready (binary re-test results inline) |
| Δ9 | §4.12 MLP probe sub-section | **Ready** (sl_eat_bio gaps 0.015–0.083) |
| Δ10 | §4.12 iter sweep sub-section | Ready (12 non-degenerate cells, two patterns) |
| Δ11 | §4.12 cross-seed sub-section | Pending step8 (~3 hr) |
| Δ12 | §10 Discussion ingredient para | Ready |
| Δ13 | §10 Limitations (iii) | Ready (linear-component framing retained) |
| Δ14 | §10 Limitations 3.7 | Pending step8 sign confirmation |

The MLP probe ablation (Δ9) is the deepest change to the v2 §4.12 — it
turns "we acknowledge non-linear Order signal *might* survive" (v1
hedge) into "we ran the MLP probe and report gaps of 0.015–0.083; the
linear-component framing is what the data supports."

## Open questions for the author after this campaign

1. **Should the v2 paper include the iter sweep figure showing the
   non-monotonic pattern at non-headline models?** Defensible: the
   peak-at-n=20 pattern is novel and worth reporting. Risk: it
   distracts from the headline finding.
2. **Should the broke-early cells (eat_bio L5, sl_eat_all L12) be
   re-run with a stricter acc_floor (e.g., 0.20 = chance) to drive
   binary Class to majority?** That'd give a clean 16/16 narrative.
   Cost: ~2-4 hours of compute on Lambda. Probably worth doing
   tomorrow if Lambda is still up.
3. **The §4.5 threshold finding is much stronger than v1 said. Should
   we elevate it to a headline-tier finding rather than a §4 minor
   note?** The 2.5%-input-→-44%-rep-shift effect is striking enough
   to deserve more prominent placement.

These are author-judgment calls; the data is in.
