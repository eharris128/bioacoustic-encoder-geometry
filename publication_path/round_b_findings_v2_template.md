# Round B + Tier 3 findings (2026-04-29 → 2026-04-30 chain)

**Status:** TEMPLATE — numbers fill in as Lambda chain results land.

## TL;DR (to fill)

The 24-hour Round B + Tier 3 chain has [PASSED|FAILED] the red-team's
proposed revisions. Specifically:

- **3.5 (binary Class re-test)**: post-INLP binary Aves-vs-Mammalia
  probe accuracy [stays at majority baseline 0.667 → headline survives]
  | [recovers to X above baseline → headline weakens to "directions
  discoverable by 5-class probe destroy Order partially"].
  - Key cell numbers (sl_eat_bio_ssl_all):
    L5 [pre 0.885 → post X.XXX], L7 [pre 0.904 → post X.XXX],
    L9 [pre 0.909 → post X.XXX], L12 [pre 0.887 → post X.XXX].
  - All 16 trained-model × layer cells: post-null binary Class median X.XX.

- **3.4 (MLP Order probe on Class-nulled activations)**: MLP recovers
  [substantially | slightly | not at all] more Order signal than the
  linear probe. Specific gaps at sl_eat_bio_ssl_all L5/L7/L9/L12:
    L5: linear X.XX, MLP X.XX (gap X.XX)
    L7: linear X.XX, MLP X.XX (gap X.XX)
    L9: linear X.XX, MLP X.XX (gap X.XX)
    L12: linear X.XX, MLP X.XX (gap X.XX)
  - Reading: [linear-component framing stands | non-linear residual is
    significant; framing should report both].

- **3.6 (INLP iteration sweep)**: at max_iters=10 (~40-D nulled,
  comparable in cumulative-D to Order-first's typical 1-10 1-D nulls),
  Order destruction at sl_eat_bio_ssl_all L9 is X.XXX (vs 0.218 at
  max_iters=80). Asymmetry [survives | fails] at low iteration count.
  - Iteration curve at sl_eat_bio_ssl_all L9: 10n → X.XXX, 20n → X.XXX,
    40n → X.XXX, 80n → X.XXX (Order linear post-nullification).

- **3.7 (random-init multi-seed)**: at random_init_eat_seed07,
  Order accuracy under Class nullification [improves +0.0X / drops
  -0.0X] across L5/L7/L9/L12 — [same sign as seed 42, contrast
  framing robust | flips sign, contrast framing weakens]. Same for
  seed 13: [+0.0X | -0.0X].

- **§4.12 caveat 2 (multi-class Order INLP)**: with `max_iters=80,
  acc_floor=0.30` taking 4-class Order (Passer + 3 Aves Orders) to
  chance (~240-D nulled), Class probe survival is [0.99–1.01 → asymmetry
  robust to symmetric depth | drops X% → asymmetry partly an
  iteration-depth artifact].

- **§4.5 (refined mixing-ratio sweep)**: 11 alphas from 0 to 1.0 reveal
  [a clear threshold at α≈0.X | a saturating non-linearity | a
  near-linear shape with residual asymmetry]. Bio-axis projection at
  α=0.05 is X.XX; at α=0.10 X.XX; at α=0.15 X.XX; at α=0.25 X.XX
  (matches existing pilot's −0.66).

- **Q6 (Pearson(null_median, eff_rank))**: source-manifest cross-cell
  Pearson r = **−0.897** (p<0.001) across 12 trained cells. Per-Order
  manifest re-run including random-init: r = X.XXX (p=X.XXX) across 15
  cells. Reviewer's parsimonious low-rank explanation [supported |
  weakened].

## Detailed numbers (filled from chain CSVs)

### Binary Aves-vs-Mammalia probe re-trained on Class-nulled activations

| Model | L5 pre→post | L7 pre→post | L9 pre→post | L12 pre→post |
|---|---|---|---|---|
| eat_all | X.XX → X.XX | X.XX → X.XX | X.XX → X.XX | X.XX → X.XX |
| eat_bio | X.XX → X.XX | X.XX → X.XX | X.XX → X.XX | X.XX → X.XX |
| sl_eat_all_ssl_all | X.XX → X.XX | X.XX → X.XX | X.XX → X.XX | X.XX → X.XX |
| sl_eat_bio_ssl_all | X.XX → X.XX | X.XX → X.XX | X.XX → X.XX | X.XX → X.XX |
| random_init_seed42 | X.XX → X.XX | X.XX → X.XX | X.XX → X.XX | X.XX → X.XX |

Majority baseline: 0.667 (Aves 400 / Mammalia 200 in 4-Order manifest).

### Order destruction by INLP iteration count (red-team 3.6)

10 cells × 4 iter counts.

| Cell | n=10 | n=20 | n=40 | n=80 |
|---|---|---|---|---|
| eat_all L7 | X.XX | X.XX | X.XX | X.XX |
| eat_all L9 | X.XX | X.XX | X.XX | X.XX |
| eat_bio L7 | X.XX | X.XX | X.XX | X.XX |
| eat_bio L9 | X.XX | X.XX | X.XX | X.XX |
| sl_eat_all_ssl_all L7 | X.XX | X.XX | X.XX | X.XX |
| sl_eat_all_ssl_all L9 | X.XX | X.XX | X.XX | X.XX |
| sl_eat_bio_ssl_all L5 | X.XX | X.XX | X.XX | X.XX |
| sl_eat_bio_ssl_all L7 | X.XX | X.XX | X.XX | X.XX |
| sl_eat_bio_ssl_all L9 | X.XX | X.XX | X.XX | X.XX |
| sl_eat_bio_ssl_all L12 | X.XX | X.XX | X.XX | X.XX |

### MLP vs linear Order probe on Class-nulled activations (red-team 3.4)

`sl_eat_bio_ssl_all` only:

| Layer | Pre-null linear | Pre-null MLP | Post-null linear | Post-null MLP |
|---|---|---|---|---|
| L5 | X.XX | X.XX | X.XX | X.XX |
| L7 | X.XX | X.XX | X.XX | X.XX |
| L9 | X.XX | X.XX | X.XX | X.XX |
| L12 | X.XX | X.XX | X.XX | X.XX |

### Multi-class Order INLP symmetric test (§4.12 caveat 2)

| Cell | 4-Order pre→post | 5-Class pre→post | Bin-Class pre→post | Bin-Order pre→post |
|---|---|---|---|---|
| eat_all L5 | X.XX→X.XX | X.XX→X.XX | X.XX→X.XX | X.XX→X.XX |
| eat_all L7 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |

### §4.12 random-init at seed=7 (and seed=13 if compute allows)

| Layer | Order pre→post (seed 42) | Order pre→post (seed 7) | Order pre→post (seed 13) |
|---|---|---|---|
| L5 | 0.695 → 0.740 (+0.045) | X.XX → X.XX (±0.0X) | X.XX → X.XX (±0.0X) |
| L7 | 0.693 → 0.742 (+0.049) | X.XX → X.XX (±0.0X) | X.XX → X.XX (±0.0X) |
| L9 | 0.685 → 0.734 (+0.049) | X.XX → X.XX (±0.0X) | X.XX → X.XX (±0.0X) |
| L12 | 0.676 → 0.728 (+0.052) | X.XX → X.XX (±0.0X) | X.XX → X.XX (±0.0X) |

### §4.5 refined mixing sweep (`sl_eat_bio_ssl_all` L9)

| α | bio-axis projection (mean ± std) | cos to bio | cos to non-bio |
|---|---|---|---|
| 0.000 | X.XX ± X.XX | X.XX | X.XX |
| 0.025 | X.XX ± X.XX | X.XX | X.XX |
| 0.050 | X.XX ± X.XX | X.XX | X.XX |
| 0.075 | X.XX ± X.XX | X.XX | X.XX |
| 0.100 | X.XX ± X.XX | X.XX | X.XX |
| 0.150 | X.XX ± X.XX | X.XX | X.XX |
| 0.200 | X.XX ± X.XX | X.XX | X.XX |
| 0.250 | X.XX ± X.XX | X.XX | X.XX |
| 0.500 | X.XX ± X.XX | X.XX | X.XX |
| 0.750 | X.XX ± X.XX | X.XX | X.XX |
| 1.000 | X.XX ± X.XX | X.XX | X.XX |

### Q6 (per-Order manifest re-run, all 15 cells including random-init)

Pearson(null_median, eff_rank) = X.XXX (p=X.XXXX), n=15
Spearman = X.XXX (p=X.XXXX)

By scope:
- All 15 cells: r = X.XXX (p=X.XXXX)
- Trained-only (12): r = −0.897 (source manifest); X.XXX (per-Order)
- With random-init eff_rank now in: random-init's empirical null
  median (~0.6) [does | does not] track its eff_rank.

### Layer-resolved INLP signature on `sl_eat_bio_ssl_all` (all 13 layers)

Order destruction (post − pre, linear probe) by layer:
L0 X.XX, L1 X.XX, L2 X.XX, L3 X.XX, L4 X.XX, L5 X.XX, L6 X.XX,
L7 X.XX, L8 X.XX, L9 X.XX, L10 X.XX, L11 X.XX, L12 X.XX.

Peak destruction at L_X (X.XXX). Tail/floor at LY (X.XXX).

## Preprint v2 deltas (proposed edits)

To be drafted once chain results are in. Specific sections expected to
update:

- **§4.5** — replace 25%-only paragraph with refined-sweep table and
  shape characterization (linear vs threshold vs saturating).
- **§4.8** — replace acoustic-features-survive-shuffling explanation
  with low-rank → correlated-random-direction explanation supported
  by Q6 (Pearson r = −0.897). Optionally cite per-Order manifest
  Q6 if it lands stronger.
- **§4.12** — update headline range using 11-cell ≥0.750 framing
  (proposed in our response). Add binary Class re-test column to the
  result table. Append sub-section on MLP probe ablation.
- **§4.12 caveat 2** — replace prose defense with the symmetric
  multi-class Order INLP result (step14): "Class probe survival under
  symmetric ~240-D Order nullification is X.XX–X.XX across all four
  trained models, confirming the asymmetry is not an iteration-depth
  artifact."
- **§4.12 caveat 3** — point at binary Class re-test result.
- **§10 Limitations (iii)** — soften / strengthen depending on MLP
  ablation result.
- **§10 Limitations (3.7)** — drop "single seed" sentence if seed=7
  (and ideally seed=13) §4.12 reads have same sign as seed=42.

## Chain provenance

Lambda 129.213.131.108 (`sentient`), launched 2026-04-29T22:42 UTC.
- `step11_round_b.py` — Round B core (3.4/3.5/3.6)
- `step8_inlp_aggressive.py` — re-run on seed=7, seed=13 (3.7)
- `step13_mixing_ratio_sweep.py` — refined α sweep (§4.5)
- `step14_multiclass_order_inlp.py` — symmetric INLP test (§4.12 caveat 2)
- `step15_layer_resolved_inlp.py` — full-network sweep on sl_eat_bio
- `step17b_per_order_effrank.py` — per-Order eff_rank for Q6 v2
- `q6_analysis.py` — local Pearson analysis
- `run_round_b_chain.sh` — orchestration
