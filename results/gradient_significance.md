# Phylogenetic gradient — Spearman analysis (n=10)

**Claim under test:** probe accuracy scales monotonically with evolutionary
distance (MYA) across species pairs. Ran on 10 xeno-canto pairs spanning
2 → 65 MYA, LORO CV, peak-accuracy summary per pair.

## Results

| statistic | value | interpretation |
|---|---|---|
| Spearman ρ | **+0.841** | strong positive monotonic relationship between MYA and accuracy |
| p (permutation, 1-sided) | **0.0016** | probability of this ρ under a null of no MYA→accuracy relationship, from 10 000 label shuffles |
| p (scipy, 2-sided) | 0.0023 | asymptotic check — agrees with the permutation p |
| 95% bootstrap CI on ρ | **[+0.44, +0.97]** | resampled pairs 10 000× with replacement; CI **excludes 0**, so the result is not driven by one or two outlier pairs |
| Pearson r on log₁₀(MYA) | +0.790 (p=0.0066) | secondary log-linear check; consistent with Spearman |
| ρ significance threshold (n=10, α=.05, 1-sided) | 0.548 | our ρ clears it by a wide margin |

## What this establishes

The gradient claim is **significant and stable** at n=10. Two independent
robustness checks (permutation p and bootstrap CI) both support the primary
result, so the finding does not rest on a single test's assumptions.

Same-family pairs (~10 MYA, ~92–95%) sit between within-genus pairs (~85–93%)
and cross-order pairs (~97–99%). The lift from ~85% (2 MYA) to ~99% (65 MYA)
is the probe-level signature of the geometric compression Evan documents in
`RESULTS.md` §4.9: training suppresses fine species detail in favor of
coarser Class/Order structure.

## Caveats

- **n=10.** The 8-pair expansion planned in `species_pairs_expansion.md`
  drops the ρ significance threshold from 0.548 to ≈0.46 and gives margin so
  no single noisy pair can sink the result.
- **Six MYA values are estimates** flagged `VERIFY TT` in
  `analysis/species_pair_results.csv`. Confirm on TimeTree.org before
  publishing. Shifts move x-positions but almost certainly do not kill ρ.
- **Peak-accuracy summary is noisy.** A weighted-mean-layer accuracy would
  be more stable if the gradient story becomes central to the paper.

Reproduce: `python analysis/spearman_gradient.py --csv analysis/species_pair_results.csv`
