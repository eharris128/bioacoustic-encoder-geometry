# Overnight findings (2026-04-28 → 2026-04-29)

## TL;DR

The chain ran cleanly. Three follow-up experiments completed, addressing the residual reviewer concerns from the first red-team round. The headline finding has shifted: **the "factored hierarchy" claim from v1 is retracted; trained EAT models exhibit an *asymmetric coupled hierarchy* — Class is structurally above Order, but Order is encoded *within* the Class subspace, not orthogonal to it.** This is a richer and more defensible claim than v1's, and it directly contrasts with Park-Choe-Veitch 2024's findings for LLMs.

## What ran

- **step6 v2** — clip-level INLP probes (replacing the leakage-inflated v1). Done before sleep.
- **step7** — manifest-resampling sensitivity. Done before sleep.
- **step8** — aggressive multi-class INLP (5-way Class probe, max_iters=80). 11 hours wall clock. Done.
- **step9** — INLP-Order-first asymmetric test. ~30 min wall clock. Done.
- **step10** — empirical Veitch permutation null (B=200). ~5 min wall clock. Done.

All five artifacts committed locally. Branch is 14 commits ahead of origin (not pushed).

## The new picture

### 1. Class is structurally above Order — but Order is encoded *within* Class.

| sl_eat_bio_ssl_all L9 | Pre | Post | Δ |
|---|---|---|---|
| Null Class → check Order | 0.807 | **0.589** | **−0.218** |
| Null Order → check Class | 0.909 | **0.911** | +0.002 |

This asymmetry is uniform across all four trained models in the family (where the test is non-degenerate): Class survives Order-nullification at 0.99–1.01; Order does not survive Class-nullification (mean drop 0.066 to 0.158 across models). Random-init shows neither effect (Order improves slightly under projection; Class essentially unchanged — both because random-init has no real Class-Order coupling).

This is NOT a factored hierarchy. It IS a hierarchy: Class is the higher-level, more-distributed feature; Order is encoded as a refinement within the Class subspace. The asymmetric INLP signature is the load-bearing evidence.

### 2. The §4.8 centroid-cosine claim is more nuanced than v1 stated.

Empirical permutation null distributions (B=200) for |cos(Aves−Mammalia, Passer−Aves)|:

- Empirical null medians range 0.077 to 0.658 across (model, layer) — *much higher* than the theoretical √(2/(πd)) ≈ 0.029 we cited in the v1 §4.8 reframe. The theoretical floor was an under-estimate of what label-shuffled centroid cosines actually produce in our data.
- Random-init reads at the **upper tail** of the null (p_lower 0.88–0.95) — its centroid axes are *more aligned* than typical permutations, because input acoustic features couple the Class and Order centroid directions.
- Trained models read at typical-null or lower-tail values, depending on (model, layer):
  - `eat_all` L7: obs **0.0032** vs null median 0.111, p_lower=**0.015** (significantly low).
  - `sl_eat_all_ssl_all` L9: obs **0.010** vs null median 0.125, p_lower=**0.045** (significantly low).
  - `sl_eat_bio_ssl_all` L9: obs **0.096** vs null median 0.077, p_lower=**0.570** (typical-null, NOT in lower tail).

So: trained models *break* the acoustic-feature alignment that random-init exhibits, but the headline model `sl_eat_bio_ssl_all` does not reach a significantly-low value at L9. The factored-hierarchy reading dies; what survives is "trained models decouple the centroid axes from the random-init alignment, but not all the way to demonstrably-orthogonal."

### 3. The two measures (centroid cosine, probe INLP) capture different geometric properties.

`eat_all` L7 has a centroid cosine in the bottom 1.5% of permutation null (looks "factored"), but its Order probe is destroyed when Class is nulled (looks "entangled"). Both can be true:

- Centroid cosine is a property of the *first moments* — where the population means sit.
- Probe INLP is a property of the *full linear subspaces* that carry probe-detectable information.

A model can have nearly-orthogonal centroid-difference axes while still encoding both features in overlapping linear subspaces. The §4.8 finding (small centroid cosine) and the step8 finding (Order doesn't survive Class-nullification) are independent — both true, neither implies the other.

### 4. v2 INLP results (clip-level splits) shrank v1's headline magnitudes.

| Model | mean Class probe acc (v2 clip-level) | mean Order probe acc (v2 clip-level) |
|---|---|---|
| `eat_all` | 0.873 | 0.757 |
| `eat_bio` | 0.858 | 0.759 |
| `sl_eat_all_ssl_all` | 0.886 | 0.764 |
| **`sl_eat_bio_ssl_all`** | **0.898** | **0.787** |
| `random_init_eat_seed42` | 0.819 | 0.687 |

`sl_eat_bio_ssl_all` is still consistently highest, but margins are 0.02–0.05, not the 0.06–0.10 that v1 (with frame-leakage) showed. The "highest at every layer" claim survives; the "uniquely high" framing does not.

### 5. step7 manifest resampling: §4.8 absolute number is robust at the headline cell.

`sl_eat_bio_ssl_all` |cos| at L12 across 5 stratified clip-swap resamples: mean 0.048 ± 0.020 (range 0.019–0.070). Smaller than the original within-manifest bootstrap CI [0.0036, 0.110]. The reviewer's predicted "several × CI" instability is refuted for the headline model. The intermediate-cos models (eat_all, sl_eat_all_ssl_all) do show wide spreads (up to 0.50), as expected for the absolute-cos metric near intermediate values.

## What this does to the paper

The center of gravity moves from "factored hierarchy" to **"asymmetric coupled hierarchy that contrasts with the LLM Veitch picture"**. New section structure:

- §4.8 — *Centroid-cosine evidence is mixed under empirical null.* Reframe entirely from "factored" to "decoupled from the random-init acoustic-feature alignment, but not significantly orthogonal in the headline cell."
- §4.11 — *Linear-probe corroboration of §4.7 with clip-level splits.* (already updated)
- **§4.12 (new) — *Asymmetric hierarchy via bidirectional INLP.*** Class survives Order-nullification (step9, class_survival ≈ 1.00 universally); Order does not survive Class-nullification (step8, drops 0.07–0.22). The asymmetry holds across all trained models, with `sl_eat_bio_ssl_all` showing the largest Class-driven Order destruction.
- §5.2 — soften further: §5.2's "thrown out everything except bio" is replaced with "Order info at L12 of `sl_eat_all_ssl_all` is at majority baseline" (already updated). The new asymmetric-hierarchy framing makes §5.2's mode-collapse story consistent with the broader picture: the SSL-fine-tune-without-bio-pretrain reaches the entanglement endpoint where Class dominates and Order is at chance.
- Discussion — replace "factored hierarchy" framing with **"asymmetric coupled hierarchy as a counter-example to the LLM Veitch picture"**. This is a stronger and more interesting claim because it identifies a domain (small audio-encoder families) where the LLM-derived geometric prediction does not transfer.

## What's now genuinely closed vs still open

**Closed by overnight chain:**
- Reviewer concern (1) on geometry-as-semantics: §4.11 v2 probe accuracies + step8 INLP destruction give probe-based + causal evidence.
- Reviewer concern (4) manifest-construction noise: step7 refutes the prediction for the headline model.
- Reviewer concern (6) §4.8 Veitch interpretation: step10 empirical null is the cleanest replacement for the theoretical-floor framing; step8 + step9 give the asymmetric-hierarchy reframe.
- Reviewer concern (7) cheapest geometry-to-causal experiment: done in step6_v2 + step8 + step9.

**Still open (would require additional work):**
- Reviewer concern (2) "preserves acoustics" control sharpening: prose-only fix in §2/§5; not closed by experiment. Would need a non-EAT control to fully address.
- Reviewer concern (3) SSL-as-new-data-domain confound: prose-only flag in §1; not closable from n=4 design.
- Reviewer concern (5) pooling-as-finding-or-choice: already a prose reframe in §3.
- §4.5 single-point threshold: would need a mixing-ratio sweep.

**New questions raised by overnight chain:**
- Why is `sl_eat_all_ssl_all` L9 in the lower tail of the empirical null (p_lower=0.045) when its centroid cosines elsewhere are not? Possibly a layer-specific concentration effect.
- Why does `eat_all` L7 reach p_lower=0.015 when its INLP entanglement is comparable to the other pretrains? Suggests centroid orthogonality and probe entanglement are independent geometric properties — worth a paragraph in the Discussion.

## Action items for when you're back

1. Read this document.
2. Read the v1 preprint with §4.11 / §5.2 / abstract already updated to v2 numbers.
3. Decide: do you want me to write the new §4.12 (asymmetric hierarchy) section directly into preprint_v1, or sketch it as a standalone v2 plan first?
4. Decide: do you want a fresh red-team round on the post-chain v1 (with §4.8 still in its current state, §4.11 fully updated)? Or hold the red team until after §4.12 is written?

## Compute and resources used

- ~12 hours total wall clock for the chain (step8 dominated due to OLMo CPU contention).
- All artifacts pulled local; nothing remains uniquely on `sentient`.
- step10 in particular ran in 5 minutes — way faster than estimated.
- The OLMo run on `sentient` is presumably still going; check `pgrep -af aitc` if you want to verify before launching anything new.
