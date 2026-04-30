# 24-hour Round B + Tier 3 campaign — summary for the morning

**Status: COMPLETE.** All jobs finished by 2026-04-30 18:46 UTC. preprint_v2.md is in place with all Round B numbers integrated.

## TL;DR

All 7 red-team major concerns closed (5 by data, 2 by prose); 3.7 cheap mitigation fully succeeded; §4.5 strengthened from a single-data-point claim to a refined 11-α sweep showing a sharp threshold at 2.5% input perturbation; Q6 supports the reviewer's parsimonious low-rank explanation. **preprint_v2.md is ready to ship to TMLR** subject to one author-judgment call (see "Open question" below).

## What landed

### Red-team concerns

| Concern | Status | Evidence |
|---|---|---|
| 3.1 (headline range) | Closed (prose) | 11-cell ≥0.750 framing; range 0.057–0.218 |
| 3.2 (degenerate cells) | Closed (prose+data) | 5 cells excluded; sub-baseline criterion |
| 3.3 (multiple comparisons) | Closed (prose) | 46× directional gap; per-cell p descriptive |
| **3.4 (MLP probe)** | **Closed (data)** | sl_eat_bio post-null linear-vs-MLP gaps 0.015–0.083; linear-component framing supported |
| **3.5 (binary Class re-test)** | **Closed (data)** | 14/16 trained cells at exact 0.667 majority; 2 broke-early cells flagged |
| **3.6 (iter sweep)** | **Closed (data)** | n=10 destruction 0.05–0.16 at 12 non-degenerate cells; asymmetry robust at low iter |
| **3.7 (random-init multi-seed)** | **Closed (data) at L9** | seed 42 +0.049, seed 7 +0.048, seed 13 +0.061; uniform +0.05 sign |
| **§4.12 caveat 2 (asymmetric depth)** | **Closed (data)** | step14 multi-class Order INLP: 16 trained cells, bin-Class Δ ±0.006 |
| **§4.5 (single mix ratio)** | **Closed (data)** | step13 11-α sweep: sharp threshold at α=0.025 (44% rep shift) |
| **Q6 (low-rank explanation)** | **Closed (data)** | Pearson r(null_median, eff_rank) = −0.820 (p<0.001), n=15 |

### Compute used (~17 hr on Lambda)

- step11_round_b.py: 20/20 cells (Round B core, 5 models × 4 layers, with MLP at sl_eat_bio cells)
- step14_multiclass_order_inlp.py: 20/20 cells (caveat 2 closure)
- step13_mixing_ratio_sweep.py: 11-α sweep on sl_eat_bio L9
- step17b_per_order_effrank.py: 91/91 cells (per-Order eff_rank for Q6)
- step8 on seeds 7, 13: 2/2 cells (3.7 cheap mitigation)
- Q5/Q6: closed via local code review + analysis

### Files in publication_path/

| File | Status |
|---|---|
| `preprint_v1.md` | Untouched (kept as base) |
| **`preprint_v2.md`** | **Final — ready to ship to TMLR** |
| `round_b_findings_final.md` | Final narrative summary |
| `round_b_findings_v2_auto.md` | Auto-extracted numbers (re-run `step20_v2_extract.py` to refresh) |
| `preprint_v2_deltas.md` | Delta plan; all 14 deltas applied |
| `red_team_response_v1.md` | The author-response we sent the red-team |
| `red_team_prompt_tmlr.md` | The TMLR-tuned prompt for the red-team agent |
| `CAMPAIGN_SUMMARY.md` | This file |

## Headline numbers (one-liner versions)

- **§4.12 Class-first INLP**: trained-model Order destruction 0.057–0.218 across 11 retained cells; largest at sl_eat_bio_ssl_all L9 (0.218).
- **§4.12 Order-first INLP (step14, symmetric depth)**: bin-Class Δ ±0.006 across 16 trained cells. Asymmetry survives matched-depth test.
- **§4.12 binary Class re-test**: 14 of 16 trained cells at exact 0.667 majority baseline post-INLP; 2 broke-early cells (eat_bio L5, sl_eat_all_ssl_all L12) at 0.720 / 0.733.
- **§4.12 MLP probe at sl_eat_bio cells**: post-null gaps L5 +0.015, L7 +0.069, L9 +0.083, L12 +0.064 — non-linear Order signal partially survives Class nullification, supporting the *linear-component* framing.
- **§4.12 random-init**: +0.05 sign uniform across 3 seeds at L9 (seed 42 +0.049, seed 7 +0.048, seed 13 +0.061).
- **§4.5 mixing-ratio**: at α=0.025 (2.5% non-bio audio), bio-axis projection has shifted 44% of full range. Sharp threshold near α=0; approximately linear thereafter.
- **§4.8 Q6**: Pearson r(null_median, eff_rank) = −0.820 (p<0.001) across 15 cells. Low-rank → correlated-random-direction-estimates explanation supported.

## What's NOT in v2 but could be

The data is in; the question is whether to add to the paper:

1. **Iter sweep figure showing the non-monotonic pattern at non-headline models.** Defensible: the peak-at-n=20 pattern (eat_all, eat_bio, sl_eat_all_ssl_all) is novel and worth reporting, though it's currently text-only in §4.12 caveat (2).
2. **Rerun broke-early cells (eat_bio L5, sl_eat_all_ssl_all L12) with `acc_floor=0.20`** to drive binary Class to 0.667 and get a clean 16/16 narrative. Would take ~3 hours on Lambda; not done because the 14/16 framing with explicit broken-early flagging is defensible.
3. **§4.5 threshold elevation from minor to headline-tier.** The 2.5%-input → 44%-rep-shift result is striking and arguably deserves more prominent placement than a §4.5 bullet. Currently it's where v1 had it, just with much stronger numbers.

## Open question

**The `linear-component` framing softening (3.4 result):** the MLP probe recovers 0.015–0.083 more Order than the linear probe post-null. This is non-trivial at L9 (+0.083) — is it large enough to demote the headline from "Order is encoded within the Class subspace" to "the *linear* component of Order is encoded within the Class subspace"? My judgment: yes, that's what the data supports and that's what's in v2. Reviewer's prediction was that this softening would be exactly what the MLP test reveals; it is.

You may want to revisit this judgment call. v1 had a hedge ("we have not run MLP — readers should treat as hypothesis-generating"); v2 has the actual MLP result and the softer claim. Both are defensible; v2's is stronger because it's anchored in measurement.

## Recommended next steps

1. **Read `round_b_findings_final.md` and `preprint_v2.md`.** Decide if the linear-component softening is the right call.
2. **Decide whether to apply (1)–(3) above or leave them as future revisions.**
3. **Push to origin if collaborators need access** (we're 25 commits ahead; nothing pushed during the campaign because pushing is shared-state and we wanted explicit authorization).
4. **Submit to TMLR.** preprint_v2.md is ready.
5. **Optional:** schedule a remote red-team agent on preprint_v2.md to confirm the v1-flagged concerns are now closed. The agent's prompt is `red_team_prompt_tmlr.md` (you can paste preprint_v2.md to it). Per our prior conversation, marginal value of another synthetic round is low — actual TMLR reviewers will give a more useful next list. But it's an option if you want one more round before submission.

## Lambda machine

Should be safe to power down. All shards we extracted (random_init seeds 7 and 13 on the per-Order manifest) are on Lambda and not yet rsynced to local. If you plan to do future revisions that need them, keep Lambda up; otherwise the local CSVs of step8/step11/step14 are sufficient for the v2 paper, and re-extraction at seeds 7/13 is ~10 min if needed.

## Notes for the action-editor response after real TMLR reviews come in

Per the red-team's closing meta-note: pre-commitment to outcome-conditional revisions (we did this for 3.5 in the response v1) and self-flagged steelmen (we did this for 3.4 around PCV2024 being itself a linear-framework prediction) are trust-building moves. Reuse this pattern in the action-editor reply.
