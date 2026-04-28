# What we need from the INLP run

Specifies what to extract from `step6_inlp_class_order.py`'s outputs once the
run on `sentient` completes, what to keep for the TMLR draft, what to
discard, and what additional INLP-related follow-ups are worth queuing if
we want to push this evidence further.

## Context

Reviewer concern (6) flagged that the §4.8 Veitch test (cos = 0.033) is at
the random-orthogonality floor in 768-d (≈ 0.029) and therefore cannot
distinguish "factored hierarchy" from "two centroid-difference directions
that are near-orthogonal by construction." Reviewer concern (7) named
INLP at L7 of `sl_eat_bio_ssl_all` as the cheapest single experiment to
convert any geometric claim in the paper into a probe-based one.

`step6_inlp_class_order.py` runs INLP on the Class probe (Aves vs
Mammalia) for layers L5/L7/L9/L12 across all 5 models, then trains an
Order probe (Passeriformes vs other-Aves) on the Class-nullspace
activations. Output: `inlp_results.csv` (per-iteration), `inlp_summary.csv`
(per-cell headline), `inlp_summary.png`.

## What worked, what didn't

### Worked: pre-INLP linear probe accuracies

The pre-INLP Class and Order probe accuracies are the most useful artifact
of the run. They give us a non-geometric, non-centroid corroboration of
§4.7. From the 16 trained-model cells (random-init still pending at time
of writing):

| Model | mean pre-Class acc | mean pre-Order acc |
|---|---|---|
| `eat_all` | 0.921 | 0.835 |
| `eat_bio` | 0.910 | 0.826 |
| `sl_eat_all_ssl_all` | 0.925 | 0.842 |
| **`sl_eat_bio_ssl_all`** | **0.957** | **0.900** |

`sl_eat_bio_ssl_all` is ~3% above the next-closest model on Class probe
and ~6% on Order probe. Random-init L5 reads Class 0.852 and Order 0.753
(Order baseline = 0.75, so essentially at chance) — confirming a clean
chance-anchor at the bottom of the table.

### Didn't work: the survival-ratio diagnostic

`order_survival_ratio = (post_order_acc - baseline) / (pre_order_acc - baseline)`
came out at 0.93–1.01 across all four trained models with no separation.
The reason is mechanical: with `max_iters=15`, INLP only drives Class
accuracy from ~0.92 to ~0.83 — a 0.09 drop, not a full nullification. The
nullspace projection is removing 1-D directions per iteration, and Class
information lives in many directions in 768-d. So "Order survives" is
partly a tautology of "Class wasn't really nulled."

The survival ratio cannot be reported as evidence for Class⊥Order
factoring without a stronger nullification regime. See "Follow-ups" below.

## What to extract for the TMLR draft

### New section: probe-based corroboration of §4.7

Add a section (call it §4.11 or §7, depending on flow) titled something
like *"Linear-probe corroboration of taxonomic structure."*

Structure:

1. Motivation: the geometric §4.7 claim relies on centroid subspace
   cosines, which the reviewer flagged as not addressing
   geometry-vs-semantics on its own. We add a linear-probe sanity check.
2. Setup: logistic regression on frame activations. Class probe = Aves vs
   Mammalia (n_aves=400, n_mammalia=200). Order probe = Passeriformes vs
   other-Aves (n_passer=100, n_other=300). 80/20 split, stratified, L2
   regularization. Trained on per-(model, layer) standardized features.
3. **Headline table:** Class and Order pre-INLP accuracies per (model,
   layer) for all 5 models. Bold the row for `sl_eat_bio_ssl_all`.
4. **Headline finding:** `sl_eat_bio_ssl_all` reads the highest Class
   probe accuracy at every layer L5/L7/L9/L12 and the highest Order probe
   accuracy at every layer in that range. The peak is L9 (Class 0.981,
   Order 0.944). Random-init reads at-or-near majority-class baseline on
   Order. The probe ranking matches the §4.7 geometric ranking
   independently.
5. **What this lets us claim:** §4.7's "strongest learned direction"
   language now has probe-based grounding, not just centroid geometry.
   This addresses reviewer concern (1) directly *for §4.7 specifically*
   (the §4.8 Veitch test still needs its own reframe).

### Methodological note (in Limitations or §4.11 itself)

State explicitly that the INLP nullification with `max_iters=15` does not
fully drive Class accuracy to chance (0.92 → 0.83 typical), so we do not
claim Class⊥Order factoring from the survival ratio. The probe-accuracy
table is reported as a probe-baseline sanity check, not as a Veitch-test
replacement.

### Files to commit alongside the draft

- `step6_inlp_class_order.py` (already committed locally, untracked)
- `artifacts/.../inlp_class_order/inlp_summary.csv`
- `artifacts/.../inlp_class_order/inlp_results.csv`
- `artifacts/.../inlp_class_order/inlp_summary.png` (or replace with a
  cleaner probe-accuracy heatmap if the survival-ratio plot is misleading)

## What to discard or downgrade

- **Don't lead with the survival ratio.** The 0.93–1.01 numbers are
  uninformative and risk creating the impression that Class⊥Order
  factoring has been disproven, which is not what the data say.
- **Don't use the survival ratio as a Veitch-test replacement.** Concern
  (6) still needs the random-orthogonality-floor reframe of §4.8;
  INLP-with-15-iters does not give us a stronger answer.
- **Don't claim the probe accuracy by itself proves "factored
  hierarchy."** It corroborates §4.7's Class direction, not §4.8's
  Class⊥Order claim. Be precise about the scope of what probe accuracy
  supports.

## Follow-ups worth queuing if we want stronger Class⊥Order evidence

These are *not* required for the v1 TMLR draft — the probe-accuracy table
plus the §4.8 reframe is sufficient. List them in case we want to push
the factored-hierarchy claim from "geometric + probe-corroborated" to
"geometric + INLP-causal-corroborated":

1. **Aggressive INLP** (~2 hours of compute). Set `max_iters=80–100` and
   `class_acc_floor=majority_baseline + 0.02`. Forces full Class
   nullification; survival ratio then becomes interpretable. Risk: at
   high iter counts INLP can over-project and degrade Order
   non-specifically; need a control where Order is the protected
   attribute and Class survival is the diagnostic.
2. **Multi-class Class probe.** Replace Aves-vs-Mammalia binary with a
   k-way probe (the 4 individual Aves Orders + Mammalia + non-bio
   sources, k=6+). Each INLP iteration nulls a (k-1)-D subspace instead
   of a 1-D direction; Class info dies faster.
3. **INLP-Order as the diagnostic instead.** Train Order probe first,
   null it, then test Class survival. If Class survives Order
   nullification but Order does not survive Class nullification, that's
   asymmetric evidence of hierarchy (Class is the higher-level, more
   distributed feature). Symmetric survival = no hierarchy. This is the
   cleanest causal version of the §4.8 claim.
4. **Activation patching** at L7 of `sl_eat_bio_ssl_all`. Patch the
   Aves-direction component of one clip's L7 activation into another
   clip's residual stream and check whether downstream layers'
   Order-direction projection is preserved. Genuinely causal evidence.
   Compute is cheap; engineering is moderate (we don't currently have a
   patching harness for EAT).

(1) is the cheapest. (3) is the highest-value. (4) is the cleanest but
the highest engineering cost.

## Decision points that depend on the random-init result (still pending)

The 16 trained-model cells already tell most of the story. Two things
random-init's results would resolve:

- **Class probe baseline.** If random-init Class accuracy is materially
  above majority baseline (say 0.85+ at every layer), then "trained
  models read 0.92+ on Class" is a smaller learned effect than it looks
  — the architecture-only network already separates Aves from Mammalia
  reasonably well, and trained models add ~7% on top of that. This
  matters for how we frame the headline.
- **Order probe baseline.** If random-init Order accuracy stays at-or-
  near 0.75 majority baseline across all layers (as L5 already suggests),
  then "trained models read 0.83–0.94 Order" is a clean learned signal
  — random-init has *no* Order discrimination linearly, while trained
  models do.

Both are likely the more interesting framings. Wait for the random-init
cells to land before finalizing the §4.11 prose.

## Concrete next-action checklist

When the INLP run completes:

- [ ] `scp` `inlp_summary.csv` and `inlp_results.csv` from `sentient`
      to `artifacts/comparisons/.../inlp_class_order/` locally.
- [ ] Build the headline probe-accuracy table (5 models × 4 layers =
      20 cells) from `inlp_summary.csv`. Cell value = pre-INLP Class
      and Order accuracy.
- [ ] Compute the across-layer mean per model for Class and Order
      probe accuracy; that's the four-row summary table for the
      §4.11 headline.
- [ ] Add §4.11 to RESULTS.md with the structure above. Cross-link to
      §4.7.
- [ ] Add the methodological note about INLP-15-iters being
      insufficient; cite as a limitation, not a finding.
- [ ] Commit `step6_inlp_class_order.py`, the CSV outputs, and the
      RESULTS.md update.
- [ ] Decide whether the §4.8 Veitch reframe (concern 6) goes in the
      same commit or a separate one.
