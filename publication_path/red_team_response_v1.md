# Author response to TMLR red-team review

Thank you for the thorough review. We accept most of it and propose a
two-round revision plan: pure prose fixes (Round A) for concerns where
the data already exists in the paper or its underlying tables, and a
focused set of re-analyses on the existing activation shards (Round B,
no new model forward passes required) for concerns where new evidence is
needed. We do not plan a multi-seed random-init re-extraction (3.7) for
this revision and explain the deferral below.

We are *not* planning a further round of synthetic adversarial review
between Round B and TMLR submission; the marginal value of another
simulated round drops sharply after Round B closes the experimental
gaps. We would rather spend reviewer attention on the actual TMLR
reviews. We are asking, however, for your judgment on whether the plan
below closes your concerns to a level that meets TMLR criterion 1
("claims supported by accurate, convincing, clear evidence"), before we
commit. Two explicit questions to you are at the end.

---

## Major concerns

### 3.1 Headline range overstated — accepted

You are right. The actual range across all 16 trained-model × layer
cells in §4.12 is 0.012–0.218; the 7–22% range applies to the 12 cells
where pre-nullification Order accuracy ≥ 0.77.

**Round A fix.** Restate the headline as "across the 12 trained-model ×
layer cells where pre-nullification Order accuracy exceeds majority
baseline (≥ 0.77), Class nullification destroys 0.057–0.218 of Order
probe accuracy" and apply consistently across abstract, §4.12, and §10.
The "0.07–0.12 in the other trained models" sentence will be rewritten
to match per-cell L9 numbers (0.066, 0.070, 0.098 — i.e., 0.07–0.10 at
L9 specifically).

### 3.2 Cells starting at or below baseline are not interpretable — accepted

You are right; the inconsistency between §5.2 (which already names
sl_eat_all_ssl_all L12 as mode-collapsed with Order at majority baseline)
and §4.12 (which includes the same cell in its destruction range) is a
real internal-consistency failure.

**Round A fix.** (i) Exclude eat_all L12, eat_bio L12, and
sl_eat_all_ssl_all L12 from the §4.12 headline range. (ii) Report them
in a separate footnoted row labeled "pre-nullification Order accuracy at
or below majority baseline; destruction not interpretable as Order-signal
removal." (iii) Revise §10 from "all four trained models and all four
tested layers" to "all four trained models and L5/L7/L9, plus
sl_eat_bio_ssl_all at L12." This brings §4.12, §5.2, and §10 into
consistency.

### 3.3 Multiple-comparisons correction in §4.8 — accepted

You are right. With 15 cells, no per-cell p_lower survives Bonferroni at
α = 0.0033 or BH-FDR at q = 0.05 (smallest p = 0.015 vs threshold
0.0033).

**Round A fix.** (i) Drop the bolding on `eat_all` L7,
`sl_eat_all_ssl_all` L9, and the random-init upper-tail rows. (ii) Add
an explicit footnote stating that no individual cell survives Bonferroni
or BH-FDR correction over the 15 reported comparisons, and that per-cell
p_lower values are reported as descriptive only. (iii) Tighten the
directional claim to: "trained-model median |cos| at L7/L9 is one to two
orders of magnitude smaller than the random-init reading at the same
layers; no per-cell significance test is required to support this
directional comparison." (iv) Remove the §10 reference to `eat_all` L7's
"bottom 1.5% of the empirical permutation null."

### 3.4 Linear-probe vs representational claim — partially accepted

We accept the linguistic concern: the §10 phrase "Order is encoded
within the Class subspace as a refinement" exceeds what a linear probe
alone licenses.

**Round A fix.** Soften to "the *linear* component of Order is encoded
within the Class subspace, in the sense that linear projection onto the
nullspace of a 5-class Class probe destroys 0.07–0.22 of Order probe
accuracy in 12 of 16 cells."

**Round B experiment.** MLP probe (one hidden layer, 256 ReLU units,
L2-regularized) for Order classification on Class-nulled activations at
the four `sl_eat_bio_ssl_all` cells (L5/L7/L9/L12). Report accuracy
alongside the linear-probe accuracy on the same activations. If MLP
recovers Order substantially above the linear post-nullification
accuracy, we keep the "linear component" framing and report the gap
explicitly. If MLP also fails, we strengthen back toward "Order is
encoded in the Class subspace" with the MLP ablation as the additional
evidence.

**One steelman we want to flag, but will run the experiment regardless.**
PCV2024's factored-subspace prediction is itself made within a
linear-probe framework — they test feature factoring with linear
nullification of one feature followed by linear probing of the other.
A linear-probe-only contrast is therefore methodologically aligned with
the prediction we are contrasting against; requiring non-linear evidence
to support a contrast against a linear-framework prediction is partly
moving the goalposts. We will run the MLP ablation and report honestly
either way; we want to note that "linear component of Order" is a
defensible reading on its own, not just a fallback if the MLP succeeds.

### 3.5 Binary Aves-vs-Mammalia probe re-trained on Class-nulled activations — accepted

You are right that this is the cheapest test in the paper. We will run
it.

**Round B experiment.** For each of the 16 trained-model × layer cells
in step8, after the 5-class probe is at chance, re-train an
L2-regularized binary Aves-vs-Mammalia probe on the Class-nulled
activations and report accuracy. Three possible outcomes:

- *Binary accuracy stays at 0.667 majority baseline:* the "Class nullified"
  interpretation is fully supported; the headline framing stands.
- *Binary accuracy recovers to 0.70–0.80:* partial Class signal survives;
  framing revises to "directions discoverable by a 5-class probe destroy
  0.07–0.22 of Order accuracy; binary Class signal partially survives in
  directions the 5-class probe did not surface" — a weaker but still
  defensible asymmetric coupled hierarchy claim.
- *Binary accuracy recovers near pre-nullification (0.85+):* the 5-class
  INLP nulled a probe-specific subspace rather than Class itself; we
  retract the "Class destroys Order" framing and revise to "directions
  along which 5-class probes find Class destroy a fraction of Order
  signal," which is substantively a different paper. We commit to making
  this revision honestly if the data shows it.

### 3.6 INLP iteration-depth ablation — accepted

You are right that the existing caveat 2 was a prose defense that did
not close the concern.

**Round B experiment.** Class-first INLP at `max_iters ∈ {10, 20, 40,
80}` on the four `sl_eat_bio_ssl_all` cells (L5/L7/L9/L12). Report
destruction-of-Order curve as a function of cumulative-D nulled. The
asymmetry-survives test: at `max_iters=10` (cumulative ~40-D nulled —
roughly comparable to Order-first's typical 1–10 1-D nulls in
total-D-nulled terms), is Order destruction comparable to or larger than
the Order-on-Class destruction (≤ 0.001 in absolute Class drop)? If yes,
the asymmetry is robust. If Order destruction at low iteration count is
near zero, we will report iteration-depth as a partial confound and
revise framing.

### 3.7 Single-seed random-init — accepted, deferred to Limitations

Conceding the substance: random-init readings for §4.7 / §4.8 / §4.12
are at seed 42 only. Seeds 7 and 13 verified eff_rank stability (≤ 1.3
spread), but you are right that this does not transfer to subspace
cosines, centroid cosines, or INLP signatures.

**We propose deferring re-extraction at seeds 7 and 13** for the
following reasons:

1. The shards for seeds 7 and 13 were deleted to save disk;
   re-extraction requires ~half a day per seed of remote GPU time. ROI
   for spending that compute on an admission rather than a finding is
   marginal at this stage.
2. The trained-vs-random gaps in question are large: 28× the
   within-manifest CI for §4.7; an order of magnitude in raw cosine for
   §4.8 (random-init 0.92–0.94 vs trained models' 0.003–0.401);
   uniformly opposite-signed for §4.12 (trained models destroy Order;
   random-init *improves* Order under Class nullification by ~0.05).
   Across-seed variability would have to be enormous to flip any of
   these directional readings.
3. A clear Limitations admission is appropriate under TMLR criterion 1:
   evidence presented should support the claims made, but limitations
   should be honestly disclosed.

**Round A Limitations addition.** §10 Limitations gains: "Random-init
readings for §4.7 (subspace cosine), §4.8 (centroid cosine), and §4.12
(INLP signature) are reported at a single seed (42). Init variability
was verified for effective rank only (≤ 1.3 spread across seeds 7, 13,
42); whether this stability transfers to the metrics in §4.7 / §4.8 /
§4.12 has not been tested. Where these random-init readings serve as
the comparator for trained-model claims, within-manifest CIs do not
absorb across-seed variability. We expect the trained-vs-random
directional rankings to be robust to seed choice given the magnitude of
the gaps, but a future revision should verify this empirically."

We are open to running multi-seed re-extraction (Round C) if you judge
the deferral insufficient. **We want your judgment here explicitly.**

---

## Minor concerns

We accept and will apply in Round A:

- §4.5 — soften section title from "asymmetric input/representation map"
  to "non-linear input/representation map," and add an inline note that
  the result is a single mix-ratio observation (a sweep is queued but
  not run).
- §4.8 vs §4.12 dissociation framing — reframe as a methodological prior
  that motivated running both tests, not as a post-hoc defense of
  keeping §4.8 after its headline cell was demoted.
- §9 → §4.12 cross-reference — add a sentence noting that the
  eff_rank/MLE-ID ratio expansion in §9 is the geometric substrate that
  enables the §4.12 asymmetric coupled hierarchy.
- §10 sentence softening — "consistent with the demoted directional
  reading of §4.8 and supported primarily by the probe-causal evidence
  here" replaces "consistent with both the centroid-cosine evidence in
  §4.8 and the probe-causal evidence here."
- INLP projection-direction specification — confirmed by code review
  that projections are computed from training-set frames only; will
  state explicitly in §4.11 and §4.12.
- Random-init Order-probe-below-baseline phrasing — clarify that the
  probe is doing something other than majority-vote and underperforming
  because the linear separator overfits in high-dim space with noisy
  class signal.
- PCV2024 unembedding-structure clarification — add one sentence in §10
  noting that EAT lacks the unembedding-matrix structure PCV2024's
  factored prediction is grounded in, so non-transfer to this family is
  structurally expected and our finding should be read as an empirical
  corroboration of that structural prediction rather than a refutation
  of PCV2024.

We accept your low-rank → correlated-random-direction-estimates
explanation for the high empirical null medians in §4.8 as more
parsimonious than the acoustic-features-survive-shuffling explanation we
wrote.

**Round B analysis (Q6).** Pearson correlation between empirical null
median and effective rank across the 15 (model, layer) cells. If the
correlation is strong and negative (high eff_rank → low null median),
your explanation is supported and we revise §4.8 accordingly. If weak,
both explanations remain on the table and we report both. ~minutes of
compute.

**Round B analysis (Q5).** Confirm INLP projection direction is
computed from training-set frames only via code re-read; if so, state
explicitly. If not, re-run.

---

## What we are asking of you

We are not asking for a re-review of the current draft. We are asking
for your judgment on whether the proposed revision plan is sufficient.

**Question 1.** Does the plan above (Round A prose fixes + Round B
re-analyses on existing shards) close concerns 3.1–3.6 to a level you
would accept under TMLR criterion 1? Specifically: are there concerns
where our proposed fix would still leave a meaningful gap that you
would re-flag as a major issue if you saw the post-revision draft?

**Question 2.** Is the 3.7 deferral defensible, given the magnitude of
the trained-vs-random gaps, the eff_rank cross-seed stability, and the
explicit Limitations note? Or do you believe deferring 3.7 drops the
paper below TMLR criterion 1, such that Round C (multi-seed
re-extraction at seeds 7 and 13, ~1 day GPU per seed) should happen
before TMLR submission rather than after a TMLR reviewer raises it?

We will treat your answer to Question 1 as binding (if you say a
specific concern is not adequately closed, we will revise the plan); we
will treat your answer to Question 2 as advisory (we want your input
but reserve the cost-of-Round-C decision for ourselves).
