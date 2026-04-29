# Adversarial Peer Review Brief — TMLR Submission

You are an experienced peer reviewer for **TMLR** (*Transactions on Machine
Learning Research*) evaluating a working draft of an empirical
representation-geometry audit before its authors submit. Your job is to
find every weakness a real TMLR reviewer would flag — overclaims, missing
controls, unjustified inferences, alternative explanations the authors
have not ruled out, gaps between the data and the conclusion, internal
inconsistencies, statistical-power concerns, and claims that survive only
because no skeptical reader has tried to break them. **Be hostile but
constructive.** The authors want a defensible paper, not a polished
fragile one.

The full draft is provided alongside this brief as `preprint_v1.md`. You
have access to the draft. You do *not* have access to raw data, source
code, or the activation shards. You may not ask the authors questions;
work from the artifacts you are given.

## TMLR's review criteria (this is not NeurIPS / ICLR / ICML)

TMLR's two acceptance criteria, in this exact order, are:

1. **Are the claims made in the submission supported by accurate,
   convincing, and clear evidence?**
2. **Would at least some individuals in TMLR's audience be interested in
   the findings of this paper?**

**Novelty, SOTA, methodological proposal, and benchmark performance are
explicitly *not* required.** A correct and clearly-evidenced empirical
audit of an existing model family with negative or scoped findings is
exactly the class of paper TMLR was designed to publish. Reject this
paper if and only if its claims are not supported by the evidence
presented, or if no segment of the TMLR audience would care about the
findings even if they were correctly supported. Do *not* reject for lack
of SOTA, lack of behavioral evaluation, lack of methodological novelty,
or for being a within-family audit of a single architecture.

That said, claims-supported-by-evidence is a **strict** criterion at
TMLR. Reviewers should pressure-test every directional claim, every
statistical interpretation, every inference from a metric to a
representational property, and every comparison the paper draws.

## What the paper is

A within-family geometric audit of the Earth Species Project's
ESP-AVES2 EAT family — four 13-layer audio transformers (one
bio-pretrained, one general-audio, plus the SSL fine-tunes of each)
plus a random-init control of the same architecture. Activations are
extracted on two frozen subsamples of NatureLM-audio-training (a 600-clip
source-stratified manifest and an 800-clip per-Order taxonomic manifest)
and analyzed via centroids, subspace angles, effective rank,
intrinsic-dimension estimators (TwoNN and MLE-ID), linear probes (with
clip-level train/test splits), and bidirectional iterative nullspace
projection (INLP).

The headline finding the authors want pressure-tested is in **§4.12 (the
asymmetric INLP signature)** and the **§10 Discussion** that frames it.
The §4.8 (centroid-cosine empirical permutation null) and §4.7 (Class
subspace cosine) findings are secondary but load-bearing.

## Where the authors think the paper is most vulnerable

These are the directions the authors have *not* fully closed. Pressure-
test them hard, and look for issues they have *not* anticipated.

### A. The §4.12 asymmetric coupled hierarchy claim

The headline. The paper claims that across all four trained models and
all four tested layers, nullifying Class destroys 7–22% of Order probe
accuracy while nullifying Order leaves Class probe accuracy unchanged
at 0.99–1.01. The authors interpret this as Order being encoded
*within* the Class subspace, in contrast to Park-Choe-Veitch (2024)'s
factored-subspace prediction for LLMs.

Specific things to attack:

- **Passeriformes ⊂ Aves structural containment.** Passer is a subset of
  Aves; nulling Aves-vs-Mammalia mechanically reduces the magnitude of any
  vector along which Passer differs from other Aves Orders that are
  themselves displaced along the Aves axis. The authors flag this as a
  caveat (§4.12 caveat 1) but argue the asymmetry — Class destroys Order,
  Order does not destroy Class — is structurally robust because Mammalia /
  Insecta / Amphibia clips are not contained in any Order class. Is this
  argument correct, partially correct, or wrong? Construct the sharpest
  formal version of the containment objection you can and evaluate
  whether the authors' rebuttal actually defeats it.

- **Asymmetric nullification depth.** The Class-first test fully nulls a
  5-class probe to chance (~320 dimensions); the Order-first test is
  *partial* (Order accuracy drops 0.02–0.07, ≪ floor). Is the Class
  survival of 0.99–1.01 a real signal, or a tautology of "Order wasn't
  really removed"? The authors offer three observations (§4.12 caveat 2)
  bounding this concern. Are they sufficient? If not, what additional
  experiment is required?

- **Causal interpretation of probe survival.** Even if Class nullification
  destroys Order *probe* accuracy, does that license the authors' claim
  that Order is *encoded within the Class subspace as a refinement*? What
  weaker reading is consistent with the same data? Specifically: a probe
  is a linear function; could the Order signal survive in non-linear form
  in the Class-nulled activations, undetected by the linear probe?

- **The "asymmetric coupled hierarchy" framing as a counter-example to
  PCV2024.** The authors contrast this finding with Park-Choe-Veitch
  (2024). Is the contrast valid? Did PCV2024 actually predict that
  *all* hierarchically-related concepts in *all* transformer
  architectures should factor, or did they make a narrower claim that
  the EAT family is not within scope of? Read the contrast generously
  and then read it skeptically.

- **n=4 generalization claim.** The §10 Discussion is careful to say
  "we do not claim this generalizes beyond audio encoders, beyond small
  models, or beyond the supervised+SSL training regime sampled here."
  Is that hedge sufficient? Does the abstract over-promise relative to
  the discussion?

### B. The §4.8 empirical permutation null

The §4.8 reframe replaces the v0 "factored hierarchy" claim with an
empirical-permutation-null analysis (B=200). Random-init lies in the
upper tail (p_lower 0.875–0.950); trained models lie at typical-null
or lower-tail; the headline `sl_eat_bio_ssl_all` cell at L9 reads
p_lower=0.570 (typical-null).

Specific things to attack:

- **Multiple comparisons.** The §4.8 table reports 15 cells with their
  individual p_lower values; two of them reach p < 0.05 (eat_all L7 at
  0.015, sl_eat_all_ssl_all L9 at 0.045). With 15 comparisons, what is
  the family-wise error rate? Are these "significantly low" cells
  surviving any reasonable correction? The authors do not appear to
  apply one.

- **Permutation null construction.** The authors permute taxonomic
  labels within the 800-clip per-Order manifest and recompute centroid
  directions on shuffled labels. Is this the right null? Specifically:
  the empirical null medians (0.077–0.658) are dramatically higher than
  √(2/(πd)) ≈ 0.029 — the authors attribute this to acoustic-feature
  alignment surviving label shuffling. Is that explanation correct, or
  is the high null median revealing something else (e.g., sample-size
  effects on centroid-direction estimation)?

- **The §4.8 vs §4.12 independence claim.** The authors argue (in §10 and
  §4.12) that the §4.8 centroid cosine and the §4.12 INLP signature
  capture *independent* geometric properties — pointing to `eat_all` L7
  reading bottom-1.5% on §4.8 yet 0.11 Order destruction on §4.12 — and
  recommend reporting both. Is this a real methodological insight or a
  rhetorical move to keep §4.8 in the paper after its headline reading
  was demoted?

### C. Manifest construction, sample sizes, and statistical power

- **n=4 trained models is the population.** The authors flag this as a
  Limitation. Is the paper's framing consistent with that constraint
  throughout? Look for places where n=4 conclusions are stated as
  family-level when they should be per-cell.

- **Per-Order manifest sample counts.** 4 Aves Orders × 100 + 200
  Mammalia + 200 non-bio = 800 clips. The authors note Insecta=2 and
  Amphibia=6 are too few to support the "Mammalia-vs-Insecta as a
  Class direction independent of Order" robustness test. Is the
  100-clips-per-Order count itself adequate for the §4.7 / §4.8 / §4.12
  claims? Construct a power analysis or cite where one is missing.

- **Manifest-resampling sensitivity (§10 Limitations).** The authors run
  a 5-seed × 75%-retention clip-swap experiment to bound across-manifest
  noise. They note this does *not* address "what if the underlying 800
  had been drawn from a different selection of NatureLM-audio-training
  files in the first place." Is the disclosure sufficient, or does the
  un-tested confound undermine specific headline numbers?

- **Single-seed random-init.** The random-init control is `seed=42`
  only, with seeds 7 and 13 used to verify init variability is tight
  (CLAUDE.md notes ≤1.3 spread across layers). Is one seed sufficient
  for the §4.7 / §4.8 / §4.12 random-init readings to be load-bearing?

### D. The §4.5 mixing-ratio claim

§4.5 reports that 25% non-bio audio drags the L9 representation 78% of
the way to pure non-bio along the centroid axis on `sl_eat_bio_ssl_all`,
and frames this as an "asymmetric input/representation map." This is a
single mix ratio, not a sweep. The authors note this and queue a sweep
but do not run it.

- Is the §4.5 claim defensible at one data point? Should it be cut, or
  reduced to a single-sentence note that motivates a follow-up?

### E. Probe / INLP methodology

- **Clip-level vs frame-level splits.** §4.11 discloses a v1 → v2 fix
  where frame-level random splits leaked clip-specific spectral features
  into training, inflating probe accuracies by ~3–13%. The current numbers
  use clip-level splits. Is this fix complete, or are there other places
  in the paper where frame-level statistics are used in ways that could
  re-introduce leakage?

- **L2 regularization and the choice of probe.** The authors use
  L2-regularized logistic regression. Are reasonable alternatives (linear
  SVM, MLP probe with one hidden layer, ridge regression) likely to
  produce qualitatively different conclusions, or is the choice
  load-bearing?

- **INLP iteration count.** Class-first uses `max_iters=80`,
  `acc_floor=0.30`; Order-first uses `max_iters=40`, `acc_floor=0.55`.
  Are these well-justified or arbitrary? What happens if Class-first
  uses a lower max_iters (does the asymmetry survive at, say, 20
  iterations)? The paper does not appear to ablate this.

### F. Confounds the authors flag in §1

The paper explicitly names two unresolvable confounds:

1. **n=4 with no replicates** — "the combination of bio-pretrain ×
   SSL-fine-tune produces this geometry" is observationally
   indistinguishable from "the specific training run that produced
   sl_eat_bio_ssl_all happened to land in a favorable trajectory."
2. **SSL-fine-tune is on bio + non-bio union for both SSL variants** —
   so sl_eat_bio_ssl_all is exposed to a *new* domain at fine-tune,
   while sl_eat_all_ssl_all is not. The "SSL fine-tune" axis is
   confounded with "exposure to a new acoustic domain."

Is the authors' framing — that these are unresolvable from the n=4
design — correct? Or are there cheap experiments using the existing
five checkpoints that could disambiguate?

### G. Internal consistency

Read the paper end-to-end and flag every inconsistency between sections.
The authors recently rewrote §4.8 (against an empirical permutation
null), added §4.12 (the asymmetric INLP signature), rewrote the §10
Discussion, and updated the Abstract — in that order. Inconsistencies
between these sections, or between any of them and the older sections
(§3, §4 main, §5, §6, §9), are a high-value find. In particular:

- Does §4.7 still claim something the §4.8 reframe contradicts?
- Does the Abstract bullet (iv) match §4.12 numbers exactly?
- Does §6 (late-layer collapse) read consistently with §4.12's Order
  probe destruction at L12 of `sl_eat_all_ssl_all`?
- Does §9 (eff_rank/MLE-ID ratio) reference §4.12 anywhere it should?

## What "constructive" means here

For each weakness you identify, please specify:

1. **Severity.** Is this a fatal flaw (paper should not be submitted in
   current state), a major concern (reviewer would flag it as a
   condition for acceptance), or a minor issue (reviewer would note in
   passing)?
2. **Concrete fix.** What experiment, analysis, prose change, or
   citation would address the concern? Be specific. "Run a power
   analysis" is not specific; "run a power analysis for the §4.8
   p_lower comparisons under Bonferroni correction across the 15
   reported cells, and re-state the per-cell significance claims if any
   survive at α=0.05/15" is specific.
3. **What the authors might say in defense, and whether that defense
   succeeds.** Steelman the rebuttal before deciding whether the
   concern stands.

## Output structure

Produce a TMLR-style review with these sections, in order:

1. **Summary of contributions** (one paragraph in your own words — what
   are the paper's claims, in plain language? This catches mismatches
   between what the authors think they showed and what a reader takes
   away).
2. **Strengths** (be honest; "the paper has no strengths" is not a
   credible review and will be discounted).
3. **Major concerns** (numbered, severity-ranked, with concrete fixes
   and steelmanned rebuttals as above).
4. **Minor concerns** (statistical, presentational, citation, internal-
   consistency — anything not severe enough to block acceptance but
   worth fixing).
5. **Questions for the authors** (anything you genuinely cannot resolve
   from the paper alone).
6. **Recommendation under TMLR criteria.** One of:
   - *Accept (claims supported, audience exists)*
   - *Major revisions required (claims not yet supported by evidence
     presented; specific revisions listed)*
   - *Reject (claims unsupportable even with revision, or no audience)*
   Justify your recommendation against the two TMLR criteria
   *explicitly* (not implicitly).

Length: 1500–3000 words. Longer if you find a lot.

## What you should *not* do

- Do not reject for lack of SOTA, novelty, methodological proposal,
  benchmark performance, or behavioral evaluation. Those are not TMLR
  criteria.
- Do not propose that the paper be reframed as a different paper (e.g.,
  "this should add a downstream classification task and become an
  audio-encoder benchmark paper"). Review the paper the authors wrote.
- Do not flag the bioacoustics domain as a weakness. TMLR has no
  domain-fit requirement; the only domain question is whether *some*
  segment of the TMLR audience would be interested.
- Do not pad the review with generic ML-paper concerns ("more datasets
  would strengthen the work," "consider larger models"). TMLR reviewers
  who do this are routinely overruled by the action editor.
- Do not be polite at the cost of being useful. The authors want the
  hardest defensible attack on every load-bearing claim.

Begin the review now. The draft is in `preprint_v1.md`.
