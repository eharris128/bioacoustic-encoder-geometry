# Next Steps

## Roadmap Section 1 — ESP-AVES2 Activations (active)

Scope: ESP-AVES2 `eat`-family only (see `open_questions.md` §2). Step 1 +
Step 2 + selected Step 3 items (audio mixing, species barycenters,
hierarchical/Veitch). **Noise dynamics is the teammate's; do not duplicate.**

### Step 1 — collection (DONE)

All four EAT-family models extracted against the frozen manifest `naturelm_by_source_100each_20260418T171459Z`. Shards live under `artifacts/roadmap_part1/<manifest_id>/<model>/shards/`:

- [x] `esp-aves2-sl-eat-bio-ssl-all`, `esp-aves2-sl-eat-all-ssl-all` (since 2026-04-18).
- [x] `esp-aves2-eat-all`, `esp-aves2-eat-bio` (after the 2026-04-20 HF re-publish + the fairseq-key loader fix in `a6498aa`).
- [x] `compare_esp_aves2_models.py`, `app_esp_aves2_compare.py`, and `nway_compare_eat_models.py` are tracked.

### Step 1 — outstanding decisions

- [x] **Storage for activations: local disk for the pilot.** Resolved
  2026-04-27 in `open_questions.md` §3. Re-evaluate only if we scale
  beyond the 600-sample manifest or add collaborators needing direct
  activation access.
- [x] Decide whether to scale beyond 100 samples × 6 sources × 4 models.
  Resolved: no scale-up. The taxonomic per-Order manifest (commit
  `26485da`, 200 samples per Order across 4 bird Orders + 200 per
  non-bird Class) gave per-(class, layer) cells with tight bootstrap
  CIs (commit `0310354`); the Round B campaign closed all 7 red-team
  concerns at this scale. Re-evaluate only if a future round needs
  finer per-Family/Genus resolution.

### Step 2 — statistics across layers and models

Done (all four models, mean-pooled, in `artifacts/comparisons/.../nway_eat_all4/step2_spectral_dim/`):

- [x] Singular-value spectra per `(model, layer)`.
- [x] Effective rank + participation ratio per `(model, layer)`.
- [x] TwoNN intrinsic dimensionality per `(model, layer)`.
- [x] Effective rank sliced by the seven `source_dataset` values + a nature-vs-non-nature split.
- [x] Cross-model CKA heatmap (in the parent `nway_eat_all4/` dir).

Outstanding:

- [x] **Within-model L2-norm distributions** per `(model, layer)` — done in `10601b9` (`step2_subspace_angles_eat.py`); histograms in `step2_subspace_angles/l2_norm_histograms.png`, percentiles in `l2_norm_per_layer.csv`.
- [x] **PCA / subspace alignment** across layers (within a model) and across models (within a layer) — done in `10601b9`; across-layer heatmaps + across-model curves in `step2_subspace_angles/`.
- [x] **Pooled vs frame-level** sanity check on `sl_eat_bio_ssl_all` — done in `11f9920` (`step2_pooled_vs_frame_eat.py`). **Pooling materially distorts geometry**: frame-level eff. rank is 2–6× larger than pooled at every layer, while frame-level TwoNN drops to ~3–5 vs ~9–11 pooled. Both directions matter — pooling overstates intrinsic dim and understates linear spread.

### Step 2 follow-ups motivated by current findings

From the `dd24541` Step 2 results:

- [x] **`sl_eat_bio_ssl_all` uses a 2–3× wider linear subspace** — confirmed in `10601b9`. Bio-vs-non-bio subspace cos drops to **0.33 at L9** (same layer as eff. rank ~148), vs 0.55–0.70 for the other three models. The bio fine-tune is genuinely separating bio from non-bio in subspace direction, not just expanding both.
- [x] **TwoNN ID stays ~8–12 everywhere** while linear effective rank swings 3–148 — addressed via the pooled-vs-frame comparison in `11f9920`. Frame-level TwoNN is ~3–5 (much lower), confirming the curved low-dim manifold story; the pooled-level TwoNN was a measurement artifact of mean-pooling.
- [x] **L0 effective rank ≈ 3 across all four models** — **rejected as a "shared tokenizer" story** in `10601b9`. Three models share L0 subspace at cos 0.91–0.98, but `eat_bio` is ~orthogonal to all of them (cos 0.28–0.32). Each model independently learned a low-dim L0 subspace; they happen to land at similar dimensionality but not the same direction.

### Step 2 outstanding (Tier 1 follow-ups)

Done in `step2_tier1_frame_level.py` (output: `step2_tier1_frame_level/`):

- [x] **TwoNN L4 dip resolved as a TwoNN(k=2) estimator artifact.** Adding MLE-ID with k=20 alongside TwoNN: 3 of 4 models show the L4 TwoNN crash (`eat_all` 0.95, `sl_eat_all_ssl_all` 1.67, `sl_eat_bio_ssl_all` 2.63) while MLE-ID(k=20) reads stable values (13.27, 13.69, 14.05). The "L4 dip" is a TwoNN failure mode, not a model phenomenon — retract from any narrative built on `step2_pooled_vs_frame/`.
- [x] **Pooled-vs-frame distortion is universal but heterogeneous.** Frame-level eff_rank > pooled eff_rank at every (model, layer); the ratio varies wildly. Pooled L0 was uniformly ~3 across all four models — frame-level L0 spreads to **11 / 26 / 44 / 17** for `eat_all` / `eat_bio` / `sl_eat_all_ssl_all` / `sl_eat_bio_ssl_all`. The "L0 effective rank ≈ 3 across all four models" finding from `dd24541` was a pooling artifact; at frame level the four L0s are visibly different.
- [x] **Bio-vs-nonbio direction story survives at frame level but is muted.** The pooled "0.33 at L9 for `sl_eat_bio_ssl_all` vs 0.55–0.70 elsewhere" becomes "**0.57 at L9 vs 0.68–0.81 elsewhere**" at frame level. `sl_eat_bio_ssl_all` is still the most-separated model and still bottoms out around L7–L9, but pooling inflated the dramatic-looking number. Restate as: "bio fine-tune separates bio from non-bio in subspace direction at frame level (mean cos drops to ~0.57 in mid-network), but the effect is smaller than mean-pooling suggested."

### Step 2 new findings from Tier 1 — folded into RESULTS.md

These three findings are documentation, not new experiments. All written
up in RESULTS.md; tracked here for historical reference only.

- [x] **Late-layer collapse splits the family by `_bio` vs not.**
  Frame-level L12 eff_rank: `eat_all` 62.6, `sl_eat_all_ssl_all` 11.2,
  vs `eat_bio` 180.4, `sl_eat_bio_ssl_all` 188.7. Documented as
  RESULTS.md §5; the mechanism question is §9.1 OPEN.
- [x] **Frame-level model rank-order at eff_rank peak flips vs pooled.**
  Pooled said `sl_eat_bio_ssl_all` ≫ `eat_bio` ≈ `sl_eat_all` > `eat_all`;
  frame says `sl_eat_bio_ssl_all` > `eat_all` ≈ `sl_eat_all_ssl_all` >
  `eat_bio`. Documented in RESULTS.md §3; pooled artifact, not a model
  property. No further action.
- [x] **MLE-ID(k=20) at 7–14 conserved → re-framed.** After the
  random-init baseline (random reads 11–15 in the same range), this is
  not a learned property of trained models. The learned property is the
  eff_rank / MLE-ID ratio (random ≈ 1, trained 17–43). Documented as
  the corrected RESULTS.md §6.

### Step 2 outstanding — taxonomic resolution of "nature vs other"

The roadmap's Step 2 explicitly asked us to "compare nature sounds to
other sound." We covered this coarsely (bio-vs-non-bio in §4 of
`RESULTS.md`) but not at finer taxonomic resolution. Our complement to
the teammate's probe-based hierarchy work is the *geometric* version
(centroids + subspace angles, no probes needed).

**Manifest enrichment is unblocked.** All 600 samples already carry full
taxonomic metadata in the parquet `metadata` JSON column —
`phylum / class / order / family / genus / species / subspecies` — verified
2026-04-27. Bio sources (Xeno-canto, iNaturalist, Animal Sound Archive,
Watkins) have populated taxonomic fields; non-bio sources (WavCaps,
NatureLM/NSynth) carry empty/placeholder values for those fields. We
can write a small script ourselves; no teammate coordination needed.

- [x] **Manifest enrichment script** (commit `d88687e`). All 600
  samples enriched with phylum / class / order / family / genus /
  species / subspecies. Coverage: bio sources 99–100% populated,
  non-bio sources empty (expected). Aves 271, Mammalia 119, Amphibia
  6, Insecta 2.
- [x] **Per-Class frame-level metrics** (commit `acbb774`).
  `sl_eat_bio_ssl_all` L7 hits Aves-vs-Mammalia cos = 0.379 — the
  strongest learned direction in the family, beating the §4
  bio-vs-nonbio L9 minimum of 0.580. Documented as RESULTS.md §4.7.
- [x] **Per-Order frame-level within Aves** (commit `acbb774`).
  Passeriformes vs other-Aves; `sl_eat_bio_ssl_all` L9 cos = 0.729.
  Order-level structure lives in many fewer dimensions than
  Class-level. Geometric peak L9 = teammate's probe peak L9.

## Roadmap Section 1 Step 3 — specific cases (now partially in scope)

Original Step 3 had three items. We're picking up two; the teammate has the third.

### Step 3a — Audio mixing along bio↔non-bio (DONE)

- [x] **bio↔non-bio mixing on `sl_eat_bio_ssl_all`** (commit `bbde771`).
  Threshold-like asymmetry: 25% non-bio audio drags the L9 representation
  78% of the way to pure non-bio along the centroid axis. 10-D subspace
  energy is more linear. Documented as RESULTS.md §4.5.
- [x] **Extended to `eat_all` / `eat_bio` / `sl_eat_all_ssl_all`** (commit
  `5793d8c`). The threshold-like asymmetry is *specific to*
  `sl_eat_bio_ssl_all`. The other three models have near-zero or partial
  bio-axis range, so "linear vs threshold" is not a meaningful question
  on them. Mechanistic conclusion: the wide bio-axis is unlocked by the
  combination of bio pretrain + SSL fine-tune. RESULTS.md §4.5 includes
  the cross-model table.

### Step 3b — Species barycenters

Unblocked: species labels are already in the parquet `metadata` JSON
(see Step 2 manifest enrichment above). Compute is trivial once the
enrichment script lands.

- [x] **Per-species centroids per (model, layer)** (commit `98d924e`).
  12 species clear the 5-sample threshold; centroids saved per
  (model, layer, species) under `species_barycenters/species_per_layer.csv`.
- [x] **Within-species vs between-species variance** (commit
  `98d924e`). Separability ratio = between / (within + between).
  Surprising result: random_init has the HIGHEST separability ratio
  at every layer (0.33 at L12) — trained models *compress* fine
  species detail to learn coarser abstractions. `sl_eat_bio_ssl_all`
  is the only trained model with substantial species structure (0.20
  peak at L10). Documented as RESULTS.md §4.9. Bootstrap-confirmed
  in commit `63eb676` (random L10 [0.286, 0.381] vs sl_eat_bio L10
  [0.206, 0.239] — no CI overlap).

### Step 3c — Hierarchical representations (Veitch)

The roadmap's "do we see hierarchical representations? See Victor
Veitch paper" item. **Does not require probes** — Veitch-style geometric
tests work on centroids + subspace angles, which we already compute.
Unblocked once Step 2 manifest enrichment lands.

- [x] **Class-direction vs Order-direction orthogonality** (commit
  `5eb044e`). Done with Aves vs Mammalia at the Class level and
  Passeriformes vs other-Aves at the Order level (within Aves).
  `sl_eat_bio_ssl_all` is the only trained model that factors the
  hierarchy: L9 cos = 0.136, **L12 cos = 0.074** (essentially
  perpendicular). None of the other trained models drop below 0.30.
  Documented as RESULTS.md §4.8. Bootstrap-confirmed in commit
  `63eb676` (sl_eat_bio L12 = 0.081 [0.021, 0.155], no CI overlap
  with any other trained model).
- [x] **Layer-resolved hierarchy** (commit `5eb044e`). Computed at
  every layer L0..L12; sl_eat_bio_ssl_all hits two minima — L9 (0.14)
  coincides with §4 / §4.5 / §4.7 / §4.9 peaks; L12 (0.07) is the
  cleanest factoring layer.
- [x] **Stronger nested-subspace test (done — `step3c_veitch_4order.py`,
  Phase 3 commit `fcb507e`; cross-check on OLD manifest commit
  `bb00af1`).** Scaled up to a per-Order manifest with 4 individual
  bird Orders × 100 samples each (Passeriformes, Charadriiformes,
  Piciformes, Strigiformes). Tested |cos(Aves−Mammalia, Order_i−Aves)|
  per (model, layer, Order). Headline: `sl_eat_bio_ssl_all` factors all
  4 Orders against the Class direction at L7, with cos 0.03–0.08 across
  every Order. None of the other trained models drop the median below
  0.30. The §4.8 Veitch finding generalizes from "Passer-vs-other-Aves"
  to all 4 individual bird Orders.

## Roadmap Section 2 — owned by teammate

Probes + attribution work is being pursued by a teammate. Out of scope
for this thread; do not duplicate.

(Earlier note about coordinating on Class / Order / Species labels has
been retracted — those labels are already in the parquet `metadata`
JSON and we can derive them ourselves; see Step 2 manifest enrichment.)

## Reviewer-driven follow-ups (preprint v0 round 1) — attend to today

Triggered by Opus 4.7 extended-thinking review of the v0 preprint draft.
Concern numbering tracks the review.

### Compute (run on `sentient` 129.213.131.108 once GPU is free)

- [x] **§4.8 INLP probe** (commits `2a79e5a` initial, `c3cd168`
  frame-level leakage fix, `d45dc01` clip-level v2). Class-then-Order
  INLP run across L5/L7/L9/L12 in `inlp_class_order/`; clip-level v2
  results integrated into preprint_v2.md §8. Round B follow-ups —
  step8 aggressive (`step8_inlp_aggressive.py`, `8e0e93c`) and step9
  Order-first (`step9_inlp_order_first.py`, `8e0e93c`) — confirmed the
  asymmetric INLP signature: Class survives Order-nullification, Order
  does not survive Class-nullification.
- [x] **Manifest-resampling sensitivity** (commit `a4005fe`,
  `step7_manifest_resampling.py`). Across-seed spread reported in
  `manifest_resampling/resampling_summary.csv` alongside
  within-manifest CIs; results integrated into preprint_v1 and v2.

### Prose-only revisions (no compute) — preprint draft v1

- [x] **Concern (1) — soften semantic vocabulary** (commits `fbcb372`
  v1 §4.8/§10/abstract reframe, `de1b87c`/`b557173` v2). v2 uses
  "exhibits a geometric property" and "asymmetric coupled hierarchy"
  framing throughout; the §5.2 "thrown out everything except its bio
  classifier" and §5.4 "bio classifier is installed at L12" phrases
  are gone.
- [x] **Concern (1) — §4.5 threshold-vs-linear is a single point**
  (commits `9e907ac` step13 added, `d34185f` 11-α refined sweep).
  `audio_mixing_refined/mixing_summary_by_alpha.csv`: at α=0.025
  (2.5% non-bio audio), bio-axis projection has shifted 44% of full
  range — sharp threshold near α=0, approximately linear thereafter.
  Integrated into preprint_v2.md §4.5.
- [x] **Concern (2) — random-init is a "preserves input acoustics"
  control** (commit `a4005fe`). Flagged in preprint_v1 §2 and §5/§6;
  v2 §10 Discussion further addresses the "got there by opposite
  routes" framing. Stronger controls (shuffled-label SSL, frequency-PCA
  init) deferred — not required by reviewer per v2 ship readiness.
- [x] **Concern (3) — the SSL-axis is confounded with new-data-domain
  exposure** (commit `a4005fe`). Confound called out in preprint_v1 §1
  and §8; carried into v2 §10 Limitations (iv). The n=4 design
  inability to resolve it is explicit.
- [x] **Concern (5) — reframe §3** (commit `a4005fe`). v2 §3 now
  reads "Frame-level and mean-pooled views diverge heterogeneously
  across (model, layer)" — "distortion"/"pathology" language dropped.
- [x] **Concern (6) — fix the Veitch null-distribution comparison**
  (commits `fbcb372` v1 §4.8 reframe, `8e0e93c`
  `step10_veitch_permutation_null.py`). Permutation-null results in
  `veitch_perm_null/`. v2 §8 reframes as "asymmetric INLP signature"
  rather than factored-hierarchy; random-init's anomalous alignment
  vs trained-model drift toward orthogonality floor is the framing.

### Framing decision (after INLP completes) — RESOLVED

INLP-Order destruction is real but partial (0.057–0.218 across 11
retained cells; largest at `sl_eat_bio_ssl_all` L9). Symmetry-test
showed Class survives Order-nullification, Order does not survive
Class-nullification (step14 `multiclass_order_inlp`, commit `9e907ac`).

- [x] Adopted: **asymmetric coupled hierarchy** framing (commit
  `fbcb372` v1 reframe, integrated into preprint_v2.md §8). Neither
  pure factored nor pure entangled — the linear component of Order is
  encoded within the Class subspace, with non-linear residue
  (MLP-probe 0.015–0.083 post-null). This is a *third* option beyond
  the binary the original framing-decision posed.

## Out of scope for the current pilot

- **Roadmap Section 1 Step 3 noise dynamics** (white/pink noise + "noise subspace") — the teammate's, do not duplicate.
- Roadmap Section 3 (dictionary learning / SAEs) — explicitly low priority in the roadmap.
- Original AVES + BirdAVES (`open_questions.md` §2).
- Cross-species call type transfer, RSA with CRCNS zebra-finch, unsupervised syllable segmentation, and the call-type discovery work from earlier exploratory phases. Revisit only after the roadmap pilot is complete.
