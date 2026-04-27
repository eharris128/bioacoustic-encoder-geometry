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

- [ ] Decide whether to scale beyond 100 samples × 7 sources × 4 models. Tied to the storage question.
- [ ] Resolve the "where do we store activations?" open question (capture the answer in `open_questions.md` §3 even if it's just "local disk for the pilot"). Gating any scale-up.

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

### Step 2 new findings from Tier 1

- [ ] **Late-layer collapse splits the family by `_bio` vs not.** Frame-level L12 eff_rank: `eat_all` 62.6, `sl_eat_all_ssl_all` **11.2**, vs `eat_bio` 180.4, `sl_eat_bio_ssl_all` 188.7. The two non-bio models collapse hard at the output; the two bio fine-tunes retain rich variance through to L12. SSL on top of `eat_all` (i.e. `sl_eat_all_ssl_all`) collapses harder than `eat_all` alone. Worth a follow-up: is this saying the bio pretrain leaves the output-side representation usable for downstream tasks while the non-bio + SSL combo over-compresses?
- [ ] **Frame-level model rank order at the eff_rank peak is `sl_eat_bio_ssl_all` > `eat_all` ≈ `sl_eat_all_ssl_all` > `eat_bio`.** Pooled said `sl_eat_bio_ssl_all` ≫ `eat_bio` ≈ `sl_eat_all` > `eat_all`. The relative position of `eat_bio` vs the other two flips between pooled and frame-level — pooled overstates `eat_bio`'s linear subspace width relative to `eat_all` and `sl_eat_all_ssl_all`. Worth restating any claims that ranked `eat_bio` high on linear width.
- [ ] **MLE-ID(k=20) frame-level intrinsic dim sits at 7–14 across all (model, layer)** vs eff_rank swings of 11–348. Confirms the curved-low-dim-manifold-inside-wide-linear-subspace story holds for every model in the family, not just `sl_eat_bio_ssl_all`. **Reframed in RESULTS.md §6 after the random-init baseline:** trained 7–14 is at-or-below the random-init baseline of 11–15. The interesting learned property is the eff_rank/MLE-ID *ratio* (random ≈ 1, trained 17–43), not the absolute manifold dim.

### Step 2 outstanding — taxonomic resolution of "nature vs other"

The roadmap's Step 2 explicitly asked us to "compare nature sounds to other sound." We covered this coarsely (bio-vs-non-bio in §4 of `RESULTS.md`) but not at finer taxonomic resolution. The teammate's Class/Order probes (PNGs from 2026-04-11 conversation) show Aves vs Amphibia vs Mammalia is decodable at L5 (82.5%) and bird Order at L9 (70.3%) — that's *behavioral* hierarchy. Our complement is the *geometric* version:

- [ ] **Per-Class (Aves / Amphibia / Mammalia / …) frame-level eff_rank, MLE-ID, and pairwise top-10 subspace overlap.** Same metrics as §3–§5 of `RESULTS.md` but sliced by taxonomic Class. Tells us where Class-level distinctions live in the network (L5 per probes — does eff_rank or subspace direction separate at the same layer?).
- [ ] **Per-Order frame-level versions of the same metrics within Aves.** Probe peak is L9 — does the directional-separation peak match?
- [ ] **Manifest enrichment.** Add Class / Order / Species columns to `naturelm_by_source_100each_20260418T171459Z` (or a new manifest variant). The teammate already has these labels; ingesting them is the prerequisite for both this and the §3 Step-3 hierarchical work below. Coordinate with teammate.

## Roadmap Section 1 Step 3 — specific cases (now partially in scope)

Original Step 3 had three items. We're picking up two; the teammate has the third.

### Step 3a — Audio mixing along bio↔non-bio (§9.7 in RESULTS.md)

- [ ] **Linear bio↔non-bio mixing.** Take a bio clip A and non-bio clip B, generate audio mixtures `M(α) = (1-α)·A + α·B` for α ∈ {0, 0.25, 0.5, 0.75, 1}, run each through `sl_eat_bio_ssl_all`, project onto the §4 top-10 bio-only and non-bio-only subspaces. Three diagnostic outcomes: (a) **smooth linear interpolation** of cos angles in α (linear feature); (b) **sharp threshold** (gating mechanism); (c) **off-manifold excursion** (model treats mixtures as OOD). Converts §4 from descriptive to mechanistic. Requires the HF NatureLM-audio-training parquet cache — *do not delete*. Compute is small (~50–100 mixtures × 1 model × 13 layers).

### Step 3b — Species barycenters

- [ ] **Per-species centroids in the 768-dim activation space, per layer, per model.** Roadmap idea: "barycenter of each species." Useful on its own (does the bio fine-tune cluster species more tightly?) and as the input to the Step 3c hierarchical test below. Requires species labels in the manifest (see Step 2 manifest enrichment above). Compute is trivial once labels exist.
- [ ] **Within-species vs between-species variance.** Per (model, layer), compare the spread of frames around their species centroid vs the spread of species centroids around the global centroid. Yields a layer-resolved "species separability" curve we can put alongside the teammate's probe accuracy.

### Step 3c — Hierarchical representations (Veitch)

The roadmap's "do we see hierarchical representations? See Victor Veitch paper" item, owned by us. **Does not require probes** — Veitch-style geometric tests work on centroids + subspace angles, which we already compute.

- [ ] **Class-direction vs Order-direction orthogonality.** Compute the centroid for Aves vs Amphibia vs Mammalia at L5 (probe-peak layer for Class). Compute the centroid for Passeriformes / Charadriiformes / Piciformes / Strigiformes at L9 (probe-peak for Order, all within Aves). Test whether the L9 within-Aves Order-directions are orthogonal to the L5 Aves-vs-other-class direction. Veitch predicts they should be (orthogonal Cartesian product of independent concepts). Pass/fail is publishable either way.
- [ ] **Nested-subspace test.** Stronger Veitch claim: the Order centroids should live in an affine subspace whose origin is roughly the Aves centroid. Compute Order-centroid - Aves-centroid vectors and check whether they span a low-dim subspace within the Aves cluster.
- [ ] **Layer-resolved hierarchy.** Repeat the orthogonality test across all layers L0…L12. Predicts which layer "factors" the hierarchy (likely between L5 and L9 based on the probe peaks).

## Roadmap Section 2 — owned by teammate

Probes + attribution work is being pursued by a teammate. Out of scope for this thread; do not duplicate. **Coordinate on:** the manifest enrichment (Class / Order / Species labels) needed by Step 2 taxonomic resolution and Step 3b/c.

## Out of scope for the current pilot

- **Roadmap Section 1 Step 3 noise dynamics** (white/pink noise + "noise subspace") — the teammate's, do not duplicate.
- Roadmap Section 3 (dictionary learning / SAEs) — explicitly low priority in the roadmap.
- Original AVES + BirdAVES (`open_questions.md` §2).
- Cross-species call type transfer, RSA with CRCNS zebra-finch, unsupervised syllable segmentation, and the call-type discovery work from earlier exploratory phases. Revisit only after the roadmap pilot is complete.
