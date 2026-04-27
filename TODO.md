# Next Steps

## Roadmap Section 1 — ESP-AVES2 Activations (active)

Scope: ESP-AVES2 `eat`-family only (see `open_questions.md` §2). Roadmap Section 3 (noise dynamics, audio mixing, barycenters) is **out of scope** for the pilot.

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

- [ ] **Within-model L2-norm distributions** per `(model, layer)`. We have `norm_by_layer_source.csv` (cross-model deltas) but no within-model histograms, so the deltas are still uninterpretable on their own.
- [ ] **PCA / subspace alignment** across layers (within a model) and across models (within a layer). CKA covers cross-model similarity but not subspace angles or top-k overlap. Doubles as the natural test for the bio-vs-non-bio subspace question (see Step 2 follow-ups below).
- [ ] **Pooled vs frame-level** sanity check on one model. Everything so far is mean-pooled; the TwoNN-vs-effective-rank gap suggests pooling understates manifold curvature. Worth a quick comparison on a few hundred items before deciding whether to commit to frame-level for any later step.

### Step 2 follow-ups motivated by current findings

From the `dd24541` Step 2 results:

- [ ] **`sl_eat_bio_ssl_all` uses a 2–3× wider linear subspace** (eff. rank ~148 at L9 vs ~75 next-best) **and shows the largest nature-vs-non-nature gap.** Test directly with subspace angles between the bio-only and non-bio-only embedding subspaces, per layer. This is the same script as the missing PCA-alignment item above — bundle them.
- [ ] **TwoNN ID stays ~8–12 everywhere** while linear effective rank swings 3–148. Curved low-dim manifold inside a wide linear subspace; motivates the pooled-vs-frame comparison above on `sl_eat_bio_ssl_all` specifically.
- [ ] **L0 effective rank ≈ 3 across all four models.** Likely shared input tokenizer; confirm with a subspace-angle check at L0 specifically (cheap once the subspace-angle script exists).

## Roadmap Section 2 — Probes + attribution (next)

Begin once Step 2 outstanding items are closed.

- [ ] **Step 2.1** — identify two (or more) species in `NatureLM-audio-training` with substantial sample counts. Probably draw from the Xeno-canto slice of the existing manifest first; expand the manifest if counts are too thin.
- [ ] **Step 2.2** — train per-layer linear probes for one-vs-one species separation across all four models. Reuse the PCA-to-50 + logistic-regression pattern from the existing `probe_species.py`.
- [ ] **Step 2.4** — add hierarchical probes: Class (Aves vs Mammalia), Order (Passeriformes / Charadriiformes / Piciformes / Strigiformes), Species. Test for hierarchical geometry — gaussian blob fit on activation centroids is the roadmap's suggested cheap version.
- [ ] **Step 2.3** — attribution methods to recover which input patches the probes rely on. Roadmap says "more details coming soon" — defer until Step 2.2/2.4 are landed and we have a concrete probe to attribute through.

## Out of scope for the current pilot

- Roadmap Section 1 Step 3 (noise dynamics, audio mixing, species barycenters).
- Roadmap Section 3 (dictionary learning / SAEs) — explicitly low priority in the roadmap.
- Original AVES + BirdAVES (`open_questions.md` §2).
- Cross-species call type transfer, RSA with CRCNS zebra-finch, unsupervised syllable segmentation, and the call-type discovery work from earlier exploratory phases. Revisit only after the roadmap pilot is complete.
