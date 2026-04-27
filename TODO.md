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

- [x] **Within-model L2-norm distributions** per `(model, layer)` — done in `10601b9` (`step2_subspace_angles_eat.py`); histograms in `step2_subspace_angles/l2_norm_histograms.png`, percentiles in `l2_norm_per_layer.csv`.
- [x] **PCA / subspace alignment** across layers (within a model) and across models (within a layer) — done in `10601b9`; across-layer heatmaps + across-model curves in `step2_subspace_angles/`.
- [x] **Pooled vs frame-level** sanity check on `sl_eat_bio_ssl_all` — done in `11f9920` (`step2_pooled_vs_frame_eat.py`). **Pooling materially distorts geometry**: frame-level eff. rank is 2–6× larger than pooled at every layer, while frame-level TwoNN drops to ~3–5 vs ~9–11 pooled. Both directions matter — pooling overstates intrinsic dim and understates linear spread.

### Step 2 follow-ups motivated by current findings

From the `dd24541` Step 2 results:

- [x] **`sl_eat_bio_ssl_all` uses a 2–3× wider linear subspace** — confirmed in `10601b9`. Bio-vs-non-bio subspace cos drops to **0.33 at L9** (same layer as eff. rank ~148), vs 0.55–0.70 for the other three models. The bio fine-tune is genuinely separating bio from non-bio in subspace direction, not just expanding both.
- [x] **TwoNN ID stays ~8–12 everywhere** while linear effective rank swings 3–148 — addressed via the pooled-vs-frame comparison in `11f9920`. Frame-level TwoNN is ~3–5 (much lower), confirming the curved low-dim manifold story; the pooled-level TwoNN was a measurement artifact of mean-pooling.
- [x] **L0 effective rank ≈ 3 across all four models** — **rejected as a "shared tokenizer" story** in `10601b9`. Three models share L0 subspace at cos 0.91–0.98, but `eat_bio` is ~orthogonal to all of them (cos 0.28–0.32). Each model independently learned a low-dim L0 subspace; they happen to land at similar dimensionality but not the same direction.

### Step 2 outstanding (new, from this round of findings)

- [ ] **TwoNN sanity check at frame level**. The frame-level TwoNN curve in `step2_pooled_vs_frame/` shows an unstable point at L4 (drops to ~2.6 sandwiched between 10.1 and 7.4). With k=2 NN over 10k subsampled rows the estimator can be jumpy. Worth re-running with a larger neighbor count or MLE-ID as a cross-check before any claim about layerwise TwoNN trends.
- [ ] **Generalize the pooled-vs-frame check to the other three models**. The story above is currently sl_eat_bio_ssl_all only. If pooling distorts geometry on the other three the same way, the original spectral-dim conclusions need to be re-stated against frame-level numbers. If only sl_eat_bio is distorted, that itself is a finding.

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
