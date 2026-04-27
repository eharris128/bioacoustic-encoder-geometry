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

### Step 2 outstanding (Tier 1 follow-ups)

Done in `step2_tier1_frame_level.py` (output: `step2_tier1_frame_level/`):

- [x] **TwoNN L4 dip resolved as a TwoNN(k=2) estimator artifact.** Adding MLE-ID with k=20 alongside TwoNN: 3 of 4 models show the L4 TwoNN crash (`eat_all` 0.95, `sl_eat_all_ssl_all` 1.67, `sl_eat_bio_ssl_all` 2.63) while MLE-ID(k=20) reads stable values (13.27, 13.69, 14.05). The "L4 dip" is a TwoNN failure mode, not a model phenomenon — retract from any narrative built on `step2_pooled_vs_frame/`.
- [x] **Pooled-vs-frame distortion is universal but heterogeneous.** Frame-level eff_rank > pooled eff_rank at every (model, layer); the ratio varies wildly. Pooled L0 was uniformly ~3 across all four models — frame-level L0 spreads to **11 / 26 / 44 / 17** for `eat_all` / `eat_bio` / `sl_eat_all_ssl_all` / `sl_eat_bio_ssl_all`. The "L0 effective rank ≈ 3 across all four models" finding from `dd24541` was a pooling artifact; at frame level the four L0s are visibly different.
- [x] **Bio-vs-nonbio direction story survives at frame level but is muted.** The pooled "0.33 at L9 for `sl_eat_bio_ssl_all` vs 0.55–0.70 elsewhere" becomes "**0.57 at L9 vs 0.68–0.81 elsewhere**" at frame level. `sl_eat_bio_ssl_all` is still the most-separated model and still bottoms out around L7–L9, but pooling inflated the dramatic-looking number. Restate as: "bio fine-tune separates bio from non-bio in subspace direction at frame level (mean cos drops to ~0.57 in mid-network), but the effect is smaller than mean-pooling suggested."

### Step 2 new findings from Tier 1

- [ ] **Late-layer collapse splits the family by `_bio` vs not.** Frame-level L12 eff_rank: `eat_all` 62.6, `sl_eat_all_ssl_all` **11.2**, vs `eat_bio` 180.4, `sl_eat_bio_ssl_all` 188.7. The two non-bio models collapse hard at the output; the two bio fine-tunes retain rich variance through to L12. SSL on top of `eat_all` (i.e. `sl_eat_all_ssl_all`) collapses harder than `eat_all` alone. Worth a follow-up: is this saying the bio pretrain leaves the output-side representation usable for downstream tasks while the non-bio + SSL combo over-compresses?
- [ ] **Frame-level model rank order at the eff_rank peak is `sl_eat_bio_ssl_all` > `eat_all` ≈ `sl_eat_all_ssl_all` > `eat_bio`.** Pooled said `sl_eat_bio_ssl_all` ≫ `eat_bio` ≈ `sl_eat_all` > `eat_all`. The relative position of `eat_bio` vs the other two flips between pooled and frame-level — pooled overstates `eat_bio`'s linear subspace width relative to `eat_all` and `sl_eat_all_ssl_all`. Worth restating any claims that ranked `eat_bio` high on linear width.
- [ ] **MLE-ID(k=20) frame-level intrinsic dim sits at 7–14 across all (model, layer)** vs eff_rank swings of 11–348. Confirms the curved-low-dim-manifold-inside-wide-linear-subspace story holds for every model in the family, not just `sl_eat_bio_ssl_all`. The intrinsic dim is fairly conserved across models; the linear envelope is what differentiates them.

## Roadmap Section 2 — owned by teammate

Probes + attribution work is being pursued by a teammate. Out of scope for this thread; do not duplicate.

## Out of scope for the current pilot

- Roadmap Section 1 Step 3 (noise dynamics, audio mixing, species barycenters).
- Roadmap Section 3 (dictionary learning / SAEs) — explicitly low priority in the roadmap.
- Original AVES + BirdAVES (`open_questions.md` §2).
- Cross-species call type transfer, RSA with CRCNS zebra-finch, unsupervised syllable segmentation, and the call-type discovery work from earlier exploratory phases. Revisit only after the roadmap pilot is complete.
