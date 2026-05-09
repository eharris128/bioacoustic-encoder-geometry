# Project 2 — Next Steps: Interpretability for Interspecies Communication

Last updated: 2026-05-08

---

## Current Status

The xeno-canto probing phase is complete (11 species pairs, 100 recordings each). The NatureLM scaling pipeline is built and ready to run on GCP. Teammate Evan Harris has completed a parallel geometry analysis on `origin/main` whose findings directly complement and explain the probe results.

---

## Immediate — NatureLM Scaling Run

Two scripts are written and ready. Run on a GCP T4 GPU instance (~$2–3 total).

**GCP setup:** Compute Engine → Deep Learning on Linux boot disk (PyTorch) → NVIDIA T4 on n1-standard-4 → SSH in → `pip install avex datasets esp-aves soundfile scipy`. Request T4 quota increase if needed.

**Step 1 — extract activations** (~3–5 hrs on GPU):
```bash
python -W ignore scripts/batch_extract_naturelm.py --rows 1000 --device cuda
```
Extracts 18 species × 1000 recordings each into `activations/naturelm/<slug>/`. Resume-safe — skips completed species.

**Step 2 — run all 10 probe pairs** (~20 min, CPU fine):
```bash
python -W ignore experiments/naturelm_probe_all_pairs.py
```
Outputs accuracy + LDA PNGs per pair to `results/probe-output/naturelm_species_vs_species/`.

Note: Great Tit vs Bokharensis is dropped — NatureLM doesn't tag Parus major subspecies.

---

## Completed Experiments

### Animals vs Music
**Script:** `experiments/animals_vs_music.py`
**Data:** Local — bullfinch + hawfinch + helmeted guinea fowl vs violin + misc music.
**Result:** Peak T5 @ 98.9%. AVES cleanly linearly separates animal vocalizations from music across all transformer layers.

### Species vs Species (xeno-canto, 100 recordings/species, LORO)

Full probe results in `results/probe-output/probes_results_README.md`. Phylogenetic gradient visualization at `results/phylogenetic_gradient.png/.pdf`.

| Pair | Taxonomy | Peak | Peak layer | Emb |
|---|---|---|---|---|
| House Sparrow vs Tree Sparrow | Same genus | 85.4% | T6 | 53.3% |
| Willow Warbler vs Chiffchaff | Same genus | 93.0% | T6 | 53.0% |
| Common vs Iberian Chiffchaff | Same genus | 91.5% | T11 | 62.5% |
| House Crow vs Carrion Crow | Same genus | 95.0% | T9 | 72.0% |
| Great Tit vs Great Tit Bokharensis | Subspecies* | 92.2% | T3 | 81.8% |
| Goldfinch vs Eurasian Siskin | Same family | 92.5% | T5 | 65.5% |
| Bullfinch vs Hawfinch | Same family | 95.0% | T2 | 61.0% |
| European Robin vs Eurasian Blackbird | Diff. families | 99.0% | T8/T11 | 67.0% |
| Chaffinch vs Great Spotted Woodpecker | Diff. orders | 97.0% | T9 | 59.1% |
| House Sparrow vs Common Swift | Diff. orders | 98.0% | T5/T6/T7 | 69.5% |
| Bullfinch vs Tawny Owl | Diff. orders | 99.0% | T3/T9 | 65.5% |

*Bokharensis result likely reflects sample bias (only 54 recordings) — treat with caution.

**Core finding:** Probe accuracy and peak layer depth both scale with phylogenetic distance. Same-genus embedding accuracy ~53% (chance); cross-order pairs established by T0–T1.

---

## Evan's Findings (origin/main)

Evan ran a systematic geometry analysis of all four ESP-AVES2 EAT checkpoints + a random-init baseline on 600 NatureLM samples (100 × 7 sources). All claims confirmed with B=50 bootstrap CIs and robustness sweeps. Source: `RESULTS.md` on `origin/main`, last updated 2026-04-28.

### Key findings for the paper

**★ `sl_eat_bio_ssl_all` learns a factored hierarchical geometry — strongest paper claim (§4.7, §4.8, §4.9)**

The only model that simultaneously develops all four properties:
- Bio-vs-non-bio directional separation (cos = 0.57 at L9 vs random-init 0.91, confirmed B=50 bootstrap)
- Aves-vs-Mammalia Class direction at L7 (cos = 0.38) — the strongest single learned direction in the family
- Orthogonal Class and Order encoding at L12 (cos = 0.074; no other trained model goes below 0.30)
- Within-Aves species structure (separability ratio 0.20 at L10)

No single training ingredient produces all four simultaneously. This is the most novel framing and the cleanest model-comparison story.

**★ Trained models compress species detail to learn coarser abstractions (§4.9) — directly explains Sid's probe results**

Random-init has *higher* per-species separability (ratio 0.33) than any trained model (peak 0.20). Training acquires Class/Order invariances by putting acoustically-distinct same-class species *closer* together. This is the geometric explanation for why LORO probes plateau and why same-genus pairs are so much harder than cross-order pairs.

**Random-init baseline is critical for the paper (§2)**

Architecture alone gives frame-level eff_rank 10–12 and bio/non-bio frames indistinguishable (cos ≥ 0.91). Every number below this floor is attributable to learning. Must include random-init as anchor in any paper draft.

**The bio↔non-bio direction is threshold-like, not linear (§4.5)**

Audio-mixing pilot: adding 25% non-bio to a bio clip pulls L9 representation 78% of the way to pure non-bio (midpoint deviation −0.30). Sharp asymmetric response, specific to `sl_eat_bio_ssl_all`.

**`sl_eat_all_ssl_all` L12 mode collapse installs the bio classifier in a single block (§5.1, §5.2, §5.4)**

At L12: 61% of variance in one direction (vs 26% at L11). That direction IS the bio centroid axis (|cos| = 0.74). The L11→L12 transition does both mode collapse and bio classification simultaneously.

**Manifold dim is set by architecture, not learning (§6)**

Trained MLE-ID(k=20, n=10k): 7–14. Random-init: 11–15. Training does not expand manifold dim. The real learned property is the eff_rank/MLE-ID ratio: random-init ≈ 1, trained 17–43×. **Always report as "MLE-ID(k=20, n=10k)" in the paper — absolute values are (n,k)-conditional.**

**Mean-pooling distorts linear geometry (§3)**

Frame-level eff_rank > pooled eff_rank at every (model, layer). The xeno-canto probing pipeline uses mean-pooled activations — probe accuracy numbers likely understate the true separability available in raw patch tokens.

### Retracted — do not use in paper
- L4 TwoNN intrinsic-dim dip — TwoNN(k=2) estimator failure; MLE-ID(k=20) shows no dip.
- "Convergent L0 eff_rank ≈ 3 across models" — mean-pooling artifact.

### How Evan's geometry connects to Sid's probe results

| Evan's finding | Sid's probe observation |
|---|---|
| §4.9: trained models suppress species-level separability | LORO accuracy plateaus; same-genus pairs hardest |
| §4.7: Aves/Mammalia is the strongest direction at L7 | Same-genus pairs only differ in suppressed signal → peak at T6 |
| §4.8: Class/Order orthogonal at L12 | Cross-order pairs separate trivially by T0–T1 |
| §3: mean-pooling understates separability | Raw-token probing (untested) would likely push accuracy higher |

### Recommended paper framings

1. **★ Factored hierarchy** — `sl_eat_bio_ssl_all` uniquely develops (a)–(d) above. Most novel, most defensible.
2. **Directional bio signature** — §4 + §4.5 + §5.1 as mechanism. Tighter scope; good fallback.
3. **Mean-pooling distorts audio-encoder geometry** — methods paper; needs non-EAT control.
4. **Manifold expansion without dim growth** — §6; exposed to MLE-ID estimator objections (documented in §6 caveat).

---

## Next Priorities (after scaling run)

### High
- **RSA with zebra finch neural recordings (CRCNS aa-4)** — 914 neurons from Field L, CLM/CMM, NCM. Present same stimuli to AVES and neural data, compute pairwise distance matrices per layer, correlate via RSA. Most publishable direction — directly maps AVES layers to biological auditory hierarchy.
- **Sparse Autoencoders (SAEs) on layer 11** — decompose 768-dim space into sparse interpretable directions. Use TopK activation (K≈20–40), not L1. Report L0, reconstruction variance explained, dead feature fraction, and max-activating examples for top-20 features.

### Medium
- **Re-run probes with raw patch tokens** (not mean-pooled) to test Evan's §3 prediction that current accuracy numbers are understated. Use `mode="raw"` in `load_species_pair` after NatureLM extraction is done.
- **Attention head ablation** — zero out heads one at a time, measure species separability collapse. Tests whether local/global head specialization is real.
- **Cross-species cluster transfer** — train k-means on Bullfinch L11 embeddings, apply to Hawfinch. Tests universality of discovered categories.

---

## Pipeline Reference

### Xeno-canto probing pipeline
```
experiments/species_vs_species.py
  → data/loader.build_xenocanto_dataset()   # fetch + extract activations live
  → probes/train.train_all_layers()          # LORO cross-validation, PCA(50) → LR
  → probes/evaluate.run_evaluation()         # accuracy PNG + LDA PNG
```

### NatureLM offline pipeline
```
scripts/batch_extract_naturelm.py           # extract once to activations/naturelm/
  → extract_species_activations.extract_species()

experiments/naturelm_probe_all_pairs.py     # probe all 10 pairs from saved activations
  → scripts/load_species_activations.load_species_pair()
  → probes/train.train_all_layers()
  → probes/evaluate.run_evaluation()
```

### Layer indexing (throughout codebase)
- Index 0 = CNN `local_encoder` output (labeled `emb` in plots)
- Indices 1–12 = transformer blocks 0–11 (labeled `T0`–`T11` in plots)
- All activations shape `(n_patches, 768)` with CLS token stripped

### Supported models
| avex name | Description |
|---|---|
| `esp_aves2_eat_all` | EAT pretrained, all data (default) |
| `esp_aves2_eat_bio` | EAT pretrained, bio-only data |
| `esp_aves2_sl_eat_all_ssl_all` | supervised fine-tune, all data |
| `esp_aves2_sl_eat_bio_ssl_all` | supervised fine-tune, bio data |

Checkpoints auto-download from HuggingFace via `avex` on first use.
