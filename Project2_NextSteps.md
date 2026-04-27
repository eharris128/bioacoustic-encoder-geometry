# Project 2 — Next Steps: Interpretability for Interspecies Communication

---

## Probe Training Phase

### Context

The mentor has already established class-level (Aves / Amphibia / Mammalia) and
order-level (Passeriformes / Charadriiformes / Piciformes / Strigiformes) probes.
This phase covers the separations not yet handled:

| Experiment | File | What it tests |
|---|---|---|
| Animals vs Music | `experiments/animals_vs_music.py` | Does AVES linearly separate biological vocalizations from structured non-biological audio (violin)? |
| Music vs Speech | `experiments/music_vs_speech.py` | Does AVES distinguish musical instruments from human speech despite being trained on animals? |
| Species pairs | `experiments/species.py` | At which layer does fine-grained species identity become linearly decodable? |
| Species vs Species (custom) | `experiments/species_vs_species.py` | Generic binary probe for any two species — populated once sample sizes are verified and pairs confirmed with mentor. |

---

### Scaffolded files and their roles

**`data/loader.py`**
Central data pipeline. All experiment configs pass their `RECORDINGS` dict here.
Handles model loading, audio loading (WAV + MP3), forward passes, frame
subsampling (seed 42, max 3000 frames per recording), and assembles the
`{ layer: (X, y) }` dataset consumed by the probe trainer.

**`probes/train.py`**
Trains one logistic regression probe per layer using leave-one-recording-out
(LORO) cross-validation. Applies StandardScaler → PCA(50 dims) → LogisticRegression
per fold. Returns `{ layer: mean_loro_accuracy }`.

*How `loro_cross_validate` works:*

`build_dataset` stacks all recordings into one contiguous array in order.
`loro_cross_validate` tracks where each recording starts via cumulative frame
offsets, then iterates over folds — each fold holds out one recording's frames,
trains the full pipeline (scaler → PCA → LR) on the rest, and applies the fitted
transforms to the held-out frames (`.transform`, never `.fit_transform` on test
data). Fold accuracies are averaged per layer to produce the final profile.

LORO is used instead of a random split because each recording carries recording-level
structure (noise floor, individual timbre, microphone) that a random split would leak
into training, artificially inflating accuracy. LORO ensures the probe only ever
evaluates on recordings it has never seen.

*How to analyze effectiveness:*

- **Layer accuracy profile** — plot `accuracy_per_layer` (layer on x, accuracy on y).
  A non-monotone profile (peak early, dip mid-network, partial recovery late) is the
  interesting signal — it means layers serve different functional roles. A flat profile
  means the probe can't detect differences across depth. A monotone rise means the
  feature is built up progressively.
- **Compare to chance** — chance = `1 / n_classes` (50% for binary probes). Layers
  barely above chance are not linearly encoding the probed feature.
- **Fold variance** — high variance across folds indicates outlier recordings (class
  imbalance, corrupted audio, or the probe overfitting to one recording's artifacts).
  Low variance = the probe generalizes stably across individuals.
- **PCA variance explained** — inspect `pca.explained_variance_ratio_.sum()` on a
  representative fold. If < 60%, consider raising `pca_components`; if > 95%, you can
  reduce components to speed up training without accuracy loss.
- **Ablate PCA** — run once with raw 768-dim input to confirm PCA compression isn't
  discarding discriminative directions. A significant accuracy drop means the relevant
  structure is spread across more than 50 dimensions.

*How `train_all_layers` works:*

`train_all_layers` is the single entry point that experiment scripts call. It does
three things:

1. **Infer class count and chance level** — reads the label array from any layer
   (`np.unique(y)`), counts distinct classes, and computes `chance = 1 / n_classes`.
   This is done once here so callers don't have to pass it in, and so it's always
   consistent with the actual data rather than a hardcoded assumption.

2. **Delegate to `loro_cross_validate`** — passes through all arguments unchanged.
   The separation exists so `loro_cross_validate` can be called independently (e.g.
   for a single experiment sweep without the metadata wrapper).

3. **Print a per-layer accuracy table** — immediately after LORO finishes, prints
   each layer's accuracy and its delta over chance so you can inspect results without
   waiting for plots. Format: `embedding | layer N | accuracy% | +delta%`.

Returns a dict with four keys:
- `accuracy_per_layer` — `{ layer_int: float }`, the main result consumed by `evaluate.py`
- `chance_level` — `1 / n_classes`, used by `evaluate.py` to draw the reference line on plots
- `n_classes` — sanity-check that the right number of classes were loaded
- `n_recordings` — sanity-check on dataset size

**`probes/evaluate.py`**
Produces two outputs per experiment:
1. Accuracy curve (`*_accuracy.png`) — per-layer LORO accuracy with chance-level reference
2. LDA projection (`*_lda.png`) — 2D discriminant projection at layers 0, 3, 6, 9, 12

All PNGs are written to `results/`. The single entry point is `run_evaluation`, which
experiment scripts call after `train_all_layers`.

*How `plot_accuracy_curve` works:*

Takes `accuracy_per_layer` (the dict from `train_all_layers`) and plots a line chart
with one point per layer. Layer 0 is labeled `emb` (CNN embedding); layers 1–12 are
labeled `T0`–`T11` (transformer layers). A dashed gray line marks chance level. A gold
dot marks the peak layer so it's immediately visible. Saved at 150 dpi.

*How `plot_lda_projection` works:*

For each layer in `layers_to_plot`, fits a `LinearDiscriminantAnalysis` on the full
dataset for that layer (standardized first, same convention as the probes) and projects
to 2D. Each class is scatter-plotted in a distinct color with `alpha=0.3, s=3` (same
style as existing project LDA plots). One subplot per layer, arranged in a single row.
For binary experiments only LD1 exists — the y-axis is zeroed so points still render
as a 2D scatter rather than a 1D strip.

*How `run_evaluation` works:*

Calls both plot functions, clips the requested `lda_layers` to layers that are actually
present in the dataset, then prints a formatted summary table to stdout showing accuracy
and delta-over-chance per layer. Experiment scripts only need to call this one function
after training — they don't interact with the individual plot functions directly.

**`experiments/animals_vs_music.py`**
Config + entry point for the binary animal-vs-music probe. Label 0 = animal, label 1 = music.

*Current config (NatureLM version — requires Lambda / stable HuggingFace connection):*

- **Animal (0):** up to 200 recordings streamed from NatureLM xeno-canto, mean-pooled (one vector per recording)
- **Music (1):** 19 local recordings (5 violin + 5 piano + 5 flute + 4 guitar), mean-pooled

Class imbalance (200 animal vs 19 music) is expected — LORO handles it but probe will be biased toward the larger class. Consider capping animals at 19 for a balanced comparison, or adding more music sources.

*Offline/local version* (20 animal vs 19 music, no network required) is available in git history (commit before the NatureLM revert).

**`experiments/music_vs_speech.py`**
Config + entry point for the binary music-vs-speech probe. Label 0 = violin,
label 1 = LibriVox speech. **Only 2 speech files available locally — collect
≥5 before running for stable LORO estimates.**

**`experiments/species.py`**
Config + entry point for all species-pair probes. Three pairs pre-configured:
`bullfinch_vs_hawfinch` (same order, fine-grained), `bullfinch_vs_guineafowl`
(cross-order), `hawfinch_vs_guineafowl` (cross-order). Add new pairs directly
in the `PAIRS` dict.

**`experiments/species_vs_species.py`**
Generic binary probe stub for a single custom species pair. `SPECIES_A`,
`SPECIES_B`, and `RECORDINGS` are left empty — populate once sample sizes have
been verified and the target pair confirmed with the mentor.

**`results/`**
Output directory. All PNGs and any future summary CSVs land here. Tracked by git
via `.gitkeep`.

---

### Recommended run order

1. **Implement `data/loader.py`** — all experiments depend on it.
2. **Implement `probes/train.py`** — depends on loader output format.
3. **Implement `probes/evaluate.py`** — depends on train output format.
4. **Run `experiments/animals_vs_music.py`** — most data available, best-balanced.
5. **Collect ≥5 speech files → run `experiments/music_vs_speech.py`**.
6. **Run `experiments/species.py`** — runs all three pairs sequentially.

---

### Expected outputs per experiment

| Experiment | PNG outputs | What to look for |
|---|---|---|
| Animals vs Music | `animals_vs_music_accuracy.png`, `animals_vs_music_lda.png` | Near-100% even at layer 0 would suggest AVES immediately segregates bio vs non-bio audio; a layer-dependent rise would be more interesting |
| Music vs Speech | `music_vs_speech_accuracy.png`, `music_vs_speech_lda.png` | Low accuracy throughout = AVES doesn't distinguish non-animal sound types; high accuracy = some generalization beyond animal domain |
| Species (each pair) | `species_{pair}_accuracy.png`, `species_{pair}_lda.png` | Fine-grained pair (bullfinch vs hawfinch) should peak later than cross-order pairs; compare peak layer to mentor's order-level results |

---

### Definition of "done" before moving to attribution

- [x] All three experiment scripts run end-to-end without error (animals_vs_music ✓)
- [x] Accuracy curves saved to `results/` for all three experiments (+ all three species pairs) (animals_vs_music ✓ → `results/probe-output/animals_vs_music/`)
- [x] LDA projection plots saved for all experiments (animals_vs_music ✓)
- [x] Peak accuracy layer identified for each experiment and noted in the summary table below (animals_vs_music: T5 @ 98.9%)
- [ ] Results compared to mentor's class/order baselines — does the layer hierarchy hold?
- [ ] `music_vs_speech` has ≥5 speech recordings (not just 2)
- [ ] `species_vs_species.py` populated with confirmed pair and run end-to-end

Once all boxes are checked, move to the attribution phase (causal tracing /
activation patching to identify which heads and layers drive each separation).

---

### Peak layer summary (fill in after running)

| Experiment | Peak layer | Peak accuracy | Chance level |
|---|---|---|---|
| Animals vs Music | T5 | 98.9% | 50% |
| Music vs Speech | TBD | TBD | 50% |
| Bullfinch vs Hawfinch | TBD | TBD | 50% |
| Bullfinch vs Guineafowl | TBD | TBD | 50% |
| Hawfinch vs Guineafowl | TBD | TBD | 50% |
