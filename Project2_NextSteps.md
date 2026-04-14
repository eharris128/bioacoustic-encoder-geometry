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

**`probes/evaluate.py`**
Produces two outputs per experiment:
1. Accuracy curve (`*_accuracy.png`) — per-layer LORO accuracy with chance-level reference
2. LDA projection (`*_lda.png`) — 2D discriminant projection at layers 0, 3, 6, 9, 11

All PNGs are written to `results/`.

**`experiments/animals_vs_music.py`**
Config + entry point for the binary animal-vs-music probe. Label 0 = bird recordings
(bullfinch, hawfinch, guineafowl), label 1 = violin. 9 animal recordings, 5 music
recordings currently configured.

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

- [ ] All three experiment scripts run end-to-end without error
- [ ] Accuracy curves saved to `results/` for all three experiments (+ all three species pairs)
- [ ] LDA projection plots saved for all experiments
- [ ] Peak accuracy layer identified for each experiment and noted in the summary table below
- [ ] Results compared to mentor's class/order baselines — does the layer hierarchy hold?
- [ ] `music_vs_speech` has ≥5 speech recordings (not just 2)
- [ ] `species_vs_species.py` populated with confirmed pair and run end-to-end

Once all boxes are checked, move to the attribution phase (causal tracing /
activation patching to identify which heads and layers drive each separation).

---

### Peak layer summary (fill in after running)

| Experiment | Peak layer | Peak accuracy | Chance level |
|---|---|---|---|
| Animals vs Music | TBD | TBD | 50% |
| Music vs Speech | TBD | TBD | 50% |
| Bullfinch vs Hawfinch | TBD | TBD | 50% |
| Bullfinch vs Guineafowl | TBD | TBD | 50% |
| Hawfinch vs Guineafowl | TBD | TBD | 50% |
