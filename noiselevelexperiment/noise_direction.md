# [noise_direction1.py](noise_direction1.py) — Code Notes

## What This Experiment Does

**This is not an activation-patching experiment.**

The previous version of this script patched random unit vectors into transformer layer
outputs mid-forward-pass to act as a control for `contrastive_patch_species.py`. That
design was measuring the model's sensitivity to arbitrary perturbations in activation
space — not anything about how real recording noise is represented.

This version asks a different question: when you add acoustic background noise to the raw
audio waveform and run it through AVES, does it move activations along a consistent linear
direction at each layer? If so, that direction is the noise direction — and we can ask
whether it overlaps with the species direction from `contrastive_patch_species.py`.

The intervention now happens entirely **before** the model. No hooks. No patching.
The forward pass is always clean.

---

## Design

```
for each recording:
    for each SNR level (40dB → 0dB):
        noisy_audio = clean_audio + calibrated_white_noise
        layer_means = mean_frame_activation(AVES(noisy_audio))  # (12, 768)

for each layer:
    X = stack all (rec, snr) layer_means  # (n_rec × n_snr, 768)
    noise_direction = PCA(X).components_[0]
    variance_explained = PCA(X).explained_variance_ratio_[0]

optional:
    species_direction = normalize(mean_hawfinch - mean_bullfinch) per layer
    orthogonality = |cos_sim(noise_direction, species_direction)| per layer
```

---

## Config

`SNR_LEVELS_DB = [40, 30, 20, 15, 10, 7, 5, 3, 1, 0]` — 10 levels from nearly clean
to signal-power-equals-noise-power. The spacing is denser at the noisy end because
perceptual differences compress at high SNR.

`RECORDINGS` — single-species audio files. Needs at least 2–3 recordings so PCA is
fitting across multiple independent noise trajectories, not just one.

`SPECIES_RECORDINGS` — optional two-species recordings. If empty or any file is missing,
the orthogonality plot is silently skipped.

---

## `add_white_noise`

```python
signal_power = np.mean(signal ** 2)
noise_power = signal_power / (10.0 ** (snr_db / 10.0))
noise = rng.normal(0.0, np.sqrt(noise_power), signal.shape)
return audio + noise
```

Noise power is calibrated to the per-recording signal power, so SNR is consistent across
files recorded at different levels. A field recording at -30dBFS and one at -10dBFS will
both get the same effective SNR — not the same absolute noise amplitude.

---

## `extract_layer_means`

Runs one clean forward pass and returns `(NUM_LAYERS, 768)` — the mean frame activation
at each transformer layer. Frames are randomly subsampled to `MAX_FRAMES_PER_RECORDING`
before averaging to keep cost bounded on long recordings.

No hooks are registered. This is a standard `model.extract_features(audio, layers=None)` call.

---

## `run_snr_sweep`

Loops over recordings and SNR levels. For each pair, adds noise then calls
`extract_layer_means`. Stores results as `(n_snr, NUM_LAYERS, 768)` per recording.

The RNG seed is fixed at 42 and advances sequentially, so every run produces the same
noise samples. Different SNR levels within the same recording get different noise draws
(the rng is not reset between levels).

---

## `compute_noise_directions`

For each layer, concatenates all `(n_rec × n_snr, 768)` mean activations into one matrix
and fits `PCA(n_components=1)`. The first principal component is the noise direction at
that layer.

**Variance explained by PC1** is the key diagnostic. If it's high, noise level moves
activations along a consistent axis — noise has a linear representation at that layer.
If it's low, noise scatters activations in multiple directions with no dominant axis,
suggesting that layer is robust to noise or encodes it nonlinearly.

---

## `compute_species_directions`

Computes `normalize(mean_species1 - mean_species0)` per layer from `SPECIES_RECORDINGS`.
Returns `None` if the dict is empty or any file path does not exist, skipping the
orthogonality analysis gracefully.

---

## `plot_results`

Three figures, produced only if their data is available:

**`noise_snr_curves.png`** — mean L2 distance from the 40dB baseline vs SNR, one curve
per layer. Steep curves = that layer reacts strongly to noise. Flat curves = that layer
is insensitive to it. Expect CNN-adjacent layers to show more sensitivity than deep
transformer layers (consistent with the known pattern that AVES transformer layers are
invariant to many low-level acoustic features).

**`noise_direction_variance.png`** — bar chart of PC1 variance explained per layer.
High variance at a layer means noise moves that layer's activations along a single
consistent direction. This is the primary output of the experiment.

**`noise_species_ortho.png`** *(optional)* — `|cosine similarity|` between the noise
direction and species direction at each layer. Values near 0 mean noise and species
occupy orthogonal axes — a denoising intervention could be applied without disturbing
species identity. Values near 1 mean the two directions are aligned, making separability
harder.
