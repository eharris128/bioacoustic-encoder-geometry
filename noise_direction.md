# [noise_direction.py](noise_direction.py) — Code Notes

## Config

`NUM_NOISE_TRIALS = 20` — number of independent random directions averaged per layer per fold. Higher reduces variance from unlucky draws that happen to partially align with the species direction. 20 is a reasonable tradeoff; drop to 5 for a quick sanity run.

`ALPHA_VALUES` and `MAX_FRAMES_PER_RECORDING` are kept identical to `contrastive_patch_species.py` so outputs are directly comparable.

---

## `extract_all_layers`

Unchanged from `contrastive_patch_species.py`. Runs one forward pass per recording, stacks all 12 layer outputs into `(n_frames, n_layers, 768)`, and subsamples to `MAX_FRAMES_PER_RECORDING` with a fixed RNG seed.

---

## `train_layer11_probe_loro`

Identical to `contrastive_patch_species.py`. Trains a logistic regression probe on layer-11 embeddings using leave-one-recording-out CV, then retrains on all data. The returned `probe` and `scaler` are reused across all patching conditions — keeping the evaluation surface constant means any difference in flip rate is attributable to the patch direction, not probe variation.

---

## `random_unit_vectors`

```python
vecs = rng.standard_normal((n, dim)).astype(np.float32)
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
return vecs / np.where(norms > 1e-8, norms, 1.0)
```

Draws `n` vectors from a standard normal and normalizes each to unit norm. This produces uniformly distributed directions on the unit sphere in R^768. The `1e-8` guard prevents division by zero on the (astronomically unlikely) all-zero draw.

All directions are pre-drawn once in `main` before the sweep loop so the same random directions are used across all recordings and folds — making per-layer averages comparable.

---

## `noise_patch_run`

Same hook pattern as `contrastive_patch_species.py`:

```python
layer_module = model.model.encoder.transformer.layers[patch_layer]
hook = layer_module.register_forward_hook(hook_fn)
```

The hook adds `alpha * direction` to the layer output. If the output is a tuple (which torchaudio transformer layers return), only `output[0]` is modified and the rest of the tuple is passed through unchanged. The hook is always removed in a `finally` block so a failed forward pass doesn't leave a dangling hook.

No `sign` argument here (unlike `contrastive_patch_run`) — random directions have no meaningful sign, so we just add them as-is.

---

## `run_noise_alpha_sweep`

Outer loop: LORO folds (one test recording at a time).
Middle loop: patch layers 0–11.
Inner loop: alpha values, then noise trials.

Alpha = 0 is handled separately — no hook is registered, just a plain forward pass. This is the baseline: mean abs shift should be near 0 if the probe classifies the recording correctly.

For alpha > 0, `NUM_NOISE_TRIALS` forward passes are run per `(fold, layer, alpha)` combination, each with a different pre-drawn random direction. Mean absolute level shifts are averaged across trials before being stored, so `results[layer][alpha]` accumulates one value per fold (not one per trial).

The metric is **mean absolute shift**: `mean(abs(preds - true_label))`. Signed shift is not used here because random directions have no meaningful sign — averaging signed shifts would cancel to ~0 regardless of alpha, making the control useless.

Final return averages across folds:
```python
{k: {a: float(np.mean(v)) for a, v in alphas.items()} for k, alphas in results.items()}
```

---

## `compute_alpha1`

Linear interpolation between the two alpha values that bracket a mean abs shift of 1.0 (one full noise level). Returns `inf` if 1.0 is never reached within the sweep range. Analogous to `compute_alpha50` in `contrastive_patch_species.py` but uses an ordinal threshold appropriate for 5-level labels.

---

## `plot_results`

Produces three figures matching the layout of `contrastive_patch_species.py`:

1. **`noise_direction_alpha_sweep.png`** — mean abs level shift vs. alpha per layer (left) + alpha_1 bar chart (right). The bar chart title notes "compare to noise-level direction" to prompt the direct comparison.
2. **`noise_direction_summary.png`** — heatmap of mean abs shift across all `(layer, alpha)` combinations, with per-cell annotations. Colormap is `RdYlGn_r` with `vmax=4` (max possible shift for 5 levels).

The gold bar marks the layer with the lowest alpha_1 under noise — this may differ from the layer identified by noise-level direction patching.
