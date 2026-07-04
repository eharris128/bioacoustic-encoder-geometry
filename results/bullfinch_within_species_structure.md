# Bullfinch within-species cluster structure across EAT layers

Model: `esp_aves2_eat_all`. Corpus: **37 Bullfinch clips** (XC1086809.mp3 skipped; ~8 recordings are format duplicates so effective unique count is ~26–30). Every clip yields exactly 512 avex patches, so per-layer matrices are (18 944, 768).

Pipeline per layer: **raw frame extraction (no pool) → PCA→50 → k-means k=2..10 (silhouette on a seeded 5 000-frame subsample) → `sklearn.cluster.HDBSCAN(min_cluster_size=50, min_samples=10)`**.

## Key pattern — a U across depth

Silhouette (best-k) traces a **U-shape** across the stack. In absolute terms no
layer produces cleanly-separated within-Bullfinch clusters (max silhouette
+0.29, standard "well-separated" territory starts around +0.5). But the
*relative* structure is highly non-uniform:

- **Endpoints (`emb`, `T11`)** sit at silhouette ≈ **+0.29** with **k=2 winning
  by a wide margin** (k=3 drops to +0.20 at `T11`). Both endpoints have a
  single dominant PC that captures **~43–45%** of variance — one axis carries
  most of the signal.
- **Middle transformer blocks (`T2`–`T4`)** flatten to silhouette ≈ **+0.09–0.11**
  with no dominant PC (top PC ≈ 9%). Cluster structure is essentially absent;
  this is the region where within-species detail is most suppressed.
- **Late transformer blocks (`T6`–`T10`)** show a **monotonic recovery** from
  silhouette +0.14 (`T6`) → +0.15 (`T9`) → +0.18 (`T10`), with the top PC
  growing 10% → 13%.

Interpretation: `emb` structure is **acoustic** — the CNN patch embedding
separates frames by raw spectral properties before any invariance is imposed.
`T11` structure looks **task-directed**: 43% of variance in one axis, k=2
strongly preferred, HDBSCAN finds a big blob plus small satellites (88% noise) —
consistent with the late-layer collapse toward a single decision direction
that RESULTS.md §5.1/§5.2 documents for `sl_eat_all_ssl_all`. The middle
valley is the invariance region.

Two HDBSCAN oddities worth flagging:
- **`T0` fragments** into 64 tiny clusters at only 4.5% noise (largest cluster
  1.6% of frames). Consistent with early transformer blocks preserving
  fine-grained per-patch local structure that k-means smooths out.
- **`T6`/`T7`** show a similar HDBSCAN pattern (20 and 18 clusters, 74–80%
  noise) — density islands appearing where the model may be starting to
  organize by call-type before collapsing into the T11 axis.

## Summary

| Layer | Label | Best k | Silhouette | PCA-50 cum var | HDBSCAN k | Noise % | Largest % | Verdict |
|-------|-------|--------|------------|----------------|-----------|---------|-----------|---------|
| 0 | `emb` | 2 | +0.2879 | 0.953 | 7 | 72.7 | 15.9 | well-separated cluster structure |
| 1 | `T0` | 10 | +0.1139 | 0.921 | 64 | 4.5 | 1.6 | weak cluster structure (borderline) |
| 2 | `T1` | 9 | +0.1007 | 0.823 | 2 | 30.6 | 68.2 | weak cluster structure (borderline) |
| 3 | `T2` | 2 | +0.0993 | 0.794 | 2 | 20.5 | 78.5 | no meaningful cluster structure |
| 4 | `T3` | 2 | +0.0952 | 0.786 | 2 | 23.6 | 76.0 | no meaningful cluster structure |
| 5 | `T4` | 2 | +0.1108 | 0.706 | 3 | 21.5 | 76.6 | weak cluster structure (borderline) |
| 6 | `T5` | 10 | +0.1125 | 0.727 | 2 | 0.7 | 98.3 | weak cluster structure (borderline) |
| 7 | `T6` | 8 | +0.1371 | 0.718 | 20 | 74.2 | 4.4 | weak cluster structure (borderline) |
| 8 | `T7` | 10 | +0.1425 | 0.712 | 18 | 80.1 | 5.9 | weak cluster structure (borderline) |
| 9 | `T8` | 10 | +0.1409 | 0.765 | 2 | 5.1 | 93.9 | weak cluster structure (borderline) |
| 10 | `T9` | 9 | +0.1515 | 0.785 | 2 | 22.8 | 76.3 | moderate cluster structure |
| 11 | `T10` | 10 | +0.1800 | 0.751 | 2 | 8.2 | 90.8 | moderate cluster structure |
| 12 | `T11` | 2 | +0.2863 | 0.823 | 7 | 87.8 | 9.5 | well-separated cluster structure |

## Per-layer notes

### Layer 0 — `emb` (best k=2, sil=+0.2879)

- Silhouette top-2: k=2→+0.2879, k=5→+0.2176.
- PCA-50 captures **95.3%** of variance; top PC alone: 45.4%.
- HDBSCAN: **7 cluster(s)**, 72.7% noise, largest cluster 15.9% of frames.
- well-separated cluster structure; HDBSCAN resolves multiple comparable groups.
- Plot: `results/bullfinch_within_layer00.png`

### Layer 1 — `T0` (best k=10, sil=+0.1139)

- Silhouette top-2: k=10→+0.1139, k=2→+0.1063.
- PCA-50 captures **92.1%** of variance; top PC alone: 12.2%.
- HDBSCAN: **64 cluster(s)**, 4.5% noise, largest cluster 1.6% of frames.
- weak cluster structure (borderline); HDBSCAN resolves multiple comparable groups.
- Plot: `results/bullfinch_within_layer01.png`

### Layer 2 — `T1` (best k=9, sil=+0.1007)

- Silhouette top-2: k=9→+0.1007, k=10→+0.0987.
- PCA-50 captures **82.3%** of variance; top PC alone: 9.2%.
- HDBSCAN: **2 cluster(s)**, 30.6% noise, largest cluster 68.2% of frames.
- weak cluster structure (borderline); HDBSCAN resolves multiple comparable groups.
- Plot: `results/bullfinch_within_layer02.png`

### Layer 3 — `T2` (best k=2, sil=+0.0993)

- Silhouette top-2: k=2→+0.0993, k=7→+0.0886.
- PCA-50 captures **79.4%** of variance; top PC alone: 9.5%.
- HDBSCAN: **2 cluster(s)**, 20.5% noise, largest cluster 78.5% of frames.
- no meaningful cluster structure; HDBSCAN resolves multiple comparable groups.
- Plot: `results/bullfinch_within_layer03.png`

### Layer 4 — `T3` (best k=2, sil=+0.0952)

- Silhouette top-2: k=2→+0.0952, k=9→+0.0940.
- PCA-50 captures **78.6%** of variance; top PC alone: 8.9%.
- HDBSCAN: **2 cluster(s)**, 23.6% noise, largest cluster 76.0% of frames.
- no meaningful cluster structure; HDBSCAN resolves multiple comparable groups.
- Plot: `results/bullfinch_within_layer04.png`

### Layer 5 — `T4` (best k=2, sil=+0.1108)

- Silhouette top-2: k=2→+0.1108, k=4→+0.1104.
- PCA-50 captures **70.6%** of variance; top PC alone: 10.2%.
- HDBSCAN: **3 cluster(s)**, 21.5% noise, largest cluster 76.6% of frames.
- weak cluster structure (borderline); HDBSCAN resolves multiple comparable groups.
- Plot: `results/bullfinch_within_layer05.png`

### Layer 6 — `T5` (best k=10, sil=+0.1125)

- Silhouette top-2: k=10→+0.1125, k=9→+0.1117.
- PCA-50 captures **72.7%** of variance; top PC alone: 9.2%.
- HDBSCAN: **2 cluster(s)**, 0.7% noise, largest cluster 98.3% of frames.
- weak cluster structure (borderline); HDBSCAN collapses to one large blob.
- Plot: `results/bullfinch_within_layer06.png`

### Layer 7 — `T6` (best k=8, sil=+0.1371)

- Silhouette top-2: k=8→+0.1371, k=9→+0.1364.
- PCA-50 captures **71.8%** of variance; top PC alone: 9.6%.
- HDBSCAN: **20 cluster(s)**, 74.2% noise, largest cluster 4.4% of frames.
- weak cluster structure (borderline); HDBSCAN resolves multiple comparable groups.
- Plot: `results/bullfinch_within_layer07.png`

### Layer 8 — `T7` (best k=10, sil=+0.1425)

- Silhouette top-2: k=10→+0.1425, k=9→+0.1359.
- PCA-50 captures **71.2%** of variance; top PC alone: 9.7%.
- HDBSCAN: **18 cluster(s)**, 80.1% noise, largest cluster 5.9% of frames.
- weak cluster structure (borderline); HDBSCAN resolves multiple comparable groups.
- Plot: `results/bullfinch_within_layer08.png`

### Layer 9 — `T8` (best k=10, sil=+0.1409)

- Silhouette top-2: k=10→+0.1409, k=9→+0.1388.
- PCA-50 captures **76.5%** of variance; top PC alone: 10.9%.
- HDBSCAN: **2 cluster(s)**, 5.1% noise, largest cluster 93.9% of frames.
- weak cluster structure (borderline); HDBSCAN finds a dominant blob plus small satellites.
- Plot: `results/bullfinch_within_layer09.png`

### Layer 10 — `T9` (best k=9, sil=+0.1515)

- Silhouette top-2: k=9→+0.1515, k=10→+0.1485.
- PCA-50 captures **78.5%** of variance; top PC alone: 11.9%.
- HDBSCAN: **2 cluster(s)**, 22.8% noise, largest cluster 76.3% of frames.
- moderate cluster structure; HDBSCAN resolves multiple comparable groups.
- Plot: `results/bullfinch_within_layer10.png`

### Layer 11 — `T10` (best k=10, sil=+0.1800)

- Silhouette top-2: k=10→+0.1800, k=9→+0.1768.
- PCA-50 captures **75.1%** of variance; top PC alone: 13.1%.
- HDBSCAN: **2 cluster(s)**, 8.2% noise, largest cluster 90.8% of frames.
- moderate cluster structure; HDBSCAN finds a dominant blob plus small satellites.
- Plot: `results/bullfinch_within_layer11.png`

### Layer 12 — `T11` (best k=2, sil=+0.2863)

- Silhouette top-2: k=2→+0.2863, k=3→+0.2023.
- PCA-50 captures **82.3%** of variance; top PC alone: 42.9%.
- HDBSCAN: **7 cluster(s)**, 87.8% noise, largest cluster 9.5% of frames.
- well-separated cluster structure; HDBSCAN resolves multiple comparable groups.
- Plot: `results/bullfinch_within_layer12.png`

## Overall reading

The templated "well-separated" verdicts in the summary above are relative to
this dataset only — no layer meets the standard silhouette threshold for
well-separated clusters. What the data actually shows:

- **Bookend structure** at `emb` and `T11` — same silhouette (+0.29), same
  best-k (2), similar top-PC dominance (43–45%). But these almost certainly
  reflect *different* axes: acoustic energy / call vs silence at `emb`; a
  task-relevant collapse direction at `T11`.
- **Middle valley** at `T2`–`T4` — training does what §4.9 predicts. Whatever
  within-species structure `emb` sees is suppressed.
- **Late-layer re-emergence** (`T6` → `T10`) suggests the model progressively
  re-organizes frames along a coarser axis before the final `T11` collapse.
  Worth checking whether the k=10 solutions at `T7`/`T8`/`T9` correlate with
  call type or with recording identity — the current pipeline doesn't label
  frames beyond "which clip they came from."

Overview plot: `results/bullfinch_within_all_layers.png` (silhouette across
layers, HDBSCAN mass, k-sweep curves per layer).

## Appendix — code and method walkthrough

### Files

- **`bullfinch_within_layer_cluster.py`** — single-layer pipeline. Exposes
  `collect_clips`, `extract_all_clips`, `save_all_layers`, `pca50`,
  `sweep_kmeans`, `hdbscan_cluster`, `plot_results`. `main()` runs the
  full path for one `--layer` (default index 6 = `T5`) and caches the
  raw activations to `activations/bullfinch_layers_raw.npz`.
- **`bullfinch_within_all_layers.py`** — driver. Loads the cached
  activations and loops the same primitives across all 13 EAT layers.
  Writes the CSV, the overview plot, and the auto-generated draft of this
  document (before manual editing of the narrative sections).
- **`data/loader.py`** — provides `load_model` (EAT via `avex`, hooks on
  local_encoder + 12 transformer blocks) and
  `extract_all_layers(model, audio, mode="raw")` which returns a
  `(13, n_frames, 768)` array — one row of patches per layer.

### Extraction (no pool)

For each clip we do:

1. `load_audio_file(path)` → mono 16 kHz float32 tensor. Uses `soundfile`
   for WAV/FLAC and falls back to `librosa` for MP3.
2. `extract_all_layers(model, audio, mode="raw")` runs one forward pass
   through EAT and captures every hooked layer's output. The CLS token
   (position 0) is stripped from every transformer block so all 13 layers
   emit **exactly (512, 768)** — 512 patches × 768 hidden dim. avex fixes
   the mel-spectrogram window to ~10 s, so clip duration doesn't change
   the patch count; longer clips get truncated inside avex.
3. Concatenate across clips: per layer we stack (512, 768) matrices from
   37 recordings → `(18 944, 768)`. We keep a parallel `rec_idx` array of
   shape `(18 944,)` marking which recording each row came from — that's
   how the right-hand PCA scatter panel colors frames by recording.
4. Save all 13 layers to `activations/bullfinch_layers_raw.npz` as float16
   (~380 MB in memory, gzipped on disk) so re-running is a load, not a
   fresh extraction.

### PCA → 50 dimensions

`pca50(X)` casts to float32 and runs `sklearn.decomposition.PCA(n_components=50, random_state=42)`.

**What PCA does.** Center the data (subtract per-column mean), then find
the 50 orthogonal directions in 768-dim space that capture the most
variance. Project each row onto those 50 directions. The result is a
`(18 944, 50)` matrix.

**Why 50 dims.** k-means and silhouette are both O(n·d) or worse per
distance calculation, and Euclidean distance in 768 dims becomes almost
uninformative because random directions carry noise. Reducing to 50 dims
retains **71–95%** of variance (per layer, reported as `pca50_cum_var`)
while dropping the tail of directions that are mostly noise.

**Watch the top-PC ratio.** If PC1 alone captures ~10% of variance, no
single direction dominates and the geometry is "diffuse." If PC1 captures
~45% (as at `emb` and `T11`), one axis carries most of the signal — that
usually means a strong binary structure along one dimension, which is why
k=2 wins silhouette at those layers.

### k-means sweep + silhouette

For each k ∈ {2, ..., 10}:

- `KMeans(n_clusters=k, random_state=42, n_init=10)` runs Lloyd's
  algorithm 10 times from different random inits and keeps the run with
  the lowest inertia (within-cluster squared distance). Output: a
  cluster label per frame.
- `silhouette_score(Xp, labels)` on a seeded 5 000-frame subsample.
  Silhouette is O(n²) so we subsample; the 5 000-frame subset is
  reproducible via `np.random.default_rng(42)`.

**Silhouette in one line.** For each point i in cluster C:

    s(i) = (b(i) - a(i)) / max(a(i), b(i))

where `a(i)` = mean distance to other points in C and `b(i)` = mean
distance to points in the *nearest other* cluster. Range −1..+1; **+1**
is a point deep in its own cluster, **0** is on the boundary, **−1** is
misclassified. Score is the mean of s(i) across the subsample.

**Convention.** Silhouette > 0.5 = well-separated. 0.25–0.5 = weak but
non-trivial. < 0.25 = no real cluster structure. Our max is +0.29 so we
present relative comparisons across layers, not absolute claims.

**Choosing k.** Best k = argmax silhouette across the sweep. Reported as
`best_k_kmeans` / `silhouette`. When silhouette is essentially flat
across k (as at `T5`), best-k picks up noise — that's a signal that the
data doesn't want to be clustered at that layer.

### HDBSCAN (secondary check)

`sklearn.cluster.HDBSCAN(min_cluster_size=50, min_samples=10)`.

**Idea.** Instead of choosing k, HDBSCAN estimates local density and
extracts clusters where density exceeds a persistent threshold. Points in
low-density regions get label −1 (noise). No parameter for "how many
clusters" — the algorithm reads it off the density landscape.

**Why we run it alongside k-means.** k-means always returns k clusters,
even from a Gaussian blob. HDBSCAN's output is diagnostic:

- **One giant blob + tiny noise fraction** (`T5`: 98% in one cluster,
  0.7% noise) means "no meaningful density islands" — k-means is
  imposing artificial structure.
- **Many small clusters with high noise** (`T0`: 64 clusters, 4.5%
  noise; `T7`: 20 clusters, 74% noise) means the geometry has fine
  local structure that k-means smooths into a few big blobs — the two
  algorithms disagree because the true topology isn't ball-shaped.

### Per-layer plots

Each `results/bullfinch_within_layer{L:02d}.png` has three panels:

1. **Silhouette vs k** with a red dashed line at best-k — shape tells
   you whether the k-choice was decisive or arbitrary.
2. **PC1 / PC2 scatter, colored by k-means (best k) assignment** —
   visual sanity check that clusters correspond to a spatial layout
   in the top-2 PCs.
3. **PC1 / PC2 scatter, colored by recording index** — critical
   confound check. If the k-means clusters look identical to the
   recording-colored panel, we're clustering "which recording" not
   any within-species structure.
