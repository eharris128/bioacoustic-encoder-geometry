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
