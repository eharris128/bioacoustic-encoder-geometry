# Probe Experiments — Results Summary

All probes use LORO cross-validation, PCA(50) → LogisticRegression per layer.
Layer index: `emb` = CNN projection, `T0–T11` = transformer layers 1–12.

---

## Animals vs Music
**Script:** `experiments/animals_vs_music.py`
**Data:** Local files — bullfinch + hawfinch + helmeted guinea fowl vs violin + misc music MP3s.
**Finding:** Strong separation across all transformer layers; AVES cleanly linearly separates animal vocalizations from musical instruments. Embedding layer already above chance, transformer pushes further.

---

## Species vs Species

### Bullfinch vs Hawfinch *(same family, Fringillidae)*
100 recordings each via xeno-canto. Peak **95.0% at T2**. Embedding: 61.0%.
Mid-network dip then partial recovery — species info encoded early but refined late.

### Bullfinch vs Tawny Owl *(different orders)*
100 recordings each. Peak **99.0% at T3 and T9**. Embedding: 65.5%.
Near-ceiling across all transformer layers. Taxonomic distance makes separation trivial.

### House Sparrow vs Tree Sparrow *(same genus, Passer)*
100 recordings each. Peak **85.4% at T6**. Embedding: 53.3% ≈ chance.
Embedding blind — separation only emerges deep in the transformer.

### Willow Warbler vs Chiffchaff *(same genus, Phylloscopus)*
100 recordings each. Peak **93.0% at T6**. Embedding: 53.0% ≈ chance.
Visually near-identical sibling species with distinct songs. Embedding again blind; transformer builds separation progressively, peaking at T6.

### Great Tit vs Great Tit Bokharensis *(subspecies, Parus major)*
100 vs 54 recordings (xeno-canto subspecies coverage limited). Peak **92.2% at T3**. Embedding: 81.8%.
Anomalous high embedding accuracy — possibly due to geographic/recordist bias in the small bokharensis sample rather than true acoustic signal. Treat with caution.

---

## Cross-experiment pattern

| Pair | Taxonomy | Peak | Peak layer | Emb |
|---|---|---|---|---|
| House Sparrow vs Tree Sparrow | Same genus | 85.4% | T6 | 53.3% |
| Willow Warbler vs Chiffchaff | Same genus | 93.0% | T6 | 53.0% |
| Great Tit vs Great Tit Bokharensis | Subspecies* | 92.2% | T3 | 81.8% |
| Bullfinch vs Hawfinch | Same family | 95.0% | T2 | 61.0% |
| Bullfinch vs Tawny Owl | Diff. orders | 99.0% | T3/T9 | 65.5% |

**Accuracy scales with taxonomic distance.** For same-genus pairs, the embedding layer is near-chance (~53%) — the transformer does the work, peaking at T6. Cross-family and cross-order pairs are easier and peak earlier (T2–T3). The subspecies result (Great Tit) is an outlier likely due to sample bias.
