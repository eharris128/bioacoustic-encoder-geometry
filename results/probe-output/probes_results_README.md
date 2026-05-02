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

### Common Chiffchaff vs Iberian Chiffchaff *(same genus, recently split sisters)*
100 recordings each. Peak **91.5% at T11**. Embedding: 62.5%.
Accuracy climbs monotonically all the way to the final layer — the most gradual build of any pair. Suggests these two are acoustically the most similar species tested.

### Great Tit vs Great Tit Bokharensis *(subspecies, Parus major)*
100 vs 54 recordings (xeno-canto subspecies coverage limited). Peak **92.2% at T3**. Embedding: 81.8%.
Anomalous high embedding accuracy — possibly due to geographic/recordist bias in the small bokharensis sample rather than true acoustic signal. Treat with caution.

### Goldfinch vs Eurasian Siskin *(cross-genus, same family Fringillidae)*
100 recordings each. Peak **92.5% at T5**. Embedding: 65.5%.
Different genera (*Carduelis* vs *Spinus*) but same family. Fits between same-genus and cross-family in difficulty.

### House Crow vs Carrion Crow *(same genus, Corvus)*
100 recordings each. Peak **95.0% at T9**. Embedding: 72.0%.
Notably high embedding for a same-genus pair — crow calls are acoustically more distinct than sparrow/warbler congeners. Peak at T9 suggests late-stage refinement still needed.

### European Robin vs Eurasian Blackbird *(different families, Muscicapidae vs Turdidae)*
100 recordings each. Peak **99.0% at T8/T11**. Embedding: 67.0%.
Near-ceiling performance. T0 already hits 92.5% — cross-family separation is established very early, with all transformer layers above 94%.

---

## Cross-experiment pattern

| Pair | Taxonomy | Peak | Peak layer | Emb |
|---|---|---|---|---|
| House Sparrow vs Tree Sparrow | Same genus | 85.4% | T6 | 53.3% |
| Willow Warbler vs Chiffchaff | Same genus | 93.0% | T6 | 53.0% |
| Common vs Iberian Chiffchaff | Same genus (sisters) | 91.5% | T11 | 62.5% |
| House Crow vs Carrion Crow | Same genus | 95.0% | T9 | 72.0% |
| Great Tit vs Great Tit Bokharensis | Subspecies* | 92.2% | T3 | 81.8% |
| Goldfinch vs Eurasian Siskin | Cross-genus, same family | 92.5% | T5 | 65.5% |
| Bullfinch vs Hawfinch | Cross-genus, same family | 95.0% | T2 | 61.0% |
| European Robin vs Eurasian Blackbird | Different families | 99.0% | T8/T11 | 67.0% |
| Bullfinch vs Tawny Owl | Different orders | 99.0% | T3/T9 | 65.5% |

**Accuracy broadly scales with taxonomic distance.** Same-genus pairs show embedding near-chance (~53–72%) with peak accuracy between T6–T11. Cross-family/order pairs hit 99% and establish separation as early as T0–T1. The subspecies result (Great Tit bokharensis) is an outlier likely due to sample bias. House Crow is an anomaly among same-genus pairs — crow calls may be inherently more distinctive than passerine congeners.
