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
Hardest pair — embedding layer is blind, separation only emerges deep in the transformer.

---

## Cross-experiment pattern

| Pair | Taxonomy | Peak | Peak layer | Emb |
|---|---|---|---|---|
| House Sparrow vs Tree Sparrow | Same genus | 85.4% | T6 | 53.3% |
| Bullfinch vs Hawfinch | Same family | 95.0% | T2 | 61.0% |
| Bullfinch vs Tawny Owl | Diff. orders | 99.0% | T3/T9 | 65.5% |

**Accuracy scales with taxonomic distance.** The embedding layer is nearly useless for same-genus discrimination — the transformer is doing the work. Peak layer shifts later for harder pairs.
