# Causal Trace — Species Information Across AVES Layers
**Run:** RUN-000003
**Date:** 2026-03-18
**Job:** causal-trace-species
**Runtime:** ~3 minutes (CPU, Lambda A10)

---

## Motivation

The species probe experiment (RUN-000001) found that layer 1 (93.6%) and layer 11 (88%)
both achieve high accuracy, but with a non-monotone profile across depth — layer 2 peaks,
layers 4–8 dip, then accuracy recovers. This is unexpected for a deep transformer, where
you'd expect later layers to be more discriminative.

Two hypotheses:

- **A (bottleneck):** Layer 1 encodes raw spectral features that are already species-diagnostic;
  later layers transform them for other purposes, reducing linear separability.
- **B (redundant encoding):** Species identity is re-encoded throughout the network; no single
  layer is the causal bottleneck.

This experiment ran causal activation patching to distinguish them.

---

## Method

**Probe accuracy (LORO):**

| Layer | Accuracy |
|-------|----------|
| 0     | 91.9%    |
| 1     | 93.1%    |
| 2     | 94.4%    |
| 3     | 92.8%    |
| 4     | 89.3%    |
| 5     | 85.7%    |
| 6     | 84.3%    |
| 7     | 91.1%    |
| 8     | 85.4%    |
| 9     | 89.6%    |
| 10    | 91.4%    |
| 11    | 93.5%    |

*Note: probe accuracy peak is layer 2 (94.4%), not layer 1 as previously reported.*
*Previous run used only 4 Bullfinch recordings; this run includes 5 Hawfinch recordings, giving a better balanced estimate.*

**Patching protocol:**

For each patch layer k ∈ {0..11}:
1. Compute opposite-species mean activation at layer k from training recordings (LORO).
2. Hook layer k's transformer output; replace the entire output tensor (all frames) with
   that constant mean vector.
3. Run layers k+1..11 normally.
4. Apply the layer-11 probe to the resulting embeddings.
5. Report transfer accuracy: fraction of frames predicting the opposite species.

---

## Result

**Transfer accuracy = 1.000 at every layer, both directions, all 9 recordings.**

Patching any layer with the opposite species mean fully redirects the layer-11 representation.

---

## Interpretation

The result is unambiguous, but it does not resolve the bottleneck question in a useful way.

**What it shows:**
Replacing all frames' activations with the opposite species mean is a massive intervention —
the model's entire representational state at layer k is overwritten with a constant vector.
Of course subsequent layers produce a representation that looks like the other species;
there is no information left from the original recording.

**What it does not show:**
Which layer is the *minimal* causal bottleneck — the layer where the smallest perturbation
in the species direction produces the largest downstream effect at layer 11.

**Implication for hypothesis B:**
The result is consistent with B (redundant encoding) but does not rule out A (bottleneck).
A true bottleneck layer would show 1.0 transfer at all scales; a non-bottleneck layer would
require a large intervention to redirect the output. The current experiment cannot distinguish
these because the intervention scale is uniform and enormous at every layer.

---

## Identified Gap

Mean replacement patching is too blunt. It tests "is this information sufficient," not
"is this information necessary at what scale."

The correct refinement: **contrastive direction patching with scale sweep.**

For each layer k and scale α:
- Compute the species direction: `d_k = normalize(mean_hawfinch_k − mean_bullfinch_k)`
- Patch: `activation + α * d_k` (additive, not replacement)
- Sweep α over a range that spans from no effect to full flip
- The layer requiring the smallest α to flip 50% of frames is the causal bottleneck

This is running as RUN-000004 (`contrastive-patch-species`).

---

## Artifacts

- `causal_trace_species.png` — transfer accuracy by patch layer (flat at 1.0)
- `species_separation.png` — probe accuracy + L2 mean separation per layer
