# Interpretability Researcher Frame

Load this file when you want to reason like a mechanistic interpretability researcher.
It synthesizes current methodology from leading practitioners and applies it to this project.

---

## Core Methodology: Pragmatic Interpretability

**Source:** Neel Nanda et al., "A Pragmatic Vision for Interpretability" (2025), Google DeepMind

The field has shifted away from ambitious reverse-engineering toward task-grounded research.
Every project should be anchored to:

1. **A North Star** — a meaningful scientific question, not just "understand the model."
   - Bad: "What are the AVES features?"
   - Good: "Does species information enter the network causally at layer 1, or is it reconstructed across layers?"

2. **A proxy task** — a concrete, falsifiable measurement that tracks progress toward the North Star.
   - Bad: "Qualitatively explore the representations."
   - Good: "Causal transfer accuracy at layer 11 as a function of which layer is patched."

3. **Method minimalism** — use the simplest method that answers the question.
   - Try black-box methods first (probing, ablation).
   - Only add complexity (SAEs, circuit analysis) when simpler methods fail or produce new questions.

---

## Method Hierarchy

Apply in order. Stop when you have an answer.

### 1. Black-box / representation methods (days)
- Linear probes per layer (leave-one-out cross-validation)
- Mean species/class separation (L2 distance between class means per layer)
- PCA / dimensionality reduction visualization
- Contrastive activation vectors (mean_A − mean_B at each layer)

### 2. Simple causal methods (days to a week)
- **Activation patching**: replace layer-k output with opposite-class mean, run forward, measure effect at output layer
- **Ablation**: zero out a layer, head, or component; measure probe accuracy drop
- **Contrastive steering**: add (mean_A − mean_B) direction at layer k; measure output probe

### 3. Complex methods (only when motivated by 1 and 2)
- Sparse autoencoders — for **unsupervised discovery** when you don't have a hypothesis yet
  - Good: "What acoustic features is AVES sensitive to at layer k?" (open-ended)
  - Bad: "Does AVES have a species feature?" (probing already answers this)
- Attention head attribution
- Circuit-level path patching

---

## SAE Guidance

**SAEs are good for:** unsupervised discovery — finding unexpected structure when you have no prior hypothesis.

**SAEs are the wrong tool when:**
- You have a clear question (use probing + patching instead)
- You're optimizing sparsity metrics without a downstream task
- You want to understand *why* a probe works (use patching instead)

**If you do run an SAE:**
- Measure quality on a downstream task, not just reconstruction loss
- Verify monosemanticity by listening to / visualizing max-activating examples
- Use TopK activation (K ≈ 20–40) rather than L1 tuning
- Report: L0, reconstruction variance explained, dead feature fraction, max-activating examples for top-20 features

---

## Causal Evidence Standards

Ranking from weakest to strongest:

| Evidence type | Example | Strength |
|---|---|---|
| Representation correlation | "The layer-1 probe reaches 93.6% accuracy" | Weak — representation ≠ causation |
| Steering / direction test | "Adding the species direction at layer 1 shifts the layer-11 probe score" | Medium |
| Activation patching | "Patching layer k with class-B mean causes layer-11 to predict class B" | Strong |
| Double dissociation | "Patching layer k flips B→A AND A→B symmetrically; no other layer does" | Very strong |

Always aim for at least "medium" before drawing mechanistic conclusions.

---

## Key Questions for This Project

When evaluating a proposed analysis, ask:

1. **What is the North Star?** What scientific claim would this confirm or disconfirm?
2. **Is there a simpler method?** Would a linear probe or mean-difference vector answer the same question?
3. **Is there a proxy task?** How will you know if you got the answer?
4. **What is the causal claim, if any?** Is the evidence correlational or interventional?
5. **What would falsify this?** What result would mean the hypothesis is wrong?

---

## Current Project North Star

> **Does species information enter the AVES residual stream causally at layer 1,
> or is it progressively constructed across layers?**

Current evidence:
- Layer-1 probe accuracy (93.6%) exceeds layer-11 (88%) — unexpected for a deep transformer
- Causal tracing experiment (`causal_trace_species.py`) is the active proxy task
- Key result pending: transfer accuracy curve by patch layer

Interpretations to test:
- **A (early-and-stable):** Layer 1 encodes a species direction from raw spectral features; later layers rotate it but preserve it. Patching layer 1 should strongly flip layer-11 predictions; patching later layers should have smaller effect.
- **B (early-then-lost):** Layer 1 encodes species, but later layers transform it away for other purposes. Patching layer 1 should have the *largest* effect, and the species direction at layer 11 should not be linearly predictable from the layer-1 direction.
- **C (distributed):** Multiple layers contribute species information redundantly. Patching any single layer has modest effect; only multi-layer patching fully flips predictions.

---

## Practitioner Reference

**Neel Nanda** (Google DeepMind)
- Core methodology: proxy tasks, North Stars, method minimalism
- SAE contribution: Towards Monosemanticity; but now explicitly deprioritizes SAE-as-default
- Key principle: "Just do what works. This includes black-box techniques when appropriate."

**Chris Olah** (Anthropic)
- Core methodology: features → circuits → universality
- Strong prior: representations are often universal across models trained on similar data
- Key principle: monosemanticity and superposition as the central structural question

**Lawrence Chan / Samuel Marks** (causal scrubbing, attribution patching)
- Core methodology: rigorously test whether a proposed circuit explains model behavior
- Key principle: the circuit should be both necessary and sufficient

**David Bau** (network dissection, causal tracing for factual recall)
- Core methodology: causal tracing in LLMs; activation patching as the standard causal test
- Key principle: correlational interpretability is necessary but not sufficient

---

## Anti-Patterns to Avoid

- Running SAEs before establishing what question they'd answer
- Reporting probe accuracy as a causal claim
- Iterating on approximation error (SAE reconstruction loss, probe accuracy) without a downstream task
- Indefinite exploration without a 2-week falsification target
- Calling a result "mechanistic" based on a single experiment type
