# Next Steps

## Roadmap Section 1 — ESP-AVES2 Activations (active)

Scope: ESP-AVES2 `eat`-family only (see `open_questions.md` §2). Section 3 of the roadmap (noise dynamics, audio mixing, barycenters, hierarchy probes) is **out of scope**.

### Step 1 gaps — collection

- [ ] Verify the freshly re-uploaded HF safetensors for `esp-aves2-eat-all` and `esp-aves2-eat-bio` (2026-04-20 re-publish) load with non-zero weights. Download in flight 2026-04-26.
- [ ] Run `collect_esp_aves2_activations.py` for `esp-aves2-eat-all` and `esp-aves2-eat-bio` against the existing frozen manifest `naturelm_by_source_100each_20260418T171459Z`. No code change required — the extraction script and manifest already exist, this just produces two more `<model>/shards/` directories alongside the two `sl-*` runs we already have.
- [ ] Decide whether the pilot is sufficient at 100 samples × 7 sources × 4 models or if we want to scale up before moving on. Tied to the storage question below.
- [ ] Resolve the "where do we store activations?" open question — capture the answer in `open_questions.md` §3 even if it's just "local disk for the pilot." Gating any scale-up.
- [ ] Commit the currently-untracked `compare_esp_aves2_models.py` and `app_esp_aves2_compare.py` so Step 2 work has a tracked baseline.

### Step 2 gaps — statistics across layers and models

Required by the roadmap, not yet produced for any of the four models:

- [ ] **L2-norm distributions** per `(model, layer)` — currently we only compute cross-model norm deltas. Need within-model norm histograms so the deltas are interpretable.
- [ ] **Singular-value spectra** of the pooled-embedding matrix per `(model, layer)`, plus effective rank / spectral entropy as a single-number summary.
- [ ] **PCA alignment** across layers (within a model) and across models (within a layer) — subspace angles or top-k overlap. CKA is a good first pass for the cross-model version.
- [ ] **Intrinsic dimensionality** per `(model, layer)` — TwoNN as a fast first estimator; participation ratio as a sanity cross-check.
- [ ] Compare each statistic across the seven `source_dataset` slices ("nature sounds vs other sound" — the roadmap's explicit ask).
- [ ] Decide pooled vs frame-level for the spectral / intrinsic-dim measurements. Pooling is what we have today; frame-level over a few hundred items would give richer SV / dim estimates without a storage blow-up. Worth a quick pooled-vs-frame comparison on one model before committing.
- [ ] Once `eat_all` and `eat_bio` extractions land, extend `compare_esp_aves2_models.py` from pairwise to 4-way (or run it pairwise across all 6 pairs and aggregate). No schema change needed — the extraction layout already supports this.

## Deepening the Mechanism

### Sparse Autoencoders (SAEs) on layer embeddings
Train a sparse autoencoder on layer 11 embeddings to decompose the 768-dim space into thousands of sparse, interpretable directions. Each direction might correspond to something specific: "rising pitch contour," "harsh broadband onset," "silence after call." Moves us from "the model has clusters" to "here are the individual features the model uses to build those clusters." Linear probes showed the representation is nonlinear — SAEs are designed to crack open exactly that kind of structure.

**Priority: High — builds directly on existing embeddings, clusters, and acoustic profiles.**

### Attention head ablation
Zero out individual heads (or pairs) and measure what changes — does species separability collapse? Do the late-layer clusters dissolve? Identifies which heads are load-bearing vs redundant, and tests whether the functional specialization we observed (local vs global heads) is real or a visualization artifact.

## Connecting to Biology

### RSA with zebra finch neural recordings
The CRCNS aa-4 dataset contains 914 neurons from zebra finch auditory brain regions (Field L, CLM/CMM, NCM) — a known hierarchy from acoustic to abstract. Present the same stimuli to both AVES and the neural data, compute pairwise distance matrices at each layer, then correlate (Representational Similarity Analysis). The layer with highest RSA to each brain region tells us which part of the model most resembles which part of the biological auditory system. Most publishable direction — directly tests "artificial network as model of biological processing."

**Priority: High — most impactful, but requires obtaining and aligning the CRCNS dataset.**

### Cross-species call type transfer
Train k-means on Bullfinch late-layer embeddings, apply to Hawfinch. If clusters transfer meaningfully, the model has discovered universal acoustic categories. If they fail, the organization is species-specific. Either result is informative.

## Toward Application

### Unsupervised syllable segmentation
Cluster transitions (where the frame-level cluster label changes) are candidate syllable boundaries. Compare to spectrogram-derived segmentation to test if the model discovers syllable structure without supervision. If so, AVES becomes a zero-shot syllable segmenter — useful for bioacoustics researchers who currently label syllables by hand.

### Call type discovery at scale
Run the pipeline on 500+ Bullfinch recordings from xeno-canto. Cluster late-layer embeddings and build a data-driven taxonomy of call types. Characterize each type with acoustic profile, temporal statistics (duration, repetition rate), and attention patterns. Becomes a tool for ornithologists.
