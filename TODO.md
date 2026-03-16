# Next Steps

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
