# Isolating Animal Vocalization directions in AVES via White noise

This experiment takes a single clean guineafowl recording and generates versions of it with increasing levels of white noise added to the raw audio, spanning from near-clean to heavily degraded across 10 levels. Each version is passed through AVES and frame-level activations are extracted at each transformer layer. By fitting a linear direction across the noise-indexed activation vectors, we identify where in the model's representational geometry recording noise lives and whether that direction is orthogonal to the species. A noise direction that is orthogonal to meaningful semantic directions can be subtracted out of downstream representations, leaving a cleaner signal for SAE feature analysis and behavioral grounding work.

## Experimental setup

The experiment starts with inputting a single clean guineafowl reference recording. This recording will be at a noise level of 1, and will be human-annotated to choose a xeno-canto recording with audibly clear bird calls, in combination with matching sonogram features. 

# AVES / BirdAVES (Earth Species Project)

**Paper:** Hagiwara (2023), ICASSP. BirdAVES update: ESP blog post, 2024.
**What it is:** Self-supervised transformer encoder for animal vocalisations (“BERT for animals”). Based on HuBERT. Pretrained on large unannotated audio datasets including animal sounds. BirdAVES adds Xeno-canto/iNaturalist bird data and scales to larger models.

This experiment will test if AVES is able to encode noise as a separate construct through investigating consistency in activation space. Essentially, it will work as a cleaning tool for future downstream experiments. Model representations are only as clean as the data going in. We can essentially project this direction in the activation space out of every activation vector in the dataset, thus representing animal signals clearer. 

# Noise_direction1.py

Noise_direction1.py takes each recording in RECORDINGS, generating 10 versions of it. Each of these versions will have added calibrated white gaussian noise at decreasing SNR levels. Levels decrease from 40dB to 0dB. The script calculates required noise power, generates a waveform at that power, adds it to the raw audio waveform, then passes it into the model. 

Noise augmentation — generating 10 audio versions via additive white gaussian noise across SNR levels

Experimental iteration

Activation extraction — frame-level embeddings at all 12 layers per noise version
Noise direction fitting — PCA across noise-indexed mean activations to find the noise axis
Monotonicity check — confirming the direction varies continuously with noise level
Orthogonality analysis — dot product against species and call-type directions across layers
Results — which layers separate noise from signal most cleanly
Conclusion — whether noise can be projected out to improve downstream semantic analysis