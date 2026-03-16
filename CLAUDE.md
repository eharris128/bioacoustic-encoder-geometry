# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interpretability research on AVES (Animal Vocalization Encoder based on Self-Supervision), a HuBERT-based transformer (12 layers, 12 heads, 768-dim) fine-tuned on animal sounds by the Earth Species Project. We systematically probe what each layer learns using Bullfinch recordings from xeno-canto.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install esp-aves torchcodec matplotlib scikit-learn
git clone https://github.com/earthspecies/aves.git
mkdir -p models
curl -L -o models/aves-base-all.torchaudio.pt \
  https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-all.torchaudio.pt
```

## Running Scripts

All scripts are standalone and run from the project root:
```bash
source venv/bin/activate
python <script_name>.py
```
No test suite — this is exploratory research. Each script produces PNG plots and prints results to stdout. Suppress sklearn convergence warnings with `python -W ignore <script>.py`.

## Architecture

### Model Access Pattern

Every script follows this pattern:
```python
from aves import load_feature_extractor
from aves.utils import load_audio

model = load_feature_extractor(
    config_path="./aves/config/default_cfg_aves-base-all.json",
    model_path="./models/aves-base-all.torchaudio.pt",
    device="cpu", for_inference=True,
)
audio = load_audio(path, mono=True, mono_avg=False)  # Returns 16kHz tensor
layer_outputs = model.extract_features(audio, layers=None)  # List of 12 tensors, each (1, n_frames, 768)
```

### Model Internals (for hooks)

The AVES wrapper (`model`) contains a torchaudio wav2vec2 model at `model.model`:
```
model.model.feature_extractor.conv_layers[0..6]  # 7 CNN layers (return tuples: (tensor, length))
model.model.encoder.feature_projection            # Linear projection 512→768
model.model.encoder.transformer.layers[0..11]     # 12 transformer layers
model.model.encoder.transformer.layers[i].attention.{q_proj, k_proj, v_proj, out_proj}
```

Attention weights are NOT exposed by torchaudio — must hook Q/K projections and compute `softmax(QK^T/sqrt(d))` manually. See `explore_attention.py` for the pattern.

CNN layer hooks receive tuples `(tensor, length)` — extract `output[0]` before processing.

### Frame Rate Alignment

AVES downsamples by 320x: 16000 Hz / 320 = 50 fps. One frame = 20ms. When computing mel spectrograms for comparison, always use `hop_length=320` to align with AVES frames.

### Data

- `aves/example_audios/` — 2 files (Guineafowl .wav, Bullfinch .mp3) from the cloned repo
- `audio/bullfinch/` — 30 recordings from xeno-canto (gitignored, ~89MB)
- `audio/hawfinch/` — 5 recordings from xeno-canto (gitignored)
- Skip `XC1086809.mp3` (35MB, dominates datasets) and `XC657517.mp3` (corrupted)

### HuBERT Comparison

HuBERT-base weights cached at `~/.cache/torch/hub/checkpoints/hubert_fairseq_base_ls960.pth`. Load via `torchaudio.pipelines.HUBERT_BASE.get_model()`. Same architecture as AVES — hooks work identically but access transformer at `model.encoder.transformer.layers[i]` (no `.model` prefix).

## Key Findings So Far

1. **CNN does the heavy lifting** — per-layer CKA of 0.10-0.23 vs transformer's 0.97-0.99
2. **Recording identity erases monotonically** across transformer layers (silhouette 0.02 → -0.003)
3. **Species probe peaks at layer 1** (94%), dips mid-network (84%), recovers at layer 11 (91%)
4. **Temporal prediction comes from CNN**, not transformer — all transformer layers predict equally
5. **Acoustic features linearly decodable only at layers 0-1** — late-layer clusters are nonlinearly acoustic
6. **AVES vs HuBERT**: broad hierarchy is architecture-driven; attention strategies are data-driven

## Conventions

- Random seed: 42 throughout
- Frame subsampling: `rng.choice(n, MAX_FRAMES, replace=False)` with sorted indices
- Train/test split: temporal (80/20), not shuffled, to respect sequential structure
- Plots: 150 dpi, `bbox_inches="tight"`, saved as PNG
- PCA to 50 dims before logistic regression probes (768-dim is too slow on CPU)
- All generated PNGs are committed; audio files and model weights are gitignored
