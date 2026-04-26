# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interpretability research on AVES (Animal Vocalization Encoder based on Self-Supervision), a HuBERT-based transformer (12 layers, 12 heads, 768-dim) fine-tuned on animal sounds by the Earth Species Project. We systematically probe what each layer learns using Bullfinch and other species recordings from xeno-canto.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install avex datasets esp-aves torchcodec matplotlib scikit-learn soundfile scipy
git clone https://github.com/earthspecies/aves.git   # only needed for legacy scripts
mkdir -p models
curl -L -o models/aves-base-all.torchaudio.pt \       # only needed for legacy scripts
  https://storage.googleapis.com/esp-public-files/ported_aves/aves-base-all.torchaudio.pt
```

EAT model checkpoints (used by the structured pipeline) auto-download from HuggingFace via `avex` on first use — no manual `curl` needed.

## Running Scripts

All scripts are standalone and run from the project root:
```bash
source venv/bin/activate
python -W ignore <script_name>.py            # legacy top-level scripts
python -W ignore experiments/<experiment>.py # structured experiments
```

Use `researchctl` wrappers for isolated workspaces and archived artifacts:
```bash
python ops/researchctl_jobs/<job>_job.py --dry-run   # preview paths
python ops/researchctl_jobs/<job>_job.py              # run and archive
```

Wrappers write `artifacts/runs/<job_id>/<run_id>/result.json` plus any PNGs. Temporary workspaces live at `.researchctl/workspaces/` and are deleted on success unless `--keep-workspace` is passed.

## Architecture

### Module Structure

The codebase has two layers: legacy top-level scripts (`explore_*.py`, `probe_species.py`, etc.) and a newer module system:

- **`data/loader.py`** — centralized model loading and activation extraction. Two dataset builders: `build_dataset` (local files) and `build_naturelm_dataset` (HuggingFace streaming). Both use the EAT pipeline via `avex` and return `{layer_index: (X, y)}`.
- **`probes/train.py`** — `train_all_layers` runs leave-one-recording-out (LORO) cross-validation across all 13 layers and prints a per-layer accuracy table.
- **`probes/evaluate.py`** — `run_evaluation` saves an accuracy curve PNG and an LDA projection PNG to `results/`.
- **`experiments/`** — per-experiment configs and entry points that wire together `data/loader`, `probes/train`, and `probes/evaluate`. Runnable: `animals_vs_music.py`, `species.py`. Stubs (need more audio): `music_vs_speech.py` (needs ≥5 speech files in `audio/speech/`), `species_vs_species.py` (template; populate `SPECIES_A`, `SPECIES_B`, `RECORDINGS` before running).
- **`ops/researchctl_jobs/`** — `common.py` provides shared utilities (`prepare_workspace`, `run_script`, `write_result`); each `*_job.py` wraps one script with symlinked inputs, structured JSON output, and regex-parsed summaries. Jobs: `bullfinch_layer11_structure_job.py`, `causal_trace_species_job.py`, `contrastive_patch_species_job.py`, `explore_clusters_job.py`, `probe_species_job.py`, `sae_layer11_job.py`.

### Layer Indexing Convention

13 layers total (used throughout `data/`, `probes/`, `experiments/`):
- Index 0 = CNN `feature_projection` output (embedding layer)
- Indices 1–12 = transformer layers 0–11

In `probes/evaluate.py`, axis labels use `"emb"` for index 0 and `"T0"`–`"T11"` for indices 1–12.

Legacy top-level scripts use 0–11 directly for transformer layers — no embedding layer at index 0.

### Model Access Pattern

```python
from data.loader import load_model, build_dataset   # structured experiments
# or (legacy pattern):
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

### Supported Models

`data/loader.EAT_MODELS` contains the 4 supported models (all EAT architecture, 12 transformer blocks, 768-dim). Checkpoints auto-download from HuggingFace on first use via `avex`:
- `"esp_aves2_eat_all"` — SSL pretrained, all data (default)
- `"esp_aves2_eat_bio"` — SSL pretrained, bio-only data
- `"esp_aves2_sl_eat_all_ssl_all"` — supervised fine-tune, all data
- `"esp_aves2_sl_eat_bio_ssl_all"` — supervised fine-tune, bio data

### Model Internals (EAT hook paths)

Hooks are registered by `load_model` on all 13 layers:
```
backbone.model.local_encoder          # index 0: patch projection, output (1, 512, 768)
backbone.model.blocks.{0..11}         # indices 1–12: transformer blocks, output (1, 513, 768)
```

The CLS token (position 0) is stripped from transformer block outputs in `extract_all_layers` so all 13 layers yield consistent `(n_patches, 768)` activations.

For legacy scripts (`explore_attention.py`, etc.) that use the old AVES API directly:
```
model.model.encoder.transformer.layers[i].attention.{q_proj, k_proj, v_proj, out_proj}
```
These scripts still hardcode `aves-base-all` and are not part of the active pipeline.

### Data

- `aves/example_audios/` — 2 files (Guineafowl .wav, Bullfinch .mp3) from the cloned repo
- `audio/bullfinch/` — 30 recordings from xeno-canto (gitignored, ~89MB)
- `audio/hawfinch/` — 5 recordings from xeno-canto (gitignored)
- `audio/helmeted-guinea-fowl/` — recordings from xeno-canto (gitignored)
- `audio/violin/`, `audio/music-misc/` — music recordings for animals-vs-music experiment (gitignored)
- `audio/speech/` — LibriVox speech recordings for music-vs-speech experiment (gitignored; needs ≥5 files before that experiment is runnable)
- Skip `XC1086809.mp3` (35MB, dominates datasets) and `XC657517.mp3` (corrupted)
- NatureLM streaming: `EarthSpeciesProject/NatureLM-audio-training` on HuggingFace (requires `datasets` package)

### HuBERT Comparison (legacy)

HuBERT-base weights cached at `~/.cache/torch/hub/checkpoints/hubert_fairseq_base_ls960.pth`. Load via `torchaudio.pipelines.HUBERT_BASE.get_model()`. Used only in `compare_hubert.py` — hooks access transformer at `model.encoder.transformer.layers[i]` (no `.model` prefix).

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
- Train/test split: temporal (80/20) for legacy scripts; LORO for structured `probes/train.py`
- Plots: 150 dpi, `bbox_inches="tight"`, saved as PNG
- PCA to 50 dims before logistic regression probes (768-dim is too slow on CPU)
- Experiment outputs: `results/<experiment_name>_accuracy.png`, `results/<experiment_name>_lda.png`
- Job artifacts: `artifacts/runs/<job_id>/<run_id>/result.json` + PNGs
- All generated PNGs are committed; audio files, model weights, and `.researchctl/workspaces/` are gitignored
- Naming: exploratory scripts `explore_<topic>.py`, comparisons `compare_<topic>.py`, experiments `experiments/<topic>.py`
