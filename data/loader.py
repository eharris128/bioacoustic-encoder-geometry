"""
data/loader.py — Load and label audio subsets for probe experiments.

Provides a unified interface for assembling (frames, labels) datasets from
local audio files. Each experiment config passes a RECORDINGS dict here;
this module handles model loading, feature extraction, frame subsampling,
and label assignment across all 12 layers.

Conventions (match existing project scripts):
- Random seed: 42
- Frame subsampling: rng.choice(n, MAX_FRAMES, replace=False), sorted
- Returns per-layer activations: shape (n_frames, 768) per layer
- Audio loaded at 16kHz mono via soundfile (WAV) or torchaudio fallback (MP3)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# RECORDINGS dict format used by all experiment configs:
#   { recording_id: (audio_path, label_int) }
RecordingsDict = dict[str, tuple[str, int]]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    config_path: str = "./aves/config/default_cfg_aves-base-all.json",
    model_path: str = "./models/aves-base-all.torchaudio.pt",
    device: str = "cpu",
):
    """
    Load and return the AVES feature extractor.

    Parameters
    ----------
    config_path : path to AVES JSON config
    model_path  : path to .torchaudio.pt checkpoint
    device      : "cpu" or "cuda"

    Returns
    -------
    model : AVES feature extractor (for_inference=True)
    """
    ...


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio_file(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load an audio file (WAV or MP3) as a mono float32 tensor at target_sr.

    Tries soundfile first (WAV/FLAC); falls back to torchaudio for MP3.

    Parameters
    ----------
    path      : absolute or relative path to audio file
    target_sr : target sample rate in Hz (default 16000)

    Returns
    -------
    waveform : (1, n_samples) float32 tensor
    """
    ...


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_all_layers(
    model,
    audio: torch.Tensor,
    max_frames: int = 3000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Run a forward pass and return subsampled frame activations for all layers.

    Parameters
    ----------
    model      : loaded AVES model
    audio      : (1, n_samples) float32 tensor at 16kHz
    max_frames : max frames to keep per recording (subsampled if exceeded)
    rng        : numpy Generator for reproducible subsampling (seed 42 default)

    Returns
    -------
    activations : (n_frames, 12, 768) float32 array
                  axis 0 = frames, axis 1 = layer index, axis 2 = embedding dim
    """
    ...


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def build_dataset(
    model,
    recordings: RecordingsDict,
    max_frames_per_recording: int = 3000,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Extract activations for all recordings and return a per-layer dataset.

    Parameters
    ----------
    model                    : loaded AVES model
    recordings               : { rec_id: (audio_path, label) }
    max_frames_per_recording : frame cap per recording before pooling

    Returns
    -------
    dataset : { layer_index: (X, y) }
              X shape: (total_frames, 768)
              y shape: (total_frames,) — integer class labels
    """
    ...
