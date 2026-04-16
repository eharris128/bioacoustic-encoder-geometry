"""
data/loader.py — Load and label audio subsets for probe experiments.

Two interfaces:

1. Local files (experiments/animals_vs_music.py, music_vs_speech.py, species*.py)
   - build_dataset(model, recordings) → {layer: (X, y)}, frames_per_recording

2. NatureLM HuggingFace streaming (large-scale probe experiments)
   - build_naturelm_dataset(model, ...) → {layer: (X, y)}, metadata_list

Both return the same {layer: (X, y)} format consumed by probes/train.py.

Activation layout
-----------------
13 layers total: index 0 = CNN feature_projection output (embedding),
                 indices 1–12 = transformer layers 0–11.
Raw mode   : (13, n_frames, 768) per item
Mean-pooled: (13, 768) per item — frame dim collapsed

Conventions (match existing project scripts)
--------------------------------------------
- Random seed: 42
- Frame subsampling: rng.choice(n, max_frames, replace=False), indices sorted
- 16 kHz mono audio throughout
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from scipy import signal as scipy_signal

NUM_LAYERS_TRANSFORMER = 12
NUM_LAYERS_TOTAL = 13          # embedding (index 0) + transformer layers 1–12
DEFAULT_MAX_FRAMES = 513       # per item for NatureLM (≈10s at 50fps)
DEFAULT_MAX_FRAMES_LOCAL = 3000
MIN_SAMPLES_DEFAULT = 500
DATASET_REPO = "EarthSpeciesProject/NatureLM-audio-training"

# Type alias used by all experiment configs
RecordingsDict = dict[str, tuple[str, int]]

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Maps short model names to (config_path, model_path).
# All 4 ESP models share the same HuBERT-base architecture and config.
# Download checkpoints to ./models/ before use:
#   curl -L -o models/<filename> <HuggingFace URL>
#
# HuggingFace repos:
#   EarthSpeciesProject/esp-aves2-eat-all
#   EarthSpeciesProject/esp-aves2-eat-bio
#   EarthSpeciesProject/esp-aves2-effnetb0-all
#   EarthSpeciesProject/esp-aves2-effnetb0-all  (alias: effnetb0-all)

_DEFAULT_CONFIG = "./aves/config/default_cfg_aves-base-all.json"

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # Original AVES base models
    "aves-base-all":    (_DEFAULT_CONFIG, "./models/aves-base-all.torchaudio.pt"),
    "aves-base-bio":    (_DEFAULT_CONFIG, "./models/aves-base-bio.torchaudio.pt"),
    "aves-base-nonbio": (_DEFAULT_CONFIG, "./models/aves-base-nonbio.torchaudio.pt"),
    "aves-base-core":   (_DEFAULT_CONFIG, "./models/aves-base-core.torchaudio.pt"),
    # ESP AVES2 models
    "esp-aves2-eat-all":      (_DEFAULT_CONFIG, "./models/esp-aves2-eat-all.torchaudio.pt"),
    "esp-aves2-eat-bio":      (_DEFAULT_CONFIG, "./models/esp-aves2-eat-bio.torchaudio.pt"),
    "esp-aves2-effnetb0-all": (_DEFAULT_CONFIG, "./models/esp-aves2-effnetb0-all.torchaudio.pt"),
    "effnetb0-all":           (_DEFAULT_CONFIG, "./models/esp-aves2-effnetb0-all.torchaudio.pt"),
}

DEFAULT_MODEL = "aves-base-all"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_name: str = DEFAULT_MODEL,
    config_path: str | None = None,
    model_path: str | None = None,
    device: str = "cpu",
):
    """
    Load and return an ESP AVES feature extractor (for_inference=True).

    Resolves config and checkpoint paths from MODEL_REGISTRY using model_name.
    config_path and model_path override the registry if provided explicitly.

    Parameters
    ----------
    model_name  : short model name, one of MODEL_REGISTRY keys.
                  Default "aves-base-all".
                  Supported:
                    "aves-base-all", "aves-base-bio", "aves-base-nonbio", "aves-base-core"
                    "esp-aves2-eat-all", "esp-aves2-eat-bio",
                    "esp-aves2-effnetb0-all", "effnetb0-all"
    config_path : override config JSON path (optional — uses registry default if None)
    model_path  : override checkpoint path (optional — uses registry default if None)
    device      : "cpu" or "cuda"

    Returns
    -------
    model : AVES feature extractor

    Raises
    ------
    KeyError  if model_name is not in MODEL_REGISTRY
    FileNotFoundError  if the resolved checkpoint does not exist locally
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Available: {sorted(MODEL_REGISTRY.keys())}"
        )

    reg_config, reg_model = MODEL_REGISTRY[model_name]
    resolved_config = config_path or reg_config
    resolved_model  = model_path  or reg_model

    if not Path(resolved_model).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {resolved_model}\n"
            f"Download from HuggingFace: EarthSpeciesProject/{model_name}"
        )

    from aves import load_feature_extractor
    return load_feature_extractor(
        config_path=resolved_config,
        model_path=resolved_model,
        device=device,
        for_inference=True,
    )


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio_file(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load a local audio file (WAV or MP3) as a mono float32 tensor at target_sr.

    Tries soundfile first (WAV/FLAC); falls back to torchaudio for MP3.

    Parameters
    ----------
    path      : absolute or relative path to audio file
    target_sr : target sample rate in Hz (default 16000)

    Returns
    -------
    waveform : (1, n_samples) float32 tensor
    """
    try:
        data, sr = sf.read(path, always_2d=True)       # (n_samples, n_channels)
        data = data.mean(axis=1)                        # mono
        if sr != target_sr:
            n_out = int(round(len(data) * target_sr / sr))
            data = scipy_signal.resample(data, n_out)
        return torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
    except Exception:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdim=True)  # mono (1, n_samples)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform.float()


def _audio_from_hf_item(item: dict, target_sr: int = 16000) -> torch.Tensor:
    """
    Convert a HuggingFace audio item dict to a (1, n_samples) float32 tensor.

    Parameters
    ----------
    item      : dataset item with item["audio"]["array"] and ["sampling_rate"]
    target_sr : target sample rate

    Returns
    -------
    waveform : (1, n_samples) float32 tensor
    """
    array = item["audio"]["array"]
    sr = item["audio"]["sampling_rate"]
    waveform = torch.tensor(array, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)               # (1, n_samples)
    if sr != target_sr:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


# ---------------------------------------------------------------------------
# Core activation 
# extraction (13 layers)
# ---------------------------------------------------------------------------

def extract_all_layers(
    model,
    audio: torch.Tensor,  
    max_frames: int = DEFAULT_MAX_FRAMES_LOCAL,
    rng: np.random.Generator | None = None,
    mode: str = "raw",
) -> np.ndarray:
    """
    Extract activations for all 13 layers from a single audio tensor.

    Registers a forward hook on model.model.encoder.feature_projection to
    capture the CNN embedding output (layer index 0), then collects the 12
    transformer layer outputs (indices 1–12) via model.extract_features.

    Parameters
    ----------
    model      : loaded AVES model
    audio      : (1, n_samples) float32 tensor at 16kHz
    max_frames : if the audio produces more frames than this, subsample randomly
    rng        : numpy Generator for reproducible subsampling (seed 42 default)
    mode       : "raw"  → returns (13, n_frames, 768)
                 "mean" → returns (13, 768), frame dimension collapsed

    Returns
    -------
    activations : float32 ndarray, shape (13, n_frames, 768) or (13, 768)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Hook the CNN feature projection to capture the embedding layer
    embedding_buf: list[torch.Tensor] = []

    def _hook(module, inp, out):
        embedding_buf.append(out.detach().cpu())

    handle = model.model.encoder.feature_projection.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            layer_outputs = model.extract_features(audio, layers=None)  # list of 12
    finally:
        handle.remove()

    # embedding_buf[0]: (1, n_frames, 768) → (n_frames, 768)
    embedding = embedding_buf[0].squeeze(0)                      # (n_frames, 768)
    transformer_layers = [lo.squeeze(0).cpu() for lo in layer_outputs]  # 12 x (n_frames, 768)

    # Stack: (13, n_frames, 768)
    stacked = torch.stack([embedding] + transformer_layers, dim=0).numpy().astype(np.float32)

    # Subsample frames if needed
    n_frames = stacked.shape[1]
    if n_frames > max_frames:
        idx = np.sort(rng.choice(n_frames, max_frames, replace=False))
        stacked = stacked[:, idx, :]

    if mode == "mean":
        return stacked.mean(axis=1)   # (13, 768)
    return stacked                    # (13, n_frames, 768)


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------

def parse_metadata(raw) -> dict:
    """
    Parse the NatureLM metadata field into a flat dict.

    The field is a JSON-encoded string containing taxonomic info
    (class, order, family, genus, species) and other attributes.
    Returns an empty dict on any parse failure.

    Parameters
    ----------
    raw : str | dict | None — raw metadata value from the dataset item

    Returns
    -------
    meta : dict with keys such as "class", "order", "species", "genus", "family"
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _item_to_meta(item: dict) -> dict:
    """
    Extract and flatten all relevant metadata from a NatureLM dataset item.

    Merges top-level fields (file_name, source_dataset, id, output, task)
    with flattened taxonomy from the JSON metadata string.

    Parameters
    ----------
    item : single HuggingFace dataset item

    Returns
    -------
    meta : flat dict with keys:
           file_name, source_dataset, id, output, task,
           class, order, family, genus, species
    """
    parsed = parse_metadata(item.get("metadata"))
    return {
        "file_name":      item.get("file_name") or "",
        "source_dataset": item.get("source_dataset") or "",
        "id":             item.get("id") or "",
        "output":         item.get("output") or "",
        "task":           item.get("task") or "",
        # Taxonomic fields from metadata JSON
        "class":          parsed.get("class") or "",
        "order":          parsed.get("order") or "",
        "family":         parsed.get("family") or "",
        "genus":          parsed.get("genus") or "",
        "species":        parsed.get("species") or "",
    }


# ---------------------------------------------------------------------------
# NatureLM filtering
# ---------------------------------------------------------------------------

def _matches_filters(
    meta: dict,
    source_dataset: list[str] | None,
    class_filter: list[str] | None,
    order_filter: list[str] | None,
    species_pair: tuple[str, str] | None,
) -> bool:
    """
    Return True if a metadata dict passes all active filters.

    Filters are ANDed together; None means no constraint on that field.

    Parameters
    ----------
    meta           : flat metadata dict from _item_to_meta
    source_dataset : whitelist of source_dataset values (e.g. ["xeno-canto"])
    class_filter   : whitelist of class values (e.g. ["Aves", "Mammalia"])
    order_filter   : whitelist of order values (e.g. ["Passeriformes"])
    species_pair   : (species_a, species_b) — keep only items matching either name

    Returns
    -------
    bool
    """
    if source_dataset is not None and meta["source_dataset"] not in source_dataset:
        return False
    if class_filter is not None and meta["class"] not in class_filter:
        return False
    if order_filter is not None and meta["order"] not in order_filter:
        return False
    if species_pair is not None and meta["species"] not in species_pair:
        return False
    return True


def _species_label(meta: dict, species_pair: tuple[str, str] | None, class_filter: list[str] | None) -> int:
    """
    Derive an integer class label from metadata.

    For binary species probes: label = index of species in species_pair (0 or 1).
    For class-level probes: label = index of class in class_filter.
    Falls back to 0 if no explicit mapping applies (should not occur if filters
    are applied before calling this function).

    Parameters
    ----------
    meta         : flat metadata dict
    species_pair : (species_a, species_b) or None
    class_filter : ordered list of class names, or None

    Returns
    -------
    label : int
    """
    if species_pair is not None:
        return list(species_pair).index(meta["species"])
    if class_filter is not None:
        try:
            return class_filter.index(meta["class"])
        except ValueError:
            return 0
    return 0


# ---------------------------------------------------------------------------
# Sample count check
# ---------------------------------------------------------------------------

def _check_sample_counts(
    label_counts: dict[int, int],
    label_names: list[str],
    min_samples: int,
) -> None:
    """
    Raise an informative ValueError if any class has fewer than min_samples.

    Parameters
    ----------
    label_counts : { label_int: count } collected so far
    label_names  : human-readable names matching label ints (for error message)
    min_samples  : minimum required samples per class
    """
    for label, count in label_counts.items():
        name = label_names[label] if label < len(label_names) else str(label)
        if count < min_samples:
            raise ValueError(
                f"Insufficient samples for class '{name}' (label={label}): "
                f"found {count}, need at least {min_samples}. "
                f"Reduce min_samples_per_class or broaden the filter."
            )


# ---------------------------------------------------------------------------
# Local-file dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    model,
    recordings: RecordingsDict,
    max_frames_per_recording: int = DEFAULT_MAX_FRAMES_LOCAL,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    """
    Build a per-layer probe dataset from local audio files.

    Iterates over the RECORDINGS dict (from experiment configs), extracts
    13-layer activations for each file, and assembles stacked (X, y) arrays.

    Parameters
    ----------
    model                    : loaded AVES model
    recordings               : { rec_id: (audio_path, label_int) }
    max_frames_per_recording : frame cap per recording (subsampled if exceeded)

    Returns
    -------
    dataset            : { layer_index: (X, y) }
                         X shape : (total_frames, 768)
                         y shape : (total_frames,) — integer class labels
    frames_per_recording : { rec_id: n_frames } — needed by LORO cross-validation
                           in probes/train.py to reconstruct fold boundaries
    """
    rng = np.random.default_rng(42)
    layer_X: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS_TOTAL)}
    layer_y: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS_TOTAL)}
    frames_per_recording: dict[str, int] = {}

    for rec_id, (path, label) in recordings.items():
        print(f"  {rec_id} ...", flush=True)
        audio = load_audio_file(path)
        acts = extract_all_layers(
            model, audio,
            max_frames=max_frames_per_recording,
            rng=rng,
            mode="raw",
        )  # (13, n_frames, 768)

        n_frames = acts.shape[1]
        frames_per_recording[rec_id] = n_frames

        for layer in range(NUM_LAYERS_TOTAL):
            layer_X[layer].append(acts[layer])                           # (n_frames, 768)
            layer_y[layer].append(np.full(n_frames, label, dtype=np.int32))

    dataset: dict[int, tuple[np.ndarray, np.ndarray]] = {
        layer: (
            np.concatenate(layer_X[layer], axis=0),
            np.concatenate(layer_y[layer], axis=0),
        )
        for layer in range(NUM_LAYERS_TOTAL)
    }
    return dataset, frames_per_recording


# ---------------------------------------------------------------------------
# NatureLM dataset builder
# ---------------------------------------------------------------------------

def build_naturelm_dataset(
    model,
    source_dataset: list[str] | None = None,
    class_filter: list[str] | None = None,
    order_filter: list[str] | None = None,
    species_pair: tuple[str, str] | None = None,
    label_names: list[str] | None = None,
    min_samples_per_class: int = MIN_SAMPLES_DEFAULT,
    max_samples_per_class: int | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    mode: str = "mean",
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[dict]]:
    """
    Build a per-layer probe dataset by streaming NatureLM-audio-training.

    Filters the stream by source_dataset, taxonomic class, order, or species
    pair; extracts AVES activations; and assembles (X, y) arrays per layer.
    Raises a ValueError if any class falls below min_samples_per_class.

    Parameters
    ----------
    model                 : loaded AVES model
    source_dataset        : whitelist of source_dataset values, 
    class_filter          : ordered list of taxonomic class names for multiclass probes,
                            e.g. ["Aves", "Mammalia", "Amphibia"]. Label = list index.
                            None = no filter.
    order_filter          : whitelist of taxonomic order names, e.g. ["Passeriformes"].
                            None = no filter.
    species_pair          : (species_a, species_b) for binary species probes.
                            Items matching species_a get label 0, species_b get label 1.
                            None = no filter.
    label_names           : human-readable class names for error messages and evaluation.
                            Inferred from class_filter or species_pair if not provided.
    min_samples_per_class : raise ValueError if any class has fewer samples after streaming.
                            Default 500.
    max_samples_per_class : stop collecting for a class once it reaches this count.
                            None = no cap (collect everything that passes filters).
    max_frames            : maximum frames to keep per audio item (default 513 ≈ 10s).
    mode                  : "mean" → X shape (n_items, 768) per layer (mean-pooled).
                            "raw"  → X shape (n_frames_total, 768) per layer
                                     (all frames concatenated, labels repeated per frame).

    Returns
    -------
    dataset       : { layer_index: (X, y) }
                    X shape: (n_items, 768)    if mode="mean"
                             (n_frames, 768)   if mode="raw"
                    y shape: (n_items,) or (n_frames,) — integer class labels
    metadata_list : list of flat metadata dicts (one per collected item),
                    in the same order as the rows in X/y.
    """
    from datasets import load_dataset

    # Infer label_names if not provided
    if label_names is None:
        if species_pair is not None:
            label_names = list(species_pair)
        elif class_filter is not None:
            label_names = list(class_filter)
        else:
            label_names = []

    rng = np.random.default_rng(42)

    # Accumulators: one list per layer
    layer_X: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS_TOTAL)}
    layer_y: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS_TOTAL)}
    metadata_list: list[dict] = []
    label_counts: dict[int, int] = {}

    print(
        f"Streaming NatureLM-audio-training "
        f"(source={source_dataset}, class={class_filter}, "
        f"order={order_filter}, species_pair={species_pair}) ...",
        flush=True,
    )

    ds = load_dataset(DATASET_REPO, split="train", streaming=True)

    for item in ds:
        meta = _item_to_meta(item)

        if not _matches_filters(meta, source_dataset, class_filter, order_filter, species_pair):
            continue

        label = _species_label(meta, species_pair, class_filter)

        # Enforce per-class cap
        if max_samples_per_class is not None:
            if label_counts.get(label, 0) >= max_samples_per_class:
                continue

        # Extract activations
        audio = _audio_from_hf_item(item)
        acts = extract_all_layers(model, audio, max_frames=max_frames, rng=rng, mode=mode)
        # mode="mean": (13, 768)   mode="raw": (13, n_frames, 768)

        if mode == "mean":
            for layer in range(NUM_LAYERS_TOTAL):
                layer_X[layer].append(acts[layer])                            # (768,)
                layer_y[layer].append(np.array([label], dtype=np.int32))
        else:
            n_frames = acts.shape[1]
            for layer in range(NUM_LAYERS_TOTAL):
                layer_X[layer].append(acts[layer])                            # (n_frames, 768)
                layer_y[layer].append(np.full(n_frames, label, dtype=np.int32))

        label_counts[label] = label_counts.get(label, 0) + 1
        metadata_list.append(meta)

        total = sum(label_counts.values())
        if total % 100 == 0:
            counts_str = ", ".join(
                f"{label_names[l] if l < len(label_names) else l}={c}"
                for l, c in sorted(label_counts.items())
            )
            print(f"  Collected {total} items ({counts_str})", flush=True)

        # Stop early if all classes have hit the cap
        if max_samples_per_class is not None:
            if all(label_counts.get(l, 0) >= max_samples_per_class for l in range(len(label_names))):
                print(f"  All classes reached cap of {max_samples_per_class}. Stopping.", flush=True)
                break

    print(
        f"Done. Total items: {len(metadata_list)}  "
        f"({', '.join(f'{label_names[l] if l < len(label_names) else l}={c}' for l, c in sorted(label_counts.items()))})",
        flush=True,
    )

    # Sample count check — raise before building arrays so error is clear
    _check_sample_counts(label_counts, label_names, min_samples_per_class)

    # Stack into arrays
    if mode == "mean":
        dataset: dict[int, tuple[np.ndarray, np.ndarray]] = {
            layer: (
                np.stack(layer_X[layer], axis=0),            # (n_items, 768)
                np.concatenate(layer_y[layer], axis=0),       # (n_items,)
            )
            for layer in range(NUM_LAYERS_TOTAL)
        }
    else:
        dataset = {
            layer: (
                np.concatenate(layer_X[layer], axis=0),       # (n_frames_total, 768)
                np.concatenate(layer_y[layer], axis=0),        # (n_frames_total,)
            )
            for layer in range(NUM_LAYERS_TOTAL)
        }

    return dataset, metadata_list
