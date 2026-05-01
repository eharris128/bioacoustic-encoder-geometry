"""
data/loader.py — Load and label audio subsets for probe experiments.

Two interfaces:

1. Local files (experiments/animals_vs_music.py, music_vs_speech.py, species*.py)
   - build_dataset(model, recordings) → {layer: (X, y)}, frames_per_recording

2. NatureLM HuggingFace streaming (large-scale probe experiments)
   - build_naturelm_dataset(model, ...) → {layer: (X, y)}, metadata_list

Both return the same {layer: (X, y)} format consumed by probes/train.py.

Models
------
All four supported models use the EAT (Environmental Audio Transformer) architecture
loaded via the avex library. Load with load_model(name):
    "esp_aves2_eat_all"           — SSL pretrained, all data
    "esp_aves2_eat_bio"           — SSL pretrained, bio-only data
    "esp_aves2_sl_eat_all_ssl_all" — supervised fine-tune, all data
    "esp_aves2_sl_eat_bio_ssl_all" — supervised fine-tune, bio data

Activation layout
-----------------
13 layers total: index 0 = local_encoder output (patch embeddings, 512 patches),
                 indices 1–12 = transformer blocks 0–11 (CLS token stripped, 512 patches).
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
NUM_LAYERS_TOTAL = 13          # local_encoder (index 0) + transformer blocks 1–12
DEFAULT_MAX_FRAMES = 513       # per item for NatureLM (≈10s at 50fps)
DEFAULT_MAX_FRAMES_LOCAL = 3000
MIN_SAMPLES_DEFAULT = 500
DATASET_REPO = "EarthSpeciesProject/NatureLM-audio-training"

# Type alias used by all experiment configs
RecordingsDict = dict[str, tuple[str, int]]

# ---------------------------------------------------------------------------
# EAT model registry
# ---------------------------------------------------------------------------
# All four models share the same HuBERT-like EAT architecture (12 transformer
# blocks, 768-dim) and hook paths. Checkpoints are auto-downloaded from
# HuggingFace on first use by avex — no local download needed.

EAT_MODELS: set[str] = {
    "esp_aves2_eat_all",
    "esp_aves2_eat_bio",
    "esp_aves2_sl_eat_all_ssl_all",
    "esp_aves2_sl_eat_bio_ssl_all",
}

DEFAULT_MODEL = "esp_aves2_eat_all"

# Hook paths registered on every EAT model (13 layers, indices 0–12):
#   index 0  = backbone.model.local_encoder (patch projection / embedding)
#   index 1+ = backbone.model.blocks.{i}   (transformer block i, 0-indexed)
_EAT_LAYER_NAMES: list[str] = (
    ["backbone.model.local_encoder"]
    + [f"backbone.model.blocks.{i}" for i in range(NUM_LAYERS_TRANSFORMER)]
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
) -> object:
    """
    Load an EAT model from the avex registry and register forward hooks.

    Hooks are registered on all 13 layers (local_encoder + 12 transformer
    blocks) so that extract_all_layers can capture intermediate activations.
    The model is set to eval mode before return.

    Parameters
    ----------
    model_name : one of EAT_MODELS. Default "esp_aves2_eat_all".
    device     : "cpu" or "cuda"

    Returns
    -------
    model : avex EATHFModel with hooks registered

    Raises
    ------
    ValueError  if model_name is not in EAT_MODELS
    """
    if model_name not in EAT_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported EAT models: {sorted(EAT_MODELS)}"
        )
    from avex import load_model as avex_load_model
    model = avex_load_model(model_name, device=device, return_features_only=True)
    model.eval()
    model.register_hooks_for_layers(_EAT_LAYER_NAMES)
    return model


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio_file(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load a local audio file (WAV or MP3) as a mono float32 tensor at target_sr.

    Tries soundfile first (WAV/FLAC); falls back to torchaudio for MP3.

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
        import librosa
        data, _ = librosa.load(path, sr=target_sr, mono=True)
        return torch.from_numpy(data).unsqueeze(0)


def _audio_from_hf_item(item: dict, target_sr: int = 16000) -> torch.Tensor:
    """
    Convert a HuggingFace audio item (decode=False) to a (1, n_samples) float32 tensor.

    Expects item["audio"] = {"bytes": bytes, "path": str} — the raw encoded file.
    Decodes via soundfile (BytesIO) to avoid torchcodec / FFmpeg version issues.
    Falls back to torchaudio for formats soundfile can't handle (e.g. MP3).
    """
    import io
    audio_field = item["audio"]
    raw_bytes = audio_field.get("bytes") or b""
    path = audio_field.get("path") or ""

    buf = io.BytesIO(raw_bytes) if raw_bytes else None

    try:
        src = buf if buf is not None else path
        data, sr = sf.read(src, always_2d=True)
        data = data.mean(axis=1)
        if sr != target_sr:
            n_out = int(round(len(data) * target_sr / sr))
            data = scipy_signal.resample(data, n_out)
        return torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
    except Exception:
        import torchaudio
        if buf is not None:
            buf.seek(0)
            waveform, sr = torchaudio.load(buf)
        else:
            waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform.float()


# ---------------------------------------------------------------------------
# Core activation extraction (13 layers)
# ---------------------------------------------------------------------------

def extract_all_layers(
    model,
    audio: torch.Tensor,
    max_frames: int = DEFAULT_MAX_FRAMES_LOCAL,
    rng: np.random.Generator | None = None,
    mode: str = "raw",
) -> np.ndarray:
    """
    Extract activations for all 13 EAT layers from a single audio tensor.

    avex handles mel spectrogram conversion internally. The local_encoder
    (index 0) produces 512 patch embeddings; each transformer block (indices
    1–12) produces 513 tokens. The CLS token (position 0) is stripped from
    transformer block outputs so all 13 layers yield the same (512, 768) shape.

    Parameters
    ----------
    model      : avex EATHFModel loaded via load_model
    audio      : (1, n_samples) float32 tensor at 16kHz
    max_frames : patch cap per audio; subsampled randomly if exceeded
    rng        : numpy Generator for reproducible subsampling (seed 42 default)
    mode       : "raw"  → returns (13, n_patches, 768)
                 "mean" → returns (13, 768), patches mean-pooled

    Returns
    -------
    activations : float32 ndarray, shape (13, n_patches, 768) or (13, 768)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # avex expects (B, T); audio is already (1, T)
    with torch.no_grad():
        layer_tensors = model.extract_embeddings(audio, aggregation="none")
    # layer_tensors: list of 13 tensors
    #   index 0  : (1, 512, 768) — local_encoder patches
    #   index 1+ : (1, 513, 768) — transformer block, position 0 is CLS token

    layers: list[np.ndarray] = []
    for i, t in enumerate(layer_tensors):
        arr = t.squeeze(0).cpu().float().numpy()  # (512 or 513, 768)
        if i > 0:
            arr = arr[1:]                         # strip CLS → (512, 768)
        layers.append(arr)

    stacked = np.stack(layers, axis=0)            # (13, 512, 768)

    n_patches = stacked.shape[1]
    if n_patches > max_frames:
        idx = np.sort(rng.choice(n_patches, max_frames, replace=False))
        stacked = stacked[:, idx, :]

    if mode == "mean":
        return stacked.mean(axis=1)   # (13, 768)
    return stacked                    # (13, n_patches, 768)


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------

def parse_metadata(raw) -> dict:
    """
    Parse the NatureLM metadata field into a flat dict.

    Returns an empty dict on any parse failure.
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
    """Extract and flatten all relevant metadata from a NatureLM dataset item."""
    parsed = parse_metadata(item.get("metadata"))
    return {
        "file_name":      item.get("file_name") or "",
        "source_dataset": item.get("source_dataset") or "",
        "id":             item.get("id") or "",
        "output":         item.get("output") or "",
        "task":           item.get("task") or "",
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
    """Return True if a metadata dict passes all active filters (ANDed)."""
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
    """Derive an integer class label from metadata."""
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
    """Raise ValueError if any class has fewer than min_samples."""
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

    Parameters
    ----------
    model                    : EAT model loaded via load_model
    recordings               : { rec_id: (audio_path, label_int) }
    max_frames_per_recording : patch cap per recording (subsampled if exceeded)

    Returns
    -------
    dataset              : { layer_index: (X, y) }
                           X shape : (total_frames, 768)
                           y shape : (total_frames,) — integer class labels
    frames_per_recording : { rec_id: n_frames } — needed by LORO cross-validation
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
            layer_X[layer].append(acts[layer])
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

    Parameters
    ----------
    model                 : EAT model loaded via load_model
    source_dataset        : whitelist of source_dataset values
    class_filter          : ordered list of taxonomic class names (label = list index)
    order_filter          : whitelist of taxonomic order names
    species_pair          : (species_a, species_b) for binary probes
    label_names           : human-readable class names; inferred from class_filter
                            or species_pair if not provided
    min_samples_per_class : raise ValueError if any class falls below this
    max_samples_per_class : stop collecting for a class once it hits this count
    max_frames            : patch cap per audio item (default 513 ≈ 10s)
    mode                  : "mean" → X shape (n_items, 768) per layer
                            "raw"  → X shape (n_frames_total, 768) per layer

    Returns
    -------
    dataset       : { layer_index: (X, y) }
    metadata_list : list of flat metadata dicts (one per collected item)
    """
    from datasets import load_dataset, Audio

    if label_names is None:
        if species_pair is not None:
            label_names = list(species_pair)
        elif class_filter is not None:
            label_names = list(class_filter)
        else:
            label_names = []

    rng = np.random.default_rng(42)

    layer_X: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS_TOTAL)}
    layer_y: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS_TOTAL)}
    metadata_list: list[dict] = []
    label_counts: dict[int, int] = {}

    import os
    import random
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_tree

    print(
        f"Loading NatureLM-audio-training via shard download "
        f"(source={source_dataset}, class={class_filter}, "
        f"order={order_filter}, species_pair={species_pair}) ...",
        flush=True,
    )

    cache_dir = ".naturelm_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Get list of all parquet shards (cached after first call)
    shard_list_path = os.path.join(cache_dir, "shard_list.txt")
    if os.path.exists(shard_list_path):
        with open(shard_list_path) as f:
            all_shards = [line.strip() for line in f if line.strip()]
    else:
        print("Fetching shard list from HuggingFace (one-time)...", flush=True)
        items = list(list_repo_tree(DATASET_REPO, repo_type="dataset", recursive=True))
        all_shards = sorted(
            getattr(it, "path", "") for it in items
            if getattr(it, "path", "").endswith(".parquet")
        )
        with open(shard_list_path, "w") as f:
            f.write("\n".join(all_shards))
        print(f"  Found {len(all_shards)} shards.", flush=True)

    # Shuffle for diverse coverage across species/sources
    random.seed(42)
    random.shuffle(all_shards)

    for shard_path in all_shards:
        # Stop early if all classes are filled
        if max_samples_per_class is not None and label_names:
            if all(label_counts.get(l, 0) >= max_samples_per_class for l in range(len(label_names))):
                print(f"  All classes reached cap of {max_samples_per_class}. Stopping.", flush=True)
                break

        # Download shard (cached by huggingface_hub)
        try:
            local_path = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=shard_path,
                repo_type="dataset",
                local_dir=cache_dir,
            )
            df = pd.read_parquet(local_path)
        except Exception:
            continue  # skip corrupt or missing shard

        # Fast pandas pre-filter before touching audio
        if source_dataset is not None:
            df = df[df["source_dataset"].isin(source_dataset)]
        if df.empty:
            continue

        for _, row in df.iterrows():
            item = row.to_dict()
            try:
                meta = _item_to_meta(item)
            except Exception:
                continue

            if not _matches_filters(meta, source_dataset, class_filter, order_filter, species_pair):
                continue

            label = _species_label(meta, species_pair, class_filter)

            if max_samples_per_class is not None:
                if label_counts.get(label, 0) >= max_samples_per_class:
                    continue

            try:
                audio = _audio_from_hf_item(item)
                acts = extract_all_layers(model, audio, max_frames=max_frames, rng=rng, mode=mode)
            except Exception:
                continue

        if mode == "mean":
            for layer in range(NUM_LAYERS_TOTAL):
                layer_X[layer].append(acts[layer])
                layer_y[layer].append(np.array([label], dtype=np.int32))
        else:
            n_frames = acts.shape[1]
            for layer in range(NUM_LAYERS_TOTAL):
                layer_X[layer].append(acts[layer])
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

        if max_samples_per_class is not None:
            if all(label_counts.get(l, 0) >= max_samples_per_class for l in range(len(label_names))):
                print(f"  All classes reached cap of {max_samples_per_class}. Stopping.", flush=True)
                break

    print(
        f"Done. Total items: {len(metadata_list)}  "
        f"({', '.join(f'{label_names[l] if l < len(label_names) else l}={c}' for l, c in sorted(label_counts.items()))})",
        flush=True,
    )

    _check_sample_counts(label_counts, label_names, min_samples_per_class)

    if mode == "mean":
        dataset: dict[int, tuple[np.ndarray, np.ndarray]] = {
            layer: (
                np.stack(layer_X[layer], axis=0),
                np.concatenate(layer_y[layer], axis=0),
            )
            for layer in range(NUM_LAYERS_TOTAL)
        }
    else:
        dataset = {
            layer: (
                np.concatenate(layer_X[layer], axis=0),
                np.concatenate(layer_y[layer], axis=0),
            )
            for layer in range(NUM_LAYERS_TOTAL)
        }

    return dataset, metadata_list


# ---------------------------------------------------------------------------
# Xeno-canto direct downloader (no HuggingFace dependency)
# ---------------------------------------------------------------------------

XENOCANTO_API = "https://xeno-canto.org/api/3/recordings"
XENOCANTO_CACHE = ".xenocanto_cache"


def build_xenocanto_dataset(
    model,
    species_pair: tuple[str, str],
    label_names: list[str] | None = None,
    n_per_species: int = 100,
    max_frames: int = DEFAULT_MAX_FRAMES,
    mode: str = "mean",
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[dict]]:
    """
    Build a per-layer probe dataset by querying xeno-canto API directly.
    No HuggingFace dependency — downloads MP3s to .xenocanto_cache/.

    Parameters
    ----------
    model         : EAT model loaded via load_model
    species_pair  : (species_a, species_b) scientific names, e.g.
                    ("Pyrrhula pyrrhula", "Coccothraustes coccothraustes")
    label_names   : human-readable names; defaults to species_pair
    n_per_species : max recordings to download per species
    mode          : "mean" → one 768-dim vector per recording
    """
    import os, requests
    from sklearn.decomposition import PCA  # noqa — only for type reference below

    if label_names is None:
        label_names = list(species_pair)

    os.makedirs(XENOCANTO_CACHE, exist_ok=True)
    rng = np.random.default_rng(42)

    layer_X: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS_TOTAL)}
    layer_y: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS_TOTAL)}
    metadata_list: list[dict] = []

    for label, species in enumerate(species_pair):
        print(f"\nFetching {n_per_species} recordings of {species} from xeno-canto...", flush=True)
        urls = _xenocanto_fetch_urls(species, n_per_species)
        print(f"  Found {len(urls)} recordings online.", flush=True)

        collected = 0
        for rec_id, url, filename in urls:
            if collected >= n_per_species:
                break
            local_path = os.path.join(XENOCANTO_CACHE, filename)
            if not os.path.exists(local_path):
                try:
                    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        f.write(r.content)
                except Exception:
                    continue
            try:
                audio = load_audio_file(local_path)
                acts = extract_all_layers(model, audio, max_frames=max_frames, rng=rng, mode=mode)
            except Exception:
                continue

            if mode == "mean":
                for layer in range(NUM_LAYERS_TOTAL):
                    layer_X[layer].append(acts[layer])
                    layer_y[layer].append(np.array([label], dtype=np.int32))
            else:
                n_frames = acts.shape[1]
                for layer in range(NUM_LAYERS_TOTAL):
                    layer_X[layer].append(acts[layer])
                    layer_y[layer].append(np.full(n_frames, label, dtype=np.int32))

            metadata_list.append({"id": rec_id, "species": species, "label": label})
            collected += 1
            print(f"  {label_names[label]} {collected}/{n_per_species} ...", flush=True)

    if mode == "mean":
        dataset: dict[int, tuple[np.ndarray, np.ndarray]] = {
            layer: (np.stack(layer_X[layer], axis=0), np.concatenate(layer_y[layer]))
            for layer in range(NUM_LAYERS_TOTAL)
        }
    else:
        dataset = {
            layer: (np.concatenate(layer_X[layer], axis=0), np.concatenate(layer_y[layer]))
            for layer in range(NUM_LAYERS_TOTAL)
        }

    return dataset, metadata_list


def _xenocanto_fetch_urls(species: str, n: int) -> list[tuple[str, str, str]]:
    """Query xeno-canto API v3 and return list of (rec_id, download_url, filename)."""
    import os, requests
    api_key = os.environ.get("XENOCANTO_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "Set XENOCANTO_API_KEY environment variable. "
            "Get your key at https://xeno-canto.org/account"
        )
    results = []
    page = 1
    # v3 requires tag syntax: gen:Pyrrhula sp:pyrrhula
    parts = species.strip().split()
    if len(parts) >= 2:
        query = f"gen:{parts[0]}+sp:{parts[1]}"
    else:
        query = f"sp:{parts[0]}"
    while len(results) < n:
        try:
            resp = requests.get(
                f"{XENOCANTO_API}?query={query}&page={page}&key={api_key}",
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            data = resp.json()
        except Exception:
            break
        if "error" in data:
            raise RuntimeError(f"xeno-canto API error: {data['message']}")
        for rec in data.get("recordings", []):
            rec_id = rec.get("id", "")
            url = rec.get("file", "")
            filename = rec.get("file-name") or f"XC{rec_id}.mp3"
            if url:
                results.append((rec_id, url, filename))
            if len(results) >= n:
                break
        if page >= int(data.get("numPages", 1)):
            break
        page += 1
    return results[:n]


# ---------------------------------------------------------------------------
# Noise subspace computation and projection
# ---------------------------------------------------------------------------

def compute_noise_subspace(
    model,
    audio_paths: list[str],
    n_components: int = 3,
    snr_levels_db: list[float] | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    n_recordings: int = 8,
) -> dict[int, np.ndarray]:
    """
    Compute the noise subspace per layer by running an SNR sweep on a subset
    of recordings and taking the top n_components PCA directions.

    Returns { layer_index: np.ndarray of shape (768, n_components) }
    — the columns are the noise basis vectors to project out.
    """
    from sklearn.decomposition import PCA

    if snr_levels_db is None:
        snr_levels_db = [20.0, 10.0, 5.0, 1.0]

    rng = np.random.default_rng(42)
    paths = list(audio_paths)
    if len(paths) > n_recordings:
        idx = rng.choice(len(paths), n_recordings, replace=False)
        paths = [paths[i] for i in sorted(idx)]

    print(f"\nComputing noise subspace ({n_components} components) from {len(paths)} recordings "
          f"× {len(snr_levels_db)} SNR levels...", flush=True)

    layer_acts: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS_TOTAL)}

    for path in paths:
        try:
            audio = load_audio_file(path)
        except Exception:
            continue
        audio_np = audio.squeeze(0).numpy()
        signal_power = float(np.mean(audio_np ** 2)) + 1e-12

        for snr_db in snr_levels_db:
            noise_power = signal_power / (10.0 ** (snr_db / 10.0))
            noise = rng.normal(0.0, np.sqrt(noise_power), audio_np.shape).astype(np.float32)
            noisy = torch.from_numpy(audio_np + noise).unsqueeze(0)
            try:
                acts = extract_all_layers(model, noisy, max_frames=max_frames, rng=rng, mode="mean")
                for layer in range(NUM_LAYERS_TOTAL):
                    layer_acts[layer].append(acts[layer])
            except Exception:
                continue

    subspaces: dict[int, np.ndarray] = {}
    for layer in range(NUM_LAYERS_TOTAL):
        X = np.stack(layer_acts[layer], axis=0)  # (n_rec * n_snr, 768)
        if X.shape[0] < n_components:
            subspaces[layer] = np.zeros((768, n_components), dtype=np.float32)
            continue
        pca = PCA(n_components=n_components)
        pca.fit(X)
        subspaces[layer] = pca.components_.T.astype(np.float32)  # (768, n_components)
        var = pca.explained_variance_ratio_
        print(f"  Layer {layer:2d}: noise PC1={var[0]:.3f}, PC2={var[1]:.3f}, PC3={var[2]:.3f}", flush=True)

    return subspaces


def project_out_subspace(
    dataset: dict[int, tuple[np.ndarray, np.ndarray]],
    subspaces: dict[int, np.ndarray],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Project the noise subspace out of every layer's activation matrix.

    For each layer: X_clean = X - X @ Q @ Q.T
    where Q is the (768, n_components) noise basis.
    """
    cleaned: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for layer, (X, y) in dataset.items():
        Q = subspaces[layer]             # (768, n_components)
        X_proj = X @ Q @ Q.T            # projection onto noise subspace
        cleaned[layer] = (X - X_proj, y)
    return cleaned
