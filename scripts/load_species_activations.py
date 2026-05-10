"""
scripts/load_species_activations.py — Load pre-extracted species activations for probing.

Reads .npy activation files saved by extract_species_activations.py and returns
them in the format expected by probes/train.py::train_all_layers().

Primary function: load_species_pair()
Helper function:  to_probe_dataset()

Example usage:
    from scripts.load_species_activations import load_species_pair, to_probe_dataset
    from probes.train import train_all_layers

    # Load both species (mean-pooled, one vector per recording per layer)
    X, y, recording_ids = load_species_pair(
        "./activations/species-parus-major-eat-all",
        "./activations/species-parus-caeruleus-eat-all",
        mode="mean",
    )
    # X shape: (n_total, 13, 768)

    # Convert to the {layer: (X_layer, y)} dict that train_all_layers expects
    dataset = to_probe_dataset(X, y)
    frames_per_recording = {rid: 1 for rid in recording_ids}

    results = train_all_layers(dataset, recording_ids, frames_per_recording)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_species_pair(
    dir_a: str | Path,
    dir_b: str | Path,
    mode: str = "mean",
    max_recordings: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load activations for two species from disk and return a combined array.

    Reads all rec_*.npy files from each directory (sorted by name, so order
    matches the extraction order). Files without a matching .json sidecar are
    still loaded — the sidecar is optional.

    Parameters
    ----------
    dir_a            : path to species A activation directory
    dir_b            : path to species B activation directory
    mode             : "mean" → mean-pool patch tokens → shape (n_recordings, 13, 768)
                       "raw"  → keep all patch tokens → shape (n_recordings, 13, n_frames, 768)
                                only valid if all recordings have the same n_frames
    max_recordings   : if set, cap each species at this many recordings (balanced)

    Returns
    -------
    X              : float32 ndarray
                     mode="mean": (n_total, 13, 768)
                     mode="raw":  (n_total, 13, n_frames, 768)
    y              : (n_total,) int32 — 0 = species_a, 1 = species_b
    recording_ids  : list of str, length n_total — "<dir_name>/<stem>"
                     suitable as LORO recording identifiers

    Raises
    ------
    FileNotFoundError  if a directory has no rec_*.npy files
    ValueError         if mode="raw" and recordings have inconsistent n_frames
    """
    arrays_a, ids_a = _load_dir(Path(dir_a), max_recordings, mode)
    arrays_b, ids_b = _load_dir(Path(dir_b), max_recordings, mode)

    if not arrays_a:
        raise FileNotFoundError(f"No rec_*.npy files found in {dir_a}")
    if not arrays_b:
        raise FileNotFoundError(f"No rec_*.npy files found in {dir_b}")

    all_arrays = arrays_a + arrays_b
    recording_ids = ids_a + ids_b
    y = np.array([0] * len(arrays_a) + [1] * len(arrays_b), dtype=np.int32)

    if mode == "raw":
        # All recordings must share the same n_frames for np.stack
        shapes = {a.shape for a in all_arrays}
        if len(shapes) > 1:
            raise ValueError(
                f"mode='raw' requires all recordings to have the same shape, "
                f"but found {len(shapes)} distinct shapes: {shapes}. "
                f"Use mode='mean' instead, or cap max_frames when extracting."
            )
        X = np.stack(all_arrays, axis=0).astype(np.float32)  # (n, 13, n_frames, 768)
    else:
        X = np.stack(all_arrays, axis=0).astype(np.float32)  # (n, 13, 768)

    return X, y, recording_ids


def to_probe_dataset(
    X: np.ndarray,
    y: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Convert mean-pooled activations to the dataset dict expected by train_all_layers().

    Parameters
    ----------
    X : (n_recordings, 13, 768) float32 — output of load_species_pair(mode="mean")
    y : (n_recordings,) int32 labels

    Returns
    -------
    dataset : { layer_index: (X_layer, y) }
              X_layer shape: (n_recordings, 768)

    Usage with train_all_layers:
        dataset = to_probe_dataset(X, y)
        frames_per_recording = {rid: 1 for rid in recording_ids}
        train_all_layers(dataset, recording_ids, frames_per_recording)
    """
    if X.ndim != 3:
        raise ValueError(
            f"Expected X.ndim == 3 (n_recordings, n_layers, 768), got shape {X.shape}. "
            "Use mode='mean' in load_species_pair."
        )
    n_layers = X.shape[1]
    return {layer: (X[:, layer, :], y) for layer in range(n_layers)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dir(
    directory: Path,
    max_recordings: int | None,
    mode: str,
) -> tuple[list[np.ndarray], list[str]]:
    """Load sorted rec_*.npy files from a directory, applying mean-pool if requested."""
    npy_files = sorted(directory.glob("rec_*.npy"))
    if max_recordings is not None:
        npy_files = npy_files[:max_recordings]

    arrays: list[np.ndarray] = []
    ids: list[str] = []

    for npy_path in npy_files:
        arr = np.load(npy_path)  # (13, n_frames, 768)
        if mode == "mean":
            arr = arr.mean(axis=1)  # (13, 768)
        arrays.append(arr)
        ids.append(f"{directory.name}/{npy_path.stem}")

    return arrays, ids


def load_manifest(directory: str | Path) -> dict:
    """Load manifest.json from an activation directory."""
    path = Path(directory) / "manifest.json"
    with open(path) as f:
        return json.load(f)


def load_sidecar(npy_path: str | Path) -> dict:
    """Load the .json sidecar for a given rec_*.npy file."""
    p = Path(npy_path).with_suffix(".json")
    with open(p) as f:
        return json.load(f)
