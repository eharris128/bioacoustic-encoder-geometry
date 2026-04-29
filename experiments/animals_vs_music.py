"""
experiments/animals_vs_music.py — Probe: Animal vocalizations vs Music.

Binary classification: does AVES linearly separate animal sounds from
musical instruments?

Labels:
    0 = animal  (NatureLM xeno-canto stream, up to 200 samples, mean-pooled)
    1 = music   (local files — all MP3s in audio/music-misc/ + audio/violin/, auto-discovered)

Note: requires a stable HuggingFace connection. Run on Lambda for best results.
      For offline use, see git history for the local-only version.

Run:
    python -W ignore experiments/animals_vs_music.py
"""

from __future__ import annotations

import glob
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from data.loader import load_model, build_dataset, build_naturelm_dataset
from probes.train import train_all_layers
from probes.evaluate import run_evaluation

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

EXPERIMENT_NAME = "animals_vs_music"
LABEL_NAMES     = ["animal", "music"]
RESULTS_DIR     = "results/probe-output/animals_vs_music"

N_ANIMAL_SAMPLES = 200  # max animal recordings streamed from NatureLM

# ---------------------------------------------------------------------------
# Music recordings — auto-populated from audio/music-misc/ and audio/violin/
# Sorted for reproducibility; label 1 = music.
# ---------------------------------------------------------------------------

MUSIC_RECORDINGS: dict[str, tuple[str, int]] = {
    f"music_{i:03d}": (path, 1)
    for i, path in enumerate(sorted(
        glob.glob("audio/music-misc/*.mp3")
        + glob.glob("audio/violin/*.mp3")
    ))
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    print("=== Animals vs Music probe ===\n")

    print("Loading model...")
    model = load_model("esp_aves2_eat_all")

    # 1. Stream animal samples from NatureLM (mean-pooled: one vector per recording)
    print(f"\nStreaming up to {N_ANIMAL_SAMPLES} animal samples from NatureLM...")
    dataset_animal, meta_animal = build_naturelm_dataset(
        model,
        source_dataset=["xeno-canto"],
        label_names=["animal"],
        min_samples_per_class=20,
        max_samples_per_class=N_ANIMAL_SAMPLES,
        mode="mean",
    )

    # 2. Build music dataset from local files (raw frames, then mean-pool per recording)
    print(f"\nLoading {len(MUSIC_RECORDINGS)} local music recordings...")
    dataset_music, frames_music = build_dataset(model, MUSIC_RECORDINGS)

    # 3. Merge: both classes represented as one mean-pooled vector per recording
    print("\nMerging datasets...")
    dataset: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for layer in range(13):
        X_animal, _ = dataset_animal[layer]                          # (n_animal, 768)
        X_music_raw, _ = dataset_music[layer]                        # (n_frames, 768)
        X_music = _mean_pool_by_recording(X_music_raw, frames_music) # (n_music, 768)

        X = np.concatenate([X_animal, X_music], axis=0)
        y = np.concatenate([
            np.zeros(len(X_animal), dtype=np.int32),
            np.ones(len(X_music),  dtype=np.int32),
        ])
        dataset[layer] = (X, y)

    # 4. Recording IDs and frame counts for LORO (each entry = 1 mean-pooled vector)
    animal_ids = [m["id"] for m in meta_animal]
    music_ids  = list(MUSIC_RECORDINGS.keys())
    all_ids    = animal_ids + music_ids
    frames_per_recording = {rid: 1 for rid in all_ids}

    # 5. Train LORO probes across all layers
    print("\nRunning LORO cross-validation...")
    results = train_all_layers(
        dataset=dataset,
        recording_ids=all_ids,
        frames_per_recording=frames_per_recording,
    )

    # 6. Evaluate and save plots
    run_evaluation(
        accuracy_per_layer=results["accuracy_per_layer"],
        dataset=dataset,
        chance_level=results["chance_level"],
        label_names=LABEL_NAMES,
        experiment_name=EXPERIMENT_NAME,
        results_dir=RESULTS_DIR,
    )
    print(f"\nDone. Results saved to {RESULTS_DIR}/")


def _mean_pool_by_recording(
    X: np.ndarray,
    frames_per_recording: dict[str, int],
) -> np.ndarray:
    """Mean-pool frame-level activations down to one vector per recording."""
    pooled = []
    idx = 0
    for n_frames in frames_per_recording.values():
        pooled.append(X[idx: idx + n_frames].mean(axis=0))
        idx += n_frames
    return np.stack(pooled, axis=0)


if __name__ == "__main__":
    run()
