"""
experiments/animals_vs_music.py — Probe: Animal vocalizations vs Music.

Binary classification: does AVES linearly separate animal sounds from
musical instruments?

Labels:
    0 = animal  (local files — bullfinch, hawfinch, helmeted guinea fowl, auto-discovered)
    1 = music   (local files — all MP3s in audio/music-misc/ + audio/violin/, auto-discovered)

Run (standard):
    python -W ignore experiments/animals_vs_music.py

Run with noise subspace subtracted (3 PCs):
    python -W ignore experiments/animals_vs_music.py --subtract-noise
"""

from __future__ import annotations

import argparse
import glob
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from data.loader import (
    load_model, build_dataset,
    compute_noise_subspace, project_out_subspace,
)
from probes.train import train_all_layers
from probes.evaluate import run_evaluation

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

LABEL_NAMES = ["animal", "music"]
RESULTS_DIR = "results/probe-output/animals_vs_music"

_SKIP = {"XC1086809.mp3", "XC657517.mp3"}  # 35MB outlier + corrupted

ANIMAL_RECORDINGS: dict[str, tuple[str, int]] = {
    f"animal_{i:03d}": (path, 0)
    for i, path in enumerate(sorted(
        glob.glob("audio/bullfinch/*.mp3")
        + glob.glob("audio/hawfinch/*.mp3")
        + glob.glob("audio/helmeted-guinea-fowl/*.mp3")
    ))
    if os.path.basename(path) not in _SKIP
}

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

def run(subtract_noise: bool = False) -> None:
    experiment_name = "animals_vs_music_noise_subtracted" if subtract_noise else "animals_vs_music"

    print("=== Animals vs Music probe ===\n")
    print(f"Animal recordings: {len(ANIMAL_RECORDINGS)}")
    print(f"Music recordings:  {len(MUSIC_RECORDINGS)}")
    if subtract_noise:
        print("Noise subtraction: ON (3-component subspace)\n")

    print("\nLoading model...")
    model = load_model("esp_aves2_eat_all")

    # 1. Build both datasets from local files (frame-level, then mean-pool per recording)
    print(f"\nProcessing {len(ANIMAL_RECORDINGS)} animal recordings...")
    dataset_animal, frames_animal = build_dataset(model, ANIMAL_RECORDINGS)

    print(f"\nProcessing {len(MUSIC_RECORDINGS)} music recordings...")
    dataset_music, frames_music = build_dataset(model, MUSIC_RECORDINGS)

    # 2. Merge: mean-pool frames -> one vector per recording, then concatenate
    print("\nMerging datasets...")
    dataset: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for layer in range(13):
        X_animal_raw, _ = dataset_animal[layer]
        X_music_raw, _  = dataset_music[layer]
        X_animal = _mean_pool_by_recording(X_animal_raw, frames_animal)
        X_music  = _mean_pool_by_recording(X_music_raw,  frames_music)

        X = np.concatenate([X_animal, X_music], axis=0)
        y = np.concatenate([
            np.zeros(len(X_animal), dtype=np.int32),
            np.ones(len(X_music),  dtype=np.int32),
        ])
        dataset[layer] = (X, y)

    # 3. Optionally subtract noise subspace
    if subtract_noise:
        animal_paths = [path for path, _ in ANIMAL_RECORDINGS.values()]
        subspaces = compute_noise_subspace(
            model,
            audio_paths=animal_paths,
            n_components=3,
            n_recordings=8,
        )
        dataset = project_out_subspace(dataset, subspaces)
        print("Noise subspace projected out.")

    # 4. Recording IDs for LORO
    all_ids = list(ANIMAL_RECORDINGS.keys()) + list(MUSIC_RECORDINGS.keys())
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
        experiment_name=experiment_name,
        results_dir=RESULTS_DIR,
    )
    print(f"\nDone. Results saved to {RESULTS_DIR}/")


def _mean_pool_by_recording(
    X: np.ndarray,
    frames_per_recording: dict[str, int],
) -> np.ndarray:
    pooled = []
    idx = 0
    for n_frames in frames_per_recording.values():
        pooled.append(X[idx: idx + n_frames].mean(axis=0))
        idx += n_frames
    return np.stack(pooled, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtract-noise", action="store_true",
                        help="Project out 3-component noise subspace before probing")
    args = parser.parse_args()
    run(subtract_noise=args.subtract_noise)
