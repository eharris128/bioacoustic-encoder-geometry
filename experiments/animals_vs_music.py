"""
experiments/animals_vs_music.py — Probe: Animal vocalizations vs Music.

Binary classification: does AVES linearly separate animal sounds from
musical instruments?

Labels:
    0 = animal  (NatureLM xeno-canto stream, up to 200 samples, mean-pooled)
    1 = music   (local files — violin + piano + flute + guitar = 19, mean-pooled)

Note: requires a stable HuggingFace connection. Run on Lambda for best results.
      For offline use, see git history for the local-only version.

Run:
    python -W ignore experiments/animals_vs_music.py
"""

from __future__ import annotations

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
# Music recordings (local files — label 1)
# ---------------------------------------------------------------------------

# fmt: off
MUSIC_RECORDINGS: dict[str, tuple[str, int]] = {
    "violin_01": ("audio/violin/good_b_music-romantic-violin-waltz-real-violin-497682.mp3",              1),
    "violin_02": ("audio/violin/nickpanekaiassets-cinematic-baroque-violin-melody-287276.mp3",           1),
    "violin_03": ("audio/violin/solarflex-emotional-inspiring-violin-499245.mp3",                        1),
    "violin_04": ("audio/violin/soulfuljamtracks-strings-violin-background-478146.mp3",                  1),
    "violin_05": ("audio/violin/vibehorn-violin-background-music-483067.mp3",                            1),
    "piano_01":  ("audio/music-misc/paulyudin-piano-piano-music-508963.mp3",                             1),
    "piano_02":  ("audio/music-misc/the_mountain-piano-piano-music-490009.mp3",                          1),
    "piano_03":  ("audio/music-misc/atlasaudio-piano-emotional-509975.mp3",                              1),
    "piano_04":  ("audio/music-misc/leberch-romantic-piano-512030.mp3",                                  1),
    "piano_05":  ("audio/music-misc/leberch-soft-piano-soft-piano-music-504418.mp3",                     1),
    "flute_01":  ("audio/music-misc/djovan-flute-of-the-silent-valley-497085.mp3",                       1),
    "flute_02":  ("audio/music-misc/bineleyas-indian-classical-flute-amp-tabla-140472.mp3",              1),
    "flute_03":  ("audio/music-misc/monosolomono-flute-dark-152088.mp3",                                 1),
    "flute_04":  ("audio/music-misc/poshpony-bansuri-flute-406082.mp3",                                  1),
    "flute_05":  ("audio/music-misc/musicwallah-solo-flute-music-relaxing-and-soothing-no-copyright-401558.mp3", 1),
    "guitar_01": ("audio/music-misc/folk_acoustic-the-beat-of-nature-122841.mp3",                        1),
    "guitar_02": ("audio/music-misc/freemusicforvideo-i-love-you-guitar-solo-guitar-music-495615.mp3",   1),
    "guitar_03": ("audio/music-misc/surprising_media-dreaming-on-guitar-strings-512774.mp3",             1),
    "guitar_04": ("audio/music-misc/andriig-wedding-romantic-love-music-471301.mp3",                     1),
}
# fmt: on


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
