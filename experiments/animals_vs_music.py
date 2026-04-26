"""
experiments/animals_vs_music.py — Probe: Animal vocalizations vs Music.

Binary classification: does AVES linearly separate animal sounds from
musical instruments?

Labels:
    0 = animal  (streamed from NatureLM xeno-canto, 25 samples)
    1 = music   (local files — violin + mixed instruments, ~25 files)

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
RESULTS_DIR     = "results"

N_SAMPLES = 25  # keep both classes balanced at ~25

# ---------------------------------------------------------------------------
# Music recordings (local files — label 1)
# Add more files here to reach ~25 total. Mix instruments for generality.
# ---------------------------------------------------------------------------

# fmt: off
MUSIC_RECORDINGS: dict[str, tuple[str, int]] = {
    "violin_01": ("audio/violin/good_b_music-romantic-violin-waltz-real-violin-497682.mp3",       1),
    "violin_02": ("audio/violin/nickpanekaiassets-cinematic-baroque-violin-melody-287276.mp3",     1),
    "violin_03": ("audio/violin/solarflex-emotional-inspiring-violin-499245.mp3",                  1),
    "violin_04": ("audio/violin/soulfuljamtracks-strings-violin-background-478146.mp3",            1),
    "violin_05": ("audio/violin/vibehorn-violin-background-music-483067.mp3",                      1),
    "piano_01": ("audio/music-misc/paulyudin-piano-piano-music-508963.mp3",                        1),
    "piano_02": ("audio/music-misc/the_mountain-piano-piano-music-490009.mp3",                     1),
    "piano_03": ("audio/music-misc/atlasaudio-piano-emotional-509975.mp3",                         1),
    "piano_04": ("audio/music-misc/leberch-romantic-piano-512030.mp3",                             1),
    "piano_05": ("audio/music-misc/leberch-soft-piano-soft-piano-music-504418.mp3",                1),
    "flute_01": ("audio/music-misc/djovan-flute-of-the-silent-valley-497085.mp3",                  1),
    "flute_02": ("audio/music-misc/bineleyas-indian-classical-flute-amp-tabla-140472.mp3",         1),
    "flute_03": ("audio/music-misc/monosolomono-flute-dark-152088.mp3",                            1),
    "flute_04": ("audio/music-misc/poshpony-bansuri-flute-406082.mp3",                             1),
    "flute_05": ("audio/music-misc/musicwallah-solo-flute-music-relaxing-and-soothing-no-copyright-401558.mp3", 1),
    "guitar_01": ("audio/music-misc/folk_acoustic-the-beat-of-nature-122841.mp3",                  1),
    "guitar_02": ("audio/music-misc/freemusicforvideo-i-love-you-guitar-solo-guitar-music-495615.mp3", 1),
    "guitar_03": ("audio/music-misc/surprising_media-dreaming-on-guitar-strings-512774.mp3",       1),
    "guitar_04": ("audio/music-misc/andriig-guitar-solo-guitar-music-471272.mp3",                  1),
    "guitar_05": ("audio/music-misc/andriig-wedding-romantic-love-music-471301.mp3",               1),

}
# fmt: on


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    print("=== Animals vs Music probe ===\n")

    # 1. Load model
    print("Loading model...")
    model = load_model("esp_aves2_eat_all")

    # 2. Stream animal samples from NatureLM
    print(f"\nStreaming {N_SAMPLES} animal samples from NatureLM...")
    dataset_animal, meta_animal = build_naturelm_dataset(
        model,
        source_dataset=["xeno-canto"],
        label_names=["animal"],
        min_samples_per_class=20,
        max_samples_per_class=N_SAMPLES,
        mode="mean",
    )

    # 3. Build music dataset from local files
    print(f"\nLoading {len(MUSIC_RECORDINGS)} local music recordings...")
    dataset_music, frames_music = build_dataset(model, MUSIC_RECORDINGS)

    # 4. Merge into single dataset
    # Animal activations are mean-pooled (n_items, 768)
    # Music activations are raw frames — mean-pool here for consistency
    print("\nMerging datasets...")
    dataset = {}
    for layer in range(13):
        X_animal, _ = dataset_animal[layer]                        # (n_animal, 768)
        X_music_raw, y_music_raw = dataset_music[layer]            # (n_frames, 768)

        # Mean-pool music frames per recording
        X_music = _mean_pool_by_recording(X_music_raw, frames_music)  # (n_music, 768)

        X = np.concatenate([X_animal, X_music], axis=0)
        y = np.concatenate([
            np.zeros(len(X_animal), dtype=np.int32),
            np.ones(len(X_music),  dtype=np.int32),
        ])
        dataset[layer] = (X, y)

    # 5. Build recording IDs and frame counts for LORO
    animal_ids = [m["id"] for m in meta_animal]
    music_ids  = list(MUSIC_RECORDINGS.keys())
    all_ids    = animal_ids + music_ids
    frames_per_recording = {
        **{aid: 1 for aid in animal_ids},   # already mean-pooled
        **{mid: 1 for mid in music_ids},    # also mean-pooled above
    }

    # 6. Train LORO probes across all layers
    print("\nRunning LORO cross-validation...")
    results = train_all_layers(
        dataset=dataset,
        recording_ids=all_ids,
        frames_per_recording=frames_per_recording,
    )

    # 7. Evaluate and save plots
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
    for rec_id, n_frames in frames_per_recording.items():
        pooled.append(X[idx: idx + n_frames].mean(axis=0))
        idx += n_frames
    return np.stack(pooled, axis=0)


if __name__ == "__main__":
    run()