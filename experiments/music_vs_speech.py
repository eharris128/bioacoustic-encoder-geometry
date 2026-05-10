"""
experiments/music_vs_speech.py — Probe: Music vs Human Speech.

Binary classification: can AVES linearly separate musical instruments from
human speech? Both are non-animal audio — this tests whether the model
encodes a music/speech distinction despite being trained on animal sounds,
or whether it treats both as undifferentiated "non-biological" audio.

Labels:
    0 = music  (violin recordings from audio/violin/)
    1 = speech (human narration from audio/speech/)

Audio sources:
    Music  — audio/violin/*.mp3
    Speech — audio/speech/*.mp3  (LibriVox recordings)

Expected output (saved to results/):
    music_vs_speech_accuracy.png  — per-layer LORO accuracy curve
    music_vs_speech_lda.png       — LDA projection at layers 0, 3, 6, 9, 11

Note: only 2 speech files are currently available locally. Add more to
audio/speech/ before running — aim for ≥5 per class for stable LORO CV.

Run:
    python -W ignore experiments/music_vs_speech.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

EXPERIMENT_NAME = "music_vs_speech"
LABEL_NAMES = ["music", "speech"]
RESULTS_DIR = "results"

# fmt: off
RECORDINGS: dict[str, tuple[str, int]] = {
    # --- Music (label 0) ---
    "violin_01": ("audio/violin/good_b_music-romantic-violin-waltz-real-violin-497682.mp3",       0),
    "violin_02": ("audio/violin/nickpanekaiassets-cinematic-baroque-violin-melody-287276.mp3",     0),
    "violin_03": ("audio/violin/solarflex-emotional-inspiring-violin-499245.mp3",                  0),
    "violin_04": ("audio/violin/soulfuljamtracks-strings-violin-background-478146.mp3",            0),
    "violin_05": ("audio/violin/vibehorn-violin-background-music-483067.mp3",                      0),
    # --- Speech (label 1) ---
    # NOTE: only 2 files available locally — add more to audio/speech/ before running
    "speech_01": ("audio/speech/librivox_childs_story.mp3",    1),
    "speech_02": ("audio/speech/librivox_skylight_room.mp3",   1),
}
# fmt: on


# ---------------------------------------------------------------------------
# Entry point (stub — pipeline not yet implemented)
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Execute the full music-vs-speech probe experiment.

    Pipeline:
        1. Load model via data.loader.load_model
        2. Build per-layer dataset via data.loader.build_dataset(RECORDINGS)
        3. Train LORO probes via probes.train.train_all_layers
        4. Evaluate and save plots via probes.evaluate.run_evaluation

    Note: LORO CV requires ≥2 recordings per class. With only 2 speech files
    the variance will be high — collect more before drawing conclusions.
    """
    ...


if __name__ == "__main__":
    run()
