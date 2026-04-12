"""
experiments/animals_vs_music.py — Probe: Animal vocalizations vs Music.

Binary classification: does AVES linearly separate animal sounds from
musical instruments? Tests whether the model's learned representations
treat music as categorically distinct from biological vocalizations,
or whether early layers treat both as structured audio.

Labels:
    0 = animal (birds: Bullfinch, Hawfinch, Helmeted Guineafowl)
    1 = music  (violin recordings from audio/violin/)

Audio sources:
    Animals — audio/bullfinch/*.wav, audio/hawfinch/*.wav,
               audio/helmeted-guinea-fowl/*.wav
    Music   — audio/violin/*.mp3

Expected output (saved to results/):
    animals_vs_music_accuracy.png  — per-layer LORO accuracy curve
    animals_vs_music_lda.png       — LDA projection at layers 0, 3, 6, 9, 11

Run:
    python -W ignore experiments/animals_vs_music.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

EXPERIMENT_NAME = "animals_vs_music"
LABEL_NAMES = ["animal", "music"]
RESULTS_DIR = "results"

# fmt: off
RECORDINGS: dict[str, tuple[str, int]] = {
    # --- Animals (label 0) ---
    # Bullfinch
    "bullfinch_01": ("audio/bullfinch/XC1077468 - Eurasian Bullfinch - Pyrrhula pyrrhula.wav", 0),
    "bullfinch_02": ("audio/bullfinch/XC965743 - Eurasian Bullfinch - Pyrrhula pyrrhula.wav",  0),
    "bullfinch_03": ("audio/bullfinch/XC938052 - Eurasian Bullfinch - Pyrrhula pyrrhula.wav",  0),
    # Hawfinch
    "hawfinch_01":  ("audio/hawfinch/XC944735 - Hawfinch - Coccothraustes coccothraustes.wav", 0),
    "hawfinch_02":  ("audio/hawfinch/XC1087947 - Hawfinch - Coccothraustes coccothraustes.wav",0),
    "hawfinch_03":  ("audio/hawfinch/XC1083076 - Hawfinch - Coccothraustes coccothraustes.wav",0),
    # Helmeted Guineafowl
    "guineafowl_01":("audio/helmeted-guinea-fowl/XC280506 - Helmeted Guineafowl - Numida meleagris.wav", 0),
    "guineafowl_02":("audio/helmeted-guinea-fowl/XC364521 - Helmeted Guineafowl - Numida meleagris.wav", 0),
    "guineafowl_03":("audio/helmeted-guinea-fowl/XC709655 - Helmeted Guineafowl - Numida meleagris.wav", 0),
    # --- Music (label 1) ---
    "violin_01":    ("audio/violin/good_b_music-romantic-violin-waltz-real-violin-497682.mp3",       1),
    "violin_02":    ("audio/violin/nickpanekaiassets-cinematic-baroque-violin-melody-287276.mp3",     1),
    "violin_03":    ("audio/violin/solarflex-emotional-inspiring-violin-499245.mp3",                  1),
    "violin_04":    ("audio/violin/soulfuljamtracks-strings-violin-background-478146.mp3",            1),
    "violin_05":    ("audio/violin/vibehorn-violin-background-music-483067.mp3",                      1),
}
# fmt: on


# ---------------------------------------------------------------------------
# Entry point (stub — pipeline not yet implemented)
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Execute the full animals-vs-music probe experiment.

    Pipeline:
        1. Load model via data.loader.load_model
        2. Build per-layer dataset via data.loader.build_dataset(RECORDINGS)
        3. Train LORO probes via probes.train.train_all_layers
        4. Evaluate and save plots via probes.evaluate.run_evaluation
    """
    ...


if __name__ == "__main__":
    run()
