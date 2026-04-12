"""
experiments/species.py — Probe: Species-level discrimination (pairs TBD).

Per-pair binary probes: for each species pair, at which AVES layer does the
representation become linearly separable? Extends mentor's class-level and
order-level probes to fine-grained species identity within the same order.

Current local species:
    Eurasian Bullfinch  (Pyrrhula pyrrhula)   — Passeriformes
    Hawfinch            (Coccothraustes)       — Passeriformes
    Helmeted Guineafowl (Numida meleagris)     — Galliformes

Pairs to run (at minimum):
    bullfinch vs hawfinch      — same order, different genus (fine-grained)
    bullfinch vs guineafowl    — different order (coarser separation)
    hawfinch vs guineafowl     — different order (coarser separation)

Additional pairs to add once more audio is downloaded:
    TBD — confirm with mentor which species pairings are most informative.

Expected output per pair (saved to results/):
    species_{pair_name}_accuracy.png — per-layer LORO accuracy curve
    species_{pair_name}_lda.png      — LDA projection at layers 0, 3, 6, 9, 11

Run:
    python -W ignore experiments/species.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Per-pair recording configs
# ---------------------------------------------------------------------------

# fmt: off
PAIRS: dict[str, dict[str, tuple[str, int]]] = {
    "bullfinch_vs_hawfinch": {
        "bullfinch_01": ("audio/bullfinch/XC1077468 - Eurasian Bullfinch - Pyrrhula pyrrhula.wav", 0),
        "bullfinch_02": ("audio/bullfinch/XC965743 - Eurasian Bullfinch - Pyrrhula pyrrhula.wav",  0),
        "bullfinch_03": ("audio/bullfinch/XC938052 - Eurasian Bullfinch - Pyrrhula pyrrhula.wav",  0),
        "hawfinch_01":  ("audio/hawfinch/XC944735 - Hawfinch - Coccothraustes coccothraustes.wav", 1),
        "hawfinch_02":  ("audio/hawfinch/XC1087947 - Hawfinch - Coccothraustes coccothraustes.wav",1),
        "hawfinch_03":  ("audio/hawfinch/XC1083076 - Hawfinch - Coccothraustes coccothraustes.wav",1),
    },
    "bullfinch_vs_guineafowl": {
        "bullfinch_01":  ("audio/bullfinch/XC1077468 - Eurasian Bullfinch - Pyrrhula pyrrhula.wav",      0),
        "bullfinch_02":  ("audio/bullfinch/XC965743 - Eurasian Bullfinch - Pyrrhula pyrrhula.wav",       0),
        "bullfinch_03":  ("audio/bullfinch/XC938052 - Eurasian Bullfinch - Pyrrhula pyrrhula.wav",       0),
        "guineafowl_01": ("audio/helmeted-guinea-fowl/XC280506 - Helmeted Guineafowl - Numida meleagris.wav", 1),
        "guineafowl_02": ("audio/helmeted-guinea-fowl/XC364521 - Helmeted Guineafowl - Numida meleagris.wav", 1),
        "guineafowl_03": ("audio/helmeted-guinea-fowl/XC709655 - Helmeted Guineafowl - Numida meleagris.wav", 1),
    },
    "hawfinch_vs_guineafowl": {
        "hawfinch_01":   ("audio/hawfinch/XC944735 - Hawfinch - Coccothraustes coccothraustes.wav",      0),
        "hawfinch_02":   ("audio/hawfinch/XC1087947 - Hawfinch - Coccothraustes coccothraustes.wav",     0),
        "hawfinch_03":   ("audio/hawfinch/XC1083076 - Hawfinch - Coccothraustes coccothraustes.wav",     0),
        "guineafowl_01": ("audio/helmeted-guinea-fowl/XC280506 - Helmeted Guineafowl - Numida meleagris.wav", 1),
        "guineafowl_02": ("audio/helmeted-guinea-fowl/XC364521 - Helmeted Guineafowl - Numida meleagris.wav", 1),
        "guineafowl_03": ("audio/helmeted-guinea-fowl/XC709655 - Helmeted Guineafowl - Numida meleagris.wav", 1),
    },
    # Add new pairs here: "species_a_vs_species_b": { ... }
}

LABEL_NAMES_PER_PAIR: dict[str, list[str]] = {
    "bullfinch_vs_hawfinch":   ["bullfinch", "hawfinch"],
    "bullfinch_vs_guineafowl": ["bullfinch", "guineafowl"],
    "hawfinch_vs_guineafowl":  ["hawfinch",  "guineafowl"],
}
# fmt: on

RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Entry point (stub — pipeline not yet implemented)
# ---------------------------------------------------------------------------

def run_pair(pair_name: str) -> None:
    """
    Execute the probe experiment for a single species pair.

    Parameters
    ----------
    pair_name : key in PAIRS, e.g. "bullfinch_vs_hawfinch"

    Pipeline:
        1. Load model via data.loader.load_model
        2. Build per-layer dataset via data.loader.build_dataset(PAIRS[pair_name])
        3. Train LORO probes via probes.train.train_all_layers
        4. Evaluate and save plots via probes.evaluate.run_evaluation
           (experiment_name = f"species_{pair_name}")
    """
    ...


def run() -> None:
    """
    Run all species-pair probe experiments sequentially.

    Iterates over all keys in PAIRS and calls run_pair for each.
    """
    ...


if __name__ == "__main__":
    run()
