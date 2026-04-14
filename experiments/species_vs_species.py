"""
experiments/species_vs_species.py — Probe: Arbitrary species pair (TBD).

Generic binary probe config for any two species. Species and recordings are
left unpopulated — fill in SPECIES_A, SPECIES_B, and RECORDINGS once sample
sizes have been verified and species pairs confirmed with the mentor.

Intended use:
    1. Check sample sizes across available species audio.
    2. Confirm target pair with mentor.
    3. Populate SPECIES_A, SPECIES_B, and RECORDINGS below.
    4. Run: python -W ignore experiments/species_vs_species.py

Expected output (saved to results/):
    species_vs_species_{SPECIES_A}_vs_{SPECIES_B}_accuracy.png
    species_vs_species_{SPECIES_A}_vs_{SPECIES_B}_lda.png
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Experiment config — populate before running
# ---------------------------------------------------------------------------

SPECIES_A: str = "TBD"   # e.g. "bullfinch"
SPECIES_B: str = "TBD"   # e.g. "chaffinch"

LABEL_NAMES: list[str] = [SPECIES_A, SPECIES_B]
RESULTS_DIR = "results"

# fmt: off
RECORDINGS: dict[str, tuple[str, int]] = {
    # --- Species A (label 0) ---
    # "species_a_01": ("audio/<species_a>/<filename>.wav", 0),

    # --- Species B (label 1) ---
    # "species_b_01": ("audio/<species_b>/<filename>.wav", 1),
}
# fmt: on


# ---------------------------------------------------------------------------
# Entry point (stub — pipeline not yet implemented)
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Execute the binary species-vs-species probe experiment.

    Pipeline:
        1. Load model via data.loader.load_model
        2. Build per-layer dataset via data.loader.build_dataset(RECORDINGS)
        3. Train LORO probes via probes.train.train_all_layers
        4. Evaluate and save plots via probes.evaluate.run_evaluation
           (experiment_name = f"species_vs_species_{SPECIES_A}_vs_{SPECIES_B}")

    Raises
    ------
    ValueError if RECORDINGS is empty or either species has fewer than 3 recordings.
    """
    ...


if __name__ == "__main__":
    run()
