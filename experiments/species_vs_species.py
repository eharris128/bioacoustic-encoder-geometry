"""
experiments/species_vs_species.py — Probe: Species A vs Species B.

Binary classification: can AVES linearly separate two species?
Default pair: Eurasian Bullfinch vs Hawfinch (closely related finches —
tests fine-grained discrimination within Fringillidae).

Labels:
    0 = SPECIES_A  (NatureLM xeno-canto stream, up to N_SAMPLES each, mean-pooled)
    1 = SPECIES_B

Note: requires a stable HuggingFace connection. Run on Lambda/GCP for best results.

Run:
    python -W ignore experiments/species_vs_species.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from data.loader import load_model, build_naturelm_dataset
from probes.train import train_all_layers
from probes.evaluate import run_evaluation

# ---------------------------------------------------------------------------
# Experiment config — change species pair here
# ---------------------------------------------------------------------------

SPECIES_A       = "Pyrrhula pyrrhula"             # Eurasian Bullfinch
SPECIES_B       = "Coccothraustes coccothraustes" # Hawfinch
LABEL_NAMES     = ["bullfinch", "hawfinch"]
N_SAMPLES       = 100  # max recordings per species streamed from NatureLM
RESULTS_DIR     = "results/probe-output/species_vs_species"
EXPERIMENT_NAME = f"species_vs_species_{LABEL_NAMES[0]}_vs_{LABEL_NAMES[1]}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    print(f"=== Species vs Species probe: {LABEL_NAMES[0]} vs {LABEL_NAMES[1]} ===\n")

    print("Loading model...")
    model = load_model("esp_aves2_eat_all")

    # Stream both species from NatureLM (mean-pooled: one vector per recording)
    print(f"\nStreaming up to {N_SAMPLES} samples per species from NatureLM...")
    dataset, meta = build_naturelm_dataset(
        model,
        source_dataset=["xeno-canto"],
        species_pair=(SPECIES_A, SPECIES_B),
        min_samples_per_class=10,
        max_samples_per_class=N_SAMPLES,
        mode="mean",
    )

    n_a = sum(1 for m in meta if m["species"] == SPECIES_A)
    n_b = sum(1 for m in meta if m["species"] == SPECIES_B)
    print(f"  {LABEL_NAMES[0]}: {n_a}  |  {LABEL_NAMES[1]}: {n_b}")

    if n_a < 5 or n_b < 5:
        raise ValueError(
            f"Too few samples: {LABEL_NAMES[0]}={n_a}, {LABEL_NAMES[1]}={n_b}. "
            "Need at least 5 per species."
        )

    # Recording IDs for LORO (one mean-pooled vector per recording)
    all_ids = [m["id"] for m in meta]
    frames_per_recording = {rid: 1 for rid in all_ids}

    # Train LORO probes across all layers
    print("\nRunning LORO cross-validation...")
    results = train_all_layers(
        dataset=dataset,
        recording_ids=all_ids,
        frames_per_recording=frames_per_recording,
    )

    # Evaluate and save plots
    run_evaluation(
        accuracy_per_layer=results["accuracy_per_layer"],
        dataset=dataset,
        chance_level=results["chance_level"],
        label_names=LABEL_NAMES,
        experiment_name=EXPERIMENT_NAME,
        results_dir=RESULTS_DIR,
    )
    print(f"\nDone. Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
