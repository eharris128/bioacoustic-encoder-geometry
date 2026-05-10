"""
experiments/species_vs_species.py — Probe: Species A vs Species B.

Binary classification: can AVES linearly separate two species?
Uses NatureLM xeno-canto stream (shard-by-shard downloader, robust to corrupt shards).

Run (default pair: bullfinch vs hawfinch):
    python -W ignore experiments/species_vs_species.py

Run with custom pair:
    python -W ignore experiments/species_vs_species.py \
        --species-a "Pyrrhula pyrrhula" --label-a bullfinch \
        --species-b "Turdus merula"     --label-b blackbird \
        --n-samples 50
"""

from __future__ import annotations

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from data.loader import load_model, build_xenocanto_dataset
from probes.train import train_all_layers
from probes.evaluate import run_evaluation

RESULTS_DIR = "results/probe-output/species_vs_species"


def run(
    species_a: str,
    species_b: str,
    label_a: str,
    label_b: str,
    n_samples: int,
) -> None:
    label_names     = [label_a, label_b]
    experiment_name = f"{label_a}_vs_{label_b}"

    print(f"=== Species vs Species probe: {label_a} vs {label_b} ===\n")
    print(f"  Species A: {species_a}")
    print(f"  Species B: {species_b}")
    print(f"  Max samples per species: {n_samples}\n")

    print("Loading model...")
    model = load_model("esp_aves2_eat_all")

    print(f"\nFetching up to {n_samples} recordings per species from xeno-canto...")
    dataset, meta = build_xenocanto_dataset(
        model,
        species_pair=(species_a, species_b),
        label_names=label_names,
        n_per_species=n_samples,
        mode="mean",
    )

    n_a = sum(1 for m in meta if m["species"] == species_a)
    n_b = sum(1 for m in meta if m["species"] == species_b)
    print(f"  {label_a}: {n_a}  |  {label_b}: {n_b}")

    if n_a < 5 or n_b < 5:
        raise ValueError(
            f"Too few samples: {label_a}={n_a}, {label_b}={n_b}. Need at least 5 per species."
        )

    all_ids = [m["id"] for m in meta]
    frames_per_recording = {rid: 1 for rid in all_ids}

    print("\nRunning LORO cross-validation...")
    results = train_all_layers(
        dataset=dataset,
        recording_ids=all_ids,
        frames_per_recording=frames_per_recording,
    )

    run_evaluation(
        accuracy_per_layer=results["accuracy_per_layer"],
        dataset=dataset,
        chance_level=results["chance_level"],
        label_names=label_names,
        experiment_name=experiment_name,
        results_dir=RESULTS_DIR,
    )
    print(f"\nDone. Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--species-a",  default="Pyrrhula pyrrhula",             help="Scientific name of species A")
    parser.add_argument("--species-b",  default="Coccothraustes coccothraustes", help="Scientific name of species B")
    parser.add_argument("--label-a",    default="bullfinch",  help="Short label for species A")
    parser.add_argument("--label-b",    default="hawfinch",   help="Short label for species B")
    parser.add_argument("--n-samples",  default=50, type=int, help="Max samples per species (default 50)")
    args = parser.parse_args()

    run(
        species_a=args.species_a,
        species_b=args.species_b,
        label_a=args.label_a,
        label_b=args.label_b,
        n_samples=args.n_samples,
    )
