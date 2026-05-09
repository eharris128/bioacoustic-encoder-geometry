"""
experiments/naturelm_probe_all_pairs.py — Run all species-vs-species probes on
pre-extracted NatureLM activations (from scripts/batch_extract_naturelm.py).

Run extraction first:
    python -W ignore scripts/batch_extract_naturelm.py --rows 1000 --device cuda

Then run probes (CPU is fine — inference already done):
    python -W ignore experiments/naturelm_probe_all_pairs.py

Results saved to:
    results/probe-output/naturelm_species_vs_species/

Note: Great Tit vs Bokharensis is omitted — NatureLM metadata does not
distinguish Parus major subspecies, making that pair unrunnable.
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.load_species_activations import load_species_pair, to_probe_dataset
from probes.train import train_all_layers
from probes.evaluate import run_evaluation

ROOT = Path(__file__).resolve().parents[1]
ACT_BASE  = ROOT / "activations" / "naturelm"
RESULTS_DIR = "results/probe-output/naturelm_species_vs_species"

# ---------------------------------------------------------------------------
# All 10 probe pairs (mirrors the xeno-canto results from probes_results_README)
# ---------------------------------------------------------------------------

PAIRS = [
    # (slug_a,                          slug_b,                            label_a,          label_b,          taxonomy)
    ("pyrrhula-pyrrhula",               "coccothraustes-coccothraustes",   "bullfinch",       "hawfinch",        "Same family"),
    ("pyrrhula-pyrrhula",               "strix-aluco",                     "bullfinch",       "tawny_owl",       "Diff. orders"),
    ("passer-domesticus",               "passer-montanus",                 "house_sparrow",   "tree_sparrow",    "Same genus"),
    ("phylloscopus-trochilus",          "phylloscopus-collybita",          "willow_warbler",  "chiffchaff",      "Same genus"),
    ("phylloscopus-collybita",          "phylloscopus-ibericus",           "common_chiffchaff","iberian_chiffchaff","Same genus"),
    ("carduelis-carduelis",             "spinus-spinus",                   "goldfinch",       "eurasian_siskin", "Same family"),
    ("corvus-splendens",                "corvus-corone",                   "house_crow",      "carrion_crow",    "Same genus"),
    ("erithacus-rubecula",              "turdus-merula",                   "european_robin",  "eurasian_blackbird","Diff. families"),
    ("passer-domesticus",               "apus-apus",                       "house_sparrow",   "common_swift",    "Diff. orders"),
    ("fringilla-coelebs",               "dendrocopos-major",               "chaffinch",       "great_spotted_woodpecker","Diff. orders"),
]


def run_pair(slug_a, slug_b, label_a, label_b, taxonomy, max_recordings=1000):
    dir_a = ACT_BASE / slug_a
    dir_b = ACT_BASE / slug_b
    experiment_name = f"{label_a}_vs_{label_b}"

    print(f"\n{'':=<52}")
    print(f"  {label_a} vs {label_b}  [{taxonomy}]")
    print(f"{'':=<52}")

    # Check both dirs exist
    for d, lbl in ((dir_a, label_a), (dir_b, label_b)):
        if not d.exists():
            print(f"  SKIP — {lbl} activations not found at {d.relative_to(ROOT)}")
            return None

    X, y, recording_ids = load_species_pair(
        dir_a, dir_b, mode="mean", max_recordings=max_recordings
    )
    n_a = int((y == 0).sum())
    n_b = int((y == 1).sum())
    print(f"  Loaded: {label_a}={n_a}  {label_b}={n_b}  (total {len(y)})")

    dataset = to_probe_dataset(X, y)
    frames_per_recording = {rid: 1 for rid in recording_ids}

    t0 = time.time()
    results = train_all_layers(
        dataset=dataset,
        recording_ids=recording_ids,
        frames_per_recording=frames_per_recording,
    )
    print(f"  LORO done in {time.time()-t0:.1f}s")

    run_evaluation(
        accuracy_per_layer=results["accuracy_per_layer"],
        dataset=dataset,
        chance_level=results["chance_level"],
        label_names=[label_a, label_b],
        experiment_name=experiment_name,
        results_dir=RESULTS_DIR,
    )
    return results


def main():
    print(f"\nNatureLM species-vs-species probing ({len(PAIRS)} pairs)")
    print(f"Activations from : {ACT_BASE.relative_to(ROOT)}")
    print(f"Results to       : {RESULTS_DIR}\n")

    summary = []
    for slug_a, slug_b, label_a, label_b, taxonomy in PAIRS:
        results = run_pair(slug_a, slug_b, label_a, label_b, taxonomy)
        if results is None:
            continue
        acc = results["accuracy_per_layer"]
        peak_layer = max(acc, key=acc.get)
        peak_acc   = acc[peak_layer]
        emb_acc    = acc[0]
        layer_label = "emb" if peak_layer == 0 else f"T{peak_layer-1}"
        summary.append({
            "pair":      f"{label_a} vs {label_b}",
            "taxonomy":  taxonomy,
            "peak":      f"{peak_acc:.1%}",
            "peak_layer": layer_label,
            "emb":       f"{emb_acc:.1%}",
        })

    if not summary:
        print("\nNo pairs ran — check that activations exist in activations/naturelm/")
        return

    print(f"\n\n{'':=<78}")
    print(f"  SUMMARY — NatureLM species probes (n=1000 per species)")
    print(f"{'':=<78}")
    print(f"  {'Pair':<42} {'Taxonomy':<18} {'Peak':>6} {'Layer':>7} {'Emb':>6}")
    print(f"  {'':─<42} {'':─<18} {'':─<6} {'':─<7} {'':─<6}")
    for r in summary:
        print(f"  {r['pair']:<42} {r['taxonomy']:<18} {r['peak']:>6} {r['peak_layer']:>7} {r['emb']:>6}")
    print(f"{'':=<78}")
    print(f"\nDone. Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
