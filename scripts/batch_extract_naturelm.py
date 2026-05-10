"""
scripts/batch_extract_naturelm.py — Batch-extract NatureLM activations for all species
used in the species-vs-species probing suite.

Skips species whose output directory already has a completed manifest (rows_saved >= --rows).
Run on GPU; CPU will take ~2-3 hours per species.

Usage:
    # GPU (recommended, ~5-15 min/species)
    python -W ignore scripts/batch_extract_naturelm.py --rows 1000 --device cuda

    # CPU (slow)
    python -W ignore scripts/batch_extract_naturelm.py --rows 1000

    # Preview what would run
    python -W ignore scripts/batch_extract_naturelm.py --rows 1000 --dry-run

Outputs:
    activations/naturelm/<genus>-<species>/
        rec_00000.npy   (13, n_frames, 768) float32
        rec_00000.json  sidecar
        ...
        manifest.json
        errors.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# All unique species across the 10 probe pairs
# ---------------------------------------------------------------------------

SPECIES = [
    # (scientific_name,              slug,                          common_name)
    ("Pyrrhula pyrrhula",            "pyrrhula-pyrrhula",           "Bullfinch"),
    ("Coccothraustes coccothraustes","coccothraustes-coccothraustes","Hawfinch"),
    ("Strix aluco",                  "strix-aluco",                 "Tawny Owl"),
    ("Passer domesticus",            "passer-domesticus",           "House Sparrow"),
    ("Passer montanus",              "passer-montanus",             "Tree Sparrow"),
    ("Parus major",                  "parus-major",                 "Great Tit"),
    ("Phylloscopus trochilus",       "phylloscopus-trochilus",      "Willow Warbler"),
    ("Phylloscopus collybita",       "phylloscopus-collybita",      "Common Chiffchaff"),
    ("Phylloscopus ibericus",        "phylloscopus-ibericus",       "Iberian Chiffchaff"),
    ("Carduelis carduelis",          "carduelis-carduelis",         "Goldfinch"),
    ("Spinus spinus",                "spinus-spinus",               "Eurasian Siskin"),
    ("Corvus splendens",             "corvus-splendens",            "House Crow"),
    ("Corvus corone",                "corvus-corone",               "Carrion Crow"),
    ("Erithacus rubecula",           "erithacus-rubecula",          "European Robin"),
    ("Turdus merula",                "turdus-merula",               "Eurasian Blackbird"),
    ("Apus apus",                    "apus-apus",                   "Common Swift"),
    ("Fringilla coelebs",            "fringilla-coelebs",           "Chaffinch"),
    ("Dendrocopos major",            "dendrocopos-major",           "Great Spotted Woodpecker"),
]


def _already_done(out_dir: Path, rows: int) -> bool:
    manifest = out_dir / "manifest.json"
    if not manifest.exists():
        return False
    with open(manifest) as f:
        m = json.load(f)
    return m.get("rows_saved", 0) >= rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch-extract NatureLM activations for all probing species.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rows",       type=int, default=1000, help="Recordings per species")
    p.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--max-frames", type=int, default=None, help="Override max patch tokens")
    p.add_argument("--base-dir",   default="activations/naturelm", help="Root output directory")
    p.add_argument("--dry-run",    action="store_true", help="Print plan without extracting")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    base = root / args.base_dir

    print(f"\n{'':=<60}")
    print(f"  NatureLM batch extraction — {len(SPECIES)} species, {args.rows} rows each")
    print(f"  Device : {args.device}   Base : {base}")
    print(f"{'':=<60}\n")

    skipped, todo = [], []
    for scientific, slug, common in SPECIES:
        out_dir = base / slug
        if _already_done(out_dir, args.rows):
            skipped.append(common)
        else:
            todo.append((scientific, slug, common, out_dir))

    if skipped:
        print(f"Skipping {len(skipped)} already-complete species:")
        for name in skipped:
            print(f"  ✓ {name}")
        print()

    if not todo:
        print("All species already extracted. Nothing to do.")
        return

    print(f"Will extract {len(todo)} species:")
    for _, slug, common, out_dir in todo:
        marker = "[dry-run] " if args.dry_run else ""
        print(f"  {marker}{common:35s} → {out_dir.relative_to(root)}")
    print()

    if args.dry_run:
        return

    from scripts.extract_species_activations import extract_species  # needs torch

    kwargs = dict(rows=args.rows, device=args.device, seed=42)
    if args.max_frames is not None:
        kwargs["max_frames"] = args.max_frames

    t_total = time.time()
    for i, (scientific, slug, common, out_dir) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] Extracting {common} ({scientific}) → {out_dir.relative_to(root)}")
        t0 = time.time()
        try:
            extract_species(species=scientific, out_dir=out_dir, **kwargs)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        elapsed = time.time() - t0
        print(f"  Done in {elapsed/60:.1f} min\n")

    print(f"\nAll extractions complete in {(time.time() - t_total)/60:.1f} min total.")


if __name__ == "__main__":
    main()
