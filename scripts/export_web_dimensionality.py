"""
scripts/export_web_dimensionality.py — Export per-layer dimensionality metrics
for the personal-website "Dimensionality across depth" explorer exhibit.

Reads the frame-level per-(model, layer) stats that already include the
random-init baseline (all five models) and writes a compact tidy CSV with just
the columns the web chart needs: effective rank, participation ratio, and
intrinsic dimension (MLE-ID), one row per (model, layer).

Usage:
    python -W ignore scripts/export_web_dimensionality.py \
        --out ~/projects/personal-website/public/explorer/dimensionality.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

SRC = (
    "artifacts/comparisons/naturelm_by_source_100each_20260418T171459Z/"
    "nway_eat_all4/random_init_baseline/frame_per_layer_stats_all5.csv"
)

# Stable display order: base encoders, SSL fine-tunes, then the baseline last.
MODEL_ORDER = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--src", default=SRC, type=Path)
    args = ap.parse_args()

    rows = []
    with open(args.src) as f:
        for r in csv.DictReader(f):
            rows.append({
                "model": r["model"],
                "layer_idx": int(r["layer_idx"]),
                "effective_rank": round(float(r["effective_rank"]), 3),
                "participation_ratio": round(float(r["participation_ratio"]), 3),
                "mle_id": round(float(r["mle_id_k20"]), 3),
            })

    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    rows.sort(key=lambda d: (order.get(d["model"], 99), d["layer_idx"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "layer_idx",
                                          "effective_rank", "participation_ratio", "mle_id"])
        w.writeheader()
        w.writerows(rows)

    models = sorted({d["model"] for d in rows}, key=lambda m: order.get(m, 99))
    print(f"Wrote {args.out}  ({len(rows)} rows, {len(models)} models: {', '.join(models)})")


if __name__ == "__main__":
    main()
