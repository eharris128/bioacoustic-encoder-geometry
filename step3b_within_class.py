"""Step 3b — within-class species barycenters.

The §4.9 finding that random-init beats trained models on per-species
separability used 12 species spanning very different acoustic domains
(whale calls vs songbirds vs dolphins). Random Gaussian projections
preserve those raw acoustic distances trivially. Question: does the
random-init advantage survive when species are *acoustically homogeneous*
— i.e. all songbirds, or all marine mammals?

This script runs the same separability metric as `step3b_species_barycenters.py`
but restricted to:
  - Within-Aves only: species whose `class == "Aves"`
  - Within-Mammalia only: species whose `class == "Mammalia"`

If random-init still has a higher separability ratio when species share
the same Class (and thus more similar acoustic statistics), the §4.9
finding generalizes: trained models compress fine-grained taxonomic
detail across the board. If the trained models match or beat random-
init within-Aves but not across Classes, the §4.9 advantage was a
between-domain artifact.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/within_class_barycenters/

Usage:
    python step3b_within_class.py
    python step3b_within_class.py --min_samples 5 --tax_manifest <...> --roadmap_dir <...>
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_bootstrap_cis import BASE_SEED, load_layer_tensor


MANIFEST_ID = "naturelm_by_order_p100_m200_n200_20260427T222756Z"
DEFAULT_TAX_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}_taxonomic.jsonl")
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "within_class_barycenters"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
TARGET_CLASSES = ("Aves", "Mammalia")
FRAMES_PER_ITEM = 50
MIN_SAMPLES_PER_SPECIES = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tax_manifest", type=Path, default=DEFAULT_TAX_MANIFEST)
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    p.add_argument("--min_samples", type=int, default=MIN_SAMPLES_PER_SPECIES)
    return p.parse_args()


def load_taxonomy(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_id[r["id"]] = r
    return by_id


def per_clip_frame_sample(
    layer_tensor: np.ndarray,
    valid_token_counts: np.ndarray,
    frames_per_item: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_items, t_max, d = layer_tensor.shape
    out = np.empty((n_items, frames_per_item, d), dtype=np.float32)
    for i in range(n_items):
        valid = max(int(min(valid_token_counts[i], t_max)), 1)
        if valid >= frames_per_item:
            f_idx = rng.choice(valid, frames_per_item, replace=False)
        else:
            f_idx = rng.choice(valid, frames_per_item, replace=True)
        out[i] = layer_tensor[i, f_idx, :]
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy(args.tax_manifest)

    # Eligibility per Class: species with >= min_samples within that Class
    species_in_class: dict[str, set[str]] = {c: set() for c in TARGET_CLASSES}
    for c in TARGET_CLASSES:
        counts = Counter(
            r.get("species", "") for r in taxonomy.values()
            if r.get("class") == c and r.get("species")
        )
        species_in_class[c] = {s for s, n in counts.items() if n >= args.min_samples}
        print(f"{c}: {len(species_in_class[c])} eligible species (>= {args.min_samples} samples)",
              flush=True)

    summary_records: list[dict] = []
    species_records: list[dict] = []

    for model_idx, model_key in enumerate(args.models):
        shard_dir = args.roadmap_dir / model_key / "shards"
        if not shard_dir.exists():
            print(f"WARN: shards missing for {model_key}, skipping", flush=True)
            continue
        print(f"\n=== {model_key} ({model_idx + 1}/{len(args.models)}) ===", flush=True)

        layer0_tensor, sample_meta = load_layer_tensor(shard_dir, layer_idx=0)
        valid_token_counts = np.array(
            [int(s.get("valid_token_count", layer0_tensor.shape[1])) for s in sample_meta]
        )
        sample_class = np.array(
            [taxonomy.get(s.get("id", ""), {}).get("class", "") for s in sample_meta]
        )
        sample_species = np.array(
            [taxonomy.get(s.get("id", ""), {}).get("species", "") for s in sample_meta]
        )
        del layer0_tensor

        for layer_idx in range(13):
            t0 = time.time()
            layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
            rng = np.random.default_rng(BASE_SEED + layer_idx)
            per_item = per_clip_frame_sample(
                layer_tensor, valid_token_counts, args.frames_per_item, rng,
            )

            for cls in TARGET_CLASSES:
                eligible = species_in_class[cls]
                if not eligible:
                    continue
                cls_centroid_arr: list[np.ndarray] = []
                cls_within_var: list[float] = []
                cls_n_clips: list[int] = []
                cls_species: list[str] = []
                for sp in eligible:
                    mask = (sample_class == cls) & (sample_species == sp)
                    if mask.sum() == 0:
                        continue
                    sf = per_item[mask].reshape(-1, per_item.shape[-1]).astype(np.float64)
                    centroid = sf.mean(axis=0)
                    within = float(np.mean(np.sum((sf - centroid) ** 2, axis=1)))
                    cls_centroid_arr.append(centroid)
                    cls_within_var.append(within)
                    cls_n_clips.append(int(mask.sum()))
                    cls_species.append(sp)

                if len(cls_centroid_arr) < 2:
                    continue
                C = np.stack(cls_centroid_arr)
                global_c = C.mean(axis=0)
                between_var = float(np.mean(np.sum((C - global_c) ** 2, axis=1)))
                mean_within = float(np.mean(cls_within_var))
                separability = between_var / max(mean_within + between_var, 1e-12)

                summary_records.append({
                    "model": model_key, "layer_idx": layer_idx, "class": cls,
                    "n_species": len(cls_centroid_arr),
                    "between_var": between_var,
                    "mean_within_var": mean_within,
                    "separability_ratio": separability,
                })
                for sp, c, w, n in zip(cls_species, cls_centroid_arr,
                                         cls_within_var, cls_n_clips):
                    species_records.append({
                        "model": model_key, "layer_idx": layer_idx,
                        "class": cls, "species": sp,
                        "n_clips": n,
                        "within_var": w,
                        "centroid_norm": float(np.linalg.norm(c)),
                    })

            del layer_tensor, per_item
            durations = time.time() - t0
            row_summary = ", ".join(
                f"{cls[:3]}={r['separability_ratio']:.3f}"
                for cls in TARGET_CLASSES
                for r in [next((rec for rec in summary_records[-len(TARGET_CLASSES):]
                                 if rec["class"] == cls and rec["model"] == model_key
                                 and rec["layer_idx"] == layer_idx), None)]
                if r is not None
            )
            print(f"  L{layer_idx:02d}  {row_summary}  ({durations:.1f}s)", flush=True)

        pd.DataFrame.from_records(summary_records).to_csv(
            args.output_dir / "within_class_separability.csv", index=False)
        pd.DataFrame.from_records(species_records).to_csv(
            args.output_dir / "within_class_species.csv", index=False)

    summary_df = pd.DataFrame.from_records(summary_records)
    species_df = pd.DataFrame.from_records(species_records)
    summary_df.to_csv(args.output_dir / "within_class_separability.csv", index=False)
    species_df.to_csv(args.output_dir / "within_class_species.csv", index=False)

    # ---------------- Plots ----------------
    cmap = plt.get_cmap("tab10")
    color_for_model = {m: cmap(i) for i, m in enumerate(args.models)}

    fig, axes = plt.subplots(1, len(TARGET_CLASSES),
                             figsize=(5.5 * len(TARGET_CLASSES), 5), sharey=True)
    if len(TARGET_CLASSES) == 1:
        axes = [axes]
    for ax, cls in zip(axes, TARGET_CLASSES):
        for model_key in args.models:
            sub = summary_df[(summary_df["model"] == model_key) &
                             (summary_df["class"] == cls)].sort_values("layer_idx")
            if sub.empty:
                continue
            is_baseline = model_key == "random_init_eat_seed42"
            ax.plot(sub["layer_idx"], sub["separability_ratio"],
                    marker="s" if is_baseline else "o",
                    linestyle="--" if is_baseline else "-",
                    color=color_for_model[model_key],
                    label=f"{model_key} (baseline)" if is_baseline else model_key)
        n_sp = len(species_in_class[cls])
        ax.set_title(f"{cls} (n_species={n_sp})", fontsize=10)
        ax.set_xlabel("layer index")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("separability ratio (between / total)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.suptitle(f"Within-class species separability (n_species_min={args.min_samples})")
    fig.savefig(args.output_dir / "within_class_separability.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Headline pivot
    print("\n=== Within-class separability ratio per (class, layer, model) ===")
    for cls in TARGET_CLASSES:
        sub = summary_df[summary_df["class"] == cls]
        if sub.empty:
            continue
        print(f"\n--- {cls} ---")
        pivot = sub.pivot(index="layer_idx", columns="model", values="separability_ratio")
        cols_in_order = [m for m in args.models if m in pivot.columns]
        pivot = pivot[cols_in_order]
        print(pivot.round(4).to_string())

    print(f"\nSaved within-class barycenter artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
