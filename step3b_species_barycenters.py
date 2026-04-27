"""Step 3b: per-species centroids and within-vs-between species variance.

For each species with at least N samples in the manifest (default
N=5), compute a 768-dim centroid per (model, layer) using the same
50-frames-per-clip protocol as the rest of the pipeline. Then ask:

  - Within-species variance: mean ||frame - species_centroid||² over
    frames belonging to that species.
  - Between-species variance: mean ||species_centroid - global_bio_centroid||²
    over species centroids, where global_bio_centroid is the mean of
    species centroids.
  - Separability ratio: between / (within + between). Bounded in
    [0, 1]; higher = species are more cleanly separated.

Restricted to bio sources (ASA, Watkins, Xeno-canto, iNaturalist) since
non-bio sources have empty species fields.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/species_barycenters/

Usage:
    python step3b_species_barycenters.py
    python step3b_species_barycenters.py --min_samples 10
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_bootstrap_cis import BASE_SEED, load_layer_tensor


MANIFEST_ID = "naturelm_by_source_100each_20260418T171459Z"
DEFAULT_TAX_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}_taxonomic.jsonl")
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "species_barycenters"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
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
    species_counts = Counter(
        r.get("species", "") for r in taxonomy.values() if r.get("species")
    )
    eligible_species = {s for s, c in species_counts.items() if c >= args.min_samples}
    print(
        f"{len(eligible_species)} species with >= {args.min_samples} samples "
        f"(out of {len(species_counts)} unique total)",
        flush=True,
    )

    species_records: list[dict] = []
    summary_records: list[dict] = []

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
        sample_species = np.array(
            [taxonomy.get(s.get("id", ""), {}).get("species", "") for s in sample_meta]
        )
        sample_class = np.array(
            [taxonomy.get(s.get("id", ""), {}).get("class", "") for s in sample_meta]
        )
        # eligibility mask
        eligible_mask = np.array([sp in eligible_species for sp in sample_species])
        del layer0_tensor

        for layer_idx in range(13):
            t0 = time.time()
            layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
            rng = np.random.default_rng(BASE_SEED + layer_idx)
            per_item = per_clip_frame_sample(
                layer_tensor, valid_token_counts, args.frames_per_item, rng,
            )

            # Per-species centroid: mean over (clips × frames-per-clip) frames
            per_species_centroid: dict[str, np.ndarray] = {}
            per_species_within_var: dict[str, float] = {}
            per_species_n_clips: dict[str, int] = {}
            for sp in eligible_species:
                mask = (sample_species == sp) & eligible_mask
                if mask.sum() == 0:
                    continue
                frames = per_item[mask].reshape(-1, per_item.shape[-1]).astype(np.float64)
                centroid = frames.mean(axis=0)
                # Within-species variance: average squared distance from centroid
                within_var = float(np.mean(np.sum((frames - centroid) ** 2, axis=1)))
                per_species_centroid[sp] = centroid
                per_species_within_var[sp] = within_var
                per_species_n_clips[sp] = int(mask.sum())

            # Global bio centroid: mean of species centroids (equal weighting)
            if not per_species_centroid:
                del layer_tensor, per_item
                continue
            centroids_arr = np.stack(list(per_species_centroid.values()))  # (S, D)
            global_bio_centroid = centroids_arr.mean(axis=0)
            # Between-species variance: average squared distance from global centroid
            between_var = float(np.mean(np.sum((centroids_arr - global_bio_centroid) ** 2, axis=1)))
            mean_within_var = float(np.mean(list(per_species_within_var.values())))
            separability = between_var / (mean_within_var + between_var + 1e-12)
            summary_records.append({
                "model": model_key, "layer_idx": layer_idx,
                "n_species": len(per_species_centroid),
                "between_var": between_var,
                "mean_within_var": mean_within_var,
                "separability_ratio": separability,
            })

            for sp, centroid in per_species_centroid.items():
                centroid_norm = float(np.linalg.norm(centroid))
                cos_to_global = float(centroid @ global_bio_centroid /
                                       max(centroid_norm * np.linalg.norm(global_bio_centroid), 1e-12))
                species_records.append({
                    "model": model_key, "layer_idx": layer_idx,
                    "species": sp,
                    "n_clips": per_species_n_clips[sp],
                    "within_var": per_species_within_var[sp],
                    "centroid_norm": centroid_norm,
                    "cos_to_global_bio_centroid": cos_to_global,
                    "class": next(
                        (taxonomy[s_id].get("class", "")
                         for s_id, r in taxonomy.items() if r.get("species") == sp),
                        ""),
                })

            del layer_tensor, per_item
            print(f"  L{layer_idx:02d}  n_species={len(per_species_centroid):>3}  "
                  f"between/within={between_var:.0f}/{mean_within_var:.0f}  "
                  f"sep={separability:.4f}  ({time.time() - t0:.1f}s)", flush=True)

        pd.DataFrame.from_records(species_records).to_csv(
            args.output_dir / "species_per_layer.csv", index=False)
        pd.DataFrame.from_records(summary_records).to_csv(
            args.output_dir / "separability_summary.csv", index=False)

    species_df = pd.DataFrame.from_records(species_records)
    summary_df = pd.DataFrame.from_records(summary_records)
    species_df.to_csv(args.output_dir / "species_per_layer.csv", index=False)
    summary_df.to_csv(args.output_dir / "separability_summary.csv", index=False)

    # ---------------- Plots ----------------
    cmap = plt.get_cmap("tab10")
    color_for_model = {m: cmap(i) for i, m in enumerate(args.models)}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model_key in args.models:
        sub = summary_df[summary_df["model"] == model_key].sort_values("layer_idx")
        if sub.empty:
            continue
        is_baseline = model_key == "random_init_eat_seed42"
        ax.plot(sub["layer_idx"], sub["separability_ratio"],
                marker="s" if is_baseline else "o",
                linestyle="--" if is_baseline else "-",
                color=color_for_model[model_key],
                label=f"{model_key} (baseline)" if is_baseline else model_key)
    ax.set_xlabel("layer index")
    ax.set_ylabel("separability ratio  =  between / (within + between)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Per-species separability by layer (n_species >= {args.min_samples} samples)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.savefig(args.output_dir / "species_separability.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also: between- and within-var on log scale, two panels
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, key, label in [(axes[0], "between_var", "between-species variance"),
                           (axes[1], "mean_within_var", "mean within-species variance")]:
        for model_key in args.models:
            sub = summary_df[summary_df["model"] == model_key].sort_values("layer_idx")
            if sub.empty:
                continue
            is_baseline = model_key == "random_init_eat_seed42"
            ax.plot(sub["layer_idx"], sub[key],
                    marker="s" if is_baseline else "o",
                    linestyle="--" if is_baseline else "-",
                    color=color_for_model[model_key],
                    label=f"{model_key} (baseline)" if is_baseline else model_key)
        ax.set_xlabel("layer index")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.suptitle("Per-species variance components (log scale)")
    fig.savefig(args.output_dir / "species_variance_components.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Headline pivot: separability ratio per (layer, model)
    print("\n=== Per-species separability ratio per (layer, model) ===")
    pivot = summary_df.pivot(index="layer_idx", columns="model", values="separability_ratio")
    pivot = pivot[args.models] if all(m in pivot.columns for m in args.models) else pivot
    print(pivot.round(4).to_string())

    print(f"\nSaved species barycenter artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
