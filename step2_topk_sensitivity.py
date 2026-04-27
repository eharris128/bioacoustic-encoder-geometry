"""Step 2: top-k sensitivity for bio-vs-non-bio subspace overlap.

§4 of RESULTS.md uses top-k=10 throughout. The 0.57 minimum at L9 for
`sl_eat_bio_ssl_all` could strengthen or weaken at smaller / larger k.
This script recomputes mean cos principal angles at k ∈ {5, 10, 20, 50}
across all 5 models × all 13 layers, frame-level (50 frames/item).

Closes RESULTS.md §9.3.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/topk_sensitivity/

Usage:
    python step2_topk_sensitivity.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_bootstrap_cis import (
    BASE_SEED, NATURE_SOURCES,
    load_layer_tensor, subspace_overlap, top_k_basis_via_cov,
)


MANIFEST_ID = "naturelm_by_source_100each_20260418T171459Z"
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "topk_sensitivity"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
DEFAULT_TOP_KS = [5, 10, 20, 50]
FRAMES_PER_ITEM = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--top_ks", nargs="+", type=int, default=DEFAULT_TOP_KS)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    return p.parse_args()


def sample_frames(
    layer_tensor: np.ndarray,
    valid_token_counts: np.ndarray,
    is_bio_per_item: np.ndarray,
    frames_per_item: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_items, t_max, d = layer_tensor.shape
    out = np.empty((n_items, frames_per_item, d), dtype=np.float32)
    is_bio = np.empty((n_items, frames_per_item), dtype=bool)
    for i in range(n_items):
        valid = max(int(min(valid_token_counts[i], t_max)), 1)
        if valid >= frames_per_item:
            f_idx = rng.choice(valid, frames_per_item, replace=False)
        else:
            f_idx = rng.choice(valid, frames_per_item, replace=True)
        out[i] = layer_tensor[i, f_idx, :]
        is_bio[i] = is_bio_per_item[i]
    return out.reshape(-1, d), is_bio.reshape(-1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Top-k sensitivity: {len(args.models)} models × 13 layers × ks={args.top_ks}", flush=True)

    records: list[dict] = []
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
        sources = np.array([s.get("source_dataset", "") for s in sample_meta])
        is_bio_per_item = np.array([s in NATURE_SOURCES for s in sources])
        del layer0_tensor

        for layer_idx in range(13):
            t0 = time.time()
            layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
            rng = np.random.default_rng(BASE_SEED + layer_idx)
            frames, is_bio = sample_frames(
                layer_tensor, valid_token_counts, is_bio_per_item,
                args.frames_per_item, rng,
            )
            frames = frames.astype(np.float64)
            bio_frames = frames[is_bio]
            nonbio_frames = frames[~is_bio]

            # Compute the largest top-k basis once, then slice for smaller k
            max_k = max(args.top_ks)
            basis_bio_full = top_k_basis_via_cov(bio_frames, max_k)
            basis_nonbio_full = top_k_basis_via_cov(nonbio_frames, max_k)

            for k in args.top_ks:
                cos = subspace_overlap(basis_bio_full[:, :k], basis_nonbio_full[:, :k])
                records.append({
                    "model": model_key,
                    "layer_idx": layer_idx,
                    "top_k": k,
                    "mean_cos_principal_angles": cos,
                    "n_bio_frames": int(bio_frames.shape[0]),
                    "n_nonbio_frames": int(nonbio_frames.shape[0]),
                })
            print(
                f"  L{layer_idx:02d}  cos@k=5={records[-len(args.top_ks)]['mean_cos_principal_angles']:.3f}  "
                f"k=10={records[-len(args.top_ks)+1]['mean_cos_principal_angles']:.3f}  "
                f"k=20={records[-len(args.top_ks)+2]['mean_cos_principal_angles']:.3f}  "
                f"k=50={records[-len(args.top_ks)+3]['mean_cos_principal_angles']:.3f}  "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
            del layer_tensor

        df = pd.DataFrame.from_records(records)
        df.to_csv(args.output_dir / "topk_sensitivity.csv", index=False)

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.output_dir / "topk_sensitivity.csv", index=False)

    # Plot: one panel per top_k, one curve per model
    fig, axes = plt.subplots(1, len(args.top_ks), figsize=(4.5 * len(args.top_ks), 5), sharey=True)
    if len(args.top_ks) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    color_for = {m: cmap(i) for i, m in enumerate(args.models)}
    for ax, k in zip(axes, args.top_ks):
        for model_key in args.models:
            sub = df[(df["model"] == model_key) & (df["top_k"] == k)].sort_values("layer_idx")
            if sub.empty:
                continue
            is_baseline = model_key == "random_init_eat_seed42"
            ax.plot(
                sub["layer_idx"], sub["mean_cos_principal_angles"],
                marker="s" if is_baseline else "o",
                linestyle="--" if is_baseline else "-",
                color=color_for[model_key],
                label=f"{model_key} (baseline)" if is_baseline else model_key,
            )
        ax.set_title(f"top-k = {k}")
        ax.set_xlabel("layer index")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean cos principal angles")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.suptitle("Bio-vs-non-bio frame-level subspace overlap, swept over top-k")
    fig.savefig(args.output_dir / "topk_sensitivity_panels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Pivot table for the §4 minimum layer (L9)
    print("\n=== Cos at L9 (§4 minimum), per (model, top_k) ===", flush=True)
    pivot = df[df["layer_idx"] == 9].pivot(index="model", columns="top_k", values="mean_cos_principal_angles")
    pivot = pivot.reindex(index=args.models)
    print(pivot.round(3).to_string(), flush=True)
    pivot.to_csv(args.output_dir / "topk_at_L9.csv")

    print(f"\nSaved top-k sensitivity artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
