"""Step 2: per-source frame-level structure.

The pooled per-source eff_rank slicing in `step2_spectral_dim/effective_rank_by_source.csv`
showed source-specific differences (e.g. Xeno-canto vs WavCaps eff_rank gap)
but at pooled-level, where pooling itself distorts geometry (§3 of RESULTS.md).
This script redoes the slicing at frame level: for each (model, layer, source),
compute frame-level eff_rank, MLE-ID(k=20), and (where 2+ sources have enough
frames) pairwise top-10 subspace overlap between sources.

Closes RESULTS.md §9.5.

Default: all 5 models × 13 layers × 7 source datasets, frame-level.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/per_source_frame_level/

Usage:
    python step2_per_source_frame_level.py
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_bootstrap_cis import (
    BASE_SEED, MLE_K, MLE_SAMPLE_SIZE, NATURE_SOURCES, SUBSPACE_TOP_K,
    cov_eigvals, effective_rank, load_layer_tensor, mle_intrinsic_dim,
    participation_ratio, subspace_overlap, top_k_basis_via_cov,
)


MANIFEST_ID = "naturelm_by_source_100each_20260418T171459Z"
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "per_source_frame_level"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
FRAMES_PER_ITEM = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    p.add_argument("--top_k", type=int, default=SUBSPACE_TOP_K)
    p.add_argument("--mle_k", type=int, default=MLE_K)
    p.add_argument("--mle_sample_size", type=int, default=MLE_SAMPLE_SIZE)
    return p.parse_args()


def sample_frames(
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
    return out  # (n_items, F, D)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Per-source frame-level: {len(args.models)} models × 13 layers × 7 sources", flush=True)

    stats_records: list[dict] = []
    pairwise_records: list[dict] = []

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
        unique_sources = sorted(set(sources))
        print(f"  sources: {unique_sources}", flush=True)
        del layer0_tensor

        for layer_idx in range(13):
            t0 = time.time()
            layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
            rng = np.random.default_rng(BASE_SEED + layer_idx)
            frames_per_clip = sample_frames(
                layer_tensor, valid_token_counts, args.frames_per_item, rng
            )  # (n_items, F, D)

            # Per-source eff_rank, PR, MLE-ID
            per_source_frames: dict[str, np.ndarray] = {}
            for src in unique_sources:
                mask = sources == src
                if mask.sum() == 0:
                    continue
                src_frames = frames_per_clip[mask].reshape(-1, frames_per_clip.shape[-1]).astype(np.float64)
                per_source_frames[src] = src_frames

                eigvals = cov_eigvals(src_frames)
                er = effective_rank(eigvals)
                pr = participation_ratio(eigvals)
                mle = mle_intrinsic_dim(
                    src_frames, k=args.mle_k,
                    sample_size=min(args.mle_sample_size, src_frames.shape[0]),
                    rng=np.random.default_rng(BASE_SEED + layer_idx + 7),
                )
                stats_records.append({
                    "model": model_key,
                    "layer_idx": layer_idx,
                    "source": src,
                    "n_clips": int(mask.sum()),
                    "n_frames": int(src_frames.shape[0]),
                    "effective_rank": er,
                    "participation_ratio": pr,
                    "mle_id_k20": mle,
                    "is_bio": src in NATURE_SOURCES,
                })

            # Pairwise top-k subspace overlap between sources at this (model, layer)
            sources_with_frames = [s for s in per_source_frames if per_source_frames[s].shape[0] >= args.top_k + 1]
            for src_a, src_b in itertools.combinations(sources_with_frames, 2):
                basis_a = top_k_basis_via_cov(per_source_frames[src_a], args.top_k)
                basis_b = top_k_basis_via_cov(per_source_frames[src_b], args.top_k)
                cos = subspace_overlap(basis_a, basis_b)
                pairwise_records.append({
                    "model": model_key,
                    "layer_idx": layer_idx,
                    "source_a": src_a,
                    "source_b": src_b,
                    "is_bio_a": src_a in NATURE_SOURCES,
                    "is_bio_b": src_b in NATURE_SOURCES,
                    "same_bio_class": (src_a in NATURE_SOURCES) == (src_b in NATURE_SOURCES),
                    "mean_cos_principal_angles": cos,
                    "k": args.top_k,
                })
            del layer_tensor, per_source_frames, frames_per_clip
            print(f"  L{layer_idx:02d} done in {time.time() - t0:.1f}s", flush=True)

        # Save incrementally per model
        pd.DataFrame.from_records(stats_records).to_csv(
            args.output_dir / "per_source_stats.csv", index=False)
        pd.DataFrame.from_records(pairwise_records).to_csv(
            args.output_dir / "per_source_pairwise.csv", index=False)

    stats_df = pd.DataFrame.from_records(stats_records)
    pairwise_df = pd.DataFrame.from_records(pairwise_records)
    stats_df.to_csv(args.output_dir / "per_source_stats.csv", index=False)
    pairwise_df.to_csv(args.output_dir / "per_source_pairwise.csv", index=False)

    # Plot: eff_rank by source vs layer, one panel per model
    cmap = plt.get_cmap("tab10")
    sources_in_order = sorted(stats_df["source"].unique())
    color_for_src = {s: cmap(i) for i, s in enumerate(sources_in_order)}

    fig, axes = plt.subplots(1, len(args.models), figsize=(4.0 * len(args.models), 5.0), sharey=True)
    if len(args.models) == 1:
        axes = [axes]
    for ax, model_key in zip(axes, args.models):
        for src in sources_in_order:
            sub = stats_df[(stats_df["model"] == model_key) & (stats_df["source"] == src)].sort_values("layer_idx")
            if sub.empty:
                continue
            ax.plot(sub["layer_idx"], sub["effective_rank"], marker="o",
                    color=color_for_src[src], label=src, linewidth=1.2)
        ax.set_title(model_key, fontsize=10)
        ax.set_xlabel("layer index")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("frame-level effective rank (per source)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.suptitle("Per-source frame-level effective rank by layer")
    fig.savefig(args.output_dir / "per_source_effective_rank.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot: bio-bio vs bio-nonbio pair cos by layer (per model)
    fig, axes = plt.subplots(1, len(args.models), figsize=(4.0 * len(args.models), 5.0), sharey=True)
    if len(args.models) == 1:
        axes = [axes]
    for ax, model_key in zip(axes, args.models):
        sub = pairwise_df[pairwise_df["model"] == model_key]
        bio_bio = sub[sub["same_bio_class"] & sub["is_bio_a"]].groupby("layer_idx")["mean_cos_principal_angles"].mean()
        bio_non = sub[~sub["same_bio_class"]].groupby("layer_idx")["mean_cos_principal_angles"].mean()
        non_non = sub[sub["same_bio_class"] & ~sub["is_bio_a"]].groupby("layer_idx")["mean_cos_principal_angles"].mean()
        if not bio_bio.empty:
            ax.plot(bio_bio.index, bio_bio.values, marker="o", color="tab:green", label="bio-bio (avg)")
        if not non_non.empty:
            ax.plot(non_non.index, non_non.values, marker="o", color="tab:orange", label="non-bio – non-bio (avg)")
        if not bio_non.empty:
            ax.plot(bio_non.index, bio_non.values, marker="s", color="tab:red", label="bio – non-bio (avg)")
        ax.set_title(model_key, fontsize=10)
        ax.set_xlabel("layer index")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean cos top-10 (per-source pairs, avg)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.suptitle("Per-source pairwise subspace overlap, frame-level")
    fig.savefig(args.output_dir / "per_source_pairwise_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved per-source frame-level artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
