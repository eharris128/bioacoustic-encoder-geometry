"""Step 2 baseline: init variability across multiple random-init seeds.

Validates that the headline random-init numbers in §2 of RESULTS.md are
stable across init by running the same Tier 1 frame-level analysis on
multiple random-init seeds and reporting the spread.

Reads shards from
`artifacts/roadmap_part1/<manifest>/random_init_eat_seed<NN>/shards/` for
each seed, computes frame-level eff_rank / TwoNN / MLE-ID and frame-level
bio-vs-non-bio top-10 subspace overlap, and writes:

- per-seed CSVs (frame_per_layer_stats_<seed>.csv, frame_bio_vs_nonbio_<seed>.csv)
- a combined summary (seed_spread_<metric>.csv) with min/mean/max across seeds
- overlay plots showing all seeds on the same axis

Seed names follow the convention `random_init_eat_seed<NN>` with a 2-digit
zero-padded suffix (e.g. seed07, seed13, seed42).

Usage:
    python step2_random_init_variability.py
    python step2_random_init_variability.py --seeds 7,13,42,100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_tier1_frame_level import (
    SUBSPACE_TOP_K,
    load_frame_level,
    compute_frame_stats,
    compute_bio_vs_nonbio,
)


MANIFEST_ID = "naturelm_by_source_100each_20260418T171459Z"
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "random_init_variability"

DEFAULT_SEEDS = [7, 13, 42]
FRAMES_PER_ITEM = 50
TWONN_SAMPLE_SIZE = 10000
MLE_K = 20
MLE_SAMPLE_SIZE = 10000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seeds", type=str,
                   default=",".join(str(s) for s in DEFAULT_SEEDS),
                   help="Comma-separated seed integers, e.g. '7,13,42'.")
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    p.add_argument("--twonn_sample_size", type=int, default=TWONN_SAMPLE_SIZE)
    p.add_argument("--mle_k", type=int, default=MLE_K)
    p.add_argument("--mle_sample_size", type=int, default=MLE_SAMPLE_SIZE)
    p.add_argument("--top_k", type=int, default=SUBSPACE_TOP_K)
    return p.parse_args()


def model_key_for_seed(seed: int) -> str:
    return f"random_init_eat_seed{seed:02d}"


def plot_seed_overlay(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    seeds: list[int],
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for idx, seed in enumerate(seeds):
        sub = df[df["seed"] == seed].sort_values("layer_idx")
        ax.plot(sub["layer_idx"], sub[metric], marker="o",
                color=cmap(idx), linewidth=1.5,
                label=f"seed={seed}")
    ax.set_xlabel("layer index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_spread(df: pd.DataFrame, metric: str, group_by: str = "layer_idx") -> pd.DataFrame:
    return (
        df.groupby(group_by)[metric]
        .agg(min="min", mean="mean", max="max", std="std")
        .reset_index()
        .rename(columns={"min": f"{metric}_min", "mean": f"{metric}_mean",
                         "max": f"{metric}_max", "std": f"{metric}_std"})
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"Loading {len(seeds)} random-init seeds: {seeds}", flush=True)

    all_frame_stats: list[pd.DataFrame] = []
    all_frame_bio: list[pd.DataFrame] = []

    for seed in seeds:
        model_key = model_key_for_seed(seed)
        stats_csv = args.output_dir / f"frame_per_layer_stats_seed{seed:02d}.csv"
        bio_csv = args.output_dir / f"frame_bio_vs_nonbio_seed{seed:02d}.csv"

        if stats_csv.exists() and bio_csv.exists():
            # Reuse cached per-seed stats so we don't need shards on disk to
            # rebuild summary tables/plots after a seed has been deleted.
            print(f"\n=== seed={seed} ({model_key}): using cached CSVs ===", flush=True)
            stats = pd.read_csv(stats_csv)
            bio = pd.read_csv(bio_csv)
            all_frame_stats.append(stats)
            all_frame_bio.append(bio)
            continue

        shard_dir = args.roadmap_dir / model_key / "shards"
        if not shard_dir.exists():
            raise SystemExit(
                f"Shards not found at {shard_dir} and no cached CSVs at "
                f"{stats_csv} / {bio_csv}. Run extraction for {model_key} first."
            )

        print(f"\n=== seed={seed} ({model_key}) ===", flush=True)
        frame_acts, sample_meta, layer_names = load_frame_level(shard_dir, args.frames_per_item)

        stats = compute_frame_stats(
            frame_acts, layer_names,
            twonn_sample_size=args.twonn_sample_size,
            mle_k=args.mle_k,
            mle_sample_size=args.mle_sample_size,
        )
        stats.insert(0, "seed", seed)
        stats.to_csv(stats_csv, index=False)
        all_frame_stats.append(stats)

        bio = compute_bio_vs_nonbio(
            frame_acts, sample_meta, args.frames_per_item, layer_names, k=args.top_k
        )
        bio.insert(0, "seed", seed)
        bio.to_csv(bio_csv, index=False)
        all_frame_bio.append(bio)

        del frame_acts

    frame_stats_all = pd.concat(all_frame_stats, ignore_index=True)
    frame_stats_all.to_csv(args.output_dir / "frame_per_layer_stats_all_seeds.csv", index=False)
    frame_bio_all = pd.concat(all_frame_bio, ignore_index=True)
    frame_bio_all.to_csv(args.output_dir / "frame_bio_vs_nonbio_all_seeds.csv", index=False)

    # ---- Seed-spread summaries ----
    print("\n=== Seed-spread per layer ===", flush=True)
    spread_records = []
    for metric in ["effective_rank", "twonn_id", "mle_id_k20", "participation_ratio"]:
        s = summarize_spread(frame_stats_all, metric)
        s["metric"] = metric
        spread_records.append(s)
    spread_df = pd.concat(spread_records, ignore_index=True)
    spread_df.to_csv(args.output_dir / "seed_spread_frame_stats.csv", index=False)

    bio_spread = summarize_spread(frame_bio_all, "mean_cos_principal_angles_frame")
    bio_spread["metric"] = "mean_cos_principal_angles_frame"
    bio_spread.to_csv(args.output_dir / "seed_spread_bio_vs_nonbio.csv", index=False)

    # ---- Plots ----
    print("\n=== Seed-overlay plots ===", flush=True)
    plot_seed_overlay(
        frame_stats_all, "effective_rank",
        ylabel="effective rank (frame-level)",
        title="Random-init init variability: frame-level effective rank",
        output_path=args.output_dir / "init_variability_effective_rank.png",
        seeds=seeds,
    )
    plot_seed_overlay(
        frame_stats_all, "mle_id_k20",
        ylabel=f"MLE-ID intrinsic dim (k={args.mle_k}, frame-level)",
        title=f"Random-init init variability: MLE-ID(k={args.mle_k})",
        output_path=args.output_dir / "init_variability_mle_id.png",
        seeds=seeds,
    )
    plot_seed_overlay(
        frame_stats_all, "twonn_id",
        ylabel="TwoNN intrinsic dim (k=2, frame-level)",
        title="Random-init init variability: TwoNN(k=2)",
        output_path=args.output_dir / "init_variability_twonn.png",
        seeds=seeds,
    )
    plot_seed_overlay(
        frame_bio_all, "mean_cos_principal_angles_frame",
        ylabel=f"mean cos principal angles (top-{args.top_k}, frame-level)",
        title=f"Random-init init variability: bio-vs-nonbio top-{args.top_k}",
        output_path=args.output_dir / "init_variability_bio_vs_nonbio.png",
        seeds=seeds,
        ylim=(0.0, 1.05),
    )

    # ---- Headline tables ----
    print("\n=== Headline numbers ===", flush=True)
    for metric, label in [
        ("effective_rank", "Frame-level effective rank"),
        ("mle_id_k20", f"Frame-level MLE-ID(k={args.mle_k})"),
    ]:
        pivot = frame_stats_all.pivot(index="layer_idx", columns="seed", values=metric)
        pivot["max-min"] = pivot.max(axis=1) - pivot.min(axis=1)
        pivot["std"] = frame_stats_all.groupby("layer_idx")[metric].std().values
        print(f"\n{label} (per layer, across seeds):")
        print(pivot.round(2).to_string())

    pivot_bio = frame_bio_all.pivot(
        index="layer_idx", columns="seed", values="mean_cos_principal_angles_frame"
    )
    pivot_bio["max-min"] = pivot_bio.max(axis=1) - pivot_bio.min(axis=1)
    print(f"\nFrame-level bio-vs-nonbio cos top-{args.top_k} (per layer, across seeds):")
    print(pivot_bio.round(3).to_string())

    print(f"\nSaved artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
