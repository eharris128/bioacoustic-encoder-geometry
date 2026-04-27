"""Step 2: frame-count sensitivity — does the 50-frame methodological choice
affect the headline numbers?

For a focal model, recompute frame-level eff_rank, MLE-ID(k=20), and bio-vs-
non-bio top-10 cos at frames_per_item ∈ {10, 30, 50, 100, 200}. If the
curves overlap (or scale predictably), the 50-frame choice in §2-§6 of
RESULTS.md is methodologically robust. If not, restate.

Default focal model: `sl_eat_bio_ssl_all` (the model with the strongest §4
finding). All 13 layers. Closes RESULTS.md §9.2.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/frame_count_sensitivity/

Usage:
    python step2_frame_count_sensitivity.py
    python step2_frame_count_sensitivity.py --model eat_all --frame_counts 10 50 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_bootstrap_cis import (
    BASE_SEED, MLE_K, MLE_SAMPLE_SIZE, NATURE_SOURCES, SUBSPACE_TOP_K,
    cov_eigvals, effective_rank, mle_intrinsic_dim, participation_ratio,
    subspace_overlap, top_k_basis_via_cov, load_layer_tensor,
)


MANIFEST_ID = "naturelm_by_source_100each_20260418T171459Z"
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "frame_count_sensitivity"

DEFAULT_MODEL = "sl_eat_bio_ssl_all"
DEFAULT_FRAME_COUNTS = [10, 30, 50, 100, 200]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--frame_counts", nargs="+", type=int, default=DEFAULT_FRAME_COUNTS)
    p.add_argument("--top_k", type=int, default=SUBSPACE_TOP_K)
    p.add_argument("--mle_k", type=int, default=MLE_K)
    p.add_argument("--mle_sample_size", type=int, default=MLE_SAMPLE_SIZE)
    return p.parse_args()


def sample_frames_per_item(
    layer_tensor: np.ndarray,
    valid_token_counts: np.ndarray,
    is_bio_per_item: np.ndarray,
    frames_per_item: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_items, t_max, d = layer_tensor.shape
    out = np.empty((n_items, frames_per_item, d), dtype=np.float32)
    is_bio_per_frame = np.empty((n_items, frames_per_item), dtype=bool)
    for i in range(n_items):
        valid = int(min(valid_token_counts[i], t_max))
        valid = max(valid, 1)
        if valid >= frames_per_item:
            f_idx = rng.choice(valid, frames_per_item, replace=False)
        else:
            f_idx = rng.choice(valid, frames_per_item, replace=True)
        out[i] = layer_tensor[i, f_idx, :]
        is_bio_per_frame[i] = is_bio_per_item[i]
    return out.reshape(-1, d), is_bio_per_frame.reshape(-1)


def compute_metrics(
    layer_tensor: np.ndarray,
    valid_token_counts: np.ndarray,
    is_bio_per_item: np.ndarray,
    frames_per_item: int,
    mle_k: int,
    mle_sample_size: int,
    top_k: int,
) -> dict:
    rng = np.random.default_rng(BASE_SEED)
    frames, is_bio = sample_frames_per_item(
        layer_tensor, valid_token_counts, is_bio_per_item, frames_per_item, rng
    )
    frames = frames.astype(np.float64)
    eigvals = cov_eigvals(frames)
    er = effective_rank(eigvals)
    pr = participation_ratio(eigvals)
    mle = mle_intrinsic_dim(frames, k=mle_k, sample_size=mle_sample_size, rng=rng)
    bio_frames = frames[is_bio]
    nonbio_frames = frames[~is_bio]
    if bio_frames.shape[0] >= top_k + 1 and nonbio_frames.shape[0] >= top_k + 1:
        basis_bio = top_k_basis_via_cov(bio_frames, top_k)
        basis_nonbio = top_k_basis_via_cov(nonbio_frames, top_k)
        cos = subspace_overlap(basis_bio, basis_nonbio)
    else:
        cos = float("nan")
    return {
        "effective_rank": er,
        "participation_ratio": pr,
        "mle_id_k20": mle,
        "bio_vs_nonbio_cos_top10": cos,
        "n_rows": int(frames.shape[0]),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shard_dir = args.roadmap_dir / args.model / "shards"
    if not shard_dir.exists():
        raise SystemExit(f"Shards missing for {args.model} at {shard_dir}")

    print(f"Frame-count sensitivity on {args.model}", flush=True)
    print(f"Frame counts: {args.frame_counts}", flush=True)

    # Need item metadata once
    layer0_tensor, sample_meta = load_layer_tensor(shard_dir, layer_idx=0)
    n_items = layer0_tensor.shape[0]
    valid_token_counts = np.array(
        [int(s.get("valid_token_count", layer0_tensor.shape[1])) for s in sample_meta]
    )
    sources = np.array([s.get("source_dataset", "") for s in sample_meta])
    is_bio_per_item = np.array([s in NATURE_SOURCES for s in sources])
    del layer0_tensor

    records: list[dict] = []
    for layer_idx in range(13):
        print(f"\nL{layer_idx:02d} loading...", flush=True, end="")
        t0 = time.time()
        layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
        print(f" {time.time() - t0:.1f}s", flush=True)
        for fc in args.frame_counts:
            metrics = compute_metrics(
                layer_tensor, valid_token_counts, is_bio_per_item,
                fc, args.mle_k, args.mle_sample_size, args.top_k,
            )
            metrics.update({"model": args.model, "layer_idx": layer_idx, "frames_per_item": fc, "n_items": n_items})
            records.append(metrics)
            print(
                f"  fc={fc:3d}  eff_rank={metrics['effective_rank']:7.2f}  "
                f"MLE-ID={metrics['mle_id_k20']:6.2f}  bio_cos={metrics['bio_vs_nonbio_cos_top10']:.3f}",
                flush=True,
            )
        del layer_tensor
        df = pd.DataFrame.from_records(records)
        df.to_csv(args.output_dir / f"frame_count_sensitivity_{args.model}.csv", index=False)

    df = pd.DataFrame.from_records(records)

    # Plots: each metric vs layer, with one curve per frame count
    cmap = plt.get_cmap("viridis")
    color_for = {fc: cmap(i / max(1, len(args.frame_counts) - 1)) for i, fc in enumerate(args.frame_counts)}

    for metric, ylabel, ylim in [
        ("effective_rank", "effective rank (frame-level)", None),
        ("mle_id_k20", "MLE-ID(k=20) intrinsic dim", None),
        ("bio_vs_nonbio_cos_top10", "bio-vs-non-bio top-10 cos", (0.0, 1.05)),
        ("participation_ratio", "participation ratio", None),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for fc in args.frame_counts:
            sub = df[df["frames_per_item"] == fc].sort_values("layer_idx")
            ax.plot(sub["layer_idx"], sub[metric], marker="o", color=color_for[fc],
                    label=f"{fc} frames/item")
        ax.set_xlabel("layer index")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{args.model} — {metric} vs layer, swept over frames/item")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.savefig(args.output_dir / f"frame_count_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved frame-count sensitivity artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
