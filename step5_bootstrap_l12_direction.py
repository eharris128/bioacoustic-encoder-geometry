"""Step 5 follow-up — bootstrap CIs on the §5.2 / §5.4 L12 bio-classifier finding.

§5.2 / §5.4 reported that sl_eat_all_ssl_all's L12 top eigenvector
aligns with the bio centroid axis at |cos|=0.74 (NEW manifest), with
top1 share 0.62 and Cohen's d 0.52. These are point estimates over the
800 items in the manifest. This script resamples items with
replacement B=30 times, recomputes top1 share, |cos(top1, bio_axis)|,
and Cohen's d on top1 each bootstrap, and reports 5/50/95 percentiles.

Tests whether the §5.2 / §5.4 numbers are larger than their CIs and
the cross-model gap (sl_eat_all_ssl_all 0.74 vs others 0.04-0.25 cos)
holds with margin.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/bootstrap_l12_direction/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from step2_bootstrap_cis import BASE_SEED, NATURE_SOURCES, load_layer_tensor


MANIFEST_ID = "naturelm_by_order_p100_m200_n200_20260427T222756Z"
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "bootstrap_l12_direction"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
DEFAULT_LAYER = 12
FRAMES_PER_ITEM = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--layer_idx", type=int, default=DEFAULT_LAYER)
    p.add_argument("--num_bootstraps", type=int, default=30)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    return p.parse_args()


def per_clip_frame_sample(
    layer_tensor: np.ndarray,
    valid_token_counts: np.ndarray,
    item_indices: np.ndarray,
    frames_per_item: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample frames for a given subset of items (item_indices)."""
    t_max = layer_tensor.shape[1]
    d = layer_tensor.shape[2]
    out = np.empty((len(item_indices), frames_per_item, d), dtype=np.float32)
    for j, i in enumerate(item_indices):
        valid = max(int(min(valid_token_counts[i], t_max)), 1)
        if valid >= frames_per_item:
            f_idx = rng.choice(valid, frames_per_item, replace=False)
        else:
            f_idx = rng.choice(valid, frames_per_item, replace=True)
        out[j] = layer_tensor[i, f_idx, :]
    return out


def compute_metrics(per_item: np.ndarray, is_bio: np.ndarray) -> dict:
    all_frames = per_item.reshape(-1, per_item.shape[-1]).astype(np.float64)
    centered = all_frames - all_frames.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    top1 = eigvecs[:, 0]
    top1_share = float(eigvals[0] / eigvals.sum())

    pooled = per_item.mean(axis=1).astype(np.float64)
    proj_top1 = pooled @ top1

    bio_centroid = pooled[is_bio].mean(axis=0)
    nonbio_centroid = pooled[~is_bio].mean(axis=0)
    bio_axis = bio_centroid - nonbio_centroid
    bio_axis_unit = bio_axis / max(np.linalg.norm(bio_axis), 1e-12)
    proj_bioaxis = pooled @ bio_axis_unit

    top1_vs_bioaxis_cos = float(abs(top1 @ bio_axis_unit))

    bio_proj = proj_top1[is_bio]
    nonbio_proj = proj_top1[~is_bio]
    pooled_std = np.sqrt(0.5 * (bio_proj.std() ** 2 + nonbio_proj.std() ** 2))
    cohens_d_top1 = float((bio_proj.mean() - nonbio_proj.mean()) / max(pooled_std, 1e-12))

    bio_proj_ax = proj_bioaxis[is_bio]
    nonbio_proj_ax = proj_bioaxis[~is_bio]
    pooled_std_ax = np.sqrt(0.5 * (bio_proj_ax.std() ** 2 + nonbio_proj_ax.std() ** 2))
    cohens_d_bioaxis = float((bio_proj_ax.mean() - nonbio_proj_ax.mean()) /
                             max(pooled_std_ax, 1e-12))

    return {
        "top1_share": top1_share,
        "top1_vs_bioaxis_abs_cos": top1_vs_bioaxis_cos,
        "cohens_d_top1": cohens_d_top1,
        "cohens_d_bioaxis": cohens_d_bioaxis,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    boot_records: list[dict] = []
    summary_records: list[dict] = []

    for model_idx, model_key in enumerate(args.models):
        shard_dir = args.roadmap_dir / model_key / "shards"
        if not shard_dir.exists():
            print(f"WARN: shards missing for {model_key}, skipping", flush=True)
            continue
        print(f"\n=== {model_key} L{args.layer_idx} ({model_idx + 1}/{len(args.models)}) ===",
              flush=True)

        t0 = time.time()
        layer_tensor, sample_meta = load_layer_tensor(shard_dir, args.layer_idx)
        valid_token_counts = np.array(
            [int(s.get("valid_token_count", layer_tensor.shape[1])) for s in sample_meta]
        )
        sources = np.array([s.get("source_dataset", "") for s in sample_meta])
        is_bio_per_item = np.array([s in NATURE_SOURCES for s in sources])
        n_items = len(sample_meta)

        for b in range(args.num_bootstraps):
            seed = BASE_SEED + b * 1000 + args.layer_idx
            rng = np.random.default_rng(seed)
            item_idx = rng.integers(0, n_items, size=n_items)
            per_item = per_clip_frame_sample(
                layer_tensor, valid_token_counts, item_idx,
                args.frames_per_item, rng,
            )
            sub_is_bio = is_bio_per_item[item_idx]
            metrics = compute_metrics(per_item, sub_is_bio)
            boot_records.append({
                "model": model_key, "layer_idx": args.layer_idx,
                "bootstrap": b, **metrics,
            })

        df_model = pd.DataFrame.from_records(
            [r for r in boot_records if r["model"] == model_key]
        )
        for col in ["top1_share", "top1_vs_bioaxis_abs_cos",
                    "cohens_d_top1", "cohens_d_bioaxis"]:
            vals = df_model[col].to_numpy()
            summary_records.append({
                "model": model_key, "layer_idx": args.layer_idx, "metric": col,
                "p05": float(np.percentile(vals, 5)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
                "n_bootstraps": int(vals.size),
            })

        print(f"  done in {time.time()-t0:.1f}s", flush=True)
        del layer_tensor

    boot_df = pd.DataFrame.from_records(boot_records)
    summary_df = pd.DataFrame.from_records(summary_records)
    boot_df.to_csv(args.output_dir / "bootstrap_l12_direction.csv", index=False)
    summary_df.to_csv(args.output_dir / "bootstrap_l12_summary.csv", index=False)

    print(f"\n=== L{args.layer_idx} bootstrap CIs (B={args.num_bootstraps}) ===")
    for metric in ["top1_share", "top1_vs_bioaxis_abs_cos",
                   "cohens_d_top1", "cohens_d_bioaxis"]:
        print(f"\n--- {metric} ---")
        sub = summary_df[summary_df["metric"] == metric]
        for _, r in sub.iterrows():
            print(f"  {r['model']:25s}  {r['p50']:+.3f}  [{r['p05']:+.3f}, {r['p95']:+.3f}]")

    print(f"\nSaved bootstrap_l12_direction artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
