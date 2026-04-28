"""Step 5 follow-up — within-clip frame structure across (model, layer).

§5.1 found L12 mode collapse along a single direction. This script asks
the complementary question: how much variance lives *within* a single
clip's 50 frames, vs *between* clips? If trained models compress each
clip's temporal trajectory into ~one point in activation space, the
within-clip variance ratio collapses; the model treats every clip as
static input rather than a 10-second temporal stream.

Per (model, layer, item):
  1. Load L_layer activations, sample 50 frames per item from valid range.
  2. For each clip, compute within-clip variance (mean squared frame
     deviation from clip centroid).
  3. For each clip, compute its centroid's distance to the per-class
     global centroid (between-clip contribution).
  4. Aggregate: ratio = mean(within) / (mean(within) + between_var).
     - 0.0 = clips are collapsed into points; all variance is between-clip.
     - 1.0 = clips are temporally rich; almost no between-clip structure.

Hypothesis (from §5.1): late layers in trained models should have low
within/total ratio (clips compressed to near-points), while random_init
should keep the ratio higher (raw acoustic variation preserved).

Default focal layers: 0, 5, 9, 12 across all 5 models.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/within_clip/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_bootstrap_cis import BASE_SEED, NATURE_SOURCES, load_layer_tensor


MANIFEST_ID = "naturelm_by_order_p100_m200_n200_20260427T222756Z"
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "within_clip"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
DEFAULT_LAYERS = list(range(13))
FRAMES_PER_ITEM = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    return p.parse_args()


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

    summary_records: list[dict] = []
    per_item_records: list[dict] = []

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

        for layer_idx in args.layers:
            t0 = time.time()
            layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
            rng = np.random.default_rng(BASE_SEED + layer_idx)
            per_item = per_clip_frame_sample(
                layer_tensor, valid_token_counts, args.frames_per_item, rng,
            )  # (n_items, F, D)

            per_item_f64 = per_item.astype(np.float64)
            clip_centroids = per_item_f64.mean(axis=1)  # (n, D)
            # within-clip variance (mean squared frame deviation from clip centroid)
            deviations = per_item_f64 - clip_centroids[:, None, :]
            within_per_clip = np.mean(np.sum(deviations ** 2, axis=2), axis=1)

            # global between-clip variance
            global_centroid = clip_centroids.mean(axis=0)
            between_per_clip = np.sum((clip_centroids - global_centroid) ** 2, axis=1)

            # bio / nonbio split
            for split_label, mask in [
                ("all", np.ones(len(sample_meta), dtype=bool)),
                ("bio", is_bio_per_item),
                ("nonbio", ~is_bio_per_item),
            ]:
                if mask.sum() < 2:
                    continue
                w = float(within_per_clip[mask].mean())
                b = float(between_per_clip[mask].mean())
                ratio = w / max(w + b, 1e-12)
                summary_records.append({
                    "model": model_key,
                    "layer_idx": layer_idx,
                    "split": split_label,
                    "n": int(mask.sum()),
                    "mean_within_clip_var": w,
                    "mean_between_clip_var": b,
                    "within_total_ratio": ratio,
                })

            for i, s in enumerate(sample_meta):
                per_item_records.append({
                    "model": model_key,
                    "layer_idx": layer_idx,
                    "id": s.get("id", ""),
                    "source_dataset": s.get("source_dataset", ""),
                    "is_bio": bool(is_bio_per_item[i]),
                    "within_var": float(within_per_clip[i]),
                    "between_dist": float(np.sqrt(between_per_clip[i])),
                })

            del layer_tensor, per_item, per_item_f64
            print(
                f"  L{layer_idx:02d}  within={summary_records[-3]['mean_within_clip_var']:.2f} "
                f"between={summary_records[-3]['mean_between_clip_var']:.2f} "
                f"ratio={summary_records[-3]['within_total_ratio']:.3f} "
                f"({time.time()-t0:.1f}s)",
                flush=True,
            )

    summary_df = pd.DataFrame.from_records(summary_records)
    per_item_df = pd.DataFrame.from_records(per_item_records)
    summary_df.to_csv(args.output_dir / "within_clip_summary.csv", index=False)
    per_item_df.to_csv(args.output_dir / "within_clip_per_item.csv", index=False)

    # ---------------- Plot: ratio per (model, layer), all-split ----------------
    cmap = plt.get_cmap("tab10")
    color_for = {m: cmap(i) for i, m in enumerate(args.models)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_key in args.models:
        sub = summary_df[(summary_df["model"] == model_key) &
                         (summary_df["split"] == "all")].sort_values("layer_idx")
        if sub.empty:
            continue
        is_baseline = model_key == "random_init_eat_seed42"
        ax.plot(sub["layer_idx"], sub["within_total_ratio"],
                marker="s" if is_baseline else "o",
                linestyle="--" if is_baseline else "-",
                color=color_for[model_key],
                label=f"{model_key} (baseline)" if is_baseline else model_key)
    ax.set_xlabel("layer index")
    ax.set_ylabel("within-clip / total variance")
    ax.set_title("Within-clip frame variance ratio — clip collapse across layers")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(args.output_dir / "within_clip_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n=== within / total variance ratio per (model, layer) [all] ===")
    pivot = summary_df[summary_df["split"] == "all"].pivot(
        index="layer_idx", columns="model", values="within_total_ratio")
    cols_in_order = [m for m in args.models if m in pivot.columns]
    print(pivot[cols_in_order].round(3).to_string())

    print(f"\nSaved within-clip artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
