"""Step 5 follow-up — bio direction across all 13 layers.

§5.2 found that sl_eat_all_ssl_all's L12 top eigenvector is the bio
classifier (|cos(top1, bio_axis)| = 0.74 NEW / 0.82 OLD). Open question:
does this direction emerge gradually across layers, or is it installed
specifically by the final transformer block?

Per (model, layer) compute:
  - top1 share (mode collapse strength)
  - |cos(top1, bio_axis)|
  - Cohen's d on top1 projection (bio vs nonbio along top1)
  - Cohen's d on bio_axis projection (bio vs nonbio along bio centroid axis)

A gradual story: top1_vs_bioaxis_cos rises smoothly from L0 to L12.
An L12-specific story: cos stays low at L0-L11, jumps at L12.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/layer_direction/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_bootstrap_cis import BASE_SEED, NATURE_SOURCES, load_layer_tensor


MANIFEST_ID = "naturelm_by_order_p100_m200_n200_20260427T222756Z"
DEFAULT_TAX_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}_taxonomic.jsonl")
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "layer_direction"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
FRAMES_PER_ITEM = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tax_manifest", type=Path, default=DEFAULT_TAX_MANIFEST)
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    return p.parse_args()


def load_taxonomy(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    if not path.exists():
        return by_id
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
        sources = np.array([s.get("source_dataset", "") for s in sample_meta])
        is_bio_per_item = np.array([s in NATURE_SOURCES for s in sources])
        del layer0_tensor

        for layer_idx in range(13):
            t0 = time.time()
            layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
            rng = np.random.default_rng(BASE_SEED + layer_idx)
            per_item = per_clip_frame_sample(
                layer_tensor, valid_token_counts, args.frames_per_item, rng,
            )

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

            bio_centroid = pooled[is_bio_per_item].mean(axis=0)
            nonbio_centroid = pooled[~is_bio_per_item].mean(axis=0)
            bio_axis = bio_centroid - nonbio_centroid
            bio_axis_norm = float(np.linalg.norm(bio_axis))
            bio_axis_unit = bio_axis / max(bio_axis_norm, 1e-12)
            proj_bioaxis = pooled @ bio_axis_unit

            top1_vs_bioaxis_cos = float(abs(top1 @ bio_axis_unit))

            bio_proj = proj_top1[is_bio_per_item]
            nonbio_proj = proj_top1[~is_bio_per_item]
            pooled_std = np.sqrt(0.5 * (bio_proj.std() ** 2 + nonbio_proj.std() ** 2))
            cohens_d_top1 = float((bio_proj.mean() - nonbio_proj.mean()) /
                                  max(pooled_std, 1e-12))

            bio_proj_ax = proj_bioaxis[is_bio_per_item]
            nonbio_proj_ax = proj_bioaxis[~is_bio_per_item]
            pooled_std_ax = np.sqrt(0.5 * (bio_proj_ax.std() ** 2 +
                                           nonbio_proj_ax.std() ** 2))
            cohens_d_bioaxis = float((bio_proj_ax.mean() - nonbio_proj_ax.mean()) /
                                     max(pooled_std_ax, 1e-12))

            summary_records.append({
                "model": model_key, "layer_idx": layer_idx,
                "top1_share": top1_share,
                "top1_vs_bioaxis_abs_cos": top1_vs_bioaxis_cos,
                "cohens_d_top1": cohens_d_top1,
                "cohens_d_bioaxis": cohens_d_bioaxis,
            })

            del layer_tensor, per_item, all_frames, centered, cov

            print(
                f"  L{layer_idx:02d}  top1_share={top1_share:.3f} "
                f"|cos|={top1_vs_bioaxis_cos:.3f} "
                f"d_top1={cohens_d_top1:+.2f} d_bio={cohens_d_bioaxis:+.2f} "
                f"({time.time()-t0:.1f}s)",
                flush=True,
            )

    summary_df = pd.DataFrame.from_records(summary_records)
    summary_df.to_csv(args.output_dir / "layer_direction_summary.csv", index=False)

    cmap = plt.get_cmap("tab10")
    color_for = {m: cmap(i) for i, m in enumerate(args.models)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    metrics = [
        ("top1_share", "top-1 eigenvalue share"),
        ("top1_vs_bioaxis_abs_cos", "|cos(top1, bio_axis)|"),
        ("cohens_d_top1", "Cohen's d (bio−nonbio) on top1"),
    ]
    for ax, (col, label) in zip(axes, metrics):
        for model_key in args.models:
            sub = summary_df[summary_df["model"] == model_key].sort_values("layer_idx")
            if sub.empty:
                continue
            is_baseline = model_key == "random_init_eat_seed42"
            ax.plot(sub["layer_idx"], sub[col],
                    marker="s" if is_baseline else "o",
                    linestyle="--" if is_baseline else "-",
                    color=color_for[model_key],
                    label=f"{model_key} (baseline)" if is_baseline else model_key)
        ax.set_xlabel("layer index")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Layer-wise bio direction across model x layer")
    fig.savefig(args.output_dir / "layer_direction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n=== |cos(top1, bio_axis)| per (model, layer) ===")
    pivot = summary_df.pivot(index="layer_idx", columns="model",
                             values="top1_vs_bioaxis_abs_cos")
    cols_in_order = [m for m in args.models if m in pivot.columns]
    print(pivot[cols_in_order].round(3).to_string())

    print(f"\nSaved layer_direction artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
