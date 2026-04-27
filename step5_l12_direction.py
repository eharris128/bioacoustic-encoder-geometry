"""Step 5 follow-up — identify what the dominant L12 direction in
sl_eat_all_ssl_all encodes.

Closes RESULTS.md §9.4: §5.1 found that sl_eat_all_ssl_all puts 61% of
its L12 covariance variance into a single direction. Likely
interpretation: an "is this animal vocalization?" classifier
direction installed by SSL fine-tuning. This script tests it directly.

Per (focal model, all manifest items):
  1. Load L12 frame-level activations (50 frames per item).
  2. Compute the top eigenvector of the centered covariance.
  3. Project each item's pooled L12 activation onto that eigenvector.
  4. Plot bio-vs-non-bio histograms of the projection.
  5. Report: separation between bio and non-bio along this single axis,
     plus comparison to the §4 (c_bio − c_nonbio) direction's projection.

If the top eigenvector strongly separates bio/nonbio, the §5.1
mode-collapse direction IS the bio classifier — confirming the SSL
fine-tune reduced the model to a single-feature classifier on this
non-bio-pretrained model. If it doesn't, something else is dominating
that direction.

Default focal model: sl_eat_all_ssl_all (the §5.1 collapsed model).
Optionally compare against eat_all (no SSL) for context.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/l12_direction/

Usage:
    python step5_l12_direction.py
    python step5_l12_direction.py --models sl_eat_all_ssl_all eat_all sl_eat_bio_ssl_all
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
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "l12_direction"

DEFAULT_MODELS = [
    "sl_eat_all_ssl_all",
    "sl_eat_bio_ssl_all",
    "eat_all",
    "eat_bio",
    "random_init_eat_seed42",
]
DEFAULT_LAYER = 12
FRAMES_PER_ITEM = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tax_manifest", type=Path, default=DEFAULT_TAX_MANIFEST)
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--layer_idx", type=int, default=DEFAULT_LAYER)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
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

    per_item_records: list[dict] = []
    summary_records: list[dict] = []

    for model_idx, model_key in enumerate(args.models):
        shard_dir = args.roadmap_dir / model_key / "shards"
        if not shard_dir.exists():
            print(f"WARN: shards missing for {model_key}, skipping", flush=True)
            continue
        print(f"\n=== {model_key} L{args.layer_idx} ({model_idx + 1}/{len(args.models)}) ===", flush=True)

        t0 = time.time()
        layer_tensor, sample_meta = load_layer_tensor(shard_dir, args.layer_idx)
        valid_token_counts = np.array(
            [int(s.get("valid_token_count", layer_tensor.shape[1])) for s in sample_meta]
        )
        sources = np.array([s.get("source_dataset", "") for s in sample_meta])
        is_bio_per_item = np.array([s in NATURE_SOURCES for s in sources])
        # Use taxonomic class as a finer label
        sample_class = np.array(
            [taxonomy.get(s.get("id", ""), {}).get("class", "") for s in sample_meta]
        )

        rng = np.random.default_rng(BASE_SEED + args.layer_idx)
        per_item = per_clip_frame_sample(
            layer_tensor, valid_token_counts, args.frames_per_item, rng,
        )  # (n_items, F, D)

        # Compute top eigenvector via covariance over all frames
        all_frames = per_item.reshape(-1, per_item.shape[-1]).astype(np.float64)
        centered = all_frames - all_frames.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        top1 = eigvecs[:, 0]
        top1_share = float(eigvals[0] / eigvals.sum())
        top2 = eigvecs[:, 1]

        # Per-item pooled activations (mean over the 50 frames)
        pooled = per_item.mean(axis=1).astype(np.float64)

        # Project pooled onto top eigenvector
        proj_top1 = pooled @ top1
        proj_top2 = pooled @ top2

        # Also: bio-vs-nonbio centroid axis (the §4 axis)
        bio_centroid = pooled[is_bio_per_item].mean(axis=0)
        nonbio_centroid = pooled[~is_bio_per_item].mean(axis=0)
        bio_axis = bio_centroid - nonbio_centroid
        bio_axis_norm = float(np.linalg.norm(bio_axis))
        bio_axis_unit = bio_axis / max(bio_axis_norm, 1e-12)
        proj_bioaxis = pooled @ bio_axis_unit

        # Cosine between top1 eigenvector and the bio-axis (signed, then |.|)
        top1_vs_bioaxis_cos = float(abs(top1 @ bio_axis_unit))

        # Diagnostic separation metrics
        bio_proj = proj_top1[is_bio_per_item]
        nonbio_proj = proj_top1[~is_bio_per_item]
        # Cohen's d on top1
        pooled_std = np.sqrt(0.5 * (bio_proj.std() ** 2 + nonbio_proj.std() ** 2))
        cohens_d = float((bio_proj.mean() - nonbio_proj.mean()) / max(pooled_std, 1e-12))

        # Same on bio-axis
        bio_proj_ax = proj_bioaxis[is_bio_per_item]
        nonbio_proj_ax = proj_bioaxis[~is_bio_per_item]
        pooled_std_ax = np.sqrt(0.5 * (bio_proj_ax.std() ** 2 + nonbio_proj_ax.std() ** 2))
        cohens_d_bioaxis = float((bio_proj_ax.mean() - nonbio_proj_ax.mean()) /
                                 max(pooled_std_ax, 1e-12))

        for i, s in enumerate(sample_meta):
            per_item_records.append({
                "model": model_key, "layer_idx": args.layer_idx,
                "id": s.get("id", ""),
                "file_name": s.get("file_name", ""),
                "source_dataset": s.get("source_dataset", ""),
                "class": sample_class[i],
                "is_bio": bool(is_bio_per_item[i]),
                "proj_top1": float(proj_top1[i]),
                "proj_top2": float(proj_top2[i]),
                "proj_bio_axis": float(proj_bioaxis[i]),
            })

        summary_records.append({
            "model": model_key, "layer_idx": args.layer_idx,
            "top1_share": top1_share,
            "top1_vs_bioaxis_abs_cos": top1_vs_bioaxis_cos,
            "cohens_d_top1": cohens_d,
            "cohens_d_bioaxis": cohens_d_bioaxis,
            "bio_proj_top1_mean": float(bio_proj.mean()),
            "nonbio_proj_top1_mean": float(nonbio_proj.mean()),
            "bio_proj_top1_std": float(bio_proj.std()),
            "nonbio_proj_top1_std": float(nonbio_proj.std()),
            "n_bio": int(is_bio_per_item.sum()),
            "n_nonbio": int((~is_bio_per_item).sum()),
        })

        print(
            f"  top1 share = {top1_share:.3f} | "
            f"|cos(top1, bio_axis)| = {top1_vs_bioaxis_cos:.3f} | "
            f"Cohen's d on top1 = {cohens_d:.2f} | "
            f"Cohen's d on bio-axis = {cohens_d_bioaxis:.2f} | "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

        del layer_tensor, per_item, all_frames, centered, cov

    per_item_df = pd.DataFrame.from_records(per_item_records)
    summary_df = pd.DataFrame.from_records(summary_records)
    per_item_df.to_csv(args.output_dir / "l12_per_item_projections.csv", index=False)
    summary_df.to_csv(args.output_dir / "l12_summary.csv", index=False)

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(1, len(args.models),
                             figsize=(4.5 * len(args.models), 5), sharey=True)
    if len(args.models) == 1:
        axes = [axes]
    for ax, model_key in zip(axes, args.models):
        sub = per_item_df[per_item_df["model"] == model_key]
        if sub.empty:
            continue
        bio_vals = sub[sub["is_bio"]]["proj_top1"].to_numpy()
        non_vals = sub[~sub["is_bio"]]["proj_top1"].to_numpy()
        bins = np.linspace(
            min(bio_vals.min(), non_vals.min()),
            max(bio_vals.max(), non_vals.max()),
            40,
        )
        ax.hist(bio_vals, bins=bins, alpha=0.55, color="tab:green",
                 label=f"bio (n={len(bio_vals)})")
        ax.hist(non_vals, bins=bins, alpha=0.55, color="tab:orange",
                 label=f"non-bio (n={len(non_vals)})")
        s = summary_df[summary_df["model"] == model_key].iloc[0]
        ax.set_title(f"{model_key}\ntop1 share={s['top1_share']:.2f}, d={s['cohens_d_top1']:.2f}",
                      fontsize=9)
        ax.set_xlabel("projection onto L12 top eigenvector")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("# items")
    fig.suptitle(f"L{args.layer_idx} top-eigenvector projections — bio vs non-bio")
    fig.savefig(args.output_dir / f"l{args.layer_idx}_top1_projections.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Headline tables
    print(f"\n=== L{args.layer_idx} top eigenvector vs bio direction ===")
    print(summary_df[["model", "top1_share", "top1_vs_bioaxis_abs_cos",
                      "cohens_d_top1", "cohens_d_bioaxis"]].round(3).to_string(index=False))

    print(f"\nSaved L{args.layer_idx} direction artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
