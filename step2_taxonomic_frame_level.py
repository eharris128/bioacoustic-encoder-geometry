"""Step 2-taxonomic: per-Class and per-Order frame-level eff_rank,
MLE-ID(k=20), and pairwise top-10 subspace overlap.

Per-Class scope: Aves (n=271) and Mammalia (n=119) — the two
well-powered classes in the 600-sample manifest. Amphibia (n=6) and
Insecta (n=2) are too thin for stable eff_rank/MLE-ID and are skipped.

Per-Order scope: Passeriformes (n=207) vs "other-Aves" (n=64, pooled
across the 17 minority orders) — at this manifest scale the individual
non-Passeriformes orders are too thin (<=11 samples each) for separate
analysis.

Geometric complement to the teammate's probes: probes peak Class at L5
(Aves vs Amphibia vs Mammalia) and Order at L9 (4 bird orders) — does
the directional separation between our coarser groupings peak at the
same layers?

Output: artifacts/comparisons/<manifest>/nway_eat_all4/taxonomic_frame_level/

Usage:
    python step2_taxonomic_frame_level.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_bootstrap_cis import (
    BASE_SEED, MLE_K, MLE_SAMPLE_SIZE, SUBSPACE_TOP_K,
    cov_eigvals, effective_rank, load_layer_tensor, mle_intrinsic_dim,
    participation_ratio, subspace_overlap, top_k_basis_via_cov,
)


MANIFEST_ID = "naturelm_by_source_100each_20260418T171459Z"
DEFAULT_TAX_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}_taxonomic.jsonl")
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "taxonomic_frame_level"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
FRAMES_PER_ITEM = 50
CLASS_TARGETS = ("Aves", "Mammalia")  # Amphibia / Insecta too thin


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tax_manifest", type=Path, default=DEFAULT_TAX_MANIFEST)
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    p.add_argument("--top_k", type=int, default=SUBSPACE_TOP_K)
    p.add_argument("--mle_k", type=int, default=MLE_K)
    p.add_argument("--mle_sample_size", type=int, default=MLE_SAMPLE_SIZE)
    return p.parse_args()


def load_taxonomy_for_manifest(tax_manifest: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with tax_manifest.open() as f:
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


def compute_group_stats(
    frames: np.ndarray, args: argparse.Namespace, rng: np.random.Generator,
) -> dict:
    if frames.shape[0] < args.top_k + 1:
        return {"effective_rank": float("nan"), "participation_ratio": float("nan"),
                "mle_id_k20": float("nan"), "n_frames": int(frames.shape[0])}
    frames = frames.astype(np.float64)
    eigvals = cov_eigvals(frames)
    er = effective_rank(eigvals)
    pr = participation_ratio(eigvals)
    mle = mle_intrinsic_dim(
        frames, k=args.mle_k,
        sample_size=min(args.mle_sample_size, frames.shape[0]),
        rng=rng,
    )
    return {"effective_rank": er, "participation_ratio": pr, "mle_id_k20": mle,
            "n_frames": int(frames.shape[0])}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy_for_manifest(args.tax_manifest)
    print(f"Taxonomic manifest: {len(taxonomy)} records", flush=True)

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
        n_items = len(sample_meta)
        # Map from manifest order to taxonomic info
        sample_class = np.array(
            [taxonomy.get(s.get("id", ""), {}).get("class", "") for s in sample_meta]
        )
        sample_order = np.array(
            [taxonomy.get(s.get("id", ""), {}).get("order", "") for s in sample_meta]
        )
        del layer0_tensor

        for layer_idx in range(13):
            t0 = time.time()
            layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
            rng = np.random.default_rng(BASE_SEED + layer_idx)
            per_item = per_clip_frame_sample(
                layer_tensor, valid_token_counts, args.frames_per_item, rng,
            )

            # Per-Class stats: Aves, Mammalia
            per_class_frames: dict[str, np.ndarray] = {}
            for cls in CLASS_TARGETS:
                mask = sample_class == cls
                if mask.sum() == 0:
                    continue
                frames = per_item[mask].reshape(-1, per_item.shape[-1])
                per_class_frames[cls] = frames.astype(np.float64)
                stats = compute_group_stats(per_class_frames[cls], args, rng)
                stats_records.append({
                    "model": model_key, "layer_idx": layer_idx,
                    "level": "class", "group": cls,
                    "n_clips": int(mask.sum()),
                    **stats,
                })

            # Pairwise Class subspace overlap (Aves vs Mammalia)
            classes_with_frames = [c for c in CLASS_TARGETS if c in per_class_frames]
            if len(classes_with_frames) == 2:
                a, b = classes_with_frames
                basis_a = top_k_basis_via_cov(per_class_frames[a], args.top_k)
                basis_b = top_k_basis_via_cov(per_class_frames[b], args.top_k)
                cos = subspace_overlap(basis_a, basis_b)
                pairwise_records.append({
                    "model": model_key, "layer_idx": layer_idx,
                    "level": "class",
                    "group_a": a, "group_b": b,
                    "k": args.top_k,
                    "mean_cos_principal_angles": cos,
                })

            # Per-Order stats within Aves: Passeriformes vs other-Aves-pooled
            aves_mask = sample_class == "Aves"
            passer_mask = (sample_order == "Passeriformes") & aves_mask
            other_aves_mask = aves_mask & ~passer_mask
            per_order_frames: dict[str, np.ndarray] = {}
            for label, mask in [("Passeriformes", passer_mask),
                                ("other-Aves", other_aves_mask)]:
                if mask.sum() == 0:
                    continue
                frames = per_item[mask].reshape(-1, per_item.shape[-1]).astype(np.float64)
                per_order_frames[label] = frames
                stats = compute_group_stats(frames, args, rng)
                stats_records.append({
                    "model": model_key, "layer_idx": layer_idx,
                    "level": "order", "group": label,
                    "n_clips": int(mask.sum()),
                    **stats,
                })

            if "Passeriformes" in per_order_frames and "other-Aves" in per_order_frames:
                basis_a = top_k_basis_via_cov(per_order_frames["Passeriformes"], args.top_k)
                basis_b = top_k_basis_via_cov(per_order_frames["other-Aves"], args.top_k)
                cos = subspace_overlap(basis_a, basis_b)
                pairwise_records.append({
                    "model": model_key, "layer_idx": layer_idx,
                    "level": "order",
                    "group_a": "Passeriformes", "group_b": "other-Aves",
                    "k": args.top_k,
                    "mean_cos_principal_angles": cos,
                })

            del layer_tensor, per_item, per_class_frames, per_order_frames
            print(f"  L{layer_idx:02d} done in {time.time() - t0:.1f}s", flush=True)

        pd.DataFrame.from_records(stats_records).to_csv(
            args.output_dir / "taxonomic_stats.csv", index=False)
        pd.DataFrame.from_records(pairwise_records).to_csv(
            args.output_dir / "taxonomic_pairwise.csv", index=False)

    stats_df = pd.DataFrame.from_records(stats_records)
    pairwise_df = pd.DataFrame.from_records(pairwise_records)
    stats_df.to_csv(args.output_dir / "taxonomic_stats.csv", index=False)
    pairwise_df.to_csv(args.output_dir / "taxonomic_pairwise.csv", index=False)

    # ---------------- Plots ----------------
    cmap_models = plt.get_cmap("tab10")
    color_for_model = {m: cmap_models(i) for i, m in enumerate(args.models)}

    # Aves vs Mammalia subspace overlap by layer
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sub = pairwise_df[pairwise_df["level"] == "class"]
    for model_key in args.models:
        m = sub[sub["model"] == model_key].sort_values("layer_idx")
        if m.empty:
            continue
        is_baseline = model_key == "random_init_eat_seed42"
        ax.plot(m["layer_idx"], m["mean_cos_principal_angles"],
                marker="s" if is_baseline else "o",
                linestyle="--" if is_baseline else "-",
                color=color_for_model[model_key],
                label=f"{model_key} (baseline)" if is_baseline else model_key)
    ax.set_xlabel("layer index")
    ax.set_ylabel("mean cos top-10 principal angles")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Aves vs Mammalia frame-level subspace overlap")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.savefig(args.output_dir / "class_aves_vs_mammalia_cos.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Passeriformes vs other-Aves
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sub = pairwise_df[pairwise_df["level"] == "order"]
    for model_key in args.models:
        m = sub[sub["model"] == model_key].sort_values("layer_idx")
        if m.empty:
            continue
        is_baseline = model_key == "random_init_eat_seed42"
        ax.plot(m["layer_idx"], m["mean_cos_principal_angles"],
                marker="s" if is_baseline else "o",
                linestyle="--" if is_baseline else "-",
                color=color_for_model[model_key],
                label=f"{model_key} (baseline)" if is_baseline else model_key)
    ax.set_xlabel("layer index")
    ax.set_ylabel("mean cos top-10 principal angles")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Passeriformes vs other-Aves frame-level subspace overlap")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.savefig(args.output_dir / "order_passer_vs_other_aves_cos.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-Class eff_rank by layer (4 panels: Aves, Mammalia, both)
    cls_sub = stats_df[stats_df["level"] == "class"]
    fig, axes = plt.subplots(1, len(CLASS_TARGETS), figsize=(5.0 * len(CLASS_TARGETS), 5.0), sharey=True)
    if len(CLASS_TARGETS) == 1:
        axes = [axes]
    for ax, cls in zip(axes, CLASS_TARGETS):
        for model_key in args.models:
            m = cls_sub[(cls_sub["model"] == model_key) & (cls_sub["group"] == cls)].sort_values("layer_idx")
            if m.empty:
                continue
            is_baseline = model_key == "random_init_eat_seed42"
            ax.plot(m["layer_idx"], m["effective_rank"],
                    marker="s" if is_baseline else "o",
                    linestyle="--" if is_baseline else "-",
                    color=color_for_model[model_key],
                    label=f"{model_key} (baseline)" if is_baseline else model_key)
        ax.set_xlabel("layer index")
        ax.set_title(f"{cls}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("frame-level effective rank")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.suptitle("Per-class effective rank by layer")
    fig.savefig(args.output_dir / "class_effective_rank.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Headline table ----------------
    print("\n=== Aves vs Mammalia mean cos top-10, per (model, layer) ===")
    pivot = pairwise_df[pairwise_df["level"] == "class"].pivot(
        index="layer_idx", columns="model", values="mean_cos_principal_angles")
    pivot = pivot[args.models] if all(m in pivot.columns for m in args.models) else pivot
    print(pivot.round(3).to_string())

    print("\n=== Passeriformes vs other-Aves mean cos top-10, per (model, layer) ===")
    pivot = pairwise_df[pairwise_df["level"] == "order"].pivot(
        index="layer_idx", columns="model", values="mean_cos_principal_angles")
    pivot = pivot[args.models] if all(m in pivot.columns for m in args.models) else pivot
    print(pivot.round(3).to_string())

    print(f"\nSaved taxonomic frame-level artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
