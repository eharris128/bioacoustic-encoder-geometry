"""Step 2 baseline: random-init EAT vs the four trained EAT-family models.

Loads the random-init shards produced by `collect_esp_aves2_activations.py`
with `--models random_init_eat_seed42` and computes:

- Frame-level eff_rank / PR / TwoNN(k=2) / MLE-ID(k=20) per layer
- Pooled (mean over valid tokens) eff_rank / PR / TwoNN per layer
- Frame-level bio-vs-non-bio top-k=10 subspace overlap per layer
- Pooled bio-vs-non-bio top-k=10 subspace overlap per layer

Then loads the existing 4-model results (Tier 1 + Step 2 spectral + Step 2
subspace) and produces 5-way comparison CSVs and plots so absolute numbers
in the trained models can be anchored against an architecture-only baseline.

Reuses geometry primitives from `step2_tier1_frame_level.py` to keep the
metric definitions identical.

Usage:
    python step2_random_init_compare.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from step2_tier1_frame_level import (
    NATURE_SOURCES,
    SUBSPACE_TOP_K,
    SEED,
    per_layer_spectrum,
    effective_rank,
    participation_ratio,
    twonn_intrinsic_dim,
    mle_intrinsic_dim,
    top_k_basis,
    subspace_overlap,
    load_frame_level,
    compute_frame_stats,
    compute_bio_vs_nonbio,
)


MANIFEST_ID = "naturelm_by_source_100each_20260418T171459Z"
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "random_init_baseline"

NEW_MODEL = "random_init_eat_seed42"
TRAINED_MODELS = ["eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all"]

FRAMES_PER_ITEM = 50
TWONN_SAMPLE_SIZE = 10000
MLE_K = 20
MLE_SAMPLE_SIZE = 10000

MODEL_COLORS = {
    "eat_all": "tab:blue",
    "eat_bio": "tab:orange",
    "sl_eat_all_ssl_all": "tab:green",
    "sl_eat_bio_ssl_all": "tab:red",
    "random_init_eat_seed42": "0.45",  # gray for the baseline
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--nway_dir", type=Path, default=DEFAULT_NWAY_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    p.add_argument("--twonn_sample_size", type=int, default=TWONN_SAMPLE_SIZE)
    p.add_argument("--mle_k", type=int, default=MLE_K)
    p.add_argument("--mle_sample_size", type=int, default=MLE_SAMPLE_SIZE)
    p.add_argument("--top_k", type=int, default=SUBSPACE_TOP_K)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pooled-level loader for the random-init shards.
# `nway_compare_eat_models.py` previously produced consolidated pooled
# embeddings for the four trained models — but we don't want to rebuild that
# n-way npz just to add one more model. Instead we mean-pool the new model's
# shards inline (over the valid-token range, exactly matching how the trained
# models were pooled) and compute pooled spectral stats locally.
# ---------------------------------------------------------------------------

def load_pooled_from_shards(shard_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Mean-pool over valid tokens and return (n_items, n_layers, D) + metadata.

    Matches `compare_esp_aves2_models.pooled_layer_vectors`: skip token 0
    (the CLS-like token EAT prepends) and average tokens 1..valid_token_count.
    Falls back to token 0 only when valid_token_count <= 1.
    """
    shard_paths = sorted(shard_dir.glob("shard_*.pt"))
    print(f"  pooling {len(shard_paths)} shards from {shard_dir.name}", flush=True)
    pooled_chunks: list[np.ndarray] = []
    metadata: list[dict] = []
    for path in shard_paths:
        s = torch.load(path, weights_only=False)
        acts = s["activations"]  # (B, L, T, D)
        b, n_layers, t_max, d = acts.shape
        chunk = np.empty((b, n_layers, d), dtype=np.float32)
        for i, sample in enumerate(s["samples"]):
            valid = int(min(sample.get("valid_token_count", t_max), t_max))
            valid = max(valid, 1)
            if valid > 1:
                chunk[i] = acts[i, :, 1:valid, :].float().mean(dim=1).numpy()
            else:
                chunk[i] = acts[i, :, 0, :].float().numpy()
            metadata.append(sample)
        pooled_chunks.append(chunk)
    pooled = np.concatenate(pooled_chunks, axis=0)
    print(f"  pooled shape: {pooled.shape}", flush=True)
    return pooled, metadata


def compute_pooled_stats(
    pooled: np.ndarray, layer_names: list[str], twonn_sample_size: int | None
) -> pd.DataFrame:
    n_samples, n_layers, d = pooled.shape
    records: list[dict] = []
    for layer_idx in range(n_layers):
        matrix = pooled[:, layer_idx, :].astype(np.float64)
        eigenvalues = per_layer_spectrum(matrix)
        er = effective_rank(eigenvalues)
        pr = participation_ratio(eigenvalues)
        twonn = twonn_intrinsic_dim(matrix, sample_size=twonn_sample_size)
        records.append({
            "layer_idx": layer_idx,
            "layer_name": layer_names[layer_idx],
            "n_samples": int(n_samples),
            "embedding_dim": int(d),
            "effective_rank": er,
            "participation_ratio": pr,
            "twonn_id": twonn,
            "var_explained_top1": float(eigenvalues[0] / eigenvalues.sum()),
            "var_explained_top10": float(eigenvalues[:10].sum() / eigenvalues.sum()),
        })
    return pd.DataFrame.from_records(records)


def compute_pooled_bio_vs_nonbio(
    pooled: np.ndarray, metadata: list[dict], layer_names: list[str], k: int
) -> pd.DataFrame:
    sources = np.array([m.get("source_dataset", "") for m in metadata])
    is_bio = np.array([s in NATURE_SOURCES for s in sources])
    n_layers = pooled.shape[1]
    records: list[dict] = []
    for layer_idx in range(n_layers):
        mat = pooled[:, layer_idx, :].astype(np.float64)
        basis_bio = top_k_basis(mat[is_bio], k)
        basis_nonbio = top_k_basis(mat[~is_bio], k)
        mean_cos, cos_angles = subspace_overlap(basis_bio, basis_nonbio)
        records.append({
            "layer_idx": layer_idx,
            "layer_name": layer_names[layer_idx],
            "mean_cos_principal_angles": mean_cos,
            "min_cos_principal_angles": float(cos_angles.min()),
            "max_cos_principal_angles": float(cos_angles.max()),
            "k": k,
            "n_bio": int(is_bio.sum()),
            "n_nonbio": int((~is_bio).sum()),
        })
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# 5-way plotting
# ---------------------------------------------------------------------------

def plot_5way_curve(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    models_in_order: list[str],
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model_key in models_in_order:
        sub = df[df["model"] == model_key].sort_values("layer_idx")
        if sub.empty:
            continue
        is_baseline = model_key == NEW_MODEL
        ax.plot(
            sub["layer_idx"], sub[metric],
            marker="s" if is_baseline else "o",
            linestyle="--" if is_baseline else "-",
            linewidth=2.0 if is_baseline else 1.5,
            color=MODEL_COLORS.get(model_key, "black"),
            label=f"{model_key} (baseline)" if is_baseline else model_key,
        )
    ax.set_xlabel("layer index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shard_dir = args.roadmap_dir / NEW_MODEL / "shards"
    if not shard_dir.exists():
        raise SystemExit(
            f"Random-init shards not found at {shard_dir}. "
            f"Run `python collect_esp_aves2_activations.py --models {NEW_MODEL} ...` first."
        )

    # ---- Frame-level stats ----
    print(f"\n=== {NEW_MODEL}: frame-level stats ===", flush=True)
    frame_acts, sample_meta, layer_names = load_frame_level(shard_dir, args.frames_per_item)
    new_frame_stats = compute_frame_stats(
        frame_acts, layer_names,
        twonn_sample_size=args.twonn_sample_size,
        mle_k=args.mle_k,
        mle_sample_size=args.mle_sample_size,
    )
    new_frame_stats.insert(0, "model", NEW_MODEL)
    new_frame_stats.to_csv(args.output_dir / f"frame_per_layer_stats_{NEW_MODEL}.csv", index=False)

    print(f"\n=== {NEW_MODEL}: frame-level bio vs non-bio ===", flush=True)
    new_frame_bio = compute_bio_vs_nonbio(
        frame_acts, sample_meta, args.frames_per_item, layer_names, k=args.top_k
    )
    new_frame_bio.insert(0, "model", NEW_MODEL)
    new_frame_bio.to_csv(args.output_dir / f"frame_bio_vs_nonbio_{NEW_MODEL}.csv", index=False)

    del frame_acts

    # ---- Pooled stats (computed inline since we don't rebuild the n-way npz) ----
    print(f"\n=== {NEW_MODEL}: pooled stats ===", flush=True)
    pooled, metadata = load_pooled_from_shards(shard_dir)
    new_pooled_stats = compute_pooled_stats(pooled, layer_names, twonn_sample_size=None)
    new_pooled_stats.insert(0, "model", NEW_MODEL)
    new_pooled_stats.to_csv(args.output_dir / f"pooled_per_layer_stats_{NEW_MODEL}.csv", index=False)

    new_pooled_bio = compute_pooled_bio_vs_nonbio(pooled, metadata, layer_names, k=args.top_k)
    new_pooled_bio.insert(0, "model", NEW_MODEL)
    new_pooled_bio.to_csv(args.output_dir / f"pooled_bio_vs_nonbio_{NEW_MODEL}.csv", index=False)

    # ---- Merge with existing 4-model results ----
    print("\n=== Merging with trained-model results ===", flush=True)
    trained_frame = pd.read_csv(args.nway_dir / "step2_tier1_frame_level" / "frame_per_layer_stats_all4.csv")
    trained_frame_bio = pd.read_csv(args.nway_dir / "step2_tier1_frame_level" / "frame_bio_vs_nonbio_all4.csv")
    trained_pooled = pd.read_csv(args.nway_dir / "step2_spectral_dim" / "per_layer_stats.csv")
    trained_pooled_bio = pd.read_csv(args.nway_dir / "step2_subspace_angles" / "bio_vs_nonbio_subspace_overlap.csv")

    all_frame = pd.concat([trained_frame, new_frame_stats], ignore_index=True)
    all_frame_bio = pd.concat([trained_frame_bio, new_frame_bio], ignore_index=True)
    all_pooled = pd.concat([trained_pooled, new_pooled_stats], ignore_index=True)
    all_pooled_bio = pd.concat([trained_pooled_bio, new_pooled_bio], ignore_index=True)

    all_frame.to_csv(args.output_dir / "frame_per_layer_stats_all5.csv", index=False)
    all_frame_bio.to_csv(args.output_dir / "frame_bio_vs_nonbio_all5.csv", index=False)
    all_pooled.to_csv(args.output_dir / "pooled_per_layer_stats_all5.csv", index=False)
    all_pooled_bio.to_csv(args.output_dir / "pooled_bio_vs_nonbio_all5.csv", index=False)

    # ---- 5-way comparison plots ----
    print("\n=== 5-way comparison plots ===", flush=True)
    models_in_order = TRAINED_MODELS + [NEW_MODEL]

    plot_5way_curve(
        all_frame,
        metric="effective_rank",
        ylabel="effective rank (frame-level)",
        title="Frame-level effective rank: trained vs random-init EAT",
        output_path=args.output_dir / "frame_effective_rank_all5.png",
        models_in_order=models_in_order,
    )
    plot_5way_curve(
        all_frame,
        metric="participation_ratio",
        ylabel="participation ratio (frame-level)",
        title="Frame-level participation ratio: trained vs random-init EAT",
        output_path=args.output_dir / "frame_participation_ratio_all5.png",
        models_in_order=models_in_order,
    )
    plot_5way_curve(
        all_frame,
        metric="mle_id_k20",
        ylabel=f"MLE-ID intrinsic dim (k={args.mle_k}, frame-level)",
        title="Frame-level MLE-ID(k=20): trained vs random-init EAT",
        output_path=args.output_dir / "frame_mle_id_all5.png",
        models_in_order=models_in_order,
    )
    plot_5way_curve(
        all_frame,
        metric="twonn_id",
        ylabel="TwoNN intrinsic dim (k=2, frame-level)",
        title="Frame-level TwoNN(k=2): trained vs random-init EAT",
        output_path=args.output_dir / "frame_twonn_id_all5.png",
        models_in_order=models_in_order,
    )
    plot_5way_curve(
        all_frame_bio,
        metric="mean_cos_principal_angles_frame",
        ylabel=f"mean cos principal angles (top-{args.top_k}, frame-level)",
        title="Frame-level bio vs non-bio subspace overlap: trained vs random-init",
        output_path=args.output_dir / "frame_bio_vs_nonbio_all5.png",
        models_in_order=models_in_order,
        ylim=(0.0, 1.05),
    )
    plot_5way_curve(
        all_pooled,
        metric="effective_rank",
        ylabel="effective rank (pooled)",
        title="Pooled effective rank: trained vs random-init EAT",
        output_path=args.output_dir / "pooled_effective_rank_all5.png",
        models_in_order=models_in_order,
    )
    plot_5way_curve(
        all_pooled_bio,
        metric="mean_cos_principal_angles",
        ylabel=f"mean cos principal angles (top-{args.top_k}, pooled)",
        title="Pooled bio vs non-bio subspace overlap: trained vs random-init",
        output_path=args.output_dir / "pooled_bio_vs_nonbio_all5.png",
        models_in_order=models_in_order,
        ylim=(0.0, 1.05),
    )

    # ---- Headline tables ----
    print("\n=== Headline numbers (frame-level) ===", flush=True)
    pivot_er = all_frame.pivot(index="layer_idx", columns="model", values="effective_rank")
    pivot_mle = all_frame.pivot(index="layer_idx", columns="model", values="mle_id_k20")
    pivot_bio = all_frame_bio.pivot(index="layer_idx", columns="model", values="mean_cos_principal_angles_frame")
    print("\nFrame-level effective rank (5 models):")
    print(pivot_er[models_in_order].round(2).to_string())
    print(f"\nFrame-level MLE-ID(k={args.mle_k}) (5 models):")
    print(pivot_mle[models_in_order].round(2).to_string())
    print(f"\nFrame-level bio-vs-nonbio mean cos top-{args.top_k} (5 models):")
    print(pivot_bio[models_in_order].round(3).to_string())

    print(f"\nSaved artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
