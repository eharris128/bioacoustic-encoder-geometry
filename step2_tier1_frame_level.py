"""Step 2 Tier 1: frame-level geometry across all four EAT-family models.

Combines three follow-ups into one script so we only load each
model's shards once:

1. **Generalize pooled-vs-frame.** Recompute effective rank, participation
   ratio, TwoNN intrinsic dim on frame-level activations (50 frames per item)
   for `eat_all`, `eat_bio`, and `sl_eat_all_ssl_all` -- the existing
   `step2_pooled_vs_frame_eat.py` only ran on `sl_eat_bio_ssl_all`.
2. **TwoNN sanity check via MLE-ID.** Add a Levina-Bickel MLE intrinsic-dim
   estimator with k=20 alongside TwoNN(k=2) so the L4 dip on
   `sl_eat_bio_ssl_all` (TwoNN ~2.6 sandwiched between ~10 and ~7) can be
   judged for stability.
3. **Bio vs non-bio subspace overlap at frame level.** Recompute the
   `sl_eat_bio_ssl_all` story (mean cos = 0.33 at L9 vs 0.55-0.70 elsewhere)
   on frame-level top-k subspaces for all four models. Mean-pooling could
   either suppress or inflate the directional separation.

Output dir: artifacts/comparisons/<manifest>/nway_eat_all4/step2_tier1_frame_level/

Usage:
    python step2_tier1_frame_level.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.linalg import subspace_angles
from sklearn.neighbors import NearestNeighbors


MANIFEST_ID = "naturelm_by_source_100each_20260418T171459Z"
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "step2_tier1_frame_level"
DEFAULT_POOLED_STATS = DEFAULT_NWAY_DIR / "step2_spectral_dim" / "per_layer_stats.csv"

MODELS = ["eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all"]
NATURE_SOURCES = {"Xeno-canto", "iNaturalist", "Animal Sound Archive", "Watkins"}
FRAMES_PER_ITEM = 50
TWONN_SAMPLE_SIZE = 10000
MLE_K = 20
MLE_SAMPLE_SIZE = 10000
SUBSPACE_TOP_K = 10
SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--nway_dir", type=Path, default=DEFAULT_NWAY_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--pooled_stats", type=Path, default=DEFAULT_POOLED_STATS)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    p.add_argument("--twonn_sample_size", type=int, default=TWONN_SAMPLE_SIZE)
    p.add_argument("--mle_k", type=int, default=MLE_K)
    p.add_argument("--mle_sample_size", type=int, default=MLE_SAMPLE_SIZE)
    p.add_argument("--top_k", type=int, default=SUBSPACE_TOP_K)
    p.add_argument("--models", nargs="+", default=MODELS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def per_layer_spectrum(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = (s ** 2) / max(centered.shape[0] - 1, 1)
    return eigenvalues.astype(np.float64)


def effective_rank(eigenvalues: np.ndarray) -> float:
    eigenvalues = eigenvalues[eigenvalues > 0]
    if eigenvalues.size == 0:
        return float("nan")
    p = eigenvalues / eigenvalues.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(eigenvalues: np.ndarray) -> float:
    eigenvalues = eigenvalues[eigenvalues > 0]
    if eigenvalues.size == 0:
        return float("nan")
    s = eigenvalues.sum()
    s2 = (eigenvalues ** 2).sum()
    return float(s * s / s2)


def twonn_intrinsic_dim(matrix: np.ndarray, sample_size: int | None) -> float:
    """Facco et al. 2017 TwoNN. Uses k=2 nearest-neighbor distance ratio."""
    if sample_size is not None and matrix.shape[0] > sample_size:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(matrix.shape[0], sample_size, replace=False)
        matrix = matrix[idx]
    if matrix.shape[0] < 3:
        return float("nan")
    nbrs = NearestNeighbors(n_neighbors=3).fit(matrix)
    dists, _ = nbrs.kneighbors(matrix)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    valid = (r1 > 0) & (r2 > r1)
    if valid.sum() < 3:
        return float("nan")
    mu = r2[valid] / r1[valid]
    log_mu = np.log(mu)
    log_mu_sorted = np.sort(log_mu)
    n = log_mu_sorted.size
    f_emp = np.arange(1, n + 1) / (n + 1)
    y = -np.log(1.0 - f_emp)
    return float((y * log_mu_sorted).sum() / (log_mu_sorted ** 2).sum())


def mle_intrinsic_dim(matrix: np.ndarray, k: int, sample_size: int | None) -> float:
    """Levina-Bickel MLE intrinsic dimension averaged over sampled points.

    Uses k+1 nearest neighbors (k+1 because the point itself is the first NN
    at distance 0). Returns the inverse-mean estimator m_hat = 1 / mean_x m_k(x)^-1
    which is the closed-form MLE recommended in Levina-Bickel 2005 over averaging
    the per-point estimates directly.
    """
    if sample_size is not None and matrix.shape[0] > sample_size:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(matrix.shape[0], sample_size, replace=False)
        matrix = matrix[idx]
    n = matrix.shape[0]
    if n < k + 2:
        return float("nan")
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(matrix)
    dists, _ = nbrs.kneighbors(matrix)
    # dists[:, 0] is self (distance 0). Use dists[:, 1:k+1] for the k NN distances.
    nn = dists[:, 1:k + 1]
    nn = np.maximum(nn, 1e-12)
    log_ratios = np.log(nn[:, k - 1:k] / nn[:, :k - 1])  # shape (n, k-1)
    inv_m = log_ratios.mean(axis=1) / 1.0  # equivalent to (1/(k-1)) sum log(T_k/T_j)
    valid = inv_m > 0
    if valid.sum() < 1:
        return float("nan")
    return float(1.0 / inv_m[valid].mean())


def top_k_basis(matrix: np.ndarray, k: int) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:k].T


def subspace_overlap(basis_a: np.ndarray, basis_b: np.ndarray) -> tuple[float, np.ndarray]:
    angles = subspace_angles(basis_a, basis_b)
    cos_angles = np.cos(angles)
    return float(cos_angles.mean()), cos_angles


# ---------------------------------------------------------------------------
# Frame-level loader
# ---------------------------------------------------------------------------

def load_frame_level(
    shard_dir: Path, frames_per_item: int
) -> tuple[np.ndarray, list[dict], list[str]]:
    """Returns
        frame_acts:  (n_items * frames_per_item, n_layers, D)
        sample_meta: list of len n_items (in row order matching frame_acts blocks)
        layer_names: list[str]
    """
    rng = np.random.default_rng(SEED)
    shard_paths = sorted(shard_dir.glob("shard_*.pt"))
    print(f"  loading {len(shard_paths)} shards from {shard_dir.name}", flush=True)
    out_chunks: list[np.ndarray] = []
    sample_meta: list[dict] = []
    layer_names: list[str] = []
    for path in shard_paths:
        s = torch.load(path, weights_only=False)
        acts = s["activations"].numpy()  # (B, L, T, D)
        samples = s["samples"]
        b, n_layers, t_max, d = acts.shape
        if not layer_names:
            layer_names = [f"layer_{i:02d}" for i in range(n_layers)]
        chunk = np.empty((b, frames_per_item, n_layers, d), dtype=np.float32)
        for i, sample in enumerate(samples):
            valid = int(min(sample.get("valid_token_count", t_max), t_max))
            valid = max(valid, 1)
            if valid >= frames_per_item:
                idx = rng.choice(valid, frames_per_item, replace=False)
            else:
                idx = rng.choice(valid, frames_per_item, replace=True)
            chunk[i] = acts[i, :, idx, :]
            sample_meta.append(sample)
        out_chunks.append(chunk)
    frame_acts = np.concatenate(out_chunks, axis=0)  # (N_items, F, L, D)
    n_items, f, n_layers, d = frame_acts.shape
    frame_acts = frame_acts.reshape(n_items * f, n_layers, d)
    print(
        f"  loaded {n_items} items x {f} frames = {frame_acts.shape[0]} rows "
        f"x {n_layers} layers x {d} dims",
        flush=True,
    )
    return frame_acts, sample_meta, layer_names


# ---------------------------------------------------------------------------
# Per-model frame stats (eff rank, PR, TwoNN, MLE-ID)
# ---------------------------------------------------------------------------

def compute_frame_stats(
    frame_acts: np.ndarray,
    layer_names: list[str],
    twonn_sample_size: int,
    mle_k: int,
    mle_sample_size: int,
) -> pd.DataFrame:
    n_rows, n_layers, d = frame_acts.shape
    records: list[dict] = []
    for layer_idx in range(n_layers):
        matrix = frame_acts[:, layer_idx, :].astype(np.float64)
        eigenvalues = per_layer_spectrum(matrix)
        er = effective_rank(eigenvalues)
        pr = participation_ratio(eigenvalues)
        twonn = twonn_intrinsic_dim(matrix, sample_size=twonn_sample_size)
        mle = mle_intrinsic_dim(matrix, k=mle_k, sample_size=mle_sample_size)
        records.append({
            "layer_idx": layer_idx,
            "layer_name": layer_names[layer_idx],
            "n_rows": int(n_rows),
            "embedding_dim": int(d),
            "effective_rank": er,
            "participation_ratio": pr,
            "twonn_id": twonn,
            "mle_id_k20": mle,
            "var_explained_top1": float(eigenvalues[0] / eigenvalues.sum()),
            "var_explained_top10": float(eigenvalues[:10].sum() / eigenvalues.sum()),
        })
        print(
            f"    L{layer_idx:02d}  eff_rank={er:7.2f}  PR={pr:7.2f}  "
            f"TwoNN={twonn:6.2f}  MLE-ID(k={mle_k})={mle:6.2f}",
            flush=True,
        )
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Frame-level bio-vs-nonbio subspace overlap (per layer)
# ---------------------------------------------------------------------------

def compute_bio_vs_nonbio(
    frame_acts: np.ndarray,
    sample_meta: list[dict],
    frames_per_item: int,
    layer_names: list[str],
    k: int,
) -> pd.DataFrame:
    n_items = len(sample_meta)
    assert frame_acts.shape[0] == n_items * frames_per_item
    sources = np.array([s.get("source_dataset", "") for s in sample_meta])
    item_is_bio = np.array([s in NATURE_SOURCES for s in sources])
    # Expand item-level mask to frame-level
    frame_is_bio = np.repeat(item_is_bio, frames_per_item)
    n_bio = int(frame_is_bio.sum())
    n_nonbio = int((~frame_is_bio).sum())
    print(f"    frame-level bio rows={n_bio}, non-bio rows={n_nonbio}", flush=True)

    records: list[dict] = []
    n_layers = frame_acts.shape[1]
    for layer_idx in range(n_layers):
        mat = frame_acts[:, layer_idx, :].astype(np.float64)
        basis_bio = top_k_basis(mat[frame_is_bio], k)
        basis_nonbio = top_k_basis(mat[~frame_is_bio], k)
        mean_cos, cos_angles = subspace_overlap(basis_bio, basis_nonbio)
        records.append({
            "layer_idx": layer_idx,
            "layer_name": layer_names[layer_idx],
            "mean_cos_principal_angles_frame": mean_cos,
            "min_cos_principal_angles_frame": float(cos_angles.min()),
            "max_cos_principal_angles_frame": float(cos_angles.max()),
            "k": k,
            "n_bio_frames": n_bio,
            "n_nonbio_frames": n_nonbio,
        })
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric_pooled_vs_frame_all_models(
    pooled_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    metric_pooled: str,
    metric_frame: str,
    ylabel: str,
    title: str,
    output_path: Path,
    models: list[str],
) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 4.0), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model_key in zip(axes, models):
        p = pooled_df[pooled_df["model"] == model_key].sort_values("layer_idx")
        f_ = frame_df[frame_df["model"] == model_key].sort_values("layer_idx")
        ax.plot(p["layer_idx"], p[metric_pooled], marker="o", label=f"pooled (n={int(p['n_samples'].iloc[0])})", color="tab:blue")
        ax.plot(f_["layer_idx"], f_[metric_frame], marker="s", label=f"frame (n={int(f_['n_rows'].iloc[0])})", color="tab:red")
        ax.set_title(model_key, fontsize=10)
        ax.set_xlabel("layer index")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_intrinsic_dim_estimators(
    frame_df: pd.DataFrame,
    output_path: Path,
    models: list[str],
    mle_k: int,
) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 4.0), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model_key in zip(axes, models):
        f_ = frame_df[frame_df["model"] == model_key].sort_values("layer_idx")
        ax.plot(f_["layer_idx"], f_["twonn_id"], marker="o", label="TwoNN (k=2)", color="tab:red")
        ax.plot(f_["layer_idx"], f_["mle_id_k20"], marker="s", label=f"MLE-ID (k={mle_k})", color="tab:green")
        ax.set_title(model_key, fontsize=10)
        ax.set_xlabel("layer index")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("intrinsic dim")
    fig.suptitle("Frame-level TwoNN vs MLE-ID — sanity check on the L4 dip")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bio_vs_nonbio_pooled_vs_frame(
    pooled_bio_df: pd.DataFrame,
    frame_bio_df: pd.DataFrame,
    output_path: Path,
    models: list[str],
) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 4.0), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model_key in zip(axes, models):
        p = pooled_bio_df[pooled_bio_df["model"] == model_key].sort_values("layer_idx")
        f_ = frame_bio_df[frame_bio_df["model"] == model_key].sort_values("layer_idx")
        ax.plot(p["layer_idx"], p["mean_cos_principal_angles"], marker="o",
                label="pooled", color="tab:blue")
        ax.plot(f_["layer_idx"], f_["mean_cos_principal_angles_frame"], marker="s",
                label="frame", color="tab:red")
        ax.set_title(model_key, fontsize=10)
        ax.set_xlabel("layer index")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"mean cos principal angles (top-{SUBSPACE_TOP_K})")
    fig.suptitle("Bio vs non-bio subspace overlap: pooled vs frame-level")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pooled_all = pd.read_csv(args.pooled_stats)
    pooled_bio_csv = args.nway_dir / "step2_subspace_angles" / "bio_vs_nonbio_subspace_overlap.csv"
    pooled_bio_all = pd.read_csv(pooled_bio_csv)

    all_frame_stats: list[pd.DataFrame] = []
    all_frame_bio: list[pd.DataFrame] = []

    for model_key in args.models:
        print(f"\n=== {model_key} ===", flush=True)
        shard_dir = args.roadmap_dir / model_key / "shards"
        frame_acts, sample_meta, layer_names = load_frame_level(shard_dir, args.frames_per_item)

        print(f"  computing per-layer frame stats...", flush=True)
        stats = compute_frame_stats(
            frame_acts,
            layer_names,
            twonn_sample_size=args.twonn_sample_size,
            mle_k=args.mle_k,
            mle_sample_size=args.mle_sample_size,
        )
        stats.insert(0, "model", model_key)
        stats.to_csv(args.output_dir / f"frame_per_layer_stats_{model_key}.csv", index=False)
        all_frame_stats.append(stats)

        print(f"  computing frame-level bio vs non-bio subspace overlap...", flush=True)
        bio = compute_bio_vs_nonbio(
            frame_acts, sample_meta, args.frames_per_item, layer_names, k=args.top_k
        )
        bio.insert(0, "model", model_key)
        bio.to_csv(args.output_dir / f"frame_bio_vs_nonbio_{model_key}.csv", index=False)
        all_frame_bio.append(bio)

        # Free the big frame_acts before loading the next model
        del frame_acts

    frame_stats_all = pd.concat(all_frame_stats, ignore_index=True)
    frame_stats_all.to_csv(args.output_dir / "frame_per_layer_stats_all4.csv", index=False)
    frame_bio_all = pd.concat(all_frame_bio, ignore_index=True)
    frame_bio_all.to_csv(args.output_dir / "frame_bio_vs_nonbio_all4.csv", index=False)

    # Combined pooled-vs-frame summary table (ratios)
    pooled_subset = pooled_all[pooled_all["model"].isin(args.models)][
        ["model", "layer_idx", "n_samples", "effective_rank", "participation_ratio", "twonn_id"]
    ].rename(columns={
        "effective_rank": "eff_rank_pooled",
        "participation_ratio": "pr_pooled",
        "twonn_id": "twonn_pooled",
    })
    frame_subset = frame_stats_all[
        ["model", "layer_idx", "n_rows", "effective_rank", "participation_ratio", "twonn_id", "mle_id_k20"]
    ].rename(columns={
        "effective_rank": "eff_rank_frame",
        "participation_ratio": "pr_frame",
        "twonn_id": "twonn_frame",
        "mle_id_k20": "mle_id_frame",
    })
    merged = pooled_subset.merge(frame_subset, on=["model", "layer_idx"])
    merged["eff_rank_ratio_frame_over_pooled"] = merged["eff_rank_frame"] / merged["eff_rank_pooled"]
    merged["twonn_ratio_frame_over_pooled"] = merged["twonn_frame"] / merged["twonn_pooled"]
    merged.to_csv(args.output_dir / "pooled_vs_frame_summary_all4.csv", index=False)

    # Plots
    print("\nGenerating plots...", flush=True)
    plot_metric_pooled_vs_frame_all_models(
        pooled_all[pooled_all["model"].isin(args.models)],
        frame_stats_all,
        metric_pooled="effective_rank",
        metric_frame="effective_rank",
        ylabel="effective rank",
        title="Effective rank: pooled vs frame-level",
        output_path=args.output_dir / "pooled_vs_frame_effective_rank_all4.png",
        models=args.models,
    )
    plot_metric_pooled_vs_frame_all_models(
        pooled_all[pooled_all["model"].isin(args.models)],
        frame_stats_all,
        metric_pooled="participation_ratio",
        metric_frame="participation_ratio",
        ylabel="participation ratio",
        title="Participation ratio: pooled vs frame-level",
        output_path=args.output_dir / "pooled_vs_frame_participation_ratio_all4.png",
        models=args.models,
    )
    plot_metric_pooled_vs_frame_all_models(
        pooled_all[pooled_all["model"].isin(args.models)],
        frame_stats_all,
        metric_pooled="twonn_id",
        metric_frame="twonn_id",
        ylabel="TwoNN intrinsic dim",
        title="TwoNN intrinsic dim: pooled vs frame-level",
        output_path=args.output_dir / "pooled_vs_frame_twonn_id_all4.png",
        models=args.models,
    )
    plot_intrinsic_dim_estimators(
        frame_stats_all,
        output_path=args.output_dir / "frame_twonn_vs_mle_id_all4.png",
        models=args.models,
        mle_k=args.mle_k,
    )
    plot_bio_vs_nonbio_pooled_vs_frame(
        pooled_bio_all,
        frame_bio_all,
        output_path=args.output_dir / "bio_vs_nonbio_pooled_vs_frame_all4.png",
        models=args.models,
    )

    # Headline numbers
    print("\n=== Headline frame-level numbers ===", flush=True)
    pivot_er = frame_stats_all.pivot(index="layer_idx", columns="model", values="effective_rank")
    pivot_twonn = frame_stats_all.pivot(index="layer_idx", columns="model", values="twonn_id")
    pivot_mle = frame_stats_all.pivot(index="layer_idx", columns="model", values="mle_id_k20")
    pivot_bio = frame_bio_all.pivot(index="layer_idx", columns="model", values="mean_cos_principal_angles_frame")
    print("\nFrame-level effective rank:")
    print(pivot_er.round(2).to_string())
    print("\nFrame-level TwoNN(k=2):")
    print(pivot_twonn.round(2).to_string())
    print(f"\nFrame-level MLE-ID(k={args.mle_k}):")
    print(pivot_mle.round(2).to_string())
    print(f"\nFrame-level bio-vs-nonbio mean cos (top-{args.top_k}):")
    print(pivot_bio.round(3).to_string())

    print(f"\nSaved artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
