"""Step 2 (continued): pooled vs frame-level geometry on sl_eat_bio_ssl_all.

The per-(model, layer) statistics from `step2_spectral_dim_eat.py` use
mean-pooled embeddings (one 768-dim vector per item). The TwoNN intrinsic
dimensionality stays small (~8-12) while linear effective rank swings
3-148, suggesting the data lives on a low-dim curved manifold inside a wide
linear subspace. Pooling almost certainly understates the manifold curvature.

This script recomputes effective rank, participation ratio, and TwoNN
intrinsic dim on `sl_eat_bio_ssl_all` using *frame-level* activations
(50 frames per item, sampled uniformly within the valid-token range).
With 600 items that gives ~30,000 rows per layer — tractable for SVD and
TwoNN on CPU.

Output: side-by-side plots vs the existing pooled metrics for the same
model. Decision artifact: does pooling materially distort the geometry on
this model?

Usage:
    python step2_pooled_vs_frame_eat.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors


DEFAULT_SHARD_DIR = Path(
    "artifacts/roadmap_part1/naturelm_by_source_100each_20260418T171459Z/sl_eat_bio_ssl_all/shards"
)
DEFAULT_POOLED_STATS = Path(
    "artifacts/comparisons/naturelm_by_source_100each_20260418T171459Z/nway_eat_all4/step2_spectral_dim/per_layer_stats.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/comparisons/naturelm_by_source_100each_20260418T171459Z/nway_eat_all4/step2_pooled_vs_frame"
)
MODEL_KEY = "sl_eat_bio_ssl_all"
FRAMES_PER_ITEM = 50
SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shard_dir", type=Path, default=DEFAULT_SHARD_DIR)
    p.add_argument("--pooled_stats", type=Path, default=DEFAULT_POOLED_STATS)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    p.add_argument("--twonn_sample_size", type=int, default=10000,
                   help="Subsample this many rows for TwoNN to keep it fast.")
    return p.parse_args()


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


def per_layer_spectrum(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = (s ** 2) / max(centered.shape[0] - 1, 1)
    return eigenvalues.astype(np.float64)


def twonn_intrinsic_dim(matrix: np.ndarray, sample_size: int | None) -> float:
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


def load_frame_level(shard_dir: Path, frames_per_item: int) -> tuple[np.ndarray, list[str]]:
    """Returns frame_acts of shape (n_items * frames_per_item, n_layers, D)
    and the list of layer names. Subsamples each item's valid-token range
    uniformly with seed=42."""
    rng = np.random.default_rng(SEED)
    shard_paths = sorted(shard_dir.glob("shard_*.pt"))
    print(f"  loading {len(shard_paths)} shards from {shard_dir.name}", flush=True)
    out_chunks: list[np.ndarray] = []
    layer_names: list[str] = []
    n_items_total = 0
    for path in shard_paths:
        s = torch.load(path, weights_only=False)
        acts = s["activations"].numpy()  # (B, L, T, D)
        samples = s["samples"]
        b, n_layers, t_max, d = acts.shape
        if not layer_names:
            # We don't have layer names in shards — borrow them from pooled embeddings layout
            layer_names = [f"layer_{i:02d}" for i in range(n_layers)]
        chunk = np.empty((b, frames_per_item, n_layers, d), dtype=np.float32)
        for i, sample in enumerate(samples):
            valid = int(min(sample.get("valid_token_count", t_max), t_max))
            valid = max(valid, 1)
            if valid >= frames_per_item:
                idx = rng.choice(valid, frames_per_item, replace=False)
            else:
                idx = rng.choice(valid, frames_per_item, replace=True)
            # NumPy advanced-indexing rule: with slices around the int array, the
            # indexed axis (frames) is moved to the front, giving (F, L, D) directly.
            chunk[i] = acts[i, :, idx, :]
        out_chunks.append(chunk)
        n_items_total += b
    frame_acts = np.concatenate(out_chunks, axis=0)  # (N_items, F, L, D)
    n_items, f, l, d = frame_acts.shape
    frame_acts = frame_acts.reshape(n_items * f, l, d)  # (N_items*F, L, D)
    print(f"  loaded {n_items} items × {f} frames = {frame_acts.shape[0]} rows × {l} layers × {d} dims", flush=True)
    return frame_acts, layer_names


def compute_frame_stats(frame_acts: np.ndarray, layer_names: list[str], twonn_sample_size: int) -> pd.DataFrame:
    n_rows, n_layers, d = frame_acts.shape
    records: list[dict] = []
    for layer_idx in range(n_layers):
        matrix = frame_acts[:, layer_idx, :].astype(np.float64)
        eigenvalues = per_layer_spectrum(matrix)
        er = effective_rank(eigenvalues)
        pr = participation_ratio(eigenvalues)
        twonn = twonn_intrinsic_dim(matrix, sample_size=twonn_sample_size)
        records.append({
            "layer_idx": layer_idx,
            "layer_name": layer_names[layer_idx],
            "n_rows": int(n_rows),
            "embedding_dim": int(d),
            "effective_rank": er,
            "participation_ratio": pr,
            "twonn_id": twonn,
            "var_explained_top1": float(eigenvalues[0] / eigenvalues.sum()),
            "var_explained_top10": float(eigenvalues[:10].sum() / eigenvalues.sum()),
        })
        print(
            f"  L{layer_idx:02d}  eff_rank={er:7.2f}  PR={pr:7.2f}  TwoNN={twonn:6.2f}",
            flush=True,
        )
    return pd.DataFrame.from_records(records)


def plot_pooled_vs_frame(
    pooled_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pooled_df["layer_idx"], pooled_df[metric], marker="o", label=f"pooled (1 vec / item, n={pooled_df['n_samples'].iloc[0]})", color="tab:blue")
    ax.plot(frame_df["layer_idx"], frame_df[metric], marker="s", label=f"frame-level (n={frame_df['n_rows'].iloc[0]})", color="tab:red")
    ax.set_xlabel("layer index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading frame-level activations for {MODEL_KEY}...", flush=True)
    frame_acts, layer_names = load_frame_level(args.shard_dir, args.frames_per_item)

    print("\nComputing per-layer frame-level statistics...", flush=True)
    frame_df = compute_frame_stats(frame_acts, layer_names, args.twonn_sample_size)
    frame_df.to_csv(args.output_dir / f"frame_per_layer_stats_{MODEL_KEY}.csv", index=False)

    pooled_all = pd.read_csv(args.pooled_stats)
    pooled_df = pooled_all[pooled_all["model"] == MODEL_KEY].sort_values("layer_idx").reset_index(drop=True)

    for metric, ylabel, title in [
        ("effective_rank", "effective rank",
         f"Effective rank: pooled vs frame-level ({MODEL_KEY})"),
        ("participation_ratio", "participation ratio",
         f"Participation ratio: pooled vs frame-level ({MODEL_KEY})"),
        ("twonn_id", "TwoNN intrinsic dim",
         f"TwoNN intrinsic dim: pooled vs frame-level ({MODEL_KEY})"),
    ]:
        out = args.output_dir / f"pooled_vs_frame_{metric}_{MODEL_KEY}.png"
        plot_pooled_vs_frame(pooled_df, frame_df, metric, ylabel, title, out)

    merged = pooled_df[["layer_idx", "effective_rank", "participation_ratio", "twonn_id"]].rename(
        columns={"effective_rank": "eff_rank_pooled", "participation_ratio": "pr_pooled", "twonn_id": "twonn_pooled"}
    ).merge(
        frame_df[["layer_idx", "effective_rank", "participation_ratio", "twonn_id"]].rename(
            columns={"effective_rank": "eff_rank_frame", "participation_ratio": "pr_frame", "twonn_id": "twonn_frame"}
        ),
        on="layer_idx",
    )
    merged["eff_rank_ratio_frame_over_pooled"] = merged["eff_rank_frame"] / merged["eff_rank_pooled"]
    merged["twonn_ratio_frame_over_pooled"] = merged["twonn_frame"] / merged["twonn_pooled"]
    merged.to_csv(args.output_dir / f"pooled_vs_frame_summary_{MODEL_KEY}.csv", index=False)

    print(f"\nSaved artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
