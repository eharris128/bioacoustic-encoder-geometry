"""Step 2: bootstrap CIs on the RESULTS.md §3-§6 headline numbers.

For each (model, layer) and each metric in {effective_rank, MLE-ID(k=20),
bio-vs-non-bio top-10 cos}, resample the 600 items with replacement B times,
re-extract 50 frames per item per bootstrap (seed 42 + bootstrap index), and
record the bootstrap distribution. Outputs 5/50/95 percentile bands.

Validates that the trained-vs-random gaps in RESULTS.md (eff_rank: ~30x,
cos: ~0.34) are larger than their CIs. Does not bootstrap seeds 7 / 13 —
those shards have been deleted; only seed 42 + the 4 trained models are
covered.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/bootstrap_cis/

Usage:
    python step2_bootstrap_cis.py
    python step2_bootstrap_cis.py --num_bootstraps 30 --models eat_all
"""

from __future__ import annotations

import argparse
import time
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
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "bootstrap_cis"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]

NATURE_SOURCES = {"Xeno-canto", "iNaturalist", "Animal Sound Archive", "Watkins"}
FRAMES_PER_ITEM = 50
MLE_K = 20
MLE_SAMPLE_SIZE = 10000
SUBSPACE_TOP_K = 10
BASE_SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--num_bootstraps", type=int, default=20)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    p.add_argument("--mle_k", type=int, default=MLE_K)
    p.add_argument("--mle_sample_size", type=int, default=MLE_SAMPLE_SIZE)
    p.add_argument("--top_k", type=int, default=SUBSPACE_TOP_K)
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Optional layer subset. Default: all 13.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Geometry primitives — covariance-based for speed (skip computing U/V_T)
# ---------------------------------------------------------------------------

def cov_eigvals(matrix: np.ndarray) -> np.ndarray:
    """Eigenvalues of (1/(n-1)) * X^T X for centered X. Faster than SVD when
    we only need eigenvalues (eff_rank, PR) — skips computing U and V."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    return eigvals[eigvals > 0][::-1]  # descending, positive only


def effective_rank(eigvals: np.ndarray) -> float:
    if eigvals.size == 0:
        return float("nan")
    p = eigvals / eigvals.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(eigvals: np.ndarray) -> float:
    if eigvals.size == 0:
        return float("nan")
    s = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    return float(s * s / s2)


def mle_intrinsic_dim(matrix: np.ndarray, k: int, sample_size: int, rng: np.random.Generator) -> float:
    if matrix.shape[0] > sample_size:
        idx = rng.choice(matrix.shape[0], sample_size, replace=False)
        matrix = matrix[idx]
    n = matrix.shape[0]
    if n < k + 2:
        return float("nan")
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(matrix)
    dists, _ = nbrs.kneighbors(matrix)
    nn = np.maximum(dists[:, 1:k + 1], 1e-12)
    log_ratios = np.log(nn[:, k - 1:k] / nn[:, :k - 1])
    inv_m = log_ratios.mean(axis=1)
    valid = inv_m > 0
    if valid.sum() < 1:
        return float("nan")
    return float(1.0 / inv_m[valid].mean())


def top_k_basis_via_cov(matrix: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors via cov eigendecomposition. Returns (D, k)."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    return eigvecs[:, -k:][:, ::-1]  # top-k, descending eigenvalue order


def subspace_overlap(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    angles = subspace_angles(basis_a, basis_b)
    return float(np.cos(angles).mean())


# ---------------------------------------------------------------------------
# Loader — one (model, layer) tensor at a time
# ---------------------------------------------------------------------------

def load_layer_tensor(shard_dir: Path, layer_idx: int) -> tuple[np.ndarray, list[dict]]:
    """Returns (n_items, T, D) float32 tensor + sample metadata."""
    shard_paths = sorted(shard_dir.glob("shard_*.pt"))
    chunks: list[np.ndarray] = []
    metadata: list[dict] = []
    for path in shard_paths:
        s = torch.load(path, weights_only=False)
        acts = s["activations"][:, layer_idx, :, :].float().numpy()  # (B, T, D)
        chunks.append(acts)
        metadata.extend(s["samples"])
    return np.concatenate(chunks, axis=0), metadata


# ---------------------------------------------------------------------------
# Bootstrap sampling — one bootstrap iteration
# ---------------------------------------------------------------------------

def sample_bootstrap_frames(
    layer_tensor: np.ndarray,  # (n_items, T, D)
    valid_token_counts: np.ndarray,  # (n_items,) ints
    is_bio_per_item: np.ndarray,  # (n_items,) bool
    frames_per_item: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample n_items items with replacement, then sample frames from each.
    Returns (n_items*F, D) frame matrix and (n_items*F,) bio-mask."""
    n_items, t_max, d = layer_tensor.shape
    item_idx = rng.integers(0, n_items, n_items)  # (n_items,) with replacement
    out = np.empty((n_items, frames_per_item, d), dtype=np.float32)
    is_bio_per_frame = np.empty((n_items, frames_per_item), dtype=bool)
    for i, src in enumerate(item_idx):
        valid = int(min(valid_token_counts[src], t_max))
        valid = max(valid, 1)
        if valid >= frames_per_item:
            f_idx = rng.choice(valid, frames_per_item, replace=False)
        else:
            f_idx = rng.choice(valid, frames_per_item, replace=True)
        out[i] = layer_tensor[src, f_idx, :]
        is_bio_per_frame[i] = is_bio_per_item[src]
    return out.reshape(-1, d), is_bio_per_frame.reshape(-1)


def run_one_bootstrap(
    layer_tensor: np.ndarray,
    valid_token_counts: np.ndarray,
    is_bio_per_item: np.ndarray,
    frames_per_item: int,
    mle_k: int,
    mle_sample_size: int,
    top_k: int,
    rng: np.random.Generator,
) -> dict:
    frames, is_bio = sample_bootstrap_frames(
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
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    layers = args.layers if args.layers is not None else list(range(13))
    print(
        f"Bootstrap CIs: {len(args.models)} models × {len(layers)} layers × "
        f"B={args.num_bootstraps} bootstraps × {args.frames_per_item} frames/item",
        flush=True,
    )

    all_records: list[dict] = []

    for model_idx, model_key in enumerate(args.models):
        shard_dir = args.roadmap_dir / model_key / "shards"
        if not shard_dir.exists():
            print(f"WARN: shards missing for {model_key}, skipping", flush=True)
            continue
        print(f"\n=== {model_key} ({model_idx + 1}/{len(args.models)}) ===", flush=True)
        # Read once to get metadata + valid_token_counts (consistent across layers)
        layer_tensor_l0, sample_meta = load_layer_tensor(shard_dir, layer_idx=0)
        n_items = layer_tensor_l0.shape[0]
        valid_token_counts = np.array(
            [int(s.get("valid_token_count", layer_tensor_l0.shape[1])) for s in sample_meta]
        )
        sources = np.array([s.get("source_dataset", "") for s in sample_meta])
        is_bio_per_item = np.array([s in NATURE_SOURCES for s in sources])
        del layer_tensor_l0  # free memory before per-layer loop

        for layer_idx in layers:
            print(f"  L{layer_idx:02d} loading...", flush=True, end="")
            t0 = time.time()
            layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
            t_load = time.time() - t0
            print(f" {t_load:.1f}s", flush=True)

            t0 = time.time()
            for b in range(args.num_bootstraps):
                seed = BASE_SEED + b * 1000 + layer_idx
                rng = np.random.default_rng(seed)
                metrics = run_one_bootstrap(
                    layer_tensor, valid_token_counts, is_bio_per_item,
                    args.frames_per_item, args.mle_k, args.mle_sample_size,
                    args.top_k, rng,
                )
                metrics.update({
                    "model": model_key,
                    "layer_idx": layer_idx,
                    "bootstrap": b,
                    "seed": seed,
                    "n_items": n_items,
                    "frames_per_item": args.frames_per_item,
                })
                all_records.append(metrics)
            t_boot = time.time() - t0
            avg = t_boot / args.num_bootstraps
            er_vals = [r["effective_rank"] for r in all_records[-args.num_bootstraps:]]
            cos_vals = [r["bio_vs_nonbio_cos_top10"] for r in all_records[-args.num_bootstraps:]]
            print(
                f"    B={args.num_bootstraps} done in {t_boot:.1f}s ({avg:.1f}s/boot) | "
                f"eff_rank median={np.median(er_vals):.2f} (5/95={np.percentile(er_vals, 5):.2f}/{np.percentile(er_vals, 95):.2f}) | "
                f"bio_cos median={np.median(cos_vals):.3f}",
                flush=True,
            )
            del layer_tensor

            # Incremental save so a crash doesn't lose progress
            df = pd.DataFrame.from_records(all_records)
            df.to_csv(args.output_dir / "bootstrap_raw.csv", index=False)

    # Final summary: percentiles per (model, layer, metric)
    df = pd.DataFrame.from_records(all_records)
    summary_records: list[dict] = []
    for (model_key, layer_idx), g in df.groupby(["model", "layer_idx"]):
        for metric in ["effective_rank", "participation_ratio", "mle_id_k20", "bio_vs_nonbio_cos_top10"]:
            vals = g[metric].dropna().to_numpy()
            if vals.size == 0:
                continue
            summary_records.append({
                "model": model_key,
                "layer_idx": int(layer_idx),
                "metric": metric,
                "n_bootstraps": int(vals.size),
                "p05": float(np.percentile(vals, 5)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
            })
    summary = pd.DataFrame.from_records(summary_records)
    summary.to_csv(args.output_dir / "bootstrap_ci_summary.csv", index=False)

    # Plots: error-bar curves per metric, all 5 models
    cmap_models = {m: c for m, c in zip(args.models, plt.get_cmap("tab10").colors)}
    for metric, ylabel in [
        ("effective_rank", "effective rank (frame-level)"),
        ("mle_id_k20", "MLE-ID(k=20) intrinsic dim"),
        ("bio_vs_nonbio_cos_top10", "bio-vs-non-bio top-10 cos"),
        ("participation_ratio", "participation ratio"),
    ]:
        sub = summary[summary["metric"] == metric]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for model_key in args.models:
            m = sub[sub["model"] == model_key].sort_values("layer_idx")
            if m.empty:
                continue
            xs = m["layer_idx"].to_numpy()
            ys = m["p50"].to_numpy()
            yerr_lo = ys - m["p05"].to_numpy()
            yerr_hi = m["p95"].to_numpy() - ys
            is_baseline = model_key == "random_init_eat_seed42"
            ax.errorbar(
                xs, ys, yerr=[yerr_lo, yerr_hi],
                marker="s" if is_baseline else "o",
                linestyle="--" if is_baseline else "-",
                color=cmap_models.get(model_key, "black"),
                label=f"{model_key} (baseline)" if is_baseline else model_key,
                capsize=3, alpha=0.85,
            )
        ax.set_xlabel("layer index")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric}: bootstrap median ± [5%, 95%], B={args.num_bootstraps}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
        if metric == "bio_vs_nonbio_cos_top10":
            ax.set_ylim(0.0, 1.05)
        fig.savefig(args.output_dir / f"bootstrap_ci_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved bootstrap CIs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
