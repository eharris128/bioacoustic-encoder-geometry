"""Step 5 / RESULTS.md §9.1: late-layer collapse mechanism.

The §5 finding: at L12, `sl_eat_all_ssl_all` collapses to eff_rank ~11
(essentially the random-init baseline 9.8), while `sl_eat_bio_ssl_all`
stays at ~180. Both share the SSL fine-tune step; they differ only in
pretrain data. What's actually happening at L12?

This script combines existing artifacts and a minimal L11+L12 shard
load to test three explanations:

  (a) "Mode collapse": L12 representation is dominated by a single
      direction. Test: top eigenvalue / total spectrum > some threshold.
  (b) "Uniform shrink": all directions decay together. Test: spectrum
      shape (eigenvalue ratios) is preserved L11 → L12; only norms
      shrink.
  (c) "Data-dependent collapse": only some inputs collapse. Test:
      per-source eff_rank at L12 — bio sources retain rank, non-bio
      collapse.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/late_layer_collapse/

Usage:
    python step5_late_layer_collapse.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_bootstrap_cis import (
    BASE_SEED, NATURE_SOURCES, cov_eigvals, load_layer_tensor,
)


MANIFEST_ID = "naturelm_by_source_100each_20260418T171459Z"
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "late_layer_collapse"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
FRAMES_PER_ITEM = 50
COMPARE_LAYERS = (10, 11, 12)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--nway_dir", type=Path, default=DEFAULT_NWAY_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
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

    # ---------- Component (c): per-source eff_rank from existing CSV ----------
    per_source_csv = args.nway_dir / "per_source_frame_level" / "per_source_stats.csv"
    if not per_source_csv.exists():
        raise SystemExit(f"Missing prerequisite: {per_source_csv}")
    per_source_df = pd.read_csv(per_source_csv)
    print(f"Loaded per-source stats: {len(per_source_df)} rows", flush=True)

    # Diagnostic: how does L11 → L12 eff_rank change per (model, source)?
    delta_records: list[dict] = []
    for model_key in args.models:
        for src in per_source_df["source"].unique():
            l11 = per_source_df[(per_source_df["model"] == model_key) &
                                (per_source_df["source"] == src) &
                                (per_source_df["layer_idx"] == 11)]["effective_rank"]
            l12 = per_source_df[(per_source_df["model"] == model_key) &
                                (per_source_df["source"] == src) &
                                (per_source_df["layer_idx"] == 12)]["effective_rank"]
            if l11.empty or l12.empty:
                continue
            l11_v, l12_v = float(l11.iloc[0]), float(l12.iloc[0])
            delta_records.append({
                "model": model_key, "source": src,
                "is_bio": src in NATURE_SOURCES,
                "eff_rank_L11": l11_v,
                "eff_rank_L12": l12_v,
                "delta": l12_v - l11_v,
                "ratio_L12_over_L11": l12_v / max(l11_v, 1e-9),
            })
    delta_df = pd.DataFrame.from_records(delta_records)
    delta_df.to_csv(args.output_dir / "per_source_L11_L12_delta.csv", index=False)

    # ---------- Components (a) and (b): full eigenvalue spectrum at L11 and L12 ----------
    spectrum_records: list[dict] = []
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
        del layer0_tensor

        for layer_idx in COMPARE_LAYERS:
            t0 = time.time()
            layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
            rng = np.random.default_rng(BASE_SEED + layer_idx)
            per_item = per_clip_frame_sample(
                layer_tensor, valid_token_counts, args.frames_per_item, rng,
            )
            frames = per_item.reshape(-1, per_item.shape[-1]).astype(np.float64)
            eigvals = cov_eigvals(frames)
            total = eigvals.sum()
            top1_share = float(eigvals[0] / total) if eigvals.size > 0 else float("nan")
            top10_share = float(eigvals[:10].sum() / total) if eigvals.size >= 10 else float("nan")
            top50_share = float(eigvals[:50].sum() / total) if eigvals.size >= 50 else float("nan")
            mean_norm = float(np.linalg.norm(frames, axis=1).mean())
            spectrum_records.append({
                "model": model_key, "layer_idx": layer_idx,
                "n_eigvals": int(eigvals.size),
                "total_variance": float(total),
                "top1_share": top1_share,
                "top10_share": top10_share,
                "top50_share": top50_share,
                "mean_l2_norm": mean_norm,
                "lambda_max": float(eigvals[0]) if eigvals.size > 0 else float("nan"),
                "lambda_min": float(eigvals[-1]) if eigvals.size > 0 else float("nan"),
                # Normalised eigenvalue head for spectrum-shape comparison
                **{f"lambda_{i:02d}_share": float(eigvals[i] / total)
                   for i in range(min(20, eigvals.size))},
            })
            del layer_tensor, per_item, frames
            print(
                f"  L{layer_idx:02d}  total_var={total:.1f}  "
                f"top1={top1_share:.3f}  top10={top10_share:.3f}  "
                f"||x||={mean_norm:.2f}  ({time.time() - t0:.1f}s)",
                flush=True,
            )

    spectrum_df = pd.DataFrame.from_records(spectrum_records)
    spectrum_df.to_csv(args.output_dir / "spectrum_L10_L11_L12.csv", index=False)

    # ---------- Plots ----------
    cmap = plt.get_cmap("tab10")
    color_for_model = {m: cmap(i) for i, m in enumerate(args.models)}

    # (1) per-source L11 → L12 ratio: does the collapse hit bio and non-bio differently?
    fig, axes = plt.subplots(1, len(args.models), figsize=(4.0 * len(args.models), 5.0), sharey=True)
    if len(args.models) == 1:
        axes = [axes]
    sources_in_order = sorted(delta_df["source"].unique())
    for ax, model_key in zip(axes, args.models):
        sub = delta_df[delta_df["model"] == model_key].sort_values("source")
        if sub.empty:
            continue
        colors = ["tab:green" if r else "tab:orange" for r in sub["is_bio"]]
        ax.bar(sub["source"], sub["ratio_L12_over_L11"], color=colors, alpha=0.85)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.7)
        ax.set_title(model_key, fontsize=10)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["source"], rotation=45, ha="right", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("eff_rank(L12) / eff_rank(L11)\n(green = bio, orange = non-bio)")
    fig.suptitle("Per-source L11 → L12 eff_rank ratio (1.0 = no change; <1.0 = collapse)")
    fig.savefig(args.output_dir / "per_source_L12_collapse.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (2) Spectrum shape comparison: lambda_i / total at L12 per model
    head_cols = [f"lambda_{i:02d}_share" for i in range(20)]
    l12 = spectrum_df[spectrum_df["layer_idx"] == 12]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model_key in args.models:
        m = l12[l12["model"] == model_key]
        if m.empty:
            continue
        is_baseline = model_key == "random_init_eat_seed42"
        shares = m.iloc[0][head_cols].to_numpy(dtype=float)
        ax.plot(range(20), shares,
                marker="s" if is_baseline else "o",
                linestyle="--" if is_baseline else "-",
                color=color_for_model[model_key],
                label=f"{model_key} (baseline)" if is_baseline else model_key)
    ax.set_xlabel("eigenvalue rank (descending)")
    ax.set_ylabel("eigenvalue / total variance")
    ax.set_yscale("log")
    ax.set_title("L12 eigenvalue head shape (log scale)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(args.output_dir / "L12_spectrum_head.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (3) L11 vs L12 mean L2 norm and total variance: norm shrink vs spectrum shrink
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, key, title in [(axes[0], "mean_l2_norm", "Mean ||x|| of frames"),
                           (axes[1], "total_variance", "Total covariance variance (sum of eigvals)")]:
        for model_key in args.models:
            sub = spectrum_df[spectrum_df["model"] == model_key].sort_values("layer_idx")
            if sub.empty:
                continue
            is_baseline = model_key == "random_init_eat_seed42"
            ax.plot(sub["layer_idx"], sub[key],
                    marker="s" if is_baseline else "o",
                    linestyle="--" if is_baseline else "-",
                    color=color_for_model[model_key],
                    label=f"{model_key} (baseline)" if is_baseline else model_key)
        ax.set_xlabel("layer index")
        ax.set_ylabel(title)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.suptitle("L10 → L11 → L12 norm and variance trajectories")
    fig.savefig(args.output_dir / "L10_L12_norm_variance.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- Headline tables ----------
    print("\n=== Spectrum top-1 share at L12 (mode-collapse signature) ===")
    pivot = spectrum_df[spectrum_df["layer_idx"] == 12].pivot(
        index="model", columns="layer_idx", values="top1_share")
    print(pivot.round(4).to_string())

    print("\n=== L11 → L12 spectrum top-1 share delta (positive = more concentrated at L12) ===")
    rows = []
    for model_key in args.models:
        m = spectrum_df[spectrum_df["model"] == model_key]
        if {11, 12}.issubset(set(m["layer_idx"])):
            l11_t1 = float(m[m["layer_idx"] == 11]["top1_share"].iloc[0])
            l12_t1 = float(m[m["layer_idx"] == 12]["top1_share"].iloc[0])
            rows.append((model_key, l11_t1, l12_t1, l12_t1 - l11_t1))
    for model_key, l11, l12, delta in rows:
        print(f"  {model_key:<28} L11 top1={l11:.4f} → L12 top1={l12:.4f}   Δ={delta:+.4f}")

    print("\n=== Per-source L12/L11 eff_rank ratio (sl_eat_all_ssl_all is the focal collapse) ===")
    pivot = delta_df.pivot(index="source", columns="model", values="ratio_L12_over_L11")
    pivot = pivot[args.models] if all(m in pivot.columns for m in args.models) else pivot
    print(pivot.round(3).to_string())

    print(f"\nSaved late-layer-collapse artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
