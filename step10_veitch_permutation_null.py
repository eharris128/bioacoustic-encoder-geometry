"""Step 10 — empirical Veitch random-orthogonality null distribution.

Reviewer concern (6) names the random-orthogonality floor in 768-d as
√(2/(πd)) ≈ 0.029, but that's the theoretical |cos| of two
*independently*-chosen unit vectors. Our parent / subord directions
are not independent — they're constructed as centroid differences over
overlapping subpopulations of the same 800 clips. The empirical null
under our actual sampling structure may differ.

This script estimates the null directly by label permutation. For each
(model, layer):

  1. Compute the original |cos(parent, subord)| as in step3c_veitch_4order.
  2. For B=200 random permutations:
       - Shuffle Class and Order labels independently across clips,
         preserving per-clip frame structure.
       - Recompute |cos(parent_shuffled, subord_shuffled)|.
  3. Report the empirical null distribution: median, 5th/95th percentiles.
  4. Compute a permutation p-value on the original |cos|.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/veitch_perm_null/
  veitch_perm_null_results.csv      # per (model, layer, permutation)
  veitch_perm_null_summary.csv      # per (model, layer): obs, p, null mean/p05/p95
  veitch_perm_null_plot.png

Usage:
    python step10_veitch_permutation_null.py
    python step10_veitch_permutation_null.py --n_perm 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_bootstrap_cis import BASE_SEED, load_layer_tensor
from step3c_veitch_4order import per_clip_frame_sample, load_taxonomy


MANIFEST_ID = "naturelm_by_order_p100_m200_n200_20260427T222756Z"
DEFAULT_TAX_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}_taxonomic.jsonl")
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "veitch_perm_null"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
DEFAULT_LAYERS = [7, 9, 12]
FRAMES_PER_ITEM = 50
PASSERIFORMES = "Passeriformes"
TARGET_ORDERS = ("Passeriformes", "Charadriiformes", "Piciformes", "Strigiformes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tax_manifest", type=Path, default=DEFAULT_TAX_MANIFEST)
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    p.add_argument("--n_perm", type=int, default=200)
    return p.parse_args()


def abs_cos(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(abs(u @ v) / (nu * nv))


def compute_veitch_cos(
    per_item: np.ndarray,
    cls_labels: np.ndarray,
    ord_labels: np.ndarray,
) -> float:
    """|cos(Aves_centroid - Mammalia_centroid, Passer_centroid - Aves_centroid)|."""
    d = per_item.shape[-1]
    mask_aves = cls_labels == "Aves"
    mask_mam = cls_labels == "Mammalia"
    mask_passer = mask_aves & (ord_labels == PASSERIFORMES)
    if mask_aves.sum() < 5 or mask_mam.sum() < 5 or mask_passer.sum() < 5:
        return float("nan")
    c_aves = per_item[mask_aves].reshape(-1, d).mean(axis=0).astype(np.float64)
    c_mam = per_item[mask_mam].reshape(-1, d).mean(axis=0).astype(np.float64)
    c_passer = per_item[mask_passer].reshape(-1, d).mean(axis=0).astype(np.float64)
    return abs_cos(c_aves - c_mam, c_passer - c_aves)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(args.tax_manifest)
    print(f"Taxonomic manifest: {len(taxonomy)} records | n_perm={args.n_perm}", flush=True)

    perm_records: list[dict] = []
    summary_records: list[dict] = []

    for model_key in args.models:
        shard_dir = args.roadmap_dir / model_key / "shards"
        if not shard_dir.exists():
            continue
        print(f"\n=== {model_key} ===", flush=True)

        l0_tensor, sample_meta = load_layer_tensor(shard_dir, layer_idx=0)
        valid_token_counts = np.array(
            [int(s.get("valid_token_count", l0_tensor.shape[1])) for s in sample_meta]
        )
        del l0_tensor

        cls = np.array([taxonomy.get(s.get("id", ""), {}).get("class", "") for s in sample_meta])
        ord_ = np.array([taxonomy.get(s.get("id", ""), {}).get("order", "") for s in sample_meta])

        for layer_idx in args.layers:
            t0 = time.time()
            layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
            rng = np.random.default_rng(BASE_SEED + layer_idx)
            per_item = per_clip_frame_sample(
                layer_tensor, valid_token_counts, args.frames_per_item, rng,
            )
            del layer_tensor

            obs = compute_veitch_cos(per_item, cls, ord_)

            null_values = []
            for b in range(args.n_perm):
                rng_b = np.random.default_rng(BASE_SEED + 10000 * (layer_idx + 1) + b)
                # Shuffle Class labels and Order labels independently
                # across the clip axis. Preserves per-clip frame
                # structure; only the (clip → label) assignment is
                # randomized.
                cls_perm = cls.copy()
                ord_perm = ord_.copy()
                rng_b.shuffle(cls_perm)
                rng_b.shuffle(ord_perm)
                v = compute_veitch_cos(per_item, cls_perm, ord_perm)
                null_values.append(v)
                perm_records.append({
                    "model": model_key, "layer": layer_idx,
                    "perm_b": b, "abs_cos": v,
                })

            null_arr = np.array(null_values, dtype=float)
            null_arr = null_arr[~np.isnan(null_arr)]
            null_median = float(np.median(null_arr))
            null_p05 = float(np.quantile(null_arr, 0.05))
            null_p95 = float(np.quantile(null_arr, 0.95))
            # Two-sided p: fraction of null at least as small as observed
            # (we care about how *low* the observed value is relative to null).
            p_lower = float((null_arr <= obs).mean())

            summary_records.append({
                "model": model_key, "layer": layer_idx,
                "observed_abs_cos": obs,
                "null_median": null_median,
                "null_p05": null_p05,
                "null_p95": null_p95,
                "p_value_lower": p_lower,
                "n_perm_valid": int(null_arr.size),
                "theoretical_floor_768d": float(np.sqrt(2.0 / (np.pi * 768))),
            })
            print(
                f"  L{layer_idx:>2}: obs={obs:.4f}  "
                f"null median={null_median:.4f} [{null_p05:.4f}, {null_p95:.4f}]  "
                f"p_lower={p_lower:.3f}  ({time.time() - t0:.1f}s)",
                flush=True,
            )

    pd.DataFrame(perm_records).to_csv(args.output_dir / "veitch_perm_null_results.csv", index=False)
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(args.output_dir / "veitch_perm_null_summary.csv", index=False)
    print(f"\nWrote {args.output_dir}/veitch_perm_null_summary.csv "
          f"({len(summary_df)} rows)", flush=True)

    if not summary_df.empty:
        fig, axes = plt.subplots(1, len(args.layers), figsize=(5 * len(args.layers), 4),
                                 sharey=True)
        if len(args.layers) == 1:
            axes = [axes]
        for ax, layer in zip(axes, args.layers):
            sub = summary_df[summary_df["layer"] == layer].sort_values("model")
            x = np.arange(len(sub))
            ax.errorbar(x, sub["null_median"],
                        yerr=[sub["null_median"] - sub["null_p05"],
                              sub["null_p95"] - sub["null_median"]],
                        fmt="o", color="grey", capsize=4, label="null (perm.)")
            ax.scatter(x, sub["observed_abs_cos"], color="red", marker="x",
                       s=100, label="observed")
            ax.axhline(np.sqrt(2.0 / (np.pi * 768)), color="blue", lw=0.5,
                       ls="--", label="theory √(2/πd)")
            ax.set_xticks(x); ax.set_xticklabels(sub["model"], rotation=30, ha="right")
            ax.set_title(f"L{layer}")
            ax.legend(fontsize=8)
        fig.suptitle("Veitch |cos(parent, subord)|: observed vs permutation null")
        fig.tight_layout()
        fig.savefig(args.output_dir / "veitch_perm_null_plot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {args.output_dir}/veitch_perm_null_plot.png", flush=True)


if __name__ == "__main__":
    main()
