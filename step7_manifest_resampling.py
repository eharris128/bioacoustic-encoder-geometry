"""Step 7 — manifest-resampling sensitivity for §4.7 / §4.8 / §4.9.

Reviewer concern (4): bootstraps see sample-selection noise *within* the
chosen 800 samples but not *across* alternative draws. Reviewer's specific
prediction: the across-manifest spread for §4.8's L12 |cos| is several ×
the [0.004, 0.110] within-manifest CI.

This script tests a cheaper version of that concern: subsample the existing
800-sample per-Order manifest at 5 seeds (75% retention, stratified),
recompute §4.7 / §4.8 / §4.9 headline numbers per resample, and report
the across-resample spread. Does NOT re-extract activations from a
genuinely different draw of NatureLM-audio-training (that would need
~19 hours of compute); it tests robustness to clip-swap within the
existing 800 only.

If the across-resample spread on this cheap test is already several ×
the bootstrap CI, the reviewer's concern is confirmed and we'd need
the full re-extraction. If the spread is comparable to the bootstrap
CI, the §4.8 number is clip-level robust and the manifest-construction
concern is at least partially defused.

Headline metrics:
  §4.7 — mean(cos(principal_angles(top10 PCA Aves frames, top10 PCA
         Mammalia frames))) at L7 of all 5 models.
  §4.8 — |cos(Aves_centroid - Mammalia_centroid, Passer_centroid -
         Aves_centroid)| at L12 of all 5 models.
  §4.9 — per-species separability ratio = between / (within + between)
         at L10 of all 5 models, restricted to species clearing a 5-
         sample threshold *within the resampled subset*.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/manifest_resampling/
  resampling_results.csv     # per (seed, model, layer, metric)
  resampling_summary.csv     # per (model, metric): mean ± spread across seeds
  resampling_summary.png     # bar plot of spread vs bootstrap CI

Usage:
    python step7_manifest_resampling.py
    python step7_manifest_resampling.py --n_seeds 5 --retention 0.75
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA

from step2_bootstrap_cis import BASE_SEED, load_layer_tensor
from step3c_veitch_4order import per_clip_frame_sample, load_taxonomy


MANIFEST_ID = "naturelm_by_order_p100_m200_n200_20260427T222756Z"
DEFAULT_TAX_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}_taxonomic.jsonl")
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "manifest_resampling"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
LAYER_47 = 7   # §4.7 Aves vs Mammalia
LAYER_48 = 12  # §4.8 Veitch L12
LAYER_49 = 10  # §4.9 species separability
ALL_HEADLINE_LAYERS = sorted({LAYER_47, LAYER_48, LAYER_49})

FRAMES_PER_ITEM = 50
PASSERIFORMES = "Passeriformes"
TARGET_ORDERS = ("Passeriformes", "Charadriiformes", "Piciformes", "Strigiformes")
SPECIES_MIN_CLIPS = 5  # threshold for §4.9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tax_manifest", type=Path, default=DEFAULT_TAX_MANIFEST)
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--retention", type=float, default=0.75,
                   help="fraction of clips retained per stratum")
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    p.add_argument("--top_k", type=int, default=10,
                   help="top-k subspace dim for §4.7 cos")
    return p.parse_args()


def stratified_subsample(
    masks: dict[str, np.ndarray],
    retention: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an indicator array (n,) selecting a stratified retention fraction
    of clips, where each named stratum is independently subsampled."""
    n = next(iter(masks.values())).size
    keep = np.zeros(n, dtype=bool)
    for name, m in masks.items():
        idx = np.where(m)[0]
        if idx.size == 0:
            continue
        n_keep = max(1, int(round(idx.size * retention)))
        chosen = rng.choice(idx, size=n_keep, replace=False)
        keep[chosen] = True
    return keep


def top_k_pca_basis(X: np.ndarray, k: int) -> np.ndarray:
    """Return the top-k right-singular vectors (k, d) of centered X."""
    X = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=k, random_state=0)
    pca.fit(X)
    return pca.components_  # (k, d)


def mean_cos_principal_angles(B1: np.ndarray, B2: np.ndarray) -> float:
    """Mean cosine of principal angles between two row-orthonormal bases."""
    return float(np.cos(subspace_angles(B1.T, B2.T)).mean())


def abs_cos(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(abs(u @ v) / (nu * nv))


def species_separability_ratio(
    species_to_centroid: dict[str, np.ndarray],
    species_to_clip_var: dict[str, float],
) -> float:
    """between / (within + between) over species centroids."""
    if len(species_to_centroid) < 2:
        return float("nan")
    centroids = np.stack(list(species_to_centroid.values()), axis=0)
    grand = centroids.mean(axis=0, keepdims=True)
    between = float(((centroids - grand) ** 2).sum(axis=1).mean())
    within = float(np.mean(list(species_to_clip_var.values())))
    if within + between < 1e-12:
        return float("nan")
    return between / (within + between)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy(args.tax_manifest)
    print(f"Taxonomic manifest: {len(taxonomy)} records", flush=True)
    print(f"Resampling with retention={args.retention} at {args.n_seeds} seeds", flush=True)

    rows: list[dict] = []

    for model_key in args.models:
        shard_dir = args.roadmap_dir / model_key / "shards"
        if not shard_dir.exists():
            print(f"WARN: shards missing for {model_key}, skipping", flush=True)
            continue
        print(f"\n=== {model_key} ===", flush=True)

        # Read sample metadata once (via L0).
        l0_tensor, sample_meta = load_layer_tensor(shard_dir, layer_idx=0)
        valid_token_counts = np.array(
            [int(s.get("valid_token_count", l0_tensor.shape[1])) for s in sample_meta]
        )
        del l0_tensor

        cls = np.array([taxonomy.get(s.get("id", ""), {}).get("class", "") for s in sample_meta])
        ord_ = np.array([taxonomy.get(s.get("id", ""), {}).get("order", "") for s in sample_meta])
        sp = np.array([taxonomy.get(s.get("id", ""), {}).get("species", "") for s in sample_meta])
        src = np.array([s.get("source_dataset", "") for s in sample_meta])

        mask_aves = cls == "Aves"
        mask_mam = cls == "Mammalia"
        order_masks = {o: (mask_aves & (ord_ == o)) for o in TARGET_ORDERS}
        # Bio sources excluding the 4 Aves orders + Mammalia bucket
        # already covered. The remaining "non-bio" stratum.
        mask_nonbio = ~(mask_aves | mask_mam)

        strata = {
            **{f"order:{o}": m for o, m in order_masks.items()},
            "mammalia": mask_mam,
            "nonbio": mask_nonbio,
        }
        sizes = {k: int(v.sum()) for k, v in strata.items()}
        print(f"  Strata: {sizes}", flush=True)

        # Pre-load shards for each headline layer to avoid reloading per seed.
        per_item_by_layer: dict[int, np.ndarray] = {}
        for layer_idx in ALL_HEADLINE_LAYERS:
            t0 = time.time()
            layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
            rng = np.random.default_rng(BASE_SEED + layer_idx)
            per_item = per_clip_frame_sample(
                layer_tensor, valid_token_counts, args.frames_per_item, rng,
            )
            del layer_tensor
            per_item_by_layer[layer_idx] = per_item
            print(f"  L{layer_idx} loaded ({time.time()-t0:.1f}s)", flush=True)

        for seed in range(1, args.n_seeds + 1):
            rng = np.random.default_rng(BASE_SEED + 100 * seed)
            keep = stratified_subsample(strata, args.retention, rng)
            n_kept = int(keep.sum())

            # §4.7 — Aves vs Mammalia top-k subspace cos at L7
            per_item = per_item_by_layer[LAYER_47]
            X_aves = per_item[keep & mask_aves].reshape(-1, per_item.shape[-1]).astype(np.float64)
            X_mam = per_item[keep & mask_mam].reshape(-1, per_item.shape[-1]).astype(np.float64)
            if X_aves.shape[0] >= args.top_k * 4 and X_mam.shape[0] >= args.top_k * 4:
                B_aves = top_k_pca_basis(X_aves, args.top_k)
                B_mam = top_k_pca_basis(X_mam, args.top_k)
                cos_47 = mean_cos_principal_angles(B_aves, B_mam)
            else:
                cos_47 = float("nan")
            rows.append({
                "model": model_key, "layer": LAYER_47, "seed": seed,
                "metric": "aves_vs_mammalia_topk_cos",
                "value": cos_47, "n_kept": n_kept,
            })

            # §4.8 — |cos(parent, subord)| at L12
            per_item = per_item_by_layer[LAYER_48]
            kept_aves = keep & mask_aves
            kept_mam = keep & mask_mam
            kept_passer = keep & mask_aves & (ord_ == PASSERIFORMES)
            if kept_aves.sum() >= 5 and kept_mam.sum() >= 5 and kept_passer.sum() >= 5:
                c_aves = per_item[kept_aves].reshape(-1, per_item.shape[-1]).mean(axis=0).astype(np.float64)
                c_mam = per_item[kept_mam].reshape(-1, per_item.shape[-1]).mean(axis=0).astype(np.float64)
                c_passer = per_item[kept_passer].reshape(-1, per_item.shape[-1]).mean(axis=0).astype(np.float64)
                parent = c_aves - c_mam
                subord = c_passer - c_aves
                cos_48 = abs_cos(parent, subord)
            else:
                cos_48 = float("nan")
            rows.append({
                "model": model_key, "layer": LAYER_48, "seed": seed,
                "metric": "veitch_abs_cos_passer",
                "value": cos_48, "n_kept": n_kept,
            })

            # §4.9 — species separability ratio at L10
            per_item = per_item_by_layer[LAYER_49]
            d = per_item.shape[-1]
            kept_clip_idx = np.where(keep)[0]
            species_to_clip_idx: dict[str, list[int]] = defaultdict(list)
            for i in kept_clip_idx:
                if sp[i]:
                    species_to_clip_idx[sp[i]].append(i)
            qualifying = {
                s: idx for s, idx in species_to_clip_idx.items()
                if len(idx) >= SPECIES_MIN_CLIPS
            }
            species_to_centroid: dict[str, np.ndarray] = {}
            species_to_clip_var: dict[str, float] = {}
            for s, idx_list in qualifying.items():
                clip_centroids = per_item[idx_list].reshape(len(idx_list), -1, d).mean(axis=1)
                species_to_centroid[s] = clip_centroids.mean(axis=0).astype(np.float64)
                species_to_clip_var[s] = float(((clip_centroids - clip_centroids.mean(axis=0, keepdims=True)) ** 2).sum(axis=1).mean())
            sep_49 = species_separability_ratio(species_to_centroid, species_to_clip_var)
            rows.append({
                "model": model_key, "layer": LAYER_49, "seed": seed,
                "metric": "species_separability_ratio",
                "value": sep_49, "n_kept": n_kept,
                "n_species_qualifying": len(qualifying),
            })

            print(
                f"  seed={seed} kept={n_kept}  "
                f"§4.7 L7 cos={cos_47:.4f}  "
                f"§4.8 L12 |cos|={cos_48:.4f}  "
                f"§4.9 L10 sep={sep_49:.4f}  "
                f"(n_sp={len(qualifying)})",
                flush=True,
            )

        # Free memory before next model
        del per_item_by_layer

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "resampling_results.csv", index=False)
    print(f"\nWrote {args.output_dir}/resampling_results.csv ({len(df)} rows)", flush=True)

    # Summary: per (model, metric) mean / min / max / std across seeds
    summary = (
        df.groupby(["model", "metric", "layer"])["value"]
          .agg(["mean", "std", "min", "max", "count"])
          .reset_index()
    )
    summary["spread_p95_p05"] = summary["max"] - summary["min"]
    summary.to_csv(args.output_dir / "resampling_summary.csv", index=False)
    print(f"Wrote {args.output_dir}/resampling_summary.csv ({len(summary)} rows)", flush=True)

    # Plot: spread per (model, metric)
    if not summary.empty:
        metrics = summary["metric"].unique()
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            sub = summary[summary["metric"] == metric].sort_values("model")
            ax.errorbar(
                sub["model"], sub["mean"],
                yerr=[sub["mean"] - sub["min"], sub["max"] - sub["mean"]],
                fmt="o", capsize=5,
            )
            ax.set_title(f"{metric}\n(layer {sub['layer'].iloc[0]})")
            ax.tick_params(axis="x", rotation=30)
            ax.set_ylabel("value (mean ± min/max across seeds)")
        fig.tight_layout()
        fig.savefig(args.output_dir / "resampling_summary.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {args.output_dir}/resampling_summary.png", flush=True)


if __name__ == "__main__":
    main()
