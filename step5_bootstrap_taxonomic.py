"""Bootstrap CIs for the new §4.7 / §4.8 / §4.9 / §5.1 numbers.

The original `step2_bootstrap_cis.py` covered §3-§6. This extends the
bootstrap to the post-2026-04-27 (latest) findings:

  §4.7  Aves vs Mammalia mean cos top-10
  §4.7  Passeriformes vs other-Aves mean cos top-10
  §4.8  |cos((Aves - Mammalia), (Passer - Aves))|  (Veitch orthogonality)
  §4.9  Per-species separability ratio
  §5.1  L12 top-1 eigenvalue share + mean L2 norm

Bootstrap design: resample the 600 manifest items with replacement,
then re-extract 50 frames per item per bootstrap. For each bootstrap,
recompute the centroid-based metrics. B=30 bootstraps. Layer subset is
the headline layers from §10 framings.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/bootstrap_taxonomic_cis/

Usage:
    python step5_bootstrap_taxonomic.py
    python step5_bootstrap_taxonomic.py --num_bootstraps 30 --layers 7 9 12
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
    BASE_SEED, NATURE_SOURCES, SUBSPACE_TOP_K,
    cov_eigvals, load_layer_tensor, subspace_overlap, top_k_basis_via_cov,
)


MANIFEST_ID = "naturelm_by_source_100each_20260418T171459Z"
DEFAULT_TAX_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}_taxonomic.jsonl")
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "bootstrap_taxonomic_cis"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
FRAMES_PER_ITEM = 50
DEFAULT_LAYERS = [5, 7, 9, 10, 11, 12]  # headline layers from §10 framings
MIN_SAMPLES_PER_SPECIES = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tax_manifest", type=Path, default=DEFAULT_TAX_MANIFEST)
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--num_bootstraps", type=int, default=30)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    p.add_argument("--top_k", type=int, default=SUBSPACE_TOP_K)
    p.add_argument("--min_samples_per_species", type=int, default=MIN_SAMPLES_PER_SPECIES)
    p.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
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


def sample_bootstrap_per_item_frames(
    layer_tensor: np.ndarray,  # (n_items, T, D)
    valid_token_counts: np.ndarray,
    frames_per_item: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample n_items items with replacement, then sample frames per item.
    Returns (n_items, F, D) and the (n_items,) bootstrap-resampled item index."""
    n_items, t_max, d = layer_tensor.shape
    item_idx = rng.integers(0, n_items, n_items)
    out = np.empty((n_items, frames_per_item, d), dtype=np.float32)
    for i, src in enumerate(item_idx):
        valid = max(int(min(valid_token_counts[src], t_max)), 1)
        if valid >= frames_per_item:
            f_idx = rng.choice(valid, frames_per_item, replace=False)
        else:
            f_idx = rng.choice(valid, frames_per_item, replace=True)
        out[i] = layer_tensor[src, f_idx, :]
    return out, item_idx


def _abs_cos(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(abs(u @ v) / (nu * nv))


def run_one_bootstrap(
    per_item_frames: np.ndarray,  # (n_items, F, D)
    item_idx: np.ndarray,         # (n_items,) -- which manifest item each row was sampled from
    sample_class: np.ndarray,
    sample_order: np.ndarray,
    sample_species: np.ndarray,
    eligible_species: set[str],
    top_k: int,
) -> dict:
    out: dict = {}
    n_items, frames_per_item, d = per_item_frames.shape

    # Per the bootstrap, the "class" of each per-item slot is the class of the
    # original sampled item.
    boot_class = sample_class[item_idx]
    boot_order = sample_order[item_idx]
    boot_species = sample_species[item_idx]

    def frames_for(mask: np.ndarray) -> np.ndarray | None:
        if mask.sum() == 0:
            return None
        return per_item_frames[mask].reshape(-1, d).astype(np.float64)

    aves_frames = frames_for(boot_class == "Aves")
    mammalia_frames = frames_for(boot_class == "Mammalia")
    passer_frames = frames_for((boot_class == "Aves") & (boot_order == "Passeriformes"))
    other_aves_frames = frames_for(
        (boot_class == "Aves") & (boot_order != "Passeriformes") & (boot_order != "")
    )

    # §4.7a: Aves vs Mammalia top-k cos
    if (aves_frames is not None and mammalia_frames is not None
            and aves_frames.shape[0] >= top_k + 1
            and mammalia_frames.shape[0] >= top_k + 1):
        ba = top_k_basis_via_cov(aves_frames, top_k)
        bm = top_k_basis_via_cov(mammalia_frames, top_k)
        out["cos_class_aves_vs_mammalia"] = subspace_overlap(ba, bm)

    # §4.7b: Passeriformes vs other-Aves top-k cos
    if (passer_frames is not None and other_aves_frames is not None
            and passer_frames.shape[0] >= top_k + 1
            and other_aves_frames.shape[0] >= top_k + 1):
        bp = top_k_basis_via_cov(passer_frames, top_k)
        bo = top_k_basis_via_cov(other_aves_frames, top_k)
        out["cos_order_passer_vs_other_aves"] = subspace_overlap(bp, bo)

    # §4.8: Veitch orthogonality
    if (aves_frames is not None and mammalia_frames is not None
            and passer_frames is not None):
        c_aves = aves_frames.mean(axis=0)
        c_mammalia = mammalia_frames.mean(axis=0)
        c_passer = passer_frames.mean(axis=0)
        out["abs_cos_veitch_class_vs_passer_subord"] = _abs_cos(
            c_aves - c_mammalia, c_passer - c_aves)

    # §4.9: Species separability ratio
    centroids: list[np.ndarray] = []
    within_vars: list[float] = []
    for sp in eligible_species:
        mask = boot_species == sp
        if mask.sum() == 0:
            continue
        sf = per_item_frames[mask].reshape(-1, d).astype(np.float64)
        c = sf.mean(axis=0)
        centroids.append(c)
        within_vars.append(float(np.mean(np.sum((sf - c) ** 2, axis=1))))
    if len(centroids) >= 2:
        carr = np.stack(centroids)
        global_c = carr.mean(axis=0)
        between_var = float(np.mean(np.sum((carr - global_c) ** 2, axis=1)))
        mean_within = float(np.mean(within_vars))
        out["species_separability_ratio"] = (
            between_var / max(mean_within + between_var, 1e-12))
        out["species_n_with_data"] = int(len(centroids))

    # §5.1: top-1 eigenvalue share + mean L2 norm (whole-population, not per-class)
    all_frames = per_item_frames.reshape(-1, d).astype(np.float64)
    eigvals = cov_eigvals(all_frames)
    if eigvals.size > 0:
        out["top1_share"] = float(eigvals[0] / eigvals.sum())
        out["top10_share"] = float(eigvals[:min(10, eigvals.size)].sum() / eigvals.sum())
    out["mean_l2_norm"] = float(np.linalg.norm(all_frames, axis=1).mean())

    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy(args.tax_manifest)
    species_counts: dict[str, int] = {}
    for r in taxonomy.values():
        sp = r.get("species", "")
        if sp:
            species_counts[sp] = species_counts.get(sp, 0) + 1
    eligible_species = {sp for sp, c in species_counts.items() if c >= args.min_samples_per_species}
    print(
        f"Bootstrap CI taxonomic: {len(args.models)} models × {len(args.layers)} layers × "
        f"B={args.num_bootstraps} bootstraps × {args.frames_per_item} frames/item",
        flush=True,
    )
    print(f"Eligible species (>= {args.min_samples_per_species} samples): {len(eligible_species)}",
          flush=True)

    all_records: list[dict] = []

    for model_idx, model_key in enumerate(args.models):
        shard_dir = args.roadmap_dir / model_key / "shards"
        if not shard_dir.exists():
            print(f"WARN: shards missing for {model_key}, skipping", flush=True)
            continue
        print(f"\n=== {model_key} ({model_idx + 1}/{len(args.models)}) ===", flush=True)

        # Load metadata once
        layer0_tensor, sample_meta = load_layer_tensor(shard_dir, layer_idx=0)
        valid_token_counts = np.array(
            [int(s.get("valid_token_count", layer0_tensor.shape[1])) for s in sample_meta]
        )
        sample_class = np.array(
            [taxonomy.get(s.get("id", ""), {}).get("class", "") for s in sample_meta]
        )
        sample_order = np.array(
            [taxonomy.get(s.get("id", ""), {}).get("order", "") for s in sample_meta]
        )
        sample_species = np.array(
            [taxonomy.get(s.get("id", ""), {}).get("species", "") for s in sample_meta]
        )
        del layer0_tensor

        for layer_idx in args.layers:
            print(f"  L{layer_idx:02d} loading...", flush=True, end="")
            t0 = time.time()
            layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
            t_load = time.time() - t0
            print(f" {t_load:.1f}s", flush=True)

            t0 = time.time()
            for b in range(args.num_bootstraps):
                seed = BASE_SEED + b * 1000 + layer_idx
                rng = np.random.default_rng(seed)
                per_item_frames, item_idx = sample_bootstrap_per_item_frames(
                    layer_tensor, valid_token_counts, args.frames_per_item, rng,
                )
                metrics = run_one_bootstrap(
                    per_item_frames, item_idx,
                    sample_class, sample_order, sample_species,
                    eligible_species, args.top_k,
                )
                metrics.update({
                    "model": model_key, "layer_idx": layer_idx,
                    "bootstrap": b, "seed": seed,
                })
                all_records.append(metrics)
            t_boot = time.time() - t0
            avg = t_boot / args.num_bootstraps
            recent = all_records[-args.num_bootstraps:]
            print(
                f"    B={args.num_bootstraps} done in {t_boot:.0f}s ({avg:.1f}s/boot)  "
                f"class_cos median={np.nanmedian([r.get('cos_class_aves_vs_mammalia', np.nan) for r in recent]):.3f}  "
                f"veitch median={np.nanmedian([r.get('abs_cos_veitch_class_vs_passer_subord', np.nan) for r in recent]):.3f}  "
                f"species_sep median={np.nanmedian([r.get('species_separability_ratio', np.nan) for r in recent]):.3f}  "
                f"top1 median={np.nanmedian([r.get('top1_share', np.nan) for r in recent]):.3f}",
                flush=True,
            )
            del layer_tensor

            df = pd.DataFrame.from_records(all_records)
            df.to_csv(args.output_dir / "bootstrap_taxonomic_raw.csv", index=False)

    # ---------------- Summary: 5/50/95 percentiles per (model, layer, metric) ----------------
    df = pd.DataFrame.from_records(all_records)
    df.to_csv(args.output_dir / "bootstrap_taxonomic_raw.csv", index=False)

    metric_cols = [
        "cos_class_aves_vs_mammalia",
        "cos_order_passer_vs_other_aves",
        "abs_cos_veitch_class_vs_passer_subord",
        "species_separability_ratio",
        "top1_share",
        "top10_share",
        "mean_l2_norm",
    ]
    summary_records: list[dict] = []
    for (model_key, layer_idx), g in df.groupby(["model", "layer_idx"]):
        for metric in metric_cols:
            if metric not in g:
                continue
            vals = g[metric].dropna().to_numpy()
            if vals.size == 0:
                continue
            summary_records.append({
                "model": model_key, "layer_idx": int(layer_idx), "metric": metric,
                "n_bootstraps": int(vals.size),
                "p05": float(np.percentile(vals, 5)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
            })
    summary = pd.DataFrame.from_records(summary_records)
    summary.to_csv(args.output_dir / "bootstrap_taxonomic_summary.csv", index=False)

    # ---------------- Plots ----------------
    cmap = plt.get_cmap("tab10")
    color_for_model = {m: cmap(i) for i, m in enumerate(args.models)}

    plot_metrics = [
        ("cos_class_aves_vs_mammalia",
         "Aves-vs-Mammalia top-10 cos (§4.7)", (0.0, 1.05)),
        ("cos_order_passer_vs_other_aves",
         "Passer-vs-other-Aves top-10 cos (§4.7)", (0.0, 1.05)),
        ("abs_cos_veitch_class_vs_passer_subord",
         "|cos((Aves-Mammalia), (Passer-Aves))| — Veitch (§4.8)", (0.0, 1.05)),
        ("species_separability_ratio",
         "Species separability ratio (§4.9)", None),
        ("top1_share",
         "Top-1 eigenvalue share (§5.1)", (0.0, 1.05)),
        ("mean_l2_norm",
         "Mean L2 norm (§5.1)", None),
    ]
    for metric, ylabel, ylim in plot_metrics:
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
                color=color_for_model[model_key],
                label=f"{model_key} (baseline)" if is_baseline else model_key,
                capsize=3, alpha=0.85,
            )
        ax.set_xlabel("layer index")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric}: median ± [5%, 95%], B={args.num_bootstraps}")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
        fig.savefig(args.output_dir / f"bootstrap_taxonomic_{metric}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved bootstrap taxonomic CIs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
