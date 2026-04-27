"""Step 3c: Veitch-style hierarchy test.

Veitch et al. (NeurIPS 2024) on hierarchical concept geometry: if a
superordinate concept (e.g. "Aves") and a subordinate concept (e.g.
"Passeriformes" within Aves) are encoded as roughly independent
features, the direction (subordinate_centroid - parent_centroid) should
be approximately orthogonal to the direction (parent_centroid -
other_parent_centroid).

Concretely, we test:
  parent_axis  = Aves_centroid - Mammalia_centroid     (Class-level direction)
  subord_axis  = Passeriformes_centroid - Aves_centroid  (Order-within-Aves)
  prediction   = cos(parent_axis, subord_axis) ≈ 0     (orthogonal)

If trained models implement the hierarchy as an orthogonal Cartesian
product of independent features, this cosine should be near zero. The
random-init baseline gives the null distribution: with no learning,
centroids are roughly random and the cos can be anywhere.

We do this at every layer L0..L12 and look for where (and whether) the
trained models cleanly factor the hierarchy.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/veitch_hierarchy/

Usage:
    python step3c_veitch_hierarchy.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_bootstrap_cis import BASE_SEED, load_layer_tensor


MANIFEST_ID = "naturelm_by_source_100each_20260418T171459Z"
DEFAULT_TAX_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}_taxonomic.jsonl")
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "veitch_hierarchy"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
FRAMES_PER_ITEM = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tax_manifest", type=Path, default=DEFAULT_TAX_MANIFEST)
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
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


def _abs_cos(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(abs(u @ v) / (nu * nv))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy(args.tax_manifest)
    print(f"Taxonomic manifest: {len(taxonomy)} records", flush=True)

    records: list[dict] = []

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
        cls = np.array(
            [taxonomy.get(s.get("id", ""), {}).get("class", "") for s in sample_meta]
        )
        order = np.array(
            [taxonomy.get(s.get("id", ""), {}).get("order", "") for s in sample_meta]
        )
        del layer0_tensor

        # Masks
        mask_aves = cls == "Aves"
        mask_mammalia = cls == "Mammalia"
        mask_passer = mask_aves & (order == "Passeriformes")
        mask_other_aves = mask_aves & (order != "Passeriformes") & (order != "")
        print(
            f"  group sizes: Aves={mask_aves.sum()}, Mammalia={mask_mammalia.sum()}, "
            f"Passeriformes={mask_passer.sum()}, other-Aves={mask_other_aves.sum()}",
            flush=True,
        )

        for layer_idx in range(13):
            t0 = time.time()
            layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
            rng = np.random.default_rng(BASE_SEED + layer_idx)
            per_item = per_clip_frame_sample(
                layer_tensor, valid_token_counts, args.frames_per_item, rng,
            )

            def centroid(mask: np.ndarray) -> np.ndarray | None:
                if mask.sum() == 0:
                    return None
                return per_item[mask].reshape(-1, per_item.shape[-1]).mean(axis=0).astype(np.float64)

            c_aves = centroid(mask_aves)
            c_mammalia = centroid(mask_mammalia)
            c_passer = centroid(mask_passer)
            c_other_aves = centroid(mask_other_aves)

            row: dict = {"model": model_key, "layer_idx": layer_idx,
                         "n_aves": int(mask_aves.sum()),
                         "n_mammalia": int(mask_mammalia.sum()),
                         "n_passer": int(mask_passer.sum()),
                         "n_other_aves": int(mask_other_aves.sum())}

            if c_aves is not None and c_mammalia is not None:
                parent_axis = c_aves - c_mammalia  # Class-level direction
                row["parent_axis_norm"] = float(np.linalg.norm(parent_axis))
                if c_passer is not None:
                    subord_passer = c_passer - c_aves
                    row["passer_subord_axis_norm"] = float(np.linalg.norm(subord_passer))
                    row["abs_cos_parent_vs_passer_subord"] = _abs_cos(parent_axis, subord_passer)
                if c_other_aves is not None:
                    subord_other = c_other_aves - c_aves
                    row["other_aves_subord_axis_norm"] = float(np.linalg.norm(subord_other))
                    row["abs_cos_parent_vs_other_aves_subord"] = _abs_cos(parent_axis, subord_other)
                # Also: angle between the two subord axes (within-Aves separation)
                if c_passer is not None and c_other_aves is not None:
                    row["abs_cos_passer_vs_other_aves"] = _abs_cos(
                        c_passer - c_aves, c_other_aves - c_aves)
            records.append(row)
            del layer_tensor, per_item
            print(f"  L{layer_idx:02d}  cos(parent, passer)={row.get('abs_cos_parent_vs_passer_subord', float('nan')):.3f}  "
                  f"cos(parent, other)={row.get('abs_cos_parent_vs_other_aves_subord', float('nan')):.3f}  "
                  f"cos(passer, other)={row.get('abs_cos_passer_vs_other_aves', float('nan')):.3f}  "
                  f"({time.time() - t0:.1f}s)",
                  flush=True)

        pd.DataFrame.from_records(records).to_csv(
            args.output_dir / "veitch_hierarchy.csv", index=False)

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.output_dir / "veitch_hierarchy.csv", index=False)

    # ---------------- Plots ----------------
    cmap = plt.get_cmap("tab10")
    color_for_model = {m: cmap(i) for i, m in enumerate(args.models)}

    metric_panels = [
        ("abs_cos_parent_vs_passer_subord",
         "|cos((Aves-Mammalia), (Passer-Aves))|"),
        ("abs_cos_parent_vs_other_aves_subord",
         "|cos((Aves-Mammalia), (other-Aves-Aves))|"),
        ("abs_cos_passer_vs_other_aves",
         "|cos((Passer-Aves), (other-Aves-Aves))|"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, (metric, ylabel) in zip(axes, metric_panels):
        for model_key in args.models:
            sub = df[df["model"] == model_key].sort_values("layer_idx")
            if sub.empty or metric not in sub:
                continue
            is_baseline = model_key == "random_init_eat_seed42"
            ax.plot(sub["layer_idx"], sub[metric],
                    marker="s" if is_baseline else "o",
                    linestyle="--" if is_baseline else "-",
                    color=color_for_model[model_key],
                    label=f"{model_key} (baseline)" if is_baseline else model_key)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, label="orthogonal (Veitch prediction)")
        ax.set_xlabel("layer index")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(ylabel.split("|cos(")[1].rstrip("|"))
        ax.grid(alpha=0.3)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.suptitle("Veitch hierarchy test — |cos| between parent and subordinate axes")
    fig.savefig(args.output_dir / "veitch_orthogonality.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Headline tables
    print("\n=== |cos((Aves-Mammalia), (Passer-Aves))| by (layer, model) ===")
    pivot = df.pivot(index="layer_idx", columns="model",
                     values="abs_cos_parent_vs_passer_subord")
    pivot = pivot[args.models] if all(m in pivot.columns for m in args.models) else pivot
    print(pivot.round(3).to_string())

    print("\n=== |cos((Aves-Mammalia), (other-Aves-Aves))| by (layer, model) ===")
    pivot = df.pivot(index="layer_idx", columns="model",
                     values="abs_cos_parent_vs_other_aves_subord")
    pivot = pivot[args.models] if all(m in pivot.columns for m in args.models) else pivot
    print(pivot.round(3).to_string())

    print(f"\nSaved Veitch hierarchy artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
