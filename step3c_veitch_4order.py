"""Step 3c — extended 4-Order Veitch hierarchy test.

The original `step3c_veitch_hierarchy.py` only had Passeriformes vs
"other-Aves" because the 600-sample manifest had ≤11 samples per
non-Passeriformes Order. The new per-Order manifest has 100 samples each
for Passeriformes / Charadriiformes / Piciformes / Strigiformes — so we
can run the Veitch hypothesis at full strength.

Veitch et al. predict that for a model encoding a hierarchy as
independent features:
  parent_axis     = Aves_centroid - Mammalia_centroid       (Class direction)
  subord_axis_o   = Order_o_centroid - Aves_centroid        (within-Aves)
  prediction (1)  cos(parent_axis, subord_axis_o) ≈ 0       for each Order o
  prediction (2)  cos(subord_axis_oi, subord_axis_oj) ≈ 0   for distinct orders

We compute both at every layer and report:
  - 4 individual subord/parent cos values per (model, layer)
  - the 4×4 mutual-cos matrix between subord directions per (model, layer)
  - mean off-diagonal mutual cos as a single scalar (the within-Aves
    "subspace dimensionality" proxy)

Compatible with the new manifest's group sizes (100 per Order, 200
Mammalia). Compatible with the old manifest too if you point it at
those shards — but with very thin per-Order data, individual cos values
will be extremely noisy.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/veitch_4order/

Usage:
    python step3c_veitch_4order.py
    python step3c_veitch_4order.py --tax_manifest <...> --roadmap_dir <...> --output_dir <...>
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_bootstrap_cis import BASE_SEED, load_layer_tensor


MANIFEST_ID = "naturelm_by_order_p100_m200_n200_20260427T222756Z"
DEFAULT_TAX_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}_taxonomic.jsonl")
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "veitch_4order"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
DEFAULT_TARGET_ORDERS = ["Passeriformes", "Charadriiformes", "Piciformes", "Strigiformes"]
FRAMES_PER_ITEM = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tax_manifest", type=Path, default=DEFAULT_TAX_MANIFEST)
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--target_orders", nargs="+", default=DEFAULT_TARGET_ORDERS)
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
    print(f"Target Orders: {args.target_orders}", flush=True)

    parent_records: list[dict] = []  # (parent vs subord_o) per (model, layer, order)
    pairwise_records: list[dict] = []  # (subord_oi vs subord_oj) per (model, layer, order_pair)

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

        mask_aves = cls == "Aves"
        mask_mammalia = cls == "Mammalia"
        order_masks = {o: (mask_aves & (order == o)) for o in args.target_orders}
        sizes = {o: int(m.sum()) for o, m in order_masks.items()}
        print(
            f"  Class sizes: Aves={int(mask_aves.sum())}, Mammalia={int(mask_mammalia.sum())}; "
            f"per-Order: {sizes}",
            flush=True,
        )
        if mask_mammalia.sum() == 0:
            print(f"  WARN: no Mammalia samples; parent direction undefined", flush=True)
            continue
        if any(v == 0 for v in sizes.values()):
            present = [o for o, v in sizes.items() if v > 0]
            print(f"  WARN: empty Orders {[o for o,v in sizes.items() if v==0]}; "
                  f"using only {present}", flush=True)

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
            order_centroids = {o: centroid(m) for o, m in order_masks.items()}
            present_orders = [o for o, c in order_centroids.items() if c is not None]

            if c_aves is None or c_mammalia is None or len(present_orders) < 2:
                del layer_tensor, per_item
                continue

            parent_axis = c_aves - c_mammalia
            subord_axes = {o: order_centroids[o] - c_aves for o in present_orders}

            for o in present_orders:
                parent_records.append({
                    "model": model_key, "layer_idx": layer_idx, "order": o,
                    "n_clips": sizes[o],
                    "abs_cos_parent_vs_subord": _abs_cos(parent_axis, subord_axes[o]),
                    "subord_norm": float(np.linalg.norm(subord_axes[o])),
                })

            for oi, oj in itertools.combinations(present_orders, 2):
                pairwise_records.append({
                    "model": model_key, "layer_idx": layer_idx,
                    "order_a": oi, "order_b": oj,
                    "abs_cos_subord_pair": _abs_cos(subord_axes[oi], subord_axes[oj]),
                })

            del layer_tensor, per_item

            cos_summary = ", ".join(
                f"{o[:5]}={_abs_cos(parent_axis, subord_axes[o]):.2f}"
                for o in present_orders
            )
            print(f"  L{layer_idx:02d}  parent vs subord: {cos_summary}  ({time.time() - t0:.1f}s)",
                  flush=True)

        pd.DataFrame.from_records(parent_records).to_csv(
            args.output_dir / "veitch_4order_parent_subord.csv", index=False)
        pd.DataFrame.from_records(pairwise_records).to_csv(
            args.output_dir / "veitch_4order_subord_pairwise.csv", index=False)

    parent_df = pd.DataFrame.from_records(parent_records)
    pairwise_df = pd.DataFrame.from_records(pairwise_records)
    parent_df.to_csv(args.output_dir / "veitch_4order_parent_subord.csv", index=False)
    pairwise_df.to_csv(args.output_dir / "veitch_4order_subord_pairwise.csv", index=False)

    # ---------------- Plots ----------------
    cmap = plt.get_cmap("tab10")
    color_for_model = {m: cmap(i) for i, m in enumerate(args.models)}

    # 1) Parent-vs-subord cos by layer, one panel per Order
    if not parent_df.empty:
        present_orders = sorted(parent_df["order"].unique(),
                                key=lambda o: args.target_orders.index(o)
                                if o in args.target_orders else 99)
        fig, axes = plt.subplots(1, len(present_orders),
                                  figsize=(4.5 * len(present_orders), 5), sharey=True)
        if len(present_orders) == 1:
            axes = [axes]
        for ax, o in zip(axes, present_orders):
            for model_key in args.models:
                sub = parent_df[(parent_df["model"] == model_key) & (parent_df["order"] == o)]
                sub = sub.sort_values("layer_idx")
                if sub.empty:
                    continue
                is_baseline = model_key == "random_init_eat_seed42"
                ax.plot(sub["layer_idx"], sub["abs_cos_parent_vs_subord"],
                        marker="s" if is_baseline else "o",
                        linestyle="--" if is_baseline else "-",
                        color=color_for_model[model_key],
                        label=f"{model_key} (baseline)" if is_baseline else model_key)
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
            ax.set_title(o, fontsize=10)
            ax.set_xlabel("layer index")
            ax.set_ylim(0.0, 1.05)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("|cos((Aves-Mammalia), (Order-Aves))|")
        axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
        fig.suptitle("Veitch parent-vs-subord cos per Order (4-Order test)")
        fig.savefig(args.output_dir / "veitch_4order_parent_subord.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 2) Mean off-diagonal mutual cos between subord directions, by layer
    if not pairwise_df.empty:
        mean_pair = pairwise_df.groupby(["model", "layer_idx"])["abs_cos_subord_pair"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for model_key in args.models:
            sub = mean_pair[mean_pair["model"] == model_key].sort_values("layer_idx")
            if sub.empty:
                continue
            is_baseline = model_key == "random_init_eat_seed42"
            ax.plot(sub["layer_idx"], sub["abs_cos_subord_pair"],
                    marker="s" if is_baseline else "o",
                    linestyle="--" if is_baseline else "-",
                    color=color_for_model[model_key],
                    label=f"{model_key} (baseline)" if is_baseline else model_key)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("layer index")
        ax.set_ylabel("mean |cos| between (Order_i - Aves) directions")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Mean within-Aves subord pairwise cos (4-Order — Veitch dimensionality)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
        fig.savefig(args.output_dir / "veitch_4order_mean_pairwise.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---------------- Headline tables ----------------
    if not parent_df.empty:
        print("\n=== Parent-vs-subord |cos| at L9 (probe-peak) per (model, order) ===")
        pivot = parent_df[parent_df["layer_idx"] == 9].pivot(
            index="model", columns="order", values="abs_cos_parent_vs_subord")
        pivot = pivot.reindex(index=args.models)
        print(pivot.round(3).to_string())
        print("\n=== Parent-vs-subord |cos| at L12 (Veitch peak) per (model, order) ===")
        pivot = parent_df[parent_df["layer_idx"] == 12].pivot(
            index="model", columns="order", values="abs_cos_parent_vs_subord")
        pivot = pivot.reindex(index=args.models)
        print(pivot.round(3).to_string())

    if not pairwise_df.empty:
        print("\n=== Mean within-Aves subord pairwise |cos| at L12 ===")
        pivot = pairwise_df[pairwise_df["layer_idx"] == 12].groupby("model")["abs_cos_subord_pair"].mean()
        pivot = pivot.reindex(args.models)
        print(pivot.round(3).to_string())

    print(f"\nSaved 4-Order Veitch artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
