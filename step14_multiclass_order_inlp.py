"""Step 14 — Multi-class Order INLP, the symmetric counterpart to step8.

Closes red-team review concern 3.6 / §4.12 caveat 2: the existing
asymmetry — Class destroys Order, Order does not destroy Class — was
read against asymmetric INLP nullification depths (5-class Class
nulled to chance via ~80 iterations × 4-D-per-iter, vs binary Order
nulled partially via 1-D-per-iter at acc_floor=0.55). This script
runs the analogous *aggressive* multi-class Order INLP and tests
whether Class still survives.

Setup:
  * 4-class Order probe over Aves clips only:
    Passeriformes vs Charadriiformes vs Piciformes vs Strigiformes
    (chance = 0.25). max_iters=80, acc_floor=0.30.
  * Each iteration nulls 3-D (one per class minus reference). Cumulative
    ~240-D nulled — comparable to step8's ~320-D.
  * After Order is at chance, evaluate:
      - 5-class Class probe (Mammalia + 4 Aves Orders)
      - Binary Aves-vs-Mammalia probe
      - Binary Order probe (Passer vs other-Aves) [sanity check that
        Order has been nulled]
  * Restricted to Aves clips (Mammalia is excluded by construction
    from the Order probe). For Class probe evaluation we add Mammalia
    clips back — applying the Aves-trained projection P to Mammalia
    activations (the projection direction is in 768-D, label-agnostic).

Output: artifacts/comparisons/<manifest>/nway_eat_all4/inlp_order_aggressive/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from step2_bootstrap_cis import BASE_SEED, load_layer_tensor
from step3c_veitch_4order import load_taxonomy
from step6_inlp_class_order import (
    nullspace_projection,
    stratified_clip_split,
    expand_clip_positions_to_frames,
    gather_frames_for_model,
)
from step8_inlp_aggressive import (
    fit_multiclass_probe,
    fit_binary_probe,
    PASSERIFORMES,
    OTHER_AVES_ORDERS,
)


MANIFEST_ID = "naturelm_by_order_p100_m200_n200_20260427T222756Z"
DEFAULT_TAX_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}_taxonomic.jsonl")
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "inlp_order_aggressive"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
DEFAULT_LAYERS = [5, 7, 9, 12]
FRAMES_PER_ITEM = 50
ORDER_LABELS_4WAY = (PASSERIFORMES, *OTHER_AVES_ORDERS)
CLASS_LABELS_5WAY = ("Mammalia", *ORDER_LABELS_4WAY)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tax_manifest", type=Path, default=DEFAULT_TAX_MANIFEST)
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    p.add_argument("--max_iters", type=int, default=80)
    p.add_argument("--order_acc_floor", type=float, default=0.30)
    p.add_argument("--c_reg", type=float, default=1.0)
    return p.parse_args()


def run_inlp_loop(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    *,
    max_iters: int,
    acc_floor: float,
    C: float,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    d = X_tr.shape[1]
    P = np.eye(d, dtype=np.float64)
    log: list[dict] = []
    for it in range(max_iters):
        Xtr_cur = X_tr @ P.T
        Xte_cur = X_te @ P.T
        _, acc, coef = fit_multiclass_probe(
            Xtr_cur, y_tr, Xte_cur, y_te, C=C, seed=seed,
        )
        log.append({"iter": it, "acc": acc})
        if acc <= acc_floor + 0.01:
            break
        P = nullspace_projection(coef) @ P
    return P, log


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(args.tax_manifest)
    print(f"Step 14 — multi-class Order INLP. {len(taxonomy)} taxonomic records | "
          f"max_iters={args.max_iters}, acc_floor={args.order_acc_floor}",
          flush=True)

    rows: list[dict] = []
    iters: list[dict] = []

    for model_key in args.models:
        shard_dir = args.roadmap_dir / model_key / "shards"
        if not shard_dir.exists():
            print(f"WARN: shards missing for {model_key}, skipping", flush=True)
            continue
        print(f"\n=== {model_key} ===", flush=True)

        l0_tensor, sample_meta = load_layer_tensor(shard_dir, layer_idx=0)
        valid_token_counts = np.array(
            [int(s.get("valid_token_count", l0_tensor.shape[1])) for s in sample_meta]
        )
        del l0_tensor

        cls = np.array([taxonomy.get(s.get("id", ""), {}).get("class", "") for s in sample_meta])
        ord_ = np.array([taxonomy.get(s.get("id", ""), {}).get("order", "") for s in sample_meta])

        # 4-class Order labels (Aves only)
        mask_aves_4order = (cls == "Aves") & np.isin(ord_, ORDER_LABELS_4WAY)
        # 5-class Class labels (Aves 4 Orders + Mammalia)
        multiclass_5 = np.full(len(sample_meta), fill_value="", dtype=object)
        for i, (c, o) in enumerate(zip(cls, ord_)):
            if c == "Aves" and o in ORDER_LABELS_4WAY:
                multiclass_5[i] = o
            elif c == "Mammalia":
                multiclass_5[i] = "Mammalia"
        mask_5class = np.isin(multiclass_5, CLASS_LABELS_5WAY)
        mask_passer = (cls == "Aves") & (ord_ == PASSERIFORMES)
        mask_other_aves = (cls == "Aves") & np.isin(ord_, OTHER_AVES_ORDERS)
        mask_binary_order = mask_passer | mask_other_aves

        n_4order = int(mask_aves_4order.sum())
        if n_4order < 80:
            print(f"  insufficient Aves Order clips ({n_4order}); skipping", flush=True)
            continue

        for layer_idx in args.layers:
            t0 = time.time()
            per_item = gather_frames_for_model(
                shard_dir, layer_idx, valid_token_counts,
                args.frames_per_item, BASE_SEED + layer_idx,
            )
            d = per_item.shape[-1]

            # 4-class Order training data (Aves only)
            order_clip_str = ord_[mask_aves_4order]
            ord_label_to_int = {l: i for i, l in enumerate(ORDER_LABELS_4WAY)}
            order4_clip_y = np.array(
                [ord_label_to_int[l] for l in order_clip_str], dtype=np.int64
            )
            order4_X = per_item[mask_aves_4order].reshape(-1, d).astype(np.float64)
            order4_y = np.repeat(order4_clip_y, args.frames_per_item)

            order_train_pos, order_test_pos = stratified_clip_split(
                order4_clip_y, test_size=0.2, seed=BASE_SEED,
            )
            order_train_idx = expand_clip_positions_to_frames(
                order_train_pos, args.frames_per_item
            )
            order_test_idx = expand_clip_positions_to_frames(
                order_test_pos, args.frames_per_item
            )

            scaler = StandardScaler().fit(order4_X[order_train_idx])
            order4_Xs = scaler.transform(order4_X)

            order4_Xs_tr = order4_Xs[order_train_idx]
            order4_y_tr = order4_y[order_train_idx]
            order4_Xs_te = order4_Xs[order_test_idx]
            order4_y_te = order4_y[order_test_idx]

            order4_chance = 0.25
            _, pre_4order_acc, _ = fit_multiclass_probe(
                order4_Xs_tr, order4_y_tr, order4_Xs_te, order4_y_te,
                C=args.c_reg, seed=BASE_SEED,
            )

            # 5-class Class data (Mammalia + 4 Aves Orders), use the
            # SAME scaler (train on Order frames) so the Order-trained
            # projection P is in the right basis for both probes.
            class5_clip_y_str = multiclass_5[mask_5class]
            cls_label_to_int = {l: i for i, l in enumerate(CLASS_LABELS_5WAY)}
            class5_clip_y = np.array(
                [cls_label_to_int[l] for l in class5_clip_y_str], dtype=np.int64
            )
            class5_X = per_item[mask_5class].reshape(-1, d).astype(np.float64)
            class5_Xs = scaler.transform(class5_X)
            class5_y = np.repeat(class5_clip_y, args.frames_per_item)
            class_train_pos, class_test_pos = stratified_clip_split(
                class5_clip_y, test_size=0.2, seed=BASE_SEED,
            )
            class_train_idx = expand_clip_positions_to_frames(
                class_train_pos, args.frames_per_item
            )
            class_test_idx = expand_clip_positions_to_frames(
                class_test_pos, args.frames_per_item
            )
            class5_Xs_tr = class5_Xs[class_train_idx]
            class5_Xs_te = class5_Xs[class_test_idx]
            class5_y_tr = class5_y[class_train_idx]
            class5_y_te = class5_y[class_test_idx]

            # Binary Aves-vs-Mammalia
            bin_class_y_per_clip = np.array(
                [0 if l == "Mammalia" else 1 for l in class5_clip_y_str],
                dtype=np.int64,
            )
            bin_class_y = np.repeat(bin_class_y_per_clip, args.frames_per_item)
            bin_class_y_tr = bin_class_y[class_train_idx]
            bin_class_y_te = bin_class_y[class_test_idx]

            # Binary Order (Passer vs other-Aves) — sanity check
            bin_ord_clip_y = (order_clip_str == PASSERIFORMES).astype(np.int64)
            bin_ord_y = np.repeat(bin_ord_clip_y, args.frames_per_item)
            bin_ord_y_tr = bin_ord_y[order_train_idx]
            bin_ord_y_te = bin_ord_y[order_test_idx]

            _, pre_class5_acc, _ = fit_multiclass_probe(
                class5_Xs_tr, class5_y_tr, class5_Xs_te, class5_y_te,
                C=args.c_reg, seed=BASE_SEED,
            )
            _, pre_bin_class_acc, _ = fit_binary_probe(
                class5_Xs_tr, bin_class_y_tr, class5_Xs_te, bin_class_y_te,
                C=args.c_reg, seed=BASE_SEED,
            )
            _, pre_bin_order_acc, _ = fit_binary_probe(
                order4_Xs_tr, bin_ord_y_tr, order4_Xs_te, bin_ord_y_te,
                C=args.c_reg, seed=BASE_SEED,
            )

            # Run the aggressive 4-class Order INLP
            P, log = run_inlp_loop(
                order4_Xs_tr, order4_y_tr, order4_Xs_te, order4_y_te,
                max_iters=args.max_iters,
                acc_floor=args.order_acc_floor,
                C=args.c_reg,
                seed=BASE_SEED,
            )
            for it_log in log:
                iters.append({
                    "model": model_key, "layer": layer_idx,
                    "iter": it_log["iter"], "order4_acc": it_log["acc"],
                })

            # Apply P to all evaluation sets, retrain probes
            order4_Xs_tr_p = order4_Xs_tr @ P.T
            order4_Xs_te_p = order4_Xs_te @ P.T
            class5_Xs_tr_p = class5_Xs_tr @ P.T
            class5_Xs_te_p = class5_Xs_te @ P.T

            _, post_4order_acc, _ = fit_multiclass_probe(
                order4_Xs_tr_p, order4_y_tr, order4_Xs_te_p, order4_y_te,
                C=args.c_reg, seed=BASE_SEED,
            )
            _, post_class5_acc, _ = fit_multiclass_probe(
                class5_Xs_tr_p, class5_y_tr, class5_Xs_te_p, class5_y_te,
                C=args.c_reg, seed=BASE_SEED,
            )
            _, post_bin_class_acc, _ = fit_binary_probe(
                class5_Xs_tr_p, bin_class_y_tr, class5_Xs_te_p, bin_class_y_te,
                C=args.c_reg, seed=BASE_SEED,
            )
            _, post_bin_order_acc, _ = fit_binary_probe(
                order4_Xs_tr_p, bin_ord_y_tr, order4_Xs_te_p, bin_ord_y_te,
                C=args.c_reg, seed=BASE_SEED,
            )

            rows.append({
                "model": model_key, "layer": layer_idx,
                "n_inlp_iters": len(log),
                "pre_4order_acc": pre_4order_acc,
                "post_4order_acc": post_4order_acc,
                "order4_chance": order4_chance,
                "pre_class5_acc": pre_class5_acc,
                "post_class5_acc": post_class5_acc,
                "pre_binary_class_acc": pre_bin_class_acc,
                "post_binary_class_acc": post_bin_class_acc,
                "pre_binary_order_acc": pre_bin_order_acc,
                "post_binary_order_acc": post_bin_order_acc,
            })
            elapsed = time.time() - t0
            print(
                f"  L{layer_idx:>2}: 4ord {pre_4order_acc:.3f} → {post_4order_acc:.3f}  "
                f"5cls {pre_class5_acc:.3f} → {post_class5_acc:.3f}  "
                f"bin-cls {pre_bin_class_acc:.3f} → {post_bin_class_acc:.3f}  "
                f"bin-ord {pre_bin_order_acc:.3f} → {post_bin_order_acc:.3f}  "
                f"iters={len(log)}  ({elapsed:.1f}s)",
                flush=True,
            )

            pd.DataFrame(rows).to_csv(
                args.output_dir / "inlp_order_aggressive_summary.csv", index=False
            )
            pd.DataFrame(iters).to_csv(
                args.output_dir / "inlp_order_aggressive_iters.csv", index=False
            )

    print(f"\nDone. Wrote {len(rows)} rows.", flush=True)


if __name__ == "__main__":
    main()
