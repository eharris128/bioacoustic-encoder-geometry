"""Step 9 — INLP-Order-first asymmetric hierarchy test.

Reviewer concern (6) follow-up named in inlp_writeup.md option (3):
the cleanest causal version of §4.8's Class⊥Order claim is asymmetric.

  - Train Order probe (Passer vs other-Aves), iteratively null its row
    space until Order is unrecoverable. Apply the resulting projection
    P to Class data. Test Class probe accuracy on the Order-nullspace.
  - Compare to step6/step8: train Class probe, null it, test Order
    survival.

Asymmetric pattern that supports a factored hierarchy:
  - Class survives Order-nullification (Class is a higher-level
    feature, distributed across many directions; nulling Order doesn't
    touch most of Class).
  - Order does NOT survive Class-nullification (Order lives "inside"
    the Class subspace; nulling Class destroys Order).

If both directions show similar survival → no hierarchy, the two
features live in mutually-overlapping subspaces.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/inlp_order_first/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from step2_bootstrap_cis import BASE_SEED, load_layer_tensor
from step3c_veitch_4order import per_clip_frame_sample, load_taxonomy
from step6_inlp_class_order import (
    nullspace_projection,
    stratified_clip_split,
    expand_clip_positions_to_frames,
    gather_frames_for_model,
    fit_probe,
    majority_baseline,
)


MANIFEST_ID = "naturelm_by_order_p100_m200_n200_20260427T222756Z"
DEFAULT_TAX_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}_taxonomic.jsonl")
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "inlp_order_first"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
DEFAULT_LAYERS = [5, 7, 9, 12]
FRAMES_PER_ITEM = 50
PASSERIFORMES = "Passeriformes"
OTHER_AVES_ORDERS = ("Charadriiformes", "Piciformes", "Strigiformes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tax_manifest", type=Path, default=DEFAULT_TAX_MANIFEST)
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    p.add_argument("--max_iters", type=int, default=40)
    p.add_argument("--order_acc_floor", type=float, default=0.55)
    p.add_argument("--c_reg", type=float, default=1.0)
    return p.parse_args()


def run_inlp_order(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    *, max_iters: int, acc_floor: float, C: float, seed: int,
) -> tuple[np.ndarray, list[dict]]:
    d = X_tr.shape[1]
    P = np.eye(d, dtype=np.float64)
    log: list[dict] = []
    baseline = majority_baseline(y_te)
    for it in range(max_iters):
        Xtr_cur = X_tr @ P.T
        Xte_cur = X_te @ P.T
        _, acc, coef = fit_probe(Xtr_cur, y_tr, Xte_cur, y_te, C=C, seed=seed)
        log.append({"iter": it, "acc": acc, "majority_baseline": baseline})
        if acc <= max(acc_floor, baseline + 0.01):
            break
        P = nullspace_projection(coef) @ P
    return P, log


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(args.tax_manifest)
    print(f"Taxonomic manifest: {len(taxonomy)} records | INLP-Order-first "
          f"(max_iters={args.max_iters}, acc_floor={args.order_acc_floor})", flush=True)

    iter_records: list[dict] = []
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

        mask_aves = cls == "Aves"
        mask_mam = cls == "Mammalia"
        mask_passer = mask_aves & (ord_ == PASSERIFORMES)
        mask_other_aves = mask_aves & np.isin(ord_, OTHER_AVES_ORDERS)
        mask_class_clips = mask_aves | mask_mam
        mask_order_clips = mask_passer | mask_other_aves

        if min(int(mask_class_clips.sum()), int(mask_order_clips.sum())) < 50:
            continue

        for layer_idx in args.layers:
            t0 = time.time()
            per_item = gather_frames_for_model(
                shard_dir, layer_idx, valid_token_counts,
                args.frames_per_item, BASE_SEED + layer_idx,
            )
            d = per_item.shape[-1]

            order_clip_y = (ord_[mask_order_clips] == PASSERIFORMES).astype(np.int64)
            order_X = per_item[mask_order_clips].reshape(-1, d).astype(np.float64)
            order_y = np.repeat(order_clip_y, args.frames_per_item)
            order_train_pos, order_test_pos = stratified_clip_split(order_clip_y, 0.2, BASE_SEED)
            order_tr_idx = expand_clip_positions_to_frames(order_train_pos, args.frames_per_item)
            order_te_idx = expand_clip_positions_to_frames(order_test_pos, args.frames_per_item)

            class_clip_y = (cls[mask_class_clips] == "Aves").astype(np.int64)
            class_X = per_item[mask_class_clips].reshape(-1, d).astype(np.float64)
            class_y = np.repeat(class_clip_y, args.frames_per_item)
            class_train_pos, class_test_pos = stratified_clip_split(class_clip_y, 0.2, BASE_SEED)
            class_tr_idx = expand_clip_positions_to_frames(class_train_pos, args.frames_per_item)
            class_te_idx = expand_clip_positions_to_frames(class_test_pos, args.frames_per_item)

            # Standardize on Order-train portion (the INLP target).
            scaler = StandardScaler().fit(order_X[order_tr_idx])
            order_Xs = scaler.transform(order_X)
            class_Xs = scaler.transform(class_X)

            order_Xs_tr = order_Xs[order_tr_idx]; order_Xs_te = order_Xs[order_te_idx]
            order_y_tr = order_y[order_tr_idx];   order_y_te = order_y[order_te_idx]
            class_Xs_tr = class_Xs[class_tr_idx]; class_Xs_te = class_Xs[class_te_idx]
            class_y_tr = class_y[class_tr_idx];   class_y_te = class_y[class_te_idx]

            class_baseline = majority_baseline(class_y_te)
            order_baseline = majority_baseline(order_y_te)

            _, pre_class_acc, _ = fit_probe(class_Xs_tr, class_y_tr, class_Xs_te, class_y_te,
                                            C=args.c_reg, seed=BASE_SEED)
            _, pre_order_acc, _ = fit_probe(order_Xs_tr, order_y_tr, order_Xs_te, order_y_te,
                                            C=args.c_reg, seed=BASE_SEED)

            # INLP on Order: null Order direction iteratively.
            P, log = run_inlp_order(
                order_Xs_tr, order_y_tr, order_Xs_te, order_y_te,
                max_iters=args.max_iters, acc_floor=args.order_acc_floor,
                C=args.c_reg, seed=BASE_SEED,
            )
            for it_log in log:
                iter_records.append({
                    "model": model_key, "layer": layer_idx,
                    "iter": it_log["iter"], "order_acc": it_log["acc"],
                    "order_baseline": it_log["majority_baseline"],
                })

            class_post_tr = class_Xs_tr @ P.T
            class_post_te = class_Xs_te @ P.T
            order_post_tr = order_Xs_tr @ P.T
            order_post_te = order_Xs_te @ P.T
            _, post_order_acc, _ = fit_probe(order_post_tr, order_y_tr, order_post_te, order_y_te,
                                             C=args.c_reg, seed=BASE_SEED)
            _, post_class_acc, _ = fit_probe(class_post_tr, class_y_tr, class_post_te, class_y_te,
                                             C=args.c_reg, seed=BASE_SEED)

            class_survival = (
                (post_class_acc - class_baseline)
                / max(pre_class_acc - class_baseline, 1e-6)
            )
            summary_records.append({
                "model": model_key, "layer": layer_idx,
                "pre_class_acc": pre_class_acc, "post_class_acc": post_class_acc,
                "class_baseline": class_baseline,
                "pre_order_acc": pre_order_acc, "post_order_acc": post_order_acc,
                "order_baseline": order_baseline,
                "class_survival_ratio": class_survival,
                "n_inlp_iters": len(log),
            })
            print(
                f"  L{layer_idx:>2}: order {pre_order_acc:.3f} → {post_order_acc:.3f}  "
                f"class {pre_class_acc:.3f} → {post_class_acc:.3f}  "
                f"class_survival={class_survival:.2f}  iters={len(log)}  "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

    pd.DataFrame(iter_records).to_csv(args.output_dir / "inlp_order_first_iters.csv", index=False)
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(args.output_dir / "inlp_order_first_summary.csv", index=False)
    print(f"\nWrote {args.output_dir}/inlp_order_first_summary.csv "
          f"({len(summary_df)} rows)", flush=True)

    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for layer in sorted(summary_df["layer"].unique()):
            sub = summary_df[summary_df["layer"] == layer].sort_values("model")
            ax.plot(sub["model"], sub["class_survival_ratio"], marker="o", label=f"L{layer}")
        ax.axhline(0.0, color="grey", lw=0.5)
        ax.axhline(1.0, color="grey", lw=0.5, ls="--")
        ax.set_ylabel("Class accuracy survival ratio")
        ax.set_title("INLP-Order-first: does Class survive Order-nullification?\n"
                     "1.0 + Order survival << 1 in step6/step8 → factored.")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Layer", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir / "inlp_order_first_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {args.output_dir}/inlp_order_first_summary.png", flush=True)


if __name__ == "__main__":
    main()
