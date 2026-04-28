"""Step 8 — aggressive multi-class INLP for a defensible Class⊥Order test.

Reviewer concern (6) reprise: step6's binary Class probe + max_iters=15
left Class accuracy at ~0.83 (drop ~0.09 from 0.92), so the Order
survival ratio is partly a tautology of "Class wasn't really nulled."

This script addresses that with two changes:

  1. Multi-class Class probe (k=5: 4 Aves Orders + Mammalia). Each INLP
     iteration nulls a (k-1)-D subspace instead of a 1-D direction, so
     Class information dies in fewer iterations.
  2. max_iters=80 with class_acc_floor very low (= 0.30, well below the
     0.20 chance baseline of 5-way), so we run until Class is genuinely
     unrecoverable.

The Order probe (Passer vs other-Aves) is then evaluated on the Class-
nullspace activations, with clip-level train/test splits.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/inlp_aggressive/

Usage:
    python step8_inlp_aggressive.py
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
)


MANIFEST_ID = "naturelm_by_order_p100_m200_n200_20260427T222756Z"
DEFAULT_TAX_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}_taxonomic.jsonl")
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "inlp_aggressive"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
DEFAULT_LAYERS = [5, 7, 9, 12]
FRAMES_PER_ITEM = 50
PASSERIFORMES = "Passeriformes"
OTHER_AVES_ORDERS = ("Charadriiformes", "Piciformes", "Strigiformes")
CLASS_LABELS_MULTICLASS = (
    "Mammalia", "Passeriformes", "Charadriiformes", "Piciformes", "Strigiformes",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tax_manifest", type=Path, default=DEFAULT_TAX_MANIFEST)
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    p.add_argument("--max_iters", type=int, default=80)
    p.add_argument("--class_acc_floor", type=float, default=0.30)
    p.add_argument("--c_reg", type=float, default=1.0)
    return p.parse_args()


def fit_multiclass_probe(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    C: float, seed: int,
) -> tuple[LogisticRegression, float, np.ndarray]:
    """Fit multinomial logistic regression; return (clf, acc, coef matrix (k-1, d))."""
    clf = LogisticRegression(
        penalty="l2", C=C, solver="lbfgs", max_iter=4000,
        multi_class="multinomial", random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))
    return clf, acc, clf.coef_.copy()


def fit_binary_probe(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    C: float, seed: int,
) -> tuple[LogisticRegression, float, np.ndarray]:
    clf = LogisticRegression(
        penalty="l2", C=C, solver="liblinear", max_iter=2000, random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))
    return clf, acc, clf.coef_.copy()


def run_aggressive_inlp(
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
    counts = np.bincount(y_tr)
    chance = 1.0 / counts.size  # uniform-class chance (multi-class)
    for it in range(max_iters):
        Xtr_cur = X_tr @ P.T
        Xte_cur = X_te @ P.T
        _, acc, coef = fit_multiclass_probe(Xtr_cur, y_tr, Xte_cur, y_te, C=C, seed=seed)
        log.append({"iter": it, "acc": acc, "chance": chance})
        if acc <= acc_floor + 0.01:
            break
        P = nullspace_projection(coef) @ P
    return P, log


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(args.tax_manifest)
    print(f"Taxonomic manifest: {len(taxonomy)} records | aggressive INLP "
          f"(multi-class Class, max_iters={args.max_iters}, "
          f"acc_floor={args.class_acc_floor})", flush=True)

    iter_records: list[dict] = []
    summary_records: list[dict] = []

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

        # Multi-class Class label: 4 Aves Orders + Mammalia. We use Order
        # for Aves clips and the literal "Mammalia" for Mammalia clips.
        multiclass_label = np.full(len(sample_meta), fill_value="", dtype=object)
        for i, (c, o) in enumerate(zip(cls, ord_)):
            if c == "Aves" and o in (PASSERIFORMES, *OTHER_AVES_ORDERS):
                multiclass_label[i] = o
            elif c == "Mammalia":
                multiclass_label[i] = "Mammalia"

        mask_class_clips = np.isin(multiclass_label, CLASS_LABELS_MULTICLASS)
        mask_aves = cls == "Aves"
        mask_passer = mask_aves & (ord_ == PASSERIFORMES)
        mask_other_aves = mask_aves & np.isin(ord_, OTHER_AVES_ORDERS)
        mask_order_clips = mask_passer | mask_other_aves

        n_class = int(mask_class_clips.sum())
        n_passer, n_other = int(mask_passer.sum()), int(mask_other_aves.sum())
        print(f"  Class clips (5-way) = {n_class}, Passer={n_passer}, "
              f"other-Aves={n_other}", flush=True)
        if min(n_class, n_passer, n_other) < 20:
            print("  WARN: too few samples", flush=True)
            continue

        for layer_idx in args.layers:
            t0 = time.time()
            per_item = gather_frames_for_model(
                shard_dir, layer_idx, valid_token_counts,
                args.frames_per_item, BASE_SEED + layer_idx,
            )
            d = per_item.shape[-1]

            # Class probe: 5-way label, clip-level split
            class_clip_y_str = multiclass_label[mask_class_clips]
            label_to_int = {l: i for i, l in enumerate(CLASS_LABELS_MULTICLASS)}
            class_clip_y = np.array([label_to_int[l] for l in class_clip_y_str], dtype=np.int64)
            class_X = per_item[mask_class_clips].reshape(-1, d).astype(np.float64)
            class_y = np.repeat(class_clip_y, args.frames_per_item)

            class_train_pos, class_test_pos = stratified_clip_split(
                class_clip_y, test_size=0.2, seed=BASE_SEED,
            )
            class_train_idx = expand_clip_positions_to_frames(class_train_pos, args.frames_per_item)
            class_test_idx = expand_clip_positions_to_frames(class_test_pos, args.frames_per_item)

            # Order probe: binary, clip-level split
            order_clip_y = (ord_[mask_order_clips] == PASSERIFORMES).astype(np.int64)
            order_X = per_item[mask_order_clips].reshape(-1, d).astype(np.float64)
            order_y = np.repeat(order_clip_y, args.frames_per_item)
            order_train_pos, order_test_pos = stratified_clip_split(
                order_clip_y, test_size=0.2, seed=BASE_SEED,
            )
            order_train_idx = expand_clip_positions_to_frames(order_train_pos, args.frames_per_item)
            order_test_idx = expand_clip_positions_to_frames(order_test_pos, args.frames_per_item)

            scaler = StandardScaler().fit(class_X[class_train_idx])
            class_Xs = scaler.transform(class_X)
            order_Xs = scaler.transform(order_X)

            class_Xs_tr = class_Xs[class_train_idx]; class_Xs_te = class_Xs[class_test_idx]
            class_y_tr = class_y[class_train_idx];  class_y_te = class_y[class_test_idx]
            order_Xs_tr = order_Xs[order_train_idx]; order_Xs_te = order_Xs[order_test_idx]
            order_y_tr = order_y[order_train_idx];  order_y_te = order_y[order_test_idx]

            class_chance = 1.0 / len(CLASS_LABELS_MULTICLASS)
            order_majority = float(np.bincount(order_y_te).max() / order_y_te.size)

            _, pre_class_acc, _ = fit_multiclass_probe(class_Xs_tr, class_y_tr, class_Xs_te, class_y_te,
                                                       C=args.c_reg, seed=BASE_SEED)
            _, pre_order_acc, _ = fit_binary_probe(order_Xs_tr, order_y_tr, order_Xs_te, order_y_te,
                                                   C=args.c_reg, seed=BASE_SEED)

            P, log = run_aggressive_inlp(
                class_Xs_tr, class_y_tr, class_Xs_te, class_y_te,
                max_iters=args.max_iters,
                acc_floor=args.class_acc_floor,
                C=args.c_reg,
                seed=BASE_SEED,
            )
            for it_log in log:
                iter_records.append({
                    "model": model_key, "layer": layer_idx,
                    "iter": it_log["iter"], "class_acc": it_log["acc"],
                    "class_chance": it_log["chance"],
                })

            class_post_te = class_Xs_te @ P.T
            order_post_tr = order_Xs_tr @ P.T
            order_post_te = order_Xs_te @ P.T
            class_post_tr = class_Xs_tr @ P.T
            _, post_class_acc, _ = fit_multiclass_probe(class_post_tr, class_y_tr, class_post_te, class_y_te,
                                                        C=args.c_reg, seed=BASE_SEED)
            _, post_order_acc, _ = fit_binary_probe(order_post_tr, order_y_tr, order_post_te, order_y_te,
                                                    C=args.c_reg, seed=BASE_SEED)

            order_survival = (
                (post_order_acc - order_majority)
                / max(pre_order_acc - order_majority, 1e-6)
            )
            summary_records.append({
                "model": model_key, "layer": layer_idx,
                "pre_class_acc": pre_class_acc, "post_class_acc": post_class_acc,
                "class_chance": class_chance,
                "pre_order_acc": pre_order_acc, "post_order_acc": post_order_acc,
                "order_majority": order_majority,
                "order_survival_ratio": order_survival,
                "n_inlp_iters": len(log),
            })
            print(
                f"  L{layer_idx:>2}: 5-class {pre_class_acc:.3f} → {post_class_acc:.3f}  "
                f"(chance {class_chance:.2f})  "
                f"order {pre_order_acc:.3f} → {post_order_acc:.3f}  "
                f"survival={order_survival:.2f}  iters={len(log)}  "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

    pd.DataFrame(iter_records).to_csv(args.output_dir / "inlp_aggressive_iters.csv", index=False)
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(args.output_dir / "inlp_aggressive_summary.csv", index=False)
    print(f"\nWrote {args.output_dir}/inlp_aggressive_summary.csv "
          f"({len(summary_df)} rows) + iters CSV", flush=True)

    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for layer in sorted(summary_df["layer"].unique()):
            sub = summary_df[summary_df["layer"] == layer].sort_values("model")
            ax.plot(sub["model"], sub["order_survival_ratio"], marker="o", label=f"L{layer}")
        ax.axhline(0.0, color="grey", lw=0.5)
        ax.axhline(1.0, color="grey", lw=0.5, ls="--")
        ax.set_ylabel("Order accuracy survival ratio")
        ax.set_title("Aggressive INLP (multi-class Class probe, max_iters=80):\n"
                     "does Order survive once Class is fully nulled?")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Layer", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir / "inlp_aggressive_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {args.output_dir}/inlp_aggressive_summary.png", flush=True)


if __name__ == "__main__":
    main()
