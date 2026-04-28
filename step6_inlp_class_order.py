"""Step 6 — INLP probe of Class⊥Order factoring.

Reviewer concern (6): the §4.8 Veitch test as written reports
|cos(parent, subord)| ≈ 0.033, which is at the random-orthogonality
floor in 768-d (≈ √(2/(πd)) ≈ 0.029). It cannot distinguish "factored
hierarchy" from "two centroid-difference directions that are
near-orthogonal by construction."

This script produces *representational* evidence by Iterative Nullspace
Projection (Ravfogel et al. 2020). For each (model, layer) we:

  1. Train a linear Class probe (Aves vs Mammalia) on frame activations.
  2. Iteratively null its row-space until Class becomes unrecoverable.
  3. On the Class-nullspace activations, train an Order probe
     (Passeriformes vs other-Aves) restricted to the within-Aves frames.
  4. Compare pre-INLP and post-INLP Order accuracy.

Interpretation:
  - If Order accuracy drops with Class accuracy, the model encodes
    Order *along* the Class direction (entangled / not factored).
  - If Order accuracy survives Class nullification, the model encodes
    Class and Order in linearly independent subspaces (factored).

The factored-hierarchy claim of §4.8 predicts the second pattern in
sl_eat_bio_ssl_all and not in the other trained models.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/inlp_class_order/
  inlp_results.csv     # per (model, layer, iteration, probe_target)
  inlp_summary.csv     # per (model, layer): pre/post Class + pre/post Order
  inlp_summary.png     # bar plot of Order-survival ratio per model

Usage:
    python step6_inlp_class_order.py
    python step6_inlp_class_order.py --layers 5 7 9 12
    python step6_inlp_class_order.py --max_iters 20 --class_acc_floor 0.55
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from step2_bootstrap_cis import BASE_SEED, load_layer_tensor
from step3c_veitch_4order import per_clip_frame_sample, load_taxonomy


MANIFEST_ID = "naturelm_by_order_p100_m200_n200_20260427T222756Z"
DEFAULT_TAX_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}_taxonomic.jsonl")
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "inlp_class_order"

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
    p.add_argument("--max_iters", type=int, default=15,
                   help="max INLP iterations before giving up")
    p.add_argument("--class_acc_floor", type=float, default=0.55,
                   help="stop INLP when Class probe acc drops below this")
    p.add_argument("--c_reg", type=float, default=1.0,
                   help="LogisticRegression C (smaller = stronger L2)")
    return p.parse_args()


def fit_probe(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    C: float, seed: int,
) -> tuple[LogisticRegression, float, np.ndarray]:
    """Fit a linear probe on a pre-split (X_tr, X_te); return (model, test-acc, coef row)."""
    clf = LogisticRegression(
        penalty="l2", C=C, solver="liblinear", max_iter=2000, random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))
    return clf, acc, clf.coef_.copy()


def majority_baseline(y: np.ndarray) -> float:
    return float(np.bincount(y).max() / y.size)


def nullspace_projection(W: np.ndarray) -> np.ndarray:
    """Project orthogonal to the row-space of W (shape (k, d)) → (d, d)."""
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    rank = int((S > 1e-8).sum())
    row_basis = Vt[:rank]
    return np.eye(W.shape[1]) - row_basis.T @ row_basis


def stratified_clip_split(
    clip_y: np.ndarray, test_size: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/test split at the *clip* level.

    Returns (train_clip_positions, test_clip_positions) — positions are
    indices into the clip_y array (i.e. positions within the masked
    subset, not original-manifest indices).
    """
    rng = np.random.default_rng(seed)
    train_pos: list[int] = []
    test_pos: list[int] = []
    for label in np.unique(clip_y):
        idx = np.where(clip_y == label)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(idx.size * test_size)))
        test_pos.extend(idx[:n_test].tolist())
        train_pos.extend(idx[n_test:].tolist())
    return np.array(sorted(train_pos)), np.array(sorted(test_pos))


def expand_clip_positions_to_frames(
    clip_positions: np.ndarray, frames_per_item: int,
) -> np.ndarray:
    """Expand a (n_clip,) position array into a (n_clip * fpi,) frame index
    array, where each clip i contributes rows [i*fpi : (i+1)*fpi]."""
    return np.concatenate(
        [np.arange(p * frames_per_item, (p + 1) * frames_per_item)
         for p in clip_positions]
    )


def run_inlp_for_target(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    *,
    max_iters: int,
    acc_floor: float,
    C: float,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """Iteratively null the linear-probe direction on the (pre-split) train
    set, evaluate test accuracy each iteration, until test acc reaches
    acc_floor or we hit max_iters. Train/test split is fixed across iters.

    Returns (final_projection (d, d), per-iteration log).
    """
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


def gather_frames_for_model(
    shard_dir: Path,
    layer_idx: int,
    valid_token_counts: np.ndarray,
    frames_per_item: int,
    seed: int,
) -> np.ndarray:
    """Frame-level subsample for one (model, layer); shape (n_items, fpi, d)."""
    layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
    rng = np.random.default_rng(seed)
    per_item = per_clip_frame_sample(
        layer_tensor, valid_token_counts, frames_per_item, rng,
    )
    del layer_tensor
    return per_item


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy(args.tax_manifest)
    print(f"Taxonomic manifest: {len(taxonomy)} records", flush=True)

    iter_records: list[dict] = []   # per-iteration probe accuracy
    summary_records: list[dict] = []  # per (model, layer) headline

    for model_key in args.models:
        shard_dir = args.roadmap_dir / model_key / "shards"
        if not shard_dir.exists():
            print(f"WARN: shards missing for {model_key}, skipping", flush=True)
            continue
        print(f"\n=== {model_key} ===", flush=True)

        # Use L0 to read sample metadata (cheap, every model has it).
        l0_tensor, sample_meta = load_layer_tensor(shard_dir, layer_idx=0)
        valid_token_counts = np.array(
            [int(s.get("valid_token_count", l0_tensor.shape[1])) for s in sample_meta]
        )
        del l0_tensor

        cls = np.array([taxonomy.get(s.get("id", ""), {}).get("class", "") for s in sample_meta])
        ord_ = np.array([taxonomy.get(s.get("id", ""), {}).get("order", "") for s in sample_meta])

        mask_aves = cls == "Aves"
        mask_mammalia = cls == "Mammalia"
        mask_passer = mask_aves & (ord_ == PASSERIFORMES)
        mask_other_aves = mask_aves & np.isin(ord_, OTHER_AVES_ORDERS)

        n_aves, n_mam = int(mask_aves.sum()), int(mask_mammalia.sum())
        n_passer, n_other = int(mask_passer.sum()), int(mask_other_aves.sum())
        print(f"  Aves={n_aves}, Mammalia={n_mam}, Passer={n_passer}, "
              f"other-Aves={n_other}", flush=True)
        if min(n_aves, n_mam, n_passer, n_other) < 20:
            print("  WARN: too few samples for one of the probe targets", flush=True)
            continue

        for layer_idx in args.layers:
            t0 = time.time()
            per_item = gather_frames_for_model(
                shard_dir, layer_idx, valid_token_counts,
                args.frames_per_item, BASE_SEED + layer_idx,
            )
            d = per_item.shape[-1]

            # ---- Class probe ----
            # Build per-clip arrays (one row per clip), then split at the
            # clip level so train and test never share clips. Frame
            # leakage was an issue in the v1 of this script.
            mask_class_clips = mask_aves | mask_mammalia
            class_clip_y = (cls[mask_class_clips] == "Aves").astype(np.int64)
            class_clip_frames = per_item[mask_class_clips]  # (n_clip, fpi, d)
            class_X = class_clip_frames.reshape(-1, d).astype(np.float64)
            class_y = np.repeat(class_clip_y, args.frames_per_item)

            class_train_pos, class_test_pos = stratified_clip_split(
                class_clip_y, test_size=0.2, seed=BASE_SEED,
            )
            class_train_frame_idx = expand_clip_positions_to_frames(
                class_train_pos, args.frames_per_item,
            )
            class_test_frame_idx = expand_clip_positions_to_frames(
                class_test_pos, args.frames_per_item,
            )

            # ---- Order probe ----
            mask_order_clips = mask_passer | mask_other_aves
            order_clip_y = (ord_[mask_order_clips] == PASSERIFORMES).astype(np.int64)
            order_clip_frames = per_item[mask_order_clips]
            order_X = order_clip_frames.reshape(-1, d).astype(np.float64)
            order_y = np.repeat(order_clip_y, args.frames_per_item)

            order_train_pos, order_test_pos = stratified_clip_split(
                order_clip_y, test_size=0.2, seed=BASE_SEED,
            )
            order_train_frame_idx = expand_clip_positions_to_frames(
                order_train_pos, args.frames_per_item,
            )
            order_test_frame_idx = expand_clip_positions_to_frames(
                order_test_pos, args.frames_per_item,
            )

            # ---- Standardize using only the train portion of the class probe
            # to avoid any test-set statistics leaking into the scaler. ----
            scaler = StandardScaler().fit(class_X[class_train_frame_idx])
            class_Xs = scaler.transform(class_X)
            order_Xs = scaler.transform(order_X)

            class_Xs_tr = class_Xs[class_train_frame_idx]
            class_Xs_te = class_Xs[class_test_frame_idx]
            class_y_tr = class_y[class_train_frame_idx]
            class_y_te = class_y[class_test_frame_idx]

            order_Xs_tr = order_Xs[order_train_frame_idx]
            order_Xs_te = order_Xs[order_test_frame_idx]
            order_y_tr = order_y[order_train_frame_idx]
            order_y_te = order_y[order_test_frame_idx]

            class_baseline = majority_baseline(class_y_te)
            order_baseline = majority_baseline(order_y_te)

            # Pre-INLP probes (clip-level test).
            _, pre_class_acc, _ = fit_probe(
                class_Xs_tr, class_y_tr, class_Xs_te, class_y_te,
                C=args.c_reg, seed=BASE_SEED,
            )
            _, pre_order_acc, _ = fit_probe(
                order_Xs_tr, order_y_tr, order_Xs_te, order_y_te,
                C=args.c_reg, seed=BASE_SEED,
            )

            # INLP on the Class probe (fixed train/test split across iters).
            P, log = run_inlp_for_target(
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
                    "class_majority_baseline": it_log["majority_baseline"],
                })

            # Post-INLP probes: apply P to both Class (sanity) and Order.
            class_Xs_post_tr = class_Xs_tr @ P.T
            class_Xs_post_te = class_Xs_te @ P.T
            order_Xs_post_tr = order_Xs_tr @ P.T
            order_Xs_post_te = order_Xs_te @ P.T
            _, post_class_acc, _ = fit_probe(
                class_Xs_post_tr, class_y_tr, class_Xs_post_te, class_y_te,
                C=args.c_reg, seed=BASE_SEED,
            )
            _, post_order_acc, _ = fit_probe(
                order_Xs_post_tr, order_y_tr, order_Xs_post_te, order_y_te,
                C=args.c_reg, seed=BASE_SEED,
            )

            order_survival = (
                (post_order_acc - order_baseline)
                / max(pre_order_acc - order_baseline, 1e-6)
            )

            summary_records.append({
                "model": model_key, "layer": layer_idx,
                "pre_class_acc": pre_class_acc,
                "post_class_acc": post_class_acc,
                "class_majority_baseline": class_baseline,
                "pre_order_acc": pre_order_acc,
                "post_order_acc": post_order_acc,
                "order_majority_baseline": order_baseline,
                "order_survival_ratio": order_survival,
                "n_inlp_iters": len(log),
                "n_class_frames": class_Xs.shape[0],
                "n_order_frames": order_Xs.shape[0],
            })
            print(
                f"  L{layer_idx:>2}: class {pre_class_acc:.3f} → {post_class_acc:.3f}  "
                f"order {pre_order_acc:.3f} → {post_order_acc:.3f}  "
                f"survival={order_survival:.2f}  iters={len(log)}  "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

    iter_df = pd.DataFrame(iter_records)
    summary_df = pd.DataFrame(summary_records)
    iter_df.to_csv(args.output_dir / "inlp_results.csv", index=False)
    summary_df.to_csv(args.output_dir / "inlp_summary.csv", index=False)
    print(f"\nWrote {args.output_dir}/inlp_results.csv "
          f"({len(iter_df)} rows) and inlp_summary.csv ({len(summary_df)} rows)", flush=True)

    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for layer in sorted(summary_df["layer"].unique()):
            sub = summary_df[summary_df["layer"] == layer].sort_values("model")
            ax.plot(sub["model"], sub["order_survival_ratio"],
                    marker="o", label=f"L{layer}")
        ax.axhline(0.0, color="grey", lw=0.5)
        ax.axhline(1.0, color="grey", lw=0.5, ls="--")
        ax.set_ylabel("Order accuracy survival ratio\n"
                      "(post-INLP − baseline) / (pre-INLP − baseline)")
        ax.set_title("INLP: does Order survive Class-nullification?\n"
                     "1.0 = factored; 0.0 = entangled with Class")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Layer", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir / "inlp_summary.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {args.output_dir}/inlp_summary.png", flush=True)


if __name__ == "__main__":
    main()
