"""Step 11 — Round B consolidated: binary Class re-test, MLP Order probe,
INLP iteration sweep, all in one INLP pass per cell.

Addresses red-team review concerns:
  3.4 — MLP probe for Order on Class-nulled activations (sl_eat_bio cells).
  3.5 — Binary Aves-vs-Mammalia probe re-trained on Class-nulled
        activations. Reviewer: "the cheapest fix in the paper."
  3.6 — Class-first INLP iteration sweep at {10, 20, 40, 80}, extended
        to L7/L9 of all four trained models (= 10 cells minimum).

Implementation: one INLP run per (model, layer) at max_iters=80 with
intermediate P checkpoints at iter counts {10, 20, 40, 80}. At each
checkpoint, evaluate four probes:
  - 5-class Class probe (multiclass, sanity check on nullification depth)
  - Binary Aves-vs-Mammalia probe (3.5)
  - Order linear probe (3.6, the iteration-curve datapoint)
  - Order MLP probe — only at sl_eat_bio_ssl_all cells (3.4)

Output: artifacts/comparisons/<manifest>/nway_eat_all4/round_b/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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
    CLASS_LABELS_MULTICLASS,
)


MANIFEST_ID = "naturelm_by_order_p100_m200_n200_20260427T222756Z"
DEFAULT_TAX_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}_taxonomic.jsonl")
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "round_b"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
DEFAULT_LAYERS = [5, 7, 9, 12]
DEFAULT_CHECKPOINTS = (10, 20, 40, 80)
FRAMES_PER_ITEM = 50
MLP_MODELS = ("sl_eat_bio_ssl_all",)


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
    p.add_argument("--mlp_models", nargs="+", default=list(MLP_MODELS))
    p.add_argument("--checkpoints", nargs="+", type=int,
                   default=list(DEFAULT_CHECKPOINTS))
    return p.parse_args()


def fit_mlp_probe(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    seed: int, hidden: int = 256,
) -> float:
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
    )
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def run_inlp_with_checkpoints(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    *,
    max_iters: int,
    acc_floor: float,
    C: float,
    seed: int,
    checkpoints: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], list[dict]]:
    d = X_tr.shape[1]
    P = np.eye(d, dtype=np.float64)
    log: list[dict] = []
    P_at_iter: dict[int, np.ndarray] = {}
    n_nulls_applied = 0
    counts = np.bincount(y_tr)
    chance = 1.0 / counts.size
    if 0 in checkpoints:
        P_at_iter[0] = P.copy()
    for it in range(max_iters):
        Xtr_cur = X_tr @ P.T
        Xte_cur = X_te @ P.T
        _, acc, coef = fit_multiclass_probe(
            Xtr_cur, y_tr, Xte_cur, y_te, C=C, seed=seed,
        )
        log.append({"iter": it, "acc": acc, "chance": chance})
        if acc <= acc_floor + 0.01:
            break
        P = nullspace_projection(coef) @ P
        n_nulls_applied += 1
        if n_nulls_applied in checkpoints:
            P_at_iter[n_nulls_applied] = P.copy()
    if n_nulls_applied not in P_at_iter:
        P_at_iter[n_nulls_applied] = P.copy()
    return P_at_iter, log


def setup_cell(
    shard_dir: Path, layer_idx: int, taxonomy: dict, sample_meta: list,
    valid_token_counts: np.ndarray, frames_per_item: int,
) -> dict:
    cls = np.array([taxonomy.get(s.get("id", ""), {}).get("class", "") for s in sample_meta])
    ord_ = np.array([taxonomy.get(s.get("id", ""), {}).get("order", "") for s in sample_meta])

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

    per_item = gather_frames_for_model(
        shard_dir, layer_idx, valid_token_counts,
        frames_per_item, BASE_SEED + layer_idx,
    )
    d = per_item.shape[-1]

    # --- 5-class Class data ---
    class_clip_y_str = multiclass_label[mask_class_clips]
    label_to_int = {l: i for i, l in enumerate(CLASS_LABELS_MULTICLASS)}
    class_clip_y = np.array([label_to_int[l] for l in class_clip_y_str], dtype=np.int64)
    class_X = per_item[mask_class_clips].reshape(-1, d).astype(np.float64)
    class_y = np.repeat(class_clip_y, frames_per_item)
    class_train_pos, class_test_pos = stratified_clip_split(
        class_clip_y, test_size=0.2, seed=BASE_SEED,
    )
    class_train_idx = expand_clip_positions_to_frames(class_train_pos, frames_per_item)
    class_test_idx = expand_clip_positions_to_frames(class_test_pos, frames_per_item)

    # --- Binary Aves-vs-Mammalia: same clips and split as 5-class, different y ---
    binary_class_y_per_clip = np.array(
        [0 if l == "Mammalia" else 1 for l in class_clip_y_str], dtype=np.int64
    )
    binary_class_y = np.repeat(binary_class_y_per_clip, frames_per_item)

    # --- Order data (Passer vs other-Aves) ---
    order_clip_y = (ord_[mask_order_clips] == PASSERIFORMES).astype(np.int64)
    order_X = per_item[mask_order_clips].reshape(-1, d).astype(np.float64)
    order_y = np.repeat(order_clip_y, frames_per_item)
    order_train_pos, order_test_pos = stratified_clip_split(
        order_clip_y, test_size=0.2, seed=BASE_SEED,
    )
    order_train_idx = expand_clip_positions_to_frames(order_train_pos, frames_per_item)
    order_test_idx = expand_clip_positions_to_frames(order_test_pos, frames_per_item)

    # --- Standardize on Class train frames; same scaler for everything ---
    scaler = StandardScaler().fit(class_X[class_train_idx])
    class_Xs = scaler.transform(class_X)
    order_Xs = scaler.transform(order_X)

    return {
        "class_Xs_tr": class_Xs[class_train_idx],
        "class_Xs_te": class_Xs[class_test_idx],
        "class_y_tr": class_y[class_train_idx],
        "class_y_te": class_y[class_test_idx],
        "bin_class_Xs_tr": class_Xs[class_train_idx],
        "bin_class_Xs_te": class_Xs[class_test_idx],
        "bin_class_y_tr": binary_class_y[class_train_idx],
        "bin_class_y_te": binary_class_y[class_test_idx],
        "order_Xs_tr": order_Xs[order_train_idx],
        "order_Xs_te": order_Xs[order_test_idx],
        "order_y_tr": order_y[order_train_idx],
        "order_y_te": order_y[order_test_idx],
        "n_class_clips": int(mask_class_clips.sum()),
        "n_passer": int(mask_passer.sum()),
        "n_other_aves": int(mask_other_aves.sum()),
        "n_mammalia": int((cls == "Mammalia").sum()),
    }


def evaluate_at_P(
    cell: dict, P: np.ndarray, *, C: float, seed: int,
    run_mlp: bool,
) -> dict:
    class_Xs_tr_p = cell["class_Xs_tr"] @ P.T
    class_Xs_te_p = cell["class_Xs_te"] @ P.T
    bin_Xs_tr_p = cell["bin_class_Xs_tr"] @ P.T
    bin_Xs_te_p = cell["bin_class_Xs_te"] @ P.T
    order_Xs_tr_p = cell["order_Xs_tr"] @ P.T
    order_Xs_te_p = cell["order_Xs_te"] @ P.T

    _, mc_acc, _ = fit_multiclass_probe(
        class_Xs_tr_p, cell["class_y_tr"], class_Xs_te_p, cell["class_y_te"],
        C=C, seed=seed,
    )
    _, bin_acc, _ = fit_binary_probe(
        bin_Xs_tr_p, cell["bin_class_y_tr"], bin_Xs_te_p, cell["bin_class_y_te"],
        C=C, seed=seed,
    )
    _, order_lin_acc, _ = fit_binary_probe(
        order_Xs_tr_p, cell["order_y_tr"], order_Xs_te_p, cell["order_y_te"],
        C=C, seed=seed,
    )
    out = {
        "multiclass_class_acc": mc_acc,
        "binary_class_acc": bin_acc,
        "order_linear_acc": order_lin_acc,
    }
    if run_mlp:
        order_mlp = fit_mlp_probe(
            order_Xs_tr_p, cell["order_y_tr"], order_Xs_te_p, cell["order_y_te"],
            seed=seed,
        )
        out["order_mlp_acc"] = order_mlp
    else:
        out["order_mlp_acc"] = np.nan
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(args.tax_manifest)
    print(f"Step 11 — Round B consolidated. {len(taxonomy)} taxonomic records | "
          f"max_iters={args.max_iters}, acc_floor={args.class_acc_floor}, "
          f"checkpoints={args.checkpoints}", flush=True)

    rows: list[dict] = []
    inlp_iter_log: list[dict] = []

    for model_key in args.models:
        shard_dir = args.roadmap_dir / model_key / "shards"
        if not shard_dir.exists():
            print(f"WARN: shards missing for {model_key}, skipping", flush=True)
            continue
        run_mlp_for_model = model_key in args.mlp_models
        print(f"\n=== {model_key} (MLP={'yes' if run_mlp_for_model else 'no'}) ===",
              flush=True)

        l0_tensor, sample_meta = load_layer_tensor(shard_dir, layer_idx=0)
        valid_token_counts = np.array(
            [int(s.get("valid_token_count", l0_tensor.shape[1])) for s in sample_meta]
        )
        del l0_tensor

        for layer_idx in args.layers:
            t0 = time.time()
            try:
                cell = setup_cell(
                    shard_dir, layer_idx, taxonomy, sample_meta,
                    valid_token_counts, args.frames_per_item,
                )
            except Exception as e:
                print(f"  L{layer_idx}: setup failed: {e}", flush=True)
                continue

            # Pre-INLP probes (P=I)
            pre = evaluate_at_P(
                cell, np.eye(cell["class_Xs_tr"].shape[1]),
                C=args.c_reg, seed=BASE_SEED, run_mlp=run_mlp_for_model,
            )
            order_majority = float(
                np.bincount(cell["order_y_te"]).max() / cell["order_y_te"].size
            )
            bin_majority = float(
                np.bincount(cell["bin_class_y_te"]).max() / cell["bin_class_y_te"].size
            )

            P_at_iter, log = run_inlp_with_checkpoints(
                cell["class_Xs_tr"], cell["class_y_tr"],
                cell["class_Xs_te"], cell["class_y_te"],
                max_iters=args.max_iters,
                acc_floor=args.class_acc_floor,
                C=args.c_reg,
                seed=BASE_SEED,
                checkpoints=tuple(args.checkpoints),
            )
            for it_log in log:
                inlp_iter_log.append({
                    "model": model_key, "layer": layer_idx,
                    "iter": it_log["iter"],
                    "class_5way_acc": it_log["acc"],
                    "class_chance": it_log["chance"],
                })

            for n_nulls in sorted(P_at_iter.keys()):
                P = P_at_iter[n_nulls]
                ev = evaluate_at_P(
                    cell, P, C=args.c_reg, seed=BASE_SEED,
                    run_mlp=run_mlp_for_model,
                )
                rows.append({
                    "model": model_key, "layer": layer_idx,
                    "n_nulls_applied": n_nulls,
                    "pre_multiclass_class_acc": pre["multiclass_class_acc"],
                    "pre_binary_class_acc": pre["binary_class_acc"],
                    "pre_order_linear_acc": pre["order_linear_acc"],
                    "pre_order_mlp_acc": pre["order_mlp_acc"],
                    "post_multiclass_class_acc": ev["multiclass_class_acc"],
                    "post_binary_class_acc": ev["binary_class_acc"],
                    "post_order_linear_acc": ev["order_linear_acc"],
                    "post_order_mlp_acc": ev["order_mlp_acc"],
                    "binary_class_majority": bin_majority,
                    "order_majority": order_majority,
                    "n_class_clips": cell["n_class_clips"],
                    "n_passer": cell["n_passer"],
                    "n_other_aves": cell["n_other_aves"],
                    "n_mammalia": cell["n_mammalia"],
                })

            elapsed = time.time() - t0
            final_n = max(P_at_iter.keys())
            final_ev = rows[-1]
            print(
                f"  L{layer_idx:>2}: pre 5cls {pre['multiclass_class_acc']:.3f} "
                f"bin {pre['binary_class_acc']:.3f} ord-lin {pre['order_linear_acc']:.3f}"
                + (f" ord-mlp {pre['order_mlp_acc']:.3f}" if run_mlp_for_model else "")
                + f"  →  @{final_n}-nulls "
                f"5cls {final_ev['post_multiclass_class_acc']:.3f} "
                f"bin {final_ev['post_binary_class_acc']:.3f} "
                f"ord-lin {final_ev['post_order_linear_acc']:.3f}"
                + (f" ord-mlp {final_ev['post_order_mlp_acc']:.3f}"
                   if run_mlp_for_model else "")
                + f"  ({elapsed:.1f}s)",
                flush=True,
            )

            # Periodic save in case of interrupt
            pd.DataFrame(rows).to_csv(
                args.output_dir / "round_b_summary.csv", index=False
            )
            pd.DataFrame(inlp_iter_log).to_csv(
                args.output_dir / "round_b_inlp_iters.csv", index=False
            )

    print(f"\nDone. Wrote {len(rows)} rows to {args.output_dir}/round_b_summary.csv",
          flush=True)


if __name__ == "__main__":
    main()
