"""Step 17b — per-Order manifest effective rank only.

Standalone replacement for step17_per_order_tier1.py (which crashed
because step2_tier1_frame_level expects bio/non-bio CSVs that don't
exist on the per-Order manifest). Just computes effective_rank,
participation_ratio, MLE-ID(k=20), and TwoNN per (model, layer) on
frame-level activations from the per-Order manifest. Used to re-run
Q6 (Pearson(null_median, eff_rank)) at apples-to-apples cells
including random-init.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from step2_bootstrap_cis import BASE_SEED, load_layer_tensor
from step6_inlp_class_order import gather_frames_for_model
from step2_tier1_frame_level import (
    per_layer_spectrum,
    effective_rank,
    participation_ratio,
    twonn_intrinsic_dim,
    mle_intrinsic_dim,
)


MANIFEST_ID = "naturelm_by_order_p100_m200_n200_20260427T222756Z"
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_OUTPUT_DIR = Path(
    f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4/per_order_effrank"
)
DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42", "random_init_eat_seed07", "random_init_eat_seed13",
]
DEFAULT_LAYERS = list(range(13))
FRAMES_PER_ITEM = 50
TWONN_SAMPLE_SIZE = 10000
MLE_K = 20
MLE_SAMPLE_SIZE = 10000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for model_key in args.models:
        shard_dir = args.roadmap_dir / model_key / "shards"
        if not shard_dir.exists():
            print(f"WARN: shards missing for {model_key}, skipping", flush=True)
            continue

        l0_tensor, sample_meta = load_layer_tensor(shard_dir, layer_idx=0)
        valid_token_counts = np.array(
            [int(s.get("valid_token_count", l0_tensor.shape[1])) for s in sample_meta]
        )
        del l0_tensor
        print(f"\n=== {model_key} ===", flush=True)

        for layer_idx in args.layers:
            t0 = time.time()
            per_item = gather_frames_for_model(
                shard_dir, layer_idx, valid_token_counts,
                FRAMES_PER_ITEM, BASE_SEED + layer_idx,
            )
            X = per_item.reshape(-1, per_item.shape[-1]).astype(np.float64)
            eigs = per_layer_spectrum(X)
            er = effective_rank(eigs)
            pr = participation_ratio(eigs)
            twonn = twonn_intrinsic_dim(X, sample_size=TWONN_SAMPLE_SIZE)
            mle = mle_intrinsic_dim(X, k=MLE_K, sample_size=MLE_SAMPLE_SIZE)
            rows.append({
                "model": model_key, "layer": layer_idx,
                "n_rows": int(X.shape[0]), "embedding_dim": int(X.shape[1]),
                "effective_rank": float(er), "participation_ratio": float(pr),
                "twonn_id": float(twonn), "mle_id_k20": float(mle),
            })
            print(
                f"  L{layer_idx:>2}: eff_rank {er:7.2f}  PR {pr:7.2f}  "
                f"TwoNN {twonn:5.2f}  MLE {mle:5.2f}  ({time.time() - t0:.1f}s)",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(
                args.output_dir / "per_order_per_layer_stats.csv", index=False
            )

    print(f"\nDone. Wrote {len(rows)} rows.", flush=True)


if __name__ == "__main__":
    main()
