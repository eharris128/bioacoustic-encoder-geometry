"""Step 5 follow-up — MLE-ID sensitivity sweep over (n, k).

§6 reports MLE-ID(k=20, n=10000) values 7-14 for trained models and
11-15 for random_init. This is a single (n, k) cell — robustness
unclear. This script sweeps:
  - k ∈ {5, 10, 20, 40, 80}
  - n ∈ {2500, 5000, 10000, 20000}
per (model, layer) using the existing 30000-frame frame matrix
(50 frames × 600 items).

Tests whether the §6 claims survive estimator-hyperparameter variation.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/mle_id_sensitivity/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from step2_bootstrap_cis import BASE_SEED, load_layer_tensor
from step2_tier1_frame_level import mle_intrinsic_dim


MANIFEST_ID = "naturelm_by_order_p100_m200_n200_20260427T222756Z"
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "mle_id_sensitivity"

DEFAULT_MODELS = [
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]
DEFAULT_LAYERS = [0, 4, 8, 9, 10, 11, 12]
DEFAULT_KS = [5, 10, 20, 40, 80]
DEFAULT_NS = [2500, 5000, 10000, 20000]
FRAMES_PER_ITEM = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    p.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS)
    p.add_argument("--ns", nargs="+", type=int, default=DEFAULT_NS)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    return p.parse_args()


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    for model_idx, model_key in enumerate(args.models):
        shard_dir = args.roadmap_dir / model_key / "shards"
        if not shard_dir.exists():
            print(f"WARN: shards missing for {model_key}, skipping", flush=True)
            continue
        print(f"\n=== {model_key} ({model_idx + 1}/{len(args.models)}) ===",
              flush=True)

        layer0_tensor, sample_meta = load_layer_tensor(shard_dir, layer_idx=0)
        valid_token_counts = np.array(
            [int(s.get("valid_token_count", layer0_tensor.shape[1])) for s in sample_meta]
        )
        del layer0_tensor

        for layer_idx in args.layers:
            t0 = time.time()
            layer_tensor, _ = load_layer_tensor(shard_dir, layer_idx)
            rng = np.random.default_rng(BASE_SEED + layer_idx)
            per_item = per_clip_frame_sample(
                layer_tensor, valid_token_counts, args.frames_per_item, rng,
            )
            frames = per_item.reshape(-1, per_item.shape[-1]).astype(np.float64)
            del layer_tensor, per_item

            for n in args.ns:
                if frames.shape[0] < n:
                    continue
                for k in args.ks:
                    mle = mle_intrinsic_dim(frames, k=k, sample_size=n)
                    records.append({
                        "model": model_key,
                        "layer_idx": layer_idx,
                        "k": k, "n": n,
                        "mle_id": float(mle),
                    })

            print(f"  L{layer_idx:02d}  swept ({time.time()-t0:.1f}s)", flush=True)
            del frames

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.output_dir / "mle_id_sensitivity.csv", index=False)

    # ----- Summary tables -----
    print("\n=== MLE-ID sensitivity per (model, layer, k, n) ===")
    for layer_idx in args.layers:
        sub = df[df["layer_idx"] == layer_idx]
        if sub.empty:
            continue
        print(f"\n--- L{layer_idx} ---")
        # pivot k vs model at n=10000 (the §6 setting)
        n_anchor = 10000 if 10000 in args.ns else max(args.ns)
        pivot = sub[sub["n"] == n_anchor].pivot(
            index="k", columns="model", values="mle_id")
        cols = [m for m in args.models if m in pivot.columns]
        print(f"n={n_anchor}, varying k:")
        print(pivot[cols].round(2).to_string())

    print(f"\nSaved MLE-ID sensitivity artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
