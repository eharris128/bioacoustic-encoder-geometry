"""Step 17 — eff_rank / MLE-ID / participation-ratio on the per-Order
manifest, mirroring step2_tier1_frame_level on the source manifest.

Closes a small gap in Q6 (Pearson(null_median, eff_rank) was strong on
trained-cell source-manifest eff_rank but missed random-init's
eff_rank). Re-running tier1 on the per-Order manifest gives an
apples-to-apples eff_rank for the same (model, layer) cells §4.8 was
computed on, including random-init.

Output: step2_tier1_frame_level/ under the per-Order manifest's
nway_eat_all4 directory.
"""

from __future__ import annotations

import sys

sys.argv = [
    "step2_tier1_frame_level.py",
    "--roadmap_dir",
    "artifacts/roadmap_part1/naturelm_by_order_p100_m200_n200_20260427T222756Z",
    "--nway_dir",
    "artifacts/comparisons/naturelm_by_order_p100_m200_n200_20260427T222756Z/nway_eat_all4",
    "--output_dir",
    "artifacts/comparisons/naturelm_by_order_p100_m200_n200_20260427T222756Z"
    "/nway_eat_all4/step2_tier1_frame_level",
    "--models",
    "eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all",
    "random_init_eat_seed42",
]

from step2_tier1_frame_level import main

if __name__ == "__main__":
    main()
