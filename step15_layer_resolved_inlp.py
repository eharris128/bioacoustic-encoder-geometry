"""Step 15 — layer-resolved Class-first INLP iteration sweep on
sl_eat_bio_ssl_all (the headline model), all 13 layers.

Reviewer concern 3.6 extension: step11's iteration sweep ran on
L5/L7/L9/L12 of the headline model + L7/L9 of the other three trained
models (10 cells). This step extends the headline model to all 13
layers — does the asymmetric-coupling signature live everywhere or
only at L7-L12?

Output: round_b_layers/ next to the main round_b/ directory.
"""

from __future__ import annotations

import sys

sys.argv = [
    "step11_round_b.py",
    "--models", "sl_eat_bio_ssl_all",
    "--layers", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
    "--mlp_models", "sl_eat_bio_ssl_all",
    "--output_dir",
    "artifacts/comparisons/naturelm_by_order_p100_m200_n200_20260427T222756Z"
    "/nway_eat_all4/round_b_layers",
]

from step11_round_b import main

if __name__ == "__main__":
    main()
