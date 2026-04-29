"""Step 13 — refined mixing-ratio sweep for §4.5.

Reviewer minor concern: §4.5's "asymmetric input/representation map"
claim rests on a single mix ratio (α=0.25 → 78% rep shift). Existing
pilot ran α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}; the 0.0 → 0.25 transition
is the load-bearing nonlinearity. Refine to characterize that
transition: run α ∈ {0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25,
0.50, 0.75, 1.0} on sl_eat_bio_ssl_all L9, with the same n_bio=5 ×
n_nonbio=5 = 25 mixtures-per-α as the pilot.

Output: a separate audio_mixing_refined/ directory next to the pilot's,
preserving original artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Override CLI args to call step3a's main with refined alphas + a new
# output dir, on GPU. This keeps step3a unchanged while producing a
# higher-resolution sweep.
sys.argv = [
    "step3a_audio_mixing_pilot.py",
    "--alphas",
    "0.0", "0.025", "0.05", "0.075", "0.10",
    "0.15", "0.20", "0.25", "0.50", "0.75", "1.0",
    "--device", "cuda",
    "--output_dir",
    "artifacts/comparisons/naturelm_by_source_100each_20260418T171459Z"
    "/nway_eat_all4/audio_mixing_refined",
]

from step3a_audio_mixing_pilot import main

if __name__ == "__main__":
    main()
