"""Q6 — Pearson(empirical null median, effective rank) across the 15 cells
of §4.8's permutation null. Tests the reviewer's parsimonious hypothesis
that high empirical null medians track LOW effective rank (because
low-rank distributions produce highly correlated random-direction
estimates) rather than the authors' "acoustic features survive label
shuffling" framing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BASE = Path("artifacts/comparisons")
NULL_CSV = BASE / "naturelm_by_order_p100_m200_n200_20260427T222756Z/nway_eat_all4/veitch_perm_null/veitch_perm_null_summary.csv"
RANK_CSV = BASE / "naturelm_by_source_100each_20260418T171459Z/nway_eat_all4/step2_tier1_frame_level/frame_per_layer_stats_all4.csv"


def main() -> None:
    nulls = pd.read_csv(NULL_CSV)
    ranks = pd.read_csv(RANK_CSV)
    ranks = ranks.rename(columns={"layer_idx": "layer"})
    merged = nulls.merge(ranks[["model", "layer", "effective_rank"]],
                         on=["model", "layer"], how="left")
    print("Merged rows (cell-level data for Q6):")
    print(merged[["model", "layer", "observed_abs_cos", "null_median",
                  "p_value_lower", "effective_rank"]].to_string(index=False))
    print()

    # All 15 cells
    pr_all, p_all = pearsonr(merged["null_median"], merged["effective_rank"])
    sr_all, sp_all = spearmanr(merged["null_median"], merged["effective_rank"])
    print(f"All 15 cells:  Pearson(null_median, eff_rank) = "
          f"{pr_all:+.3f} (p={p_all:.4f})")
    print(f"               Spearman                          = "
          f"{sr_all:+.3f} (p={sp_all:.4f})")

    # Trained-only (drop random-init)
    trained = merged[~merged["model"].str.startswith("random_init")]
    pr_t, p_t = pearsonr(trained["null_median"], trained["effective_rank"])
    sr_t, sp_t = spearmanr(trained["null_median"], trained["effective_rank"])
    print(f"\nTrained only (12 cells):  "
          f"Pearson = {pr_t:+.3f} (p={p_t:.4f})  "
          f"Spearman = {sr_t:+.3f} (p={sp_t:.4f})")

    # Random-init only (3 cells)
    rand = merged[merged["model"].str.startswith("random_init")]
    print(f"\nRandom-init alone (3 cells):  "
          f"null_median range {rand['null_median'].min():.3f}–{rand['null_median'].max():.3f}, "
          f"eff_rank range {rand['effective_rank'].min():.1f}–{rand['effective_rank'].max():.1f}")

    # Within trained, by model
    print("\nPer-model trained Pearson (4 layers each):")
    for m in sorted(trained["model"].unique()):
        sub = trained[trained["model"] == m]
        if len(sub) >= 3:
            pr, p = pearsonr(sub["null_median"], sub["effective_rank"])
            print(f"  {m:<28s}  r={pr:+.3f}  (n={len(sub)})  null_med "
                  f"{sub['null_median'].min():.3f}–{sub['null_median'].max():.3f}  "
                  f"eff_rank {sub['effective_rank'].min():.1f}–{sub['effective_rank'].max():.1f}")

    out = Path("artifacts/comparisons/naturelm_by_order_p100_m200_n200_20260427T222756Z/nway_eat_all4/q6_null_vs_effrank.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    summary = pd.DataFrame([
        {"scope": "all", "n": len(merged), "pearson_r": pr_all, "pearson_p": p_all,
         "spearman_r": sr_all, "spearman_p": sp_all},
        {"scope": "trained_only", "n": len(trained), "pearson_r": pr_t, "pearson_p": p_t,
         "spearman_r": sr_t, "spearman_p": sp_t},
    ])
    summary.to_csv(out.with_name("q6_null_vs_effrank_summary.csv"), index=False)
    print(f"\nWrote {out}")
    print(f"Wrote {out.with_name('q6_null_vs_effrank_summary.csv')}")


if __name__ == "__main__":
    main()
