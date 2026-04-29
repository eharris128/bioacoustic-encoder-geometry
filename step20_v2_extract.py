"""Step 20 — auto-extract preprint v2 numbers from Round B + step8_seed7
+ step8_seed13 + step14 CSVs.

Runs after the chain to populate publication_path/round_b_findings_v2.md
with concrete numbers replacing the X.XX placeholders. Idempotent — if
some CSVs are missing, fills in only the available sections.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
ARTS = ROOT / "artifacts/comparisons/naturelm_by_order_p100_m200_n200_20260427T222756Z/nway_eat_all4"

ROUND_B = ARTS / "round_b/round_b_summary.csv"
ROUND_B_LAYERS = ARTS / "round_b_layers/round_b_summary.csv"
ROUND_B_SEEDS = ARTS / "round_b_random_seeds/round_b_summary.csv"
INLP_SEED7 = ARTS / "inlp_aggressive_seed7/inlp_aggressive_summary.csv"
INLP_SEED13 = ARTS / "inlp_aggressive_seed13/inlp_aggressive_summary.csv"
MULTICLASS_ORDER = ARTS / "inlp_order_aggressive/inlp_order_aggressive_summary.csv"
PER_ORDER_RANK = ARTS / "per_order_effrank/per_order_per_layer_stats.csv"
NULL_CSV = ARTS / "veitch_perm_null/veitch_perm_null_summary.csv"


def safe(df_path: Path):
    if df_path.exists():
        return pd.read_csv(df_path)
    return None


def fmt(x: float, sig: str = ".3f") -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:{sig}}"


def section_round_b_summary(rb: pd.DataFrame) -> list[str]:
    if rb is None:
        return ["### Round B core: NOT YET LANDED"]
    out = ["### Round B core (n_nulls = 80, headline cells)"]
    final = rb[rb["n_nulls_applied"] == rb.groupby(["model", "layer"])["n_nulls_applied"].transform("max")]
    out.append("")
    out.append("| Model | Layer | pre 5cls | post 5cls | pre bin | post bin | pre ord-lin | post ord-lin | pre ord-mlp | post ord-mlp |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in final.iterrows():
        out.append(
            f"| {r['model']} | L{int(r['layer'])} | "
            f"{fmt(r['pre_multiclass_class_acc'])} | {fmt(r['post_multiclass_class_acc'])} | "
            f"{fmt(r['pre_binary_class_acc'])} | {fmt(r['post_binary_class_acc'])} | "
            f"{fmt(r['pre_order_linear_acc'])} | {fmt(r['post_order_linear_acc'])} | "
            f"{fmt(r['pre_order_mlp_acc'])} | {fmt(r['post_order_mlp_acc'])} |"
        )

    out.append("")
    out.append("**3.5 (binary Class re-test):** post values across all 16 trained-model cells")
    trained = final[~final["model"].str.startswith("random_init")]
    if len(trained):
        post_bin = trained["post_binary_class_acc"]
        out.append(f"  median: {post_bin.median():.3f}, range: {post_bin.min():.3f}–{post_bin.max():.3f}")
        out.append(f"  baseline: {final['binary_class_majority'].iloc[0]:.3f}")
        out.append(f"  cells at exactly baseline (within ±0.005): "
                   f"{int((np.abs(post_bin - final['binary_class_majority'].iloc[0]) < 0.005).sum())}/{len(trained)}")

    out.append("")
    out.append("**3.4 (MLP vs linear Order on Class-nulled):**")
    sl = final[final["model"] == "sl_eat_bio_ssl_all"]
    if len(sl) and "post_order_mlp_acc" in sl and not sl["post_order_mlp_acc"].isna().all():
        for _, r in sl.iterrows():
            mlp_gap = r["post_order_mlp_acc"] - r["post_order_linear_acc"]
            out.append(
                f"  L{int(r['layer'])}: linear post={fmt(r['post_order_linear_acc'])}  "
                f"MLP post={fmt(r['post_order_mlp_acc'])}  gap={mlp_gap:+.3f}"
            )

    return out


def section_iter_sweep(rb: pd.DataFrame) -> list[str]:
    if rb is None:
        return ["### Iter sweep (3.6): NOT YET LANDED"]
    out = ["", "### 3.6 — INLP iteration sweep (Order linear post-null)"]
    out.append("")
    out.append("| Model | Layer | n=10 | n=20 | n=40 | n=80 |")
    out.append("|---|---|---|---|---|---|")
    pivot = rb.pivot_table(
        index=["model", "layer"], columns="n_nulls_applied",
        values="post_order_linear_acc", aggfunc="first",
    )
    for (m, l), row in pivot.iterrows():
        v10 = fmt(row.get(10))
        v20 = fmt(row.get(20))
        v40 = fmt(row.get(40))
        v80 = fmt(row.get(80))
        out.append(f"| {m} | L{int(l)} | {v10} | {v20} | {v40} | {v80} |")
    return out


def section_q6_v2(per_order: pd.DataFrame, nulls: pd.DataFrame) -> list[str]:
    if per_order is None or nulls is None:
        return ["### Q6 v2: NOT YET LANDED"]
    from scipy.stats import pearsonr
    merged = nulls.merge(per_order[["model", "layer", "effective_rank"]],
                         on=["model", "layer"], how="inner")
    if len(merged) < 3:
        return [f"### Q6 v2 (per-Order eff_rank): only {len(merged)} cells available"]
    pr, p = pearsonr(merged["null_median"], merged["effective_rank"])
    out = ["", f"### Q6 v2 — Pearson(null_median, eff_rank) on per-Order manifest, n={len(merged)}"]
    out.append("")
    out.append(f"  Pearson r = {pr:+.3f} (p = {p:.4f})")
    out.append(f"  cells: {sorted(merged['model'].unique())}")
    return out


def section_seed_inlp(seed7: pd.DataFrame, seed13: pd.DataFrame) -> list[str]:
    out = ["", "### 3.7 — random-init §4.12 reads at seeds 7 and 13"]
    out.append("")
    out.append("| Layer | seed42 (existing) | seed7 | seed13 |")
    out.append("|---|---|---|---|")
    seed42_post = {5: 0.740, 7: 0.742, 9: 0.734, 12: 0.728}
    seed42_pre = {5: 0.695, 7: 0.693, 9: 0.685, 12: 0.676}
    for layer in (5, 7, 9, 12):
        s42 = f"{seed42_pre[layer]:.3f} → {seed42_post[layer]:.3f}"
        s7 = "—"
        s13 = "—"
        if seed7 is not None:
            r = seed7[seed7["layer"] == layer]
            if len(r):
                s7 = f"{r['pre_order_acc'].iloc[0]:.3f} → {r['post_order_acc'].iloc[0]:.3f}"
        if seed13 is not None:
            r = seed13[seed13["layer"] == layer]
            if len(r):
                s13 = f"{r['pre_order_acc'].iloc[0]:.3f} → {r['post_order_acc'].iloc[0]:.3f}"
        out.append(f"| L{layer} | {s42} | {s7} | {s13} |")
    return out


def section_multiclass_order(mc: pd.DataFrame) -> list[str]:
    if mc is None:
        return ["", "### §4.12 caveat 2 (multi-class Order INLP): NOT YET LANDED"]
    out = ["", "### §4.12 caveat 2 — multi-class Order INLP (symmetric depth test)"]
    out.append("")
    out.append("| Model | Layer | 4-Order pre→post | 5-Class pre→post | Bin-Class pre→post |")
    out.append("|---|---|---|---|---|")
    for _, r in mc.iterrows():
        out.append(
            f"| {r['model']} | L{int(r['layer'])} | "
            f"{fmt(r['pre_4order_acc'])} → {fmt(r['post_4order_acc'])} | "
            f"{fmt(r['pre_class5_acc'])} → {fmt(r['post_class5_acc'])} | "
            f"{fmt(r['pre_binary_class_acc'])} → {fmt(r['post_binary_class_acc'])} |"
        )
    out.append("")
    if len(mc):
        bin_cls_drop = mc["post_binary_class_acc"] - mc["pre_binary_class_acc"]
        out.append(f"Binary Class survival under Order nullification: "
                   f"mean drop {bin_cls_drop.mean():+.4f}, range "
                   f"{bin_cls_drop.min():+.4f}–{bin_cls_drop.max():+.4f} across "
                   f"{len(mc)} cells.")
    return out


def main() -> None:
    rb = safe(ROUND_B)
    rb_layers = safe(ROUND_B_LAYERS)
    rb_seeds = safe(ROUND_B_SEEDS)
    seed7 = safe(INLP_SEED7)
    seed13 = safe(INLP_SEED13)
    mc = safe(MULTICLASS_ORDER)
    per_order = safe(PER_ORDER_RANK)
    nulls = safe(NULL_CSV)

    out = ["# Round B + Tier 3 v2 numbers (auto-extracted)\n"]
    out.append(f"Generated by step20_v2_extract.py at {pd.Timestamp.utcnow():%Y-%m-%dT%H:%M:%SZ} UTC\n")
    out.extend(section_round_b_summary(rb))
    out.extend(section_iter_sweep(rb))
    out.extend(section_seed_inlp(seed7, seed13))
    out.extend(section_multiclass_order(mc))
    out.extend(section_q6_v2(per_order, nulls))

    target = ROOT / "publication_path/round_b_findings_v2_auto.md"
    target.write_text("\n".join(out))
    print(f"Wrote {target} ({len(out)} lines)")
    print("\n".join(out[:60]))


if __name__ == "__main__":
    main()
