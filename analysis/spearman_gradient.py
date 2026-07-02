"""
spearman_gradient.py — Phylogenetic gradient statistics.

Tests the primary claim of the paper: probe accuracy scales monotonically with
evolutionary distance (MYA) across species pairs.

Computes, on the (MYA, accuracy) table of species pairs:
  1. Spearman rho (rank correlation — monotonic, not linear)
  2. Asymptotic p-value (scipy) + permutation p-value (honest for small n)
  3. Bootstrap 95% CI on rho (resample pairs with replacement)
  4. Pearson r on (log10 MYA, accuracy) as a secondary log-linear check
Saves a scatter plot (log-x) and a JSON results dump.

INPUT: a CSV with columns  pair,mya,accuracy   (extra columns ignored).
  - `accuracy` should be a single robust per-pair summary. Peak accuracy is fine
    to start; a weighted-mean-layer accuracy is more stable (argmax is noisy).
  - Rows with a blank `accuracy` are skipped with a warning (fill them from your
    LaTeX appendix / probe stdout logs).

USAGE:
    source venv/bin/activate
    python analysis/spearman_gradient.py --csv analysis/species_pair_results.csv
    # options: --alt greater|two-sided   --boot 10000   --out results/gradient_stats

The directional hypothesis is that accuracy INCREASES with MYA, so the default
alternative is one-sided 'greater'. Two-sided is reported alongside it regardless.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_pairs(csv_path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read pair,mya,accuracy from CSV. Skip rows with missing/blank accuracy."""
    names: list[str] = []
    mya: list[float] = []
    acc: list[float] = []
    skipped: list[str] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"pair", "mya", "accuracy"}
        missing_cols = required - set(h.strip() for h in (reader.fieldnames or []))
        if missing_cols:
            raise ValueError(
                f"CSV missing required column(s): {sorted(missing_cols)}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            name = (row.get("pair") or "").strip()
            mya_raw = (row.get("mya") or "").strip()
            acc_raw = (row.get("accuracy") or "").strip()
            if not name:
                continue
            if not acc_raw or not mya_raw:
                skipped.append(name or "<unnamed>")
                continue
            try:
                mya.append(float(mya_raw))
                acc.append(float(acc_raw))
                names.append(name)
            except ValueError:
                skipped.append(name)

    if skipped:
        print(f"[warn] skipped {len(skipped)} row(s) with blank/invalid data: "
              f"{', '.join(skipped)}")

    return names, np.asarray(mya, float), np.asarray(acc, float)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def permutation_pvalue(x: np.ndarray, y: np.ndarray, n_perm: int,
                       alternative: str, rng: np.random.Generator) -> float:
    """P-value by permuting y against fixed x and recomputing Spearman rho."""
    obs = stats.spearmanr(x, y).statistic
    yp = y.copy()
    count = 0
    for _ in range(n_perm):
        rng.shuffle(yp)
        r = stats.spearmanr(x, yp).statistic
        if alternative == "greater":
            count += r >= obs
        else:  # two-sided
            count += abs(r) >= abs(obs)
    # +1 correction (Phipson & Smyth) so p is never exactly 0
    return (count + 1) / (n_perm + 1)


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int,
                 rng: np.random.Generator) -> tuple[float, float, np.ndarray, int]:
    """Resample PAIRS with replacement; recompute rho. Return CI + distribution."""
    n = len(x)
    rhos = np.empty(n_boot)
    degenerate = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        # degenerate resample: no variance in one variable -> rho undefined
        if np.all(xb == xb[0]) or np.all(yb == yb[0]):
            rhos[b] = np.nan
            degenerate += 1
            continue
        rhos[b] = stats.spearmanr(xb, yb).statistic
    valid = rhos[~np.isnan(rhos)]
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return float(lo), float(hi), valid, degenerate


def spearman_threshold(n: int, alpha: float = 0.05, two_sided: bool = False) -> float:
    """Approx critical rho for significance at this n (normal approx: z/sqrt(n-1))."""
    z = stats.norm.ppf(1 - (alpha / 2 if two_sided else alpha))
    return z / np.sqrt(n - 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(csv_path: str, alternative: str, n_boot: int, n_perm: int,
        out_prefix: str, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    names, mya, acc = load_pairs(csv_path)
    n = len(names)

    if n < 4:
        raise SystemExit(
            f"Only {n} usable pair(s) after skipping blanks. Fill in the accuracy "
            f"column in {csv_path} (need the numbers from your appendix/logs)."
        )

    sp = stats.spearmanr(mya, acc)
    rho, p_asym = float(sp.statistic), float(sp.pvalue)  # scipy p is two-sided
    p_perm = permutation_pvalue(mya, acc, n_perm, alternative, rng)
    lo, hi, boot_dist, degen = bootstrap_ci(mya, acc, n_boot, rng)

    # secondary: log-linear
    pear = stats.pearsonr(np.log10(mya), acc)
    r_log, p_log = float(pear.statistic), float(pear.pvalue)

    thr_1 = spearman_threshold(n, two_sided=False)
    thr_2 = spearman_threshold(n, two_sided=True)

    result = {
        "n_pairs": n,
        "pairs": names,
        "spearman_rho": rho,
        "p_two_sided_asymptotic": p_asym,
        f"p_{alternative}_permutation": p_perm,
        "bootstrap_ci_95": [lo, hi],
        "bootstrap_degenerate_frac": degen / n_boot,
        "pearson_r_on_log10_mya": r_log,
        "pearson_p_two_sided": p_log,
        "sig_threshold_rho_1sided_p05": float(thr_1),
        "sig_threshold_rho_2sided_p05": float(thr_2),
        "significant_1sided_p05": bool(rho >= thr_1 and p_perm < 0.05),
    }

    # ---- report ----
    print("\n" + "=" * 60)
    print(" PHYLOGENETIC GRADIENT — Spearman analysis")
    print("=" * 60)
    print(f" n pairs                : {n}")
    print(f" Spearman rho           : {rho:+.3f}")
    print(f" p (two-sided, scipy)   : {p_asym:.4f}")
    print(f" p ({alternative}, perm)  : {p_perm:.4f}   [{n_perm} permutations]")
    print(f" 95% bootstrap CI on rho: [{lo:+.3f}, {hi:+.3f}]   "
          f"[{n_boot} resamples, {100*degen/n_boot:.1f}% degenerate]")
    print(f" sig threshold (rho)    : >= {thr_1:.3f} (1-sided p<.05), "
          f">= {thr_2:.3f} (2-sided)")
    print(f" log-linear Pearson r   : {r_log:+.3f} (p={p_log:.4f})")
    print("-" * 60)
    verdict = "SIGNIFICANT" if result["significant_1sided_p05"] else "NOT significant"
    ci_note = "excludes 0" if lo > 0 else "includes 0 — unstable"
    print(f" VERDICT: gradient is {verdict} at n={n}. Bootstrap CI {ci_note}.")
    print("=" * 60 + "\n")

    # ---- outputs ----
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    with open(out_prefix + ".json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_prefix}.json")

    _plot(names, mya, acc, rho, p_perm, lo, hi, alternative, out_prefix + ".png")
    print(f"[saved] {out_prefix}.png")

    return result


def _plot(names, mya, acc, rho, p_perm, lo, hi, alternative, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mya, acc, s=45, zorder=3, color="#2A788E", edgecolor="k", linewidth=0.5)
    for x, y, nm in zip(mya, acc, names):
        ax.annotate(nm, (x, y), fontsize=6, xytext=(4, 3),
                    textcoords="offset points", alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("Divergence time (MYA, log scale)")
    ax.set_ylabel("Probe accuracy (per-pair summary)")
    ax.axhline(0.5, ls="--", c="gray", lw=1, label="chance (binary)")
    ax.set_title(
        f"Phylogenetic gradient  |  Spearman rho={rho:+.3f} "
        f"(p_{alternative}={p_perm:.3f})\n95% bootstrap CI on rho "
        f"[{lo:+.2f}, {hi:+.2f}], n={len(names)}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Spearman gradient statistics.")
    ap.add_argument("--csv", default="analysis/species_pair_results.csv")
    ap.add_argument("--alt", choices=["greater", "two-sided"], default="greater",
                    help="directional hypothesis (default: greater = accuracy rises with MYA)")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--perm", type=int, default=10000, help="permutations")
    ap.add_argument("--out", default="results/gradient_stats", help="output prefix")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.csv, args.alt, args.boot, args.perm, args.out, args.seed)


if __name__ == "__main__":
    main()
