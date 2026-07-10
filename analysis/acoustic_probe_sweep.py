"""
analysis/acoustic_probe_sweep.py — layer sweep of linear probes predicting
acoustic features from pooled EAT time-slice activations (Bullfinch).

Inputs (rows already aligned 1-to-1, verified upstream):
  activations/bullfinch_timeslices_pooled.npz  — layer_00..layer_12, (2365, 768)
  activations/bullfinch_acoustic_targets.npz   — Y, feature_names, voiced_mask, slice_rec_idx

For each layer (0..12) x each of 7 continuous acoustic targets we fit a
StandardScaler + Ridge probe under leave-one-recording-out (LORO) CV
(37 recordings = 37 folds), following the scaler/LORO discipline in
probes/train.py (scaler fit on TRAIN only, seed 42). We regress:
  rms_energy, spectral_centroid, spectral_bandwidth, spectral_rolloff,
  zero_crossing_rate, spectral_flatness, f0
We do NOT regress voiced_flag / voiced_prob (voicing metadata, not acoustic
targets). f0 is probed only on voiced_mask==True rows (unvoiced f0 is NaN).

====================================================================
Metric choice — POOLED LORO R2 (important)
====================================================================
The primary R2 is the POOLED held-out R2: gather every fold's held-out
predictions and score once, r2_score(y_all, yhat_all). This is the standard,
stable LORO regression metric.

We deliberately do NOT headline the mean of per-fold R2. A per-fold R2 scores
a single held-out recording, and R2 = 1 - SS_res/SS_tot blows up when that one
recording has near-constant target variance (denominator -> 0). Empirically the
per-fold mean is dominated by a few such folds (e.g. rms fold-mean ~ -16 +/- 55)
and is uninterpretable. We still report the per-fold mean/std as a transparency
column (fold_mean_R2, fold_std_R2), clearly labelled, but the heatmap, curves
and peak-layer summary all use the pooled R2.

alpha: fixed Ridge alpha = 1000.0 (stated). Chosen so the shuffle null sits
tight around 0: at alpha=100 the 768-dim probe over-fits shuffled labels and the
LORO null floors at ~ -0.2 (over-fit, NOT leakage); raising alpha to 1000 pulls
the null to ~[-0.08, 0] while the real R2 barely moves (e.g. centroid T5
0.820 -> 0.810, rms T1 0.669 -> 0.615). A fixed alpha keeps each fold
leakage-free and fast (per-fold RidgeCV was ~20x slower for no material gain).

Controls per (layer, target):
  * shuffle null — permute y once (seed 42), rerun one LORO pass -> pooled R2
    should be ~0. Reported as shuffle_R2. Leakage would push this POSITIVE; a
    small negative null is the expected LORO over-fit floor, not leakage.
  * random-init EAT baseline — only if a pooled random-init cache exists in the
    same format (we do NOT extract). Skipped with a message otherwise.

Separate optional probe: voiced_flag (binary) via LogisticRegression, same LORO,
reported as accuracy + AUC in its own CSV — kept apart from the R2 regression.

Outputs (results/):
  acoustic_probe_R2.csv           (layer, target, mean_R2, std_R2, shuffle_R2,
                                   n_samples, alpha, fold_mean_R2)
  acoustic_probe_R2_heatmap.png   (layer x target, pooled R2)
  acoustic_probe_R2_curves.png    (R2 vs layer, one line per target, shuffle ref)
  acoustic_probe_voiced_flag.csv  (layer, accuracy, auc, shuffle_accuracy)

CPU-only, seed 42. Follows CLAUDE.md.

Usage:
    python -W ignore analysis/acoustic_probe_sweep.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.linear_model import Ridge, LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from probes.evaluate import _layer_label  # noqa: E402  (reuse layer-label convention)

SEED = 42
ALPHA = 1000.0
PCA_COMPONENTS = 50  # for the voiced_flag logistic probe (CLAUDE.md convention)
N_LAYERS = 13

POOLED_NPZ = Path("activations/bullfinch_timeslices_pooled.npz")
TARGETS_NPZ = Path("activations/bullfinch_acoustic_targets.npz")
# Optional random-init baseline in the identical pooled format (not extracted here):
RANDOM_INIT_CANDIDATES = [
    Path("activations/bullfinch_timeslices_pooled_random_init_eat_seed42.npz"),
    Path("activations/bullfinch_timeslices_pooled_random_init.npz"),
]

RESULTS_DIR = Path("results")
OUT_R2_CSV = RESULTS_DIR / "acoustic_probe_R2.csv"
OUT_HEATMAP = RESULTS_DIR / "acoustic_probe_R2_heatmap.png"
OUT_CURVES = RESULTS_DIR / "acoustic_probe_R2_curves.png"
OUT_VOICED_CSV = RESULTS_DIR / "acoustic_probe_voiced_flag.csv"

REGRESSION_TARGETS = [
    "rms_energy",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "zero_crossing_rate",
    "spectral_flatness",
    "f0",
]
# colorblind-friendly, one per target
TARGET_COLORS = {
    "rms_energy": "#4C78A8",
    "spectral_centroid": "#F58518",
    "spectral_bandwidth": "#54A24B",
    "spectral_rolloff": "#E45756",
    "zero_crossing_rate": "#72B7B2",
    "spectral_flatness": "#B279A2",
    "f0": "#9D755D",
}


def loro_regress(X, y_dict, rec_idx, alpha=ALPHA):
    """LORO Ridge for several targets sharing the same folds / scaling.

    y_dict : {key: y}. Returns {key: (pooled_R2, fold_mean_R2, fold_std_R2)}.
    Scaler is fit on TRAIN rows only each fold; no fold trains on its own
    held-out recording (train mask = rec_idx != r).
    """
    recs = np.unique(rec_idx)
    preds = {k: np.full(y.shape[0], np.nan, dtype=np.float64) for k, y in y_dict.items()}
    fold_r2 = {k: [] for k in y_dict}
    for r in recs:
        tr = rec_idx != r
        te = ~tr
        assert not (rec_idx[tr] == r).any(), "fold trained on its own held-out recording"
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        for k, y in y_dict.items():
            model = Ridge(alpha=alpha, random_state=SEED).fit(Xtr, y[tr])
            pr = model.predict(Xte)
            preds[k][te] = pr
            fold_r2[k].append(r2_score(y[te], pr))
    return {
        k: (r2_score(y_dict[k], preds[k]), float(np.mean(fold_r2[k])), float(np.std(fold_r2[k])))
        for k in y_dict
    }


def loro_logistic(X, y, rec_idx):
    """LORO LogisticRegression -> (pooled accuracy, pooled AUC) on held-out preds.

    Uses StandardScaler + PCA-50 (fit on train only) before LR, per the
    CLAUDE.md probe convention (768-dim LR is slow on CPU).
    """
    recs = np.unique(rec_idx)
    pred = np.full(y.shape[0], -1, dtype=np.int64)
    prob = np.full(y.shape[0], np.nan, dtype=np.float64)
    for r in recs:
        tr = rec_idx != r
        te = ~tr
        scaler = StandardScaler().fit(X[tr])
        pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED).fit(scaler.transform(X[tr]))
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(pca.transform(scaler.transform(X[tr])), y[tr])
        Xte = pca.transform(scaler.transform(X[te]))
        pred[te] = clf.predict(Xte)
        prob[te] = clf.predict_proba(Xte)[:, 1]
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else float("nan")
    return acc, auc


def run_regression_sweep(pooled, Y, feat_names, voiced_mask, rec_idx, rng):
    """Sweep all layers x regression targets. Returns results[(layer,target)] = dict."""
    results: dict[tuple[int, str], dict] = {}
    full_targets = [t for t in REGRESSION_TARGETS if t != "f0"]

    for layer in range(N_LAYERS):
        X = pooled[f"layer_{layer:02d}"].astype(np.float32)

        # --- 6 full-row targets share folds/scaling; include their shuffles ---
        y_dict = {}
        for t in full_targets:
            y = Y[:, feat_names.index(t)].astype(np.float64)
            y_dict[t] = y
            y_dict[t + "__shuffle"] = y[rng.permutation(y.shape[0])]
        out = loro_regress(X, y_dict, rec_idx)
        for t in full_targets:
            pooled_r2, fmean, fstd = out[t]
            results[(layer, t)] = {
                "mean_R2": pooled_r2, "std_R2": fstd, "fold_mean_R2": fmean,
                "shuffle_R2": out[t + "__shuffle"][0], "n_samples": y_dict[t].shape[0],
            }

        # --- f0: voiced rows only, separate mask ------------------------------
        yv = Y[voiced_mask, feat_names.index("f0")].astype(np.float64)
        Xv = X[voiced_mask]
        rv = rec_idx[voiced_mask]
        f0_dict = {"f0": yv, "f0__shuffle": yv[rng.permutation(yv.shape[0])]}
        of = loro_regress(Xv, f0_dict, rv)
        pooled_r2, fmean, fstd = of["f0"]
        results[(layer, "f0")] = {
            "mean_R2": pooled_r2, "std_R2": fstd, "fold_mean_R2": fmean,
            "shuffle_R2": of["f0__shuffle"][0], "n_samples": yv.shape[0],
        }
        peak = max(REGRESSION_TARGETS, key=lambda t: results[(layer, t)]["mean_R2"])
        print(f"  layer {_layer_label(layer):>3} done  "
              f"(best: {peak} R2={results[(layer, peak)]['mean_R2']:+.3f})", flush=True)
    return results


def plot_heatmap(results, path):
    M = np.array([[results[(l, t)]["mean_R2"] for l in range(N_LAYERS)] for t in REGRESSION_TARGETS])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    vmax = float(np.nanmax(M))
    vmin = min(0.0, float(np.nanmin(M)))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(N_LAYERS))
    ax.set_xticklabels([_layer_label(l) for l in range(N_LAYERS)])
    ax.set_yticks(range(len(REGRESSION_TARGETS)))
    ax.set_yticklabels(REGRESSION_TARGETS)
    ax.set_xlabel("EAT layer")
    ax.set_title("Pooled LORO $R^2$ — acoustic feature probes across EAT layers")
    for i in range(len(REGRESSION_TARGETS)):
        for j in range(N_LAYERS):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v < vmin + 0.55 * (vmax - vmin) else "black")
    fig.colorbar(im, ax=ax, label="pooled $R^2$")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def plot_curves(results, path):
    layers = list(range(N_LAYERS))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    max_shuffle = max(abs(results[(l, t)]["shuffle_R2"])
                      for l in layers for t in REGRESSION_TARGETS)
    ax.axhspan(-max_shuffle, max_shuffle, color="gray", alpha=0.15,
               label=f"shuffle null band (|R2|<={max_shuffle:.3f})")
    ax.axhline(0.0, color="gray", ls="--", lw=1)
    for t in REGRESSION_TARGETS:
        ys = [results[(l, t)]["mean_R2"] for l in layers]
        ax.plot(layers, ys, marker="o", ms=4, lw=1.6, color=TARGET_COLORS[t], label=t)
    ax.set_xticks(layers)
    ax.set_xticklabels([_layer_label(l) for l in layers])
    ax.set_xlabel("EAT layer")
    ax.set_ylabel("pooled LORO $R^2$")
    ax.set_title("Acoustic feature decodability across EAT layers (37-recording LORO)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def main():
    rng = np.random.default_rng(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pooled = np.load(POOLED_NPZ, allow_pickle=True)
    targets = np.load(TARGETS_NPZ, allow_pickle=True)
    feat_names = [str(n) for n in targets["feature_names"]]
    Y = targets["Y"]
    voiced_mask = targets["voiced_mask"].astype(bool)
    rec_idx = targets["slice_rec_idx"].astype(np.int64)

    # alignment guard
    if not np.array_equal(rec_idx, pooled["slice_rec_idx"].astype(np.int64)):
        raise RuntimeError("pooled and target slice_rec_idx disagree — rows not aligned.")
    n_rows = rec_idx.shape[0]
    n_recs = int(np.unique(rec_idx).shape[0])
    print(f"[data] rows={n_rows}  recordings={n_recs}  targets={REGRESSION_TARGETS}")
    print(f"[probe] StandardScaler + Ridge(alpha={ALPHA}), LORO over {n_recs} recordings, "
          f"pooled held-out R2\n")

    results = run_regression_sweep(pooled, Y, feat_names, voiced_mask, rec_idx, rng)

    # --- write R2 CSV ----------------------------------------------------------
    with OUT_R2_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "layer_label", "target", "mean_R2", "std_R2",
                    "shuffle_R2", "n_samples", "alpha", "fold_mean_R2"])
        for layer in range(N_LAYERS):
            for t in REGRESSION_TARGETS:
                r = results[(layer, t)]
                w.writerow([layer, _layer_label(layer), t, f"{r['mean_R2']:.5f}",
                            f"{r['std_R2']:.5f}", f"{r['shuffle_R2']:.5f}",
                            r["n_samples"], ALPHA, f"{r['fold_mean_R2']:.5f}"])
    print(f"\n[saved] {OUT_R2_CSV}")

    plot_heatmap(results, OUT_HEATMAP)
    plot_curves(results, OUT_CURVES)

    # --- optional random-init baseline (only if a cache exists) ---------------
    ri_path = next((p for p in RANDOM_INIT_CANDIDATES if p.exists()), None)
    if ri_path is None:
        print("[random-init] no pooled random-init cache found — skipping baseline "
              "(not extracting).")
    else:
        print(f"[random-init] found {ri_path} — running architecture baseline sweep.")
        ri_pooled = np.load(ri_path, allow_pickle=True)
        ri_results = run_regression_sweep(ri_pooled, Y, feat_names, voiced_mask, rec_idx, rng)
        ri_csv = RESULTS_DIR / "acoustic_probe_R2_random_init.csv"
        with ri_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "target", "mean_R2", "shuffle_R2", "n_samples"])
            for layer in range(N_LAYERS):
                for t in REGRESSION_TARGETS:
                    r = ri_results[(layer, t)]
                    w.writerow([layer, t, f"{r['mean_R2']:.5f}", f"{r['shuffle_R2']:.5f}", r["n_samples"]])
        print(f"[saved] {ri_csv}")

    # --- separate binary voiced_flag probe ------------------------------------
    print("\n[voiced_flag] binary LogisticRegression probe (separate from R2 regression):")
    y_voiced = Y[:, feat_names.index("voiced_flag")].astype(np.int64)
    voiced_rows = []
    for layer in range(N_LAYERS):
        X = pooled[f"layer_{layer:02d}"].astype(np.float32)
        acc, auc = loro_logistic(X, y_voiced, rec_idx)
        y_shuf = y_voiced[rng.permutation(y_voiced.shape[0])]
        acc_shuf, _ = loro_logistic(X, y_shuf, rec_idx)
        voiced_rows.append((layer, acc, auc, acc_shuf))
        print(f"  layer {_layer_label(layer):>3}  acc={acc:.3f}  auc={auc:.3f}  "
              f"shuffle_acc={acc_shuf:.3f}")
    with OUT_VOICED_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "layer_label", "accuracy", "auc", "shuffle_accuracy"])
        for layer, acc, auc, acc_shuf in voiced_rows:
            w.writerow([layer, _layer_label(layer), f"{acc:.5f}", f"{auc:.5f}", f"{acc_shuf:.5f}"])
    print(f"[saved] {OUT_VOICED_CSV}")

    # --- summary: peak layer per target ---------------------------------------
    print("\n[summary] peak layer per target (pooled LORO R2):")
    print(f"  {'target':<20s} {'peak_layer':>10s} {'peak_R2':>8s} {'shuffle':>8s}")
    for t in REGRESSION_TARGETS:
        best_layer = max(range(N_LAYERS), key=lambda l: results[(l, t)]["mean_R2"])
        r = results[(best_layer, t)]
        print(f"  {t:<20s} {_layer_label(best_layer):>10s} {r['mean_R2']:>+8.3f} "
              f"{r['shuffle_R2']:>+8.3f}")

    # --- sanity checks --------------------------------------------------------
    print("\n[sanity]")
    shuffles = [results[(l, t)]["shuffle_R2"] for l in range(N_LAYERS) for t in REGRESSION_TARGETS]
    max_shuffle = max(shuffles)   # positive side = leakage signature
    min_shuffle = min(shuffles)   # small negative = expected over-fit floor
    leak = max_shuffle > 0.05
    print(f"  shuffle_R2 range over all (layer,target): [{min_shuffle:+.3f}, {max_shuffle:+.3f}]")
    print(f"  leakage check (shuffle not meaningfully positive): "
          f"max={max_shuffle:+.3f} -> {'LEAKAGE SUSPECTED' if leak else 'OK, no leakage'}")
    print(f"  (small negative nulls are the LORO over-fit floor, not leakage; "
          f"real peaks sit far above.)")
    f0_n = results[(0, 'f0')]["n_samples"]
    print(f"  f0 probe n_samples == voiced count: {f0_n} == {int(voiced_mask.sum())} "
          f"-> {f0_n == int(voiced_mask.sum())}  (expected 1446: {f0_n == 1446})")
    print(f"  LORO folds = recordings = {n_recs}; train mask excludes held-out rec "
          f"(asserted per fold in loro_regress).")


if __name__ == "__main__":
    main()
