"""
analysis/midlayer_calltype_sweep.py — search T5..T9 (or any subset) for fine
call-type structure that lives mid-network.

Pipeline per layer:
    X (n_frames, 768) float16
      → StandardScaler (z-score per feature) — REQUIRED so silhouette isn't
        inflated by activation-scale differences across layers
      → pca50 (reuse from bullfinch_within_layer_cluster)
      → k-means sweep k=3..8 + silhouette (reuse sweep_kmeans)
      → best-k clustering
      → recording-artifact metrics (reuse from analysis.recording_id_recolor):
          ARI + AMI between cluster and recording labels
          per-cluster purity, per-recording containment
      → UMAP-2 (seed 42, cached to activations/bullfinch_umap_coords_zscored.npz
        — separate cache from the recolor script since the z-scored PCA
        embedding is different)
      → 2-panel plot at 150 dpi

Verdict per layer:
  ARTIFACT   if recording_ARI > 0.2                    (clusters track recordings)
  COARSE?    if best_k <= 3 AND silhouette >= 0.20     (e.g. T11 sound-vs-silence)
  CANDIDATE  if 4 <= best_k <= 7 AND recording_ARI < 0.1
             (best_k ≤ 7 enforces "local peak" — the sweep top-out at k=8
             means silhouette may still be climbing and we should widen the
             sweep before declaring a peak)
  NULL       otherwise

Ranking: among CANDIDATEs, highest best_k first, ties broken by silhouette.

CPU-only, seed 42. Full frame set used for k-means and UMAP; silhouette
subsamples to 5000 rows with a seeded RNG (existing `sweep_kmeans`
convention) — that subsample is index-independent so the ARI check on
best_labels vs rec_idx uses the full-length arrays.

Usage:
    python -W ignore analysis/midlayer_calltype_sweep.py
    python -W ignore analysis/midlayer_calltype_sweep.py --layers 6,7,8,9,10
    python -W ignore analysis/midlayer_calltype_sweep.py --layers 6,7 --ks 3,4,5,6,7,8,9,10
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import NUM_LAYERS_TOTAL
from bullfinch_within_layer_cluster import (
    ACTIVATIONS_PATH,
    RESULTS_DIR,
    pca50,
    sweep_kmeans,
)
from analysis.recording_id_recolor import (
    layer_label,
    per_cluster_purity,
    per_recording_containment,
)

CSV_PATH = RESULTS_DIR / "midlayer_calltype_sweep.csv"
UMAP_COORDS_PATH = Path("activations/bullfinch_umap_coords_zscored.npz")
SEED = 42

COARSE_SIL_THRESHOLD = 0.20
CANDIDATE_ARI_MAX = 0.10
ARTIFACT_ARI_MIN = 0.20


def zscore(X: np.ndarray) -> np.ndarray:
    """Per-feature z-score (StandardScaler). Float32 cast for sklearn."""
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X.astype(np.float32))


def get_umap_coords_zscored(L: int, X_pca: np.ndarray) -> np.ndarray:
    """
    Load or compute+cache the z-scored-pipeline UMAP for layer L.
    Separate cache from bullfinch_umap_coords.npz because the input PCA
    embedding is different.
    """
    import umap
    key = f"layer_{L:02d}"
    cache: dict[str, np.ndarray] = {}
    if UMAP_COORDS_PATH.exists():
        cache = dict(np.load(UMAP_COORDS_PATH))
        if key in cache:
            print(f"  [reused] {UMAP_COORDS_PATH}[{key}]")
            return cache[key]
    print("  UMAP (seed=42, n_neighbors=15, min_dist=0.1) ...")
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, random_state=SEED,
    )
    Xu = reducer.fit_transform(X_pca)
    cache[key] = Xu
    UMAP_COORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(UMAP_COORDS_PATH, **cache)
    print(f"  [saved] {UMAP_COORDS_PATH}[{key}]")
    return Xu


def classify_verdict(best_k: int, best_sil: float, ari: float) -> str:
    if ari > ARTIFACT_ARI_MIN:
        return "ARTIFACT"
    if best_k <= 3 and best_sil >= COARSE_SIL_THRESHOLD:
        return "COARSE?"
    if 4 <= best_k <= 7 and ari < CANDIDATE_ARI_MAX:
        return "CANDIDATE"
    return "NULL"


def plot_side_by_side(Xu: np.ndarray, labels: np.ndarray, rec_idx: np.ndarray,
                      L: int, k: int, sil: float, ari: float,
                      verdict: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_recs = int(rec_idx.max()) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle(
        f"Midlayer call-type sweep — layer {L} ({layer_label(L)})  "
        f"best k={k}, sil={sil:+.3f}, ARI(cluster, recording)={ari:+.3f}  "
        f"→ {verdict}",
        fontsize=11, fontweight="bold",
    )

    cmap_c = plt.get_cmap("tab10", k)
    sc0 = axes[0].scatter(Xu[:, 0], Xu[:, 1], c=labels, s=2, cmap=cmap_c,
                          alpha=0.5, rasterized=True)
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[0].set_title(f"colored by k-means (k={k})")
    fig.colorbar(sc0, ax=axes[0], ticks=range(k), shrink=0.85).set_label("cluster")

    cmap_r = plt.get_cmap("hsv", n_recs)
    sc1 = axes[1].scatter(Xu[:, 0], Xu[:, 1], c=rec_idx, s=2, cmap=cmap_r,
                          alpha=0.5, rasterized=True)
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    axes[1].set_title(f"colored by recording ID ({n_recs} recordings)")
    fig.colorbar(sc1, ax=axes[1], shrink=0.85).set_label("recording index")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"midlayer_calltype_layer{L:02d}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run_one_layer(L: int, X: np.ndarray, rec_idx: np.ndarray,
                  rec_names: list[str], ks: Iterable[int]) -> dict:
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

    print(f"\n===== Layer {L:2d} ({layer_label(L)})  X={X.shape} =====")

    print("  z-scoring (StandardScaler, per feature) ...")
    Xz = zscore(X)

    print("  PCA → 50 ...")
    Xp, evr = pca50(Xz)
    print(f"  cum var: {evr.sum():.3f}   top-3: {[f'{v:.3f}' for v in evr[:3]]}")

    ks_list = list(ks)
    print(f"  k-means sweep k={ks_list[0]}..{ks_list[-1]}, silhouette on"
          f" 5000-frame subsample (seed=42) ...")
    sweep = sweep_kmeans(Xp, ks=ks_list)
    best_k, best_sil, best_labels = max(sweep, key=lambda r: r[1])
    print(f"  → best k = {best_k}, silhouette = {best_sil:+.4f}")

    ari = float(adjusted_rand_score(rec_idx, best_labels))
    ami = float(adjusted_mutual_info_score(rec_idx, best_labels))
    per_cluster, mean_purity = per_cluster_purity(best_labels, rec_idx, rec_names)
    per_rec, mean_containment = per_recording_containment(best_labels, rec_idx, rec_names)
    print(f"  recording overlap: ARI={ari:+.4f}, AMI={ami:+.4f}, "
          f"mean containment={mean_containment:.3f}, "
          f"mean per-cluster purity={mean_purity:.3f}")

    verdict = classify_verdict(best_k, best_sil, ari)
    print(f"  verdict = {verdict}")

    Xu = get_umap_coords_zscored(L, Xp)
    out = plot_side_by_side(Xu, best_labels, rec_idx, L, best_k, best_sil,
                            ari, verdict)
    print(f"  [saved] {out}")

    return {
        "layer_idx": L,
        "label": layer_label(L),
        "best_k": best_k,
        "best_silhouette": round(best_sil, 4),
        "recording_ARI": round(ari, 4),
        "recording_AMI": round(ami, 4),
        "mean_containment": round(mean_containment, 4),
        "mean_purity": round(mean_purity, 4),
        "pca50_cum_var": round(float(evr.sum()), 4),
        "verdict": verdict,
        "sweep_k_and_sil": ";".join(f"{k}:{round(s, 4)}" for k, s, _ in sweep),
    }


def write_csv(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[saved] {CSV_PATH}")


def print_summary(rows: list[dict]) -> None:
    print("\n=== Cross-layer summary ===")
    print(f"{'layer':<8} {'best_k':>7} {'best_silhouette':>16} "
          f"{'recording_ARI':>14} {'verdict':<10}")
    print(f"{'-'*8} {'-'*7} {'-'*16} {'-'*14} {'-'*10}")
    for r in rows:
        print(f"{r['label']:<8} {r['best_k']:>7} "
              f"{r['best_silhouette']:>+16.4f} "
              f"{r['recording_ARI']:>+14.4f} {r['verdict']:<10}")

    candidates = [r for r in rows if r["verdict"] == "CANDIDATE"]
    if not candidates:
        print("\nNo CANDIDATE layer found. Check ARTIFACT/COARSE?/NULL flags.")
        return
    # Rank: higher best_k first, then higher silhouette
    candidates.sort(key=lambda r: (-r["best_k"], -r["best_silhouette"]))
    top = candidates[0]
    print(f"\n>>> TOP CANDIDATE: {top['label']} "
          f"(loader idx {top['layer_idx']})  "
          f"best_k = {top['best_k']}, silhouette = {top['best_silhouette']:+.4f}, "
          f"recording_ARI = {top['recording_ARI']:+.4f}")


def parse_layers(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        L = int(part)
        if not 0 <= L < NUM_LAYERS_TOTAL:
            raise ValueError(f"layer {L} out of range 0..{NUM_LAYERS_TOTAL - 1}")
        out.append(L)
    if not out:
        raise ValueError("--layers parsed to empty list")
    return out


def parse_ks(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run(layers: list[int], ks: Iterable[int]) -> None:
    print(f"[cache] loading {ACTIVATIONS_PATH}")
    data = np.load(ACTIVATIONS_PATH, allow_pickle=True)
    rec_idx = np.asarray(data["rec_idx"])
    rec_names = list(data["rec_names"])
    n_recs = len(rec_names)
    print(f"        rec_idx shape {rec_idx.shape}, {n_recs} unique recordings")

    rows: list[dict] = []
    for L in layers:
        X = data[f"layer_{L:02d}"]
        row = run_one_layer(L, X, rec_idx, rec_names, ks)
        rows.append(row)

    write_csv(rows)
    print_summary(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=parse_layers, default=[6, 7, 8, 9, 10],
                    help="comma-separated loader indices. Default: 6,7,8,9,10 "
                         "(T5,T6,T7,T8,T9).")
    ap.add_argument("--ks", type=parse_ks, default=[3, 4, 5, 6, 7, 8],
                    help="comma-separated k values. Default: 3,4,5,6,7,8.")
    args = ap.parse_args()
    run(args.layers, args.ks)


if __name__ == "__main__":
    main()
