"""
analysis/recording_id_recolor.py — recording-artifact check for within-species
clustering.

Renders the same UMAP scatter twice, side by side, using ONE 2D embedding:
  left  = colored by k-means cluster label
  right = colored by recording ID

If most clusters are >0.8 dominated by a single recording, the "structure"
we're reporting is a recording-identity artifact. If clusters draw roughly
evenly across many recordings, the structure is real (call type, temporal
phase, etc.).

Defaults to T11 (loader index 12) since that's where the U-shape peaks and
the "look how clustered!" claim is most suspect. Reuses cached activations,
best-k from the prior sweep CSV, and pca50 from
bullfinch_within_layer_cluster. UMAP is refit with the same params
(seed=42, n_neighbors=15, min_dist=0.1) as bullfinch_umap_all_layers.py
because coords weren't persisted upstream — deterministic under fixed seed,
so byte-identical to the prior render.

CPU-only, seed 42.

Usage:
    python -W ignore analysis/recording_id_recolor.py             # T11
    python -W ignore analysis/recording_id_recolor.py --layer 6   # T5
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import NUM_LAYERS_TOTAL
from bullfinch_within_layer_cluster import (
    ACTIVATIONS_PATH,
    RESULTS_DIR,
    pca50,
)

CSV_PATH = RESULTS_DIR / "bullfinch_within_all_layers.csv"
UMAP_COORDS_PATH = Path("activations/bullfinch_umap_coords.npz")
SEED = 42


def layer_label(L: int) -> str:
    return "emb" if L == 0 else f"T{L - 1}"


def load_best_k(L: int) -> int:
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if int(r["layer_idx"]) == L:
                return int(r["best_k_kmeans"])
    raise ValueError(f"no row for layer {L} in {CSV_PATH}")


def compute_umap(X_pca: np.ndarray) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=SEED,
    )
    return reducer.fit_transform(X_pca)


def kmeans_labels(X_pca: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    return km.fit_predict(X_pca)


def get_umap_coords(L: int, X_pca: np.ndarray) -> np.ndarray:
    """
    Load cached UMAP coords for this layer if present; otherwise compute and
    cache them into UMAP_COORDS_PATH.
    """
    key = f"layer_{L:02d}"
    if UMAP_COORDS_PATH.exists():
        cache = dict(np.load(UMAP_COORDS_PATH))
        if key in cache:
            print(f"  [reused] {UMAP_COORDS_PATH}[{key}]")
            return cache[key]
    else:
        cache = {}
    print("  UMAP (seed=42, n_neighbors=15, min_dist=0.1) ...")
    Xu = compute_umap(X_pca)
    cache[key] = Xu
    UMAP_COORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(UMAP_COORDS_PATH, **cache)
    print(f"  [saved] {UMAP_COORDS_PATH}[{key}]")
    return Xu


def per_cluster_purity(labels: np.ndarray, rec_idx: np.ndarray,
                       rec_names: list[str]) -> tuple[dict[int, dict], float]:
    """
    For each cluster: fraction of its frames from its single most-common
    recording. Return per-cluster stats and mean purity across clusters.
    """
    per_cluster: dict[int, dict] = {}
    for c in sorted(np.unique(labels)):
        mask = labels == c
        recs = rec_idx[mask]
        counter = Counter(recs)
        top_rec, top_count = counter.most_common(1)[0]
        per_cluster[int(c)] = {
            "size": int(mask.sum()),
            "purity": top_count / int(mask.sum()),
            "top_rec_idx": int(top_rec),
            "top_rec_name": rec_names[int(top_rec)],
            "top_rec_count": int(top_count),
            "n_unique_recordings": int(len(counter)),
        }
    mean_purity = float(np.mean([d["purity"] for d in per_cluster.values()]))
    return per_cluster, mean_purity


def per_recording_containment(labels: np.ndarray, rec_idx: np.ndarray,
                              rec_names: list[str]) -> tuple[dict[int, float], float]:
    """
    Complementary metric: for each recording, fraction of its frames in the
    single cluster where the recording is most concentrated. If ≈1 for most
    recordings, the clustering is recording-driven even when per-cluster
    purity looks low (typical for k << n_recordings).
    """
    per_rec: dict[int, float] = {}
    for r in range(len(rec_names)):
        mask = rec_idx == r
        if mask.sum() == 0:
            continue
        top_count = Counter(labels[mask]).most_common(1)[0][1]
        per_rec[r] = top_count / int(mask.sum())
    mean_cont = float(np.mean(list(per_rec.values())))
    return per_rec, mean_cont


def print_recording_by_cluster(labels: np.ndarray, rec_idx: np.ndarray,
                               rec_names: list[str]) -> None:
    """
    Wide table: one row per recording, columns per cluster (counts).
    Transposed from the literal request because k is small and n_rec is 37 —
    reads better on a terminal.
    """
    ks = sorted(np.unique(labels))
    n_recs = len(rec_names)
    counts = np.zeros((n_recs, len(ks)), dtype=np.int32)
    for r in range(n_recs):
        rmask = rec_idx == r
        if not rmask.any():
            continue
        for c, cnt in Counter(labels[rmask]).items():
            counts[r, ks.index(c)] = cnt

    header = " ".join(f"c{c:<5d}" for c in ks)
    print(f"\n=== Recording x Cluster count table ({n_recs} recordings × "
          f"{len(ks)} clusters) ===")
    print(f"  {'recording':40s}  total  {header}  top_cluster_share")
    for r in range(n_recs):
        total = int(counts[r].sum())
        if total == 0:
            continue
        row = " ".join(f"{c:<6d}" for c in counts[r])
        top_share = counts[r].max() / total
        name = rec_names[r][:40]
        print(f"  {name:40s}  {total:5d}  {row}  {top_share:.3f}")


def make_verdict(mean_purity: float, mean_containment: float,
                 ari: float, ami: float, high_purity_frac: float) -> str:
    """
    Combine three signals to decide:
      - per-cluster purity (Sid's spec)
      - per-recording containment (works when k << n_recordings)
      - ARI/AMI (label-set-agnostic)
    """
    if high_purity_frac >= 0.8 or mean_purity >= 0.8 or ari >= 0.5:
        return (f"LIKELY RECORDING ARTIFACT — mean cluster purity "
                f"{mean_purity:.2f}, ARI(cluster, recording)={ari:.2f}, "
                f"mean recording containment {mean_containment:.2f}")
    if ari < 0.05:
        return (f"clusters mix recordings — structure looks real "
                f"(mean purity {mean_purity:.2f}, ARI={ari:.2f}, containment "
                f"{mean_containment:.2f}). Cluster labels are ~independent of "
                f"recording identity.")
    return (f"MIXED — partial recording overlap "
            f"(mean purity {mean_purity:.2f}, ARI={ari:.2f}, "
            f"containment {mean_containment:.2f}). "
            f"Inspect the plot before believing the clusters.")


def plot_side_by_side(Xu: np.ndarray, labels: np.ndarray, rec_idx: np.ndarray,
                      L: int, k: int, mean_purity: float, verdict: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_recs = int(rec_idx.max()) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle(
        f"Recording-artifact check — layer {L} ({layer_label(L)}), "
        f"same UMAP coords, two colorings\n{verdict}",
        fontsize=11, fontweight="bold",
    )

    cluster_cmap = plt.get_cmap("tab10", k)
    sc0 = axes[0].scatter(Xu[:, 0], Xu[:, 1], c=labels, s=2, cmap=cluster_cmap,
                          alpha=0.5, rasterized=True)
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[0].set_title(f"colored by k-means (k={k})   "
                      f"mean per-cluster purity = {mean_purity:.3f}",
                      fontsize=10)
    cbar0 = fig.colorbar(sc0, ax=axes[0], ticks=range(k), shrink=0.85)
    cbar0.set_label("cluster")

    rec_cmap = plt.get_cmap("hsv", n_recs)
    sc1 = axes[1].scatter(Xu[:, 0], Xu[:, 1], c=rec_idx, s=2, cmap=rec_cmap,
                          alpha=0.5, rasterized=True)
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    axes[1].set_title(f"colored by recording ID ({n_recs} recordings)",
                      fontsize=10)
    cbar1 = fig.colorbar(sc1, ax=axes[1], shrink=0.85)
    cbar1.set_label("recording index")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"bullfinch_recording_recolor_layer{L:02d}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run(L: int) -> None:
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

    print(f"[cache] loading {ACTIVATIONS_PATH}")
    data = np.load(ACTIVATIONS_PATH, allow_pickle=True)
    rec_idx = np.asarray(data["rec_idx"])
    rec_names = list(data["rec_names"])
    n_recs = len(rec_names)

    X = data[f"layer_{L:02d}"]
    print(f"\nLayer {L} ({layer_label(L)}) X={X.shape} across {n_recs} recordings")

    print("  PCA → 50 ...")
    Xp, evr = pca50(X)
    print(f"  cum var: {evr.sum():.3f}   top-3: {[f'{v:.3f}' for v in evr[:3]]}")

    k = load_best_k(L)
    print(f"  k-means k={k}  (best from prior sweep)")
    labels = kmeans_labels(Xp, k)

    Xu = get_umap_coords(L, Xp)

    per_cluster, mean_purity = per_cluster_purity(labels, rec_idx, rec_names)
    per_rec, mean_cont = per_recording_containment(labels, rec_idx, rec_names)
    ari = float(adjusted_rand_score(rec_idx, labels))
    ami = float(adjusted_mutual_info_score(rec_idx, labels))
    high_purity_frac = sum(1 for d in per_cluster.values() if d["purity"] > 0.8) / len(per_cluster)

    print("\n=== Per-cluster purity (Sid's spec: fraction from top-1 recording) ===")
    for c, d in per_cluster.items():
        print(f"  cluster {c:2d}  size={d['size']:5d}  purity={d['purity']:.3f}  "
              f"top_rec='{d['top_rec_name'][:40]}' "
              f"({d['top_rec_count']}/{d['size']})  "
              f"unique_recordings={d['n_unique_recordings']}/{n_recs}")
    print(f"  mean per-cluster purity = {mean_purity:.3f}   "
          f"(fraction of clusters with purity>0.8: {high_purity_frac:.2f})")

    print("\n=== Per-recording containment (complementary; robust to k << n_recs) ===")
    print(f"  mean recording containment (fraction in top-1 cluster) = {mean_cont:.3f}")
    high_cont_recs = sum(1 for v in per_rec.values() if v > 0.9)
    print(f"  recordings with >90% of frames in a single cluster: "
          f"{high_cont_recs}/{n_recs}")

    print(f"\n=== Label-set-agnostic overlap ===")
    print(f"  ARI(cluster, recording) = {ari:+.3f}   "
          f"(0 = independent, 1 = identical partition)")
    print(f"  AMI(cluster, recording) = {ami:+.3f}   "
          f"(same idea, adjusted mutual info)")

    print_recording_by_cluster(labels, rec_idx, rec_names)

    verdict = make_verdict(mean_purity, mean_cont, ari, ami, high_purity_frac)
    print(f"\n>>> VERDICT: {verdict}")

    out = plot_side_by_side(Xu, labels, rec_idx, L, k, mean_purity, verdict)
    print(f"\n[saved] {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12,
                    help="loader index (0=emb, 1..12=T0..T11). Default 12 (=T11).")
    args = ap.parse_args()
    if not 0 <= args.layer < NUM_LAYERS_TOTAL:
        raise SystemExit(f"layer must be in 0..{NUM_LAYERS_TOTAL - 1}")
    run(args.layer)


if __name__ == "__main__":
    main()
