"""
bullfinch_umap_all_layers.py — UMAP visualization of the within-species
clustering experiment across all 13 EAT layers.

Pipeline per layer (using cached activations):
  raw (n_frames, 768) → PCA-50 → UMAP-2

Outputs:
  results/bullfinch_umap_layer{L:02d}.png    (per-layer 2-panel plots:
                                              k-means best-k / recording)
  results/bullfinch_umap_all_layers.png       (3x5 grid, all layers,
                                              colored by recording)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from data.loader import NUM_LAYERS_TOTAL
from bullfinch_within_layer_cluster import (
    ACTIVATIONS_PATH,
    RESULTS_DIR,
    pca50,
    sweep_kmeans,
)

CSV_PATH = RESULTS_DIR / "bullfinch_within_all_layers.csv"
OVERVIEW_PNG = RESULTS_DIR / "bullfinch_umap_all_layers.png"
UMAP_SEED = 42


def layer_label(L: int) -> str:
    return "emb" if L == 0 else f"T{L - 1}"


def load_best_ks() -> dict[int, int]:
    with open(CSV_PATH) as f:
        return {int(r["layer_idx"]): int(r["best_k_kmeans"]) for r in csv.DictReader(f)}


def umap_reduce(X_pca: np.ndarray) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=UMAP_SEED,
    )
    return reducer.fit_transform(X_pca)


def kmeans_labels(X_pca: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=UMAP_SEED, n_init=10)
    return km.fit_predict(X_pca)


def save_per_layer_plot(Xu: np.ndarray, km: np.ndarray, rec_idx: np.ndarray,
                        L: int, k: int, silhouette: float) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    axes[0].scatter(Xu[:, 0], Xu[:, 1], c=km, s=2, cmap="tab10",
                    alpha=0.5, rasterized=True)
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[0].set_title(f"Layer {L} ({layer_label(L)})  —  "
                      f"k-means (best k={k}, sil={silhouette:+.3f})")

    n_recs = len(np.unique(rec_idx))
    axes[1].scatter(Xu[:, 0], Xu[:, 1], c=rec_idx, s=2, cmap="hsv",
                    alpha=0.5, rasterized=True)
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    axes[1].set_title(f"Layer {L} ({layer_label(L)})  —  "
                      f"colored by recording ({n_recs} clips)")

    out = RESULTS_DIR / f"bullfinch_umap_layer{L:02d}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def save_grid(umaps: dict[int, np.ndarray], rec_idx: np.ndarray) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle("Bullfinch within-species UMAP across EAT layers  "
                 "(colored by recording, 37 clips)",
                 fontsize=14, fontweight="bold")

    for L in range(NUM_LAYERS_TOTAL):
        ax = axes[L // 5, L % 5]
        Xu = umaps[L]
        ax.scatter(Xu[:, 0], Xu[:, 1], c=rec_idx, s=1, cmap="hsv",
                   alpha=0.5, rasterized=True)
        ax.set_title(f"Layer {L}  ({layer_label(L)})", fontsize=11,
                     fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # hide the 2 unused panels
    for ax_idx in (13, 14):
        axes[ax_idx // 5, ax_idx % 5].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OVERVIEW_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return OVERVIEW_PNG


def run() -> None:
    print(f"[cache] loading {ACTIVATIONS_PATH}")
    data = np.load(ACTIVATIONS_PATH, allow_pickle=True)
    rec_idx = data["rec_idx"]

    best_ks = load_best_ks()

    # read the per-layer silhouette we already computed, for plot titles
    with open(CSV_PATH) as f:
        sils = {int(r["layer_idx"]): float(r["silhouette"]) for r in csv.DictReader(f)}

    umaps: dict[int, np.ndarray] = {}
    for L in range(NUM_LAYERS_TOTAL):
        X = data[f"layer_{L:02d}"]
        print(f"\n===== Layer {L:2d} ({layer_label(L)})  X={X.shape} =====")
        Xp, _ = pca50(X)
        print("  running UMAP ...", flush=True)
        Xu = umap_reduce(Xp)
        umaps[L] = Xu

        k = best_ks[L]
        km = kmeans_labels(Xp, k)
        out = save_per_layer_plot(Xu, km, rec_idx, L, k, sils[L])
        print(f"  [saved] {out}")

    out = save_grid(umaps, rec_idx)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    run()
