"""
bullfinch_within_layer_cluster.py — Frame-level within-species clustering.

Extract raw (no-pool) frame activations from all Bullfinch clips using EAT
(esp_aves2_eat_all). Concatenate per layer into (total_frames, 768) plus a
parallel per-frame recording index. On the target layer (default loader
index 6), PCA to 50 dims and run:
  - k-means for k in [2..10], pick k by silhouette score
  - sklearn.cluster.HDBSCAN (default settings) for a density-based view

Outputs:
  activations/bullfinch_layers_raw.npz  (all 13 layers, float16)
  results/bullfinch_within_layer{L}.png (silhouette curve + PCA scatter)
  stdout: silhouette table, chosen k, HDBSCAN summary.

Layer indexing (per data/loader.py):
  index 0     = local_encoder patch embedding
  index 1..12 = transformer blocks 0..11  (== T0..T11)

Usage:
  python -W ignore bullfinch_within_layer_cluster.py
  python -W ignore bullfinch_within_layer_cluster.py --layer 7   # T6
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from data.loader import load_model, load_audio_file, extract_all_layers, NUM_LAYERS_TOTAL

AUDIO_DIR = Path("audio/bullfinch")
SKIP = {"XC1086809.mp3"}   # 36 MB file the old explore_individuals.py excluded
ACTIVATIONS_PATH = Path("activations/bullfinch_layers_raw.npz")
RESULTS_DIR = Path("results")
SEED = 42


def collect_clips(audio_dir: Path) -> list[Path]:
    paths = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in {".mp3", ".wav"})
    paths = [p for p in paths if p.name not in SKIP]
    return paths


def extract_all_clips(model, paths: list[Path]) -> tuple[dict[int, np.ndarray], np.ndarray, list[str]]:
    """
    Returns:
      per_layer_X : {layer: (total_frames, 768) float16}
      rec_idx     : (total_frames,) int — recording index per row
      rec_names   : list[str] of length n_recordings
    """
    per_clip: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS_TOTAL)}
    rec_idx_pieces: list[np.ndarray] = []
    rec_names: list[str] = []

    for i, path in enumerate(paths):
        t0 = time.time()
        try:
            audio = load_audio_file(str(path))
        except Exception as exc:
            print(f"  [{i+1:2d}/{len(paths)}] {path.name}: DECODE FAIL ({exc})", flush=True)
            continue
        try:
            acts = extract_all_layers(model, audio, mode="raw")  # (13, n_frames, 768)
        except Exception as exc:
            print(f"  [{i+1:2d}/{len(paths)}] {path.name}: FORWARD FAIL ({exc})", flush=True)
            continue

        n_frames = acts.shape[1]
        rec_names.append(path.stem)
        rec_idx_pieces.append(np.full(n_frames, len(rec_names) - 1, dtype=np.int32))
        for layer in range(NUM_LAYERS_TOTAL):
            per_clip[layer].append(acts[layer].astype(np.float16))
        elapsed = time.time() - t0
        print(f"  [{i+1:2d}/{len(paths)}] {path.name}  n_frames={n_frames}  {elapsed:.1f}s", flush=True)

    rec_idx = np.concatenate(rec_idx_pieces, axis=0)
    per_layer_X = {layer: np.concatenate(per_clip[layer], axis=0) for layer in range(NUM_LAYERS_TOTAL)}
    return per_layer_X, rec_idx, rec_names


def save_all_layers(per_layer_X: dict[int, np.ndarray], rec_idx: np.ndarray, rec_names: list[str]) -> None:
    ACTIVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"layer_{i:02d}": per_layer_X[i] for i in range(NUM_LAYERS_TOTAL)}
    payload["rec_idx"] = rec_idx
    payload["rec_names"] = np.array(rec_names)
    np.savez_compressed(ACTIVATIONS_PATH, **payload)
    total_mb = sum(x.nbytes for x in per_layer_X.values()) / 1e6
    print(f"\n[saved] {ACTIVATIONS_PATH}  (~{total_mb:.0f} MB in-memory, compressed on disk)")


def pca50(X: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA
    Xf = X.astype(np.float32)
    n_comp = min(50, Xf.shape[0], Xf.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    Xp = pca.fit_transform(Xf)
    return Xp, pca.explained_variance_ratio_


def sweep_kmeans(Xp: np.ndarray, ks: range, seed: int = SEED) -> list[tuple[int, float, np.ndarray]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    if len(Xp) > 5000:
        sil_idx = rng.choice(len(Xp), 5000, replace=False)
    else:
        sil_idx = np.arange(len(Xp))

    results: list[tuple[int, float, np.ndarray]] = []
    print(f"\n{'k':>3}  {'silhouette':>11}  labels_summary")
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(Xp)
        sil = silhouette_score(Xp[sil_idx], labels[sil_idx])
        counts = np.bincount(labels).tolist()
        results.append((k, sil, labels))
        print(f"{k:>3}  {sil:>+11.4f}  sizes={counts}")
    return results


def hdbscan_cluster(Xp: np.ndarray) -> np.ndarray:
    from sklearn.cluster import HDBSCAN
    hdb = HDBSCAN(min_cluster_size=50, min_samples=10)
    labels = hdb.fit_predict(Xp)
    n_clusters = int((labels >= 0).any() and labels.max() + 1) if labels.size else 0
    n_noise = int(np.sum(labels < 0))
    print(f"\nHDBSCAN: {n_clusters} clusters, {n_noise} noise "
          f"({100*n_noise/len(labels):.1f}%)  sizes="
          f"{[int(np.sum(labels == c)) for c in range(n_clusters)]}")
    return labels


def plot_results(Xp: np.ndarray, results: list[tuple[int, float, np.ndarray]],
                 best_k: int, best_labels: np.ndarray,
                 hdb_labels: np.ndarray, rec_idx: np.ndarray, layer_idx: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r[0] for r in results]
    sils = [r[1] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].plot(ks, sils, marker="o")
    axes[0].axvline(best_k, ls="--", c="red", alpha=0.5, label=f"best k={best_k}")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("silhouette (subsample 5000)")
    axes[0].set_title(f"K-means silhouette sweep, layer index {layer_idx}")
    axes[0].grid(True, ls=":", alpha=0.4)
    axes[0].legend()

    axes[1].scatter(Xp[:, 0], Xp[:, 1], c=best_labels, s=2, cmap="tab10", alpha=0.5, rasterized=True)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title(f"K-means k={best_k} on layer {layer_idx} (PCA-50 → PC1/PC2)")

    axes[2].scatter(Xp[:, 0], Xp[:, 1], c=rec_idx, s=2, cmap="hsv", alpha=0.5, rasterized=True)
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].set_title(f"Colored by recording ({len(np.unique(rec_idx))} clips)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"bullfinch_within_layer{layer_idx:02d}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=6,
                    help="loader index (0=local_encoder, 1..12=T0..T11). Default 6 (=T5).")
    ap.add_argument("--model", default="esp_aves2_eat_all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force-reextract", action="store_true")
    args = ap.parse_args()

    if ACTIVATIONS_PATH.exists() and not args.force_reextract:
        print(f"[cache] loading {ACTIVATIONS_PATH}")
        data = np.load(ACTIVATIONS_PATH, allow_pickle=True)
        per_layer_X = {i: data[f"layer_{i:02d}"] for i in range(NUM_LAYERS_TOTAL)}
        rec_idx = data["rec_idx"]
        rec_names = list(data["rec_names"])
    else:
        clips = collect_clips(AUDIO_DIR)
        print(f"Found {len(clips)} Bullfinch clips (skipped: {SKIP})")
        print(f"Loading model {args.model} on {args.device}...")
        model = load_model(args.model, device=args.device)
        print(f"Extracting frame-level activations (mode='raw', all 13 layers)...")
        per_layer_X, rec_idx, rec_names = extract_all_clips(model, clips)
        save_all_layers(per_layer_X, rec_idx, rec_names)

    L = args.layer
    X = per_layer_X[L]
    n_frames, dim = X.shape
    n_recs = len(rec_names)
    print(f"\nLayer index {L}  →  X shape ({n_frames}, {dim})  from {n_recs} recordings"
          f"  (avg {n_frames // n_recs} frames/rec)")

    print("\nPCA → 50 dims ...")
    Xp, evr = pca50(X)
    print(f"  explained variance top-10: {[f'{v:.3f}' for v in evr[:10]]}"
          f"  cumulative(50): {evr.sum():.3f}")

    kmeans_results = sweep_kmeans(Xp, ks=range(2, 11))
    best_k, best_sil, best_labels = max(kmeans_results, key=lambda r: r[1])
    print(f"\n→ Best k = {best_k}  silhouette = {best_sil:+.4f}")

    hdb_labels = hdbscan_cluster(Xp)

    out_png = plot_results(Xp, kmeans_results, best_k, best_labels, hdb_labels, rec_idx, L)
    print(f"\n[saved] {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
