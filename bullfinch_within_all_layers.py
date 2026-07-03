"""
bullfinch_within_all_layers.py — Run the within-species clustering sweep
across every EAT layer using cached activations from
activations/bullfinch_layers_raw.npz.

Pipeline per layer:
  raw frame matrix → PCA-50 → k-means k=2..10 + silhouette → sklearn HDBSCAN

Outputs:
  results/bullfinch_within_layer{L:02d}.png    (13 per-layer 3-panel plots)
  results/bullfinch_within_all_layers.csv      (summary row per layer)
  results/bullfinch_within_all_layers.png      (silhouette-vs-layer overview)
  results/bullfinch_within_species_structure.md
                                               (narrative per layer)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from data.loader import NUM_LAYERS_TOTAL
from bullfinch_within_layer_cluster import (
    ACTIVATIONS_PATH,
    RESULTS_DIR,
    hdbscan_cluster,
    pca50,
    plot_results,
    sweep_kmeans,
)

CSV_PATH = RESULTS_DIR / "bullfinch_within_all_layers.csv"
OVERVIEW_PNG = RESULTS_DIR / "bullfinch_within_all_layers.png"
DOC_PATH = RESULTS_DIR / "bullfinch_within_species_structure.md"


def layer_label(L: int) -> str:
    return "emb" if L == 0 else f"T{L - 1}"


def describe(sil: float, hdb_noise_pct: float, hdb_largest_pct: float) -> str:
    if sil >= 0.25:
        cluster_desc = "well-separated cluster structure"
    elif sil >= 0.15:
        cluster_desc = "moderate cluster structure"
    elif sil >= 0.10:
        cluster_desc = "weak cluster structure (borderline)"
    else:
        cluster_desc = "no meaningful cluster structure"
    if hdb_largest_pct >= 95:
        hdb_desc = "HDBSCAN collapses to one large blob"
    elif hdb_largest_pct >= 80:
        hdb_desc = "HDBSCAN finds a dominant blob plus small satellites"
    else:
        hdb_desc = "HDBSCAN resolves multiple comparable groups"
    return f"{cluster_desc}; {hdb_desc}"


def run() -> None:
    data = np.load(ACTIVATIONS_PATH, allow_pickle=True)
    rec_idx = data["rec_idx"]
    rec_names = list(data["rec_names"])
    n_recs = len(rec_names)

    rows: list[dict] = []
    per_layer_sils: dict[int, list[tuple[int, float]]] = {}
    per_layer_evr: dict[int, list[float]] = {}

    for L in range(NUM_LAYERS_TOTAL):
        X = data[f"layer_{L:02d}"]
        label = layer_label(L)
        print(f"\n===== Layer {L:2d} ({label})  X={X.shape} =====")

        Xp, evr = pca50(X)
        print(f"  PCA-50 cum var: {evr.sum():.3f}   top-5 ratios: "
              f"{[f'{v:.3f}' for v in evr[:5]]}")

        km_results = sweep_kmeans(Xp, ks=range(2, 11))
        best_k, best_sil, best_labels = max(km_results, key=lambda r: r[1])
        print(f"  → best k = {best_k}   silhouette = {best_sil:+.4f}")

        hdb_labels = hdbscan_cluster(Xp)
        n_hdb = int(hdb_labels.max() + 1) if (hdb_labels >= 0).any() else 0
        n_noise = int((hdb_labels < 0).sum())
        if n_hdb > 0:
            cluster_sizes = np.bincount(hdb_labels[hdb_labels >= 0])
            largest = int(cluster_sizes.max())
        else:
            largest = 0
        noise_pct = 100.0 * n_noise / len(hdb_labels)
        largest_pct = 100.0 * largest / len(hdb_labels)

        out_png = plot_results(Xp, km_results, best_k, best_labels, hdb_labels,
                               rec_idx, L)
        print(f"  [saved] {out_png}")

        rows.append({
            "layer_idx": L,
            "label": label,
            "n_frames": int(X.shape[0]),
            "best_k_kmeans": best_k,
            "silhouette": round(best_sil, 4),
            "silhouette_k2": round(km_results[0][1], 4),
            "pca50_cum_var": round(float(evr.sum()), 4),
            "top_pc_var": round(float(evr[0]), 4),
            "hdbscan_clusters": n_hdb,
            "hdbscan_noise_pct": round(noise_pct, 2),
            "hdbscan_largest_pct": round(largest_pct, 2),
        })
        per_layer_sils[L] = [(k, s) for k, s, _ in km_results]
        per_layer_evr[L] = evr[:10].tolist()

    write_csv(rows)
    write_overview_plot(rows, per_layer_sils)
    write_markdown(rows, per_layer_sils, per_layer_evr, n_recs)


def write_csv(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[saved] {CSV_PATH}")


def write_overview_plot(rows: list[dict],
                        per_layer_sils: dict[int, list[tuple[int, float]]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ls = [r["layer_idx"] for r in rows]
    labels = [r["label"] for r in rows]
    best_sils = [r["silhouette"] for r in rows]
    noise = [r["hdbscan_noise_pct"] for r in rows]
    largest = [r["hdbscan_largest_pct"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(Ls, best_sils, marker="o", color="#2A788E")
    axes[0].set_xticks(Ls)
    axes[0].set_xticklabels(labels, rotation=45, fontsize=8)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Best k-means silhouette (k=2..10)")
    axes[0].axhline(0.10, ls=":", c="gray", alpha=0.7, label="weak (0.10)")
    axes[0].axhline(0.25, ls=":", c="red", alpha=0.5, label="well-sep (0.25)")
    axes[0].set_title("K-means silhouette across layers")
    axes[0].grid(True, ls=":", alpha=0.4)
    axes[0].legend(fontsize=8)

    axes[1].plot(Ls, largest, marker="s", color="#440154", label="largest HDBSCAN cluster %")
    axes[1].plot(Ls, noise, marker="^", color="#F0932B", label="HDBSCAN noise %")
    axes[1].set_xticks(Ls)
    axes[1].set_xticklabels(labels, rotation=45, fontsize=8)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("% of frames")
    axes[1].set_title("HDBSCAN mass across layers")
    axes[1].grid(True, ls=":", alpha=0.4)
    axes[1].legend(fontsize=8)

    for L, sils in per_layer_sils.items():
        ks = [k for k, _ in sils]
        ss = [s for _, s in sils]
        axes[2].plot(ks, ss, alpha=0.6, label=layer_label(L))
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("silhouette")
    axes[2].set_title("k-sweep per layer")
    axes[2].grid(True, ls=":", alpha=0.4)
    axes[2].legend(fontsize=7, ncol=2, loc="upper right")

    fig.tight_layout()
    fig.savefig(OVERVIEW_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OVERVIEW_PNG}")


def write_markdown(rows: list[dict],
                   per_layer_sils: dict[int, list[tuple[int, float]]],
                   per_layer_evr: dict[int, list[float]],
                   n_recs: int) -> None:
    lines: list[str] = []
    lines.append("# Bullfinch within-species cluster structure across EAT layers")
    lines.append("")
    lines.append(f"Model: `esp_aves2_eat_all`. Corpus: **{n_recs} Bullfinch clips** "
                 "(XC1086809.mp3 skipped; ~8 recordings are format duplicates so "
                 "effective unique count is ~26–30). Every clip yields exactly 512 "
                 "avex patches, so per-layer matrices are (18 944, 768).")
    lines.append("")
    lines.append("Pipeline per layer: **raw frame extraction (no pool) → PCA→50 → "
                 "k-means k=2..10 (silhouette on a seeded 5 000-frame subsample) → "
                 "`sklearn.cluster.HDBSCAN(min_cluster_size=50, min_samples=10)`**.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Layer | Label | Best k | Silhouette | PCA-50 cum var | HDBSCAN k | Noise % | Largest % | Verdict |")
    lines.append("|-------|-------|--------|------------|----------------|-----------|---------|-----------|---------|")
    for r in rows:
        verdict_short = describe(r["silhouette"], r["hdbscan_noise_pct"],
                                 r["hdbscan_largest_pct"]).split(";")[0]
        lines.append(
            f"| {r['layer_idx']} | `{r['label']}` | {r['best_k_kmeans']} | "
            f"{r['silhouette']:+.4f} | {r['pca50_cum_var']:.3f} | "
            f"{r['hdbscan_clusters']} | {r['hdbscan_noise_pct']:.1f} | "
            f"{r['hdbscan_largest_pct']:.1f} | {verdict_short} |"
        )
    lines.append("")

    lines.append("## Per-layer notes")
    lines.append("")
    for r in rows:
        L = r["layer_idx"]
        sils = per_layer_sils[L]
        # 2nd best k for context
        sorted_sils = sorted(sils, key=lambda kv: -kv[1])
        top2 = sorted_sils[:2]
        top_pc = per_layer_evr[L][0]
        lines.append(f"### Layer {L} — `{r['label']}`"
                     f" (best k={r['best_k_kmeans']}, sil={r['silhouette']:+.4f})")
        lines.append("")
        lines.append(f"- Silhouette top-2: "
                     f"k={top2[0][0]}→{top2[0][1]:+.4f}, "
                     f"k={top2[1][0]}→{top2[1][1]:+.4f}.")
        lines.append(f"- PCA-50 captures **{r['pca50_cum_var']*100:.1f}%** of variance; "
                     f"top PC alone: {top_pc*100:.1f}%.")
        lines.append(f"- HDBSCAN: **{r['hdbscan_clusters']} cluster(s)**, "
                     f"{r['hdbscan_noise_pct']:.1f}% noise, "
                     f"largest cluster {r['hdbscan_largest_pct']:.1f}% of frames.")
        lines.append(f"- {describe(r['silhouette'], r['hdbscan_noise_pct'], r['hdbscan_largest_pct'])}.")
        lines.append(f"- Plot: `results/bullfinch_within_layer{L:02d}.png`")
        lines.append("")

    lines.append("## Overall reading")
    lines.append("")
    best_layer = max(rows, key=lambda r: r["silhouette"])
    worst_layer = min(rows, key=lambda r: r["silhouette"])
    lines.append(
        f"- Strongest within-species structure: **{best_layer['label']}** "
        f"(silhouette {best_layer['silhouette']:+.4f}, k={best_layer['best_k_kmeans']}).")
    lines.append(
        f"- Weakest: **{worst_layer['label']}** "
        f"(silhouette {worst_layer['silhouette']:+.4f}).")
    top_pcs = [(r["label"], r["top_pc_var"]) for r in rows]
    hi_pc = max(top_pcs, key=lambda t: t[1])
    lines.append(
        f"- Highest top-PC concentration: **{hi_pc[0]}** "
        f"({hi_pc[1]*100:.1f}%). Layers where PC1 dominates suggest a single "
        f"salient signal direction (e.g. bio vs non-bio, or a call-type axis) "
        f"rather than many balanced sub-clusters.")
    lines.append("")
    lines.append("Overview plot: `results/bullfinch_within_all_layers.png` "
                 "(silhouette across layers, HDBSCAN mass, k-sweep curves).")
    lines.append("")

    DOC_PATH.write_text("\n".join(lines))
    print(f"[saved] {DOC_PATH}")


if __name__ == "__main__":
    run()
