"""
scripts/plot_phylogenetic_gradient.py — Publication-quality phylogenetic gradient figure.

Numerical data is hardcoded from LORO cross-validation results stored in this session.
evaluate.py saves only PNGs (no JSON outputs), so accuracy values are embedded here.
Phylogenetic distances from TimeTree + literature.

Usage:
    python -W ignore scripts/plot_phylogenetic_gradient.py

Outputs:
    results/phylogenetic_gradient.png  (300 dpi)
    results/phylogenetic_gradient.pdf
    results/phylogenetic_gradient_data.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Probe accuracy data (hardcoded — per-layer LORO cross-validation results)
# Layer order: index 0 = emb, 1 = T0, 2 = T1, ..., 12 = T11
# ---------------------------------------------------------------------------

PROBE_ACCURACY: dict[str, dict] = {
    "great_tit_vs_great_tit_bokharensis": {
        "label": "Gt. Tit/Bokharensis*",
        "acc": [81.8, 86.4, 89.6, 88.3, 92.2, 90.3, 89.0, 89.0, 89.6, 90.3, 89.6, 86.4, 90.9],
        # * anomalous high embedding — likely sample bias (only 54 bokharensis recordings)
    },
    "common_chiffchaff_vs_iberian_chiffchaff": {
        "label": "Com./Iberian Chiffchaff",
        "acc": [62.5, 72.0, 82.0, 84.5, 82.0, 82.5, 84.5, 83.5, 83.0, 86.5, 89.0, 86.5, 91.5],
    },
    "willow_warbler_vs_chiffchaff": {
        "label": "Willow Warbler/Chiffchaff",
        "acc": [53.0, 71.5, 83.5, 91.5, 87.5, 87.0, 89.0, 93.0, 91.0, 91.0, 91.5, 92.0, 90.5],
    },
    "house_sparrow_vs_tree_sparrow": {
        "label": "House/Tree Sparrow",
        "acc": [53.3, 74.9, 79.4, 83.9, 82.9, 81.9, 80.9, 85.4, 82.4, 81.9, 79.9, 81.9, 83.9],
    },
    "goldfinch_vs_eurasian_siskin": {
        "label": "Goldfinch/Siskin",
        "acc": [65.5, 78.0, 88.0, 92.0, 91.0, 89.0, 92.5, 92.0, 89.5, 90.5, 92.0, 90.5, 85.0],
    },
    "house_crow_vs_carrion_crow": {
        "label": "House/Carrion Crow",
        "acc": [72.0, 84.0, 85.0, 91.5, 93.5, 92.5, 90.5, 89.5, 93.5, 93.0, 95.0, 90.5, 87.5],
    },
    "bullfinch_vs_hawfinch": {
        "label": "Bullfinch/Hawfinch",
        "acc": [61.0, 87.0, 90.0, 95.0, 91.0, 89.0, 89.0, 92.0, 92.0, 91.0, 92.0, 94.0, 94.0],
    },
    "european_robin_vs_eurasian_blackbird": {
        "label": "Robin/Blackbird",
        "acc": [67.0, 92.5, 98.0, 96.0, 97.0, 94.5, 95.5, 96.5, 98.0, 99.0, 97.5, 97.5, 99.0],
    },
    "house_sparrow_vs_common_swift": {
        "label": "Sparrow/Swift",
        "acc": [69.5, 92.0, 96.5, 96.5, 96.0, 95.0, 98.0, 98.0, 98.0, 97.5, 96.5, 97.5, 97.5],
    },
    "bullfinch_vs_tawny_owl": {
        "label": "Bullfinch/Tawny Owl",
        "acc": [65.5, 87.0, 95.5, 97.0, 99.0, 96.5, 98.0, 98.5, 98.0, 98.5, 99.0, 98.0, 96.0],
    },
    "chaffinch_vs_great_spotted_woodpecker": {
        "label": "Chaffinch/Woodpecker",
        "acc": [59.1, 87.9, 91.9, 95.5, 96.0, 96.5, 94.9, 93.9, 94.9, 95.5, 97.0, 96.5, 95.5],
    },
}

# ---------------------------------------------------------------------------
# Phylogenetic distances (TimeTree + literature, MYA)
# ---------------------------------------------------------------------------

PHYLO_DATA: dict[str, dict] = {
    "great_tit_vs_great_tit_bokharensis":         {"mya": 0.5,  "level": "Subspecies"},
    "common_chiffchaff_vs_iberian_chiffchaff":    {"mya": 2.0,  "level": "Same genus"},
    "willow_warbler_vs_chiffchaff":               {"mya": 2.0,  "level": "Same genus"},
    "house_sparrow_vs_tree_sparrow":              {"mya": 4.4,  "level": "Same genus"},
    "goldfinch_vs_eurasian_siskin":               {"mya": 6.0,  "level": "Same family"},
    "house_crow_vs_carrion_crow":                 {"mya": 10.0, "level": "Same genus"},
    "bullfinch_vs_hawfinch":                      {"mya": 12.5, "level": "Same family"},
    "european_robin_vs_eurasian_blackbird":       {"mya": 17.0, "level": "Diff. families"},
    "house_sparrow_vs_common_swift":              {"mya": 65.0, "level": "Diff. orders"},
    "bullfinch_vs_tawny_owl":                     {"mya": 65.0, "level": "Diff. orders"},
    "chaffinch_vs_great_spotted_woodpecker":      {"mya": 70.0, "level": "Diff. orders"},
}

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

DARK_TEAL    = "#1A4A5C"
ACCENT_ORANGE = "#E8741A"
GRAY         = "#8C9BA8"

LEVEL_COLORS = {
    "Subspecies":      "#2ECC71",
    "Same genus":      "#3498DB",
    "Same family":     "#9B59B6",
    "Diff. families":  "#E67E22",
    "Diff. orders":    "#E74C3C",
}

LAYER_LABELS = ["emb", "T0", "T1", "T2", "T3", "T4",
                "T5", "T6", "T7", "T8", "T9", "T10", "T11"]


# ---------------------------------------------------------------------------
# Build merged dataset
# ---------------------------------------------------------------------------

def build_records() -> list[dict]:
    records = []
    for key, phylo in PHYLO_DATA.items():
        probe = PROBE_ACCURACY[key]
        acc = probe["acc"]
        peak_layer = int(np.argmax(acc))
        peak_accuracy = float(acc[peak_layer])
        records.append({
            "pair":            key,
            "label":           probe["label"],
            "mya":             phylo["mya"],
            "level":           phylo["level"],
            "peak_accuracy":   peak_accuracy,
            "peak_layer":      peak_layer,
            "acc":             acc,
        })
    return records


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

def save_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["pair", "mya", "taxonomic_level", "peak_accuracy", "peak_layer"]
        + ["acc_emb"] + [f"acc_{lbl}" for lbl in LAYER_LABELS[1:]]
    )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {
                "pair":            r["pair"],
                "mya":             r["mya"],
                "taxonomic_level": r["level"],
                "peak_accuracy":   r["peak_accuracy"],
                "peak_layer":      r["peak_layer"],
            }
            for i, lbl in enumerate(LAYER_LABELS):
                col = "acc_emb" if i == 0 else f"acc_{lbl}"
                row[col] = r["acc"][i]
            writer.writerow(row)
    print(f"Saved CSV → {path}")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _add_trend(ax, x_log: np.ndarray, y: np.ndarray) -> None:
    """Fit and draw a linear trend on log-transformed x."""
    coeffs = np.polyfit(x_log, y, 1)
    x_fit = np.linspace(x_log.min(), x_log.max(), 200)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(
        10 ** x_fit, y_fit,
        linestyle="--", color=DARK_TEAL, linewidth=1.5, alpha=0.55,
        zorder=2, label=f"trend  (slope {coeffs[0]:+.2f}/log₁₀ MYA)",
    )


def _style_ax(ax, xlabel: str, title: str, subtitle: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(DARK_TEAL)
    ax.spines["bottom"].set_color(DARK_TEAL)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=11, color=DARK_TEAL, fontweight="bold", labelpad=6)
    ax.tick_params(colors=DARK_TEAL, labelsize=9)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=13, color=DARK_TEAL, fontweight="bold", pad=10)
    ax.text(
        0.5, 1.01, subtitle,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8, color=GRAY, style="italic",
    )


def _label_points(ax, xs, ys, labels, x_log):
    """Place text labels with simple heuristic offsets to reduce overlap."""
    # Manual per-point offsets (dx in log-data fraction, dy in data units)
    # Positive dx = right, negative = left; positive dy = up, negative = down
    OFFSETS = {
        "Gt. Tit/Bokharensis*":       (-0.25,  1.8),
        "Com./Iberian Chiffchaff":    ( 0.08, -2.5),
        "Willow Warbler/Chiffchaff":  ( 0.08,  1.5),
        "House/Tree Sparrow":         ( 0.08, -2.2),
        "Goldfinch/Siskin":           ( 0.08,  1.5),
        "House/Carrion Crow":         (-0.35,  1.8),
        "Bullfinch/Hawfinch":         ( 0.08,  1.5),
        "Robin/Blackbird":            ( 0.08, -2.2),
        "Sparrow/Swift":              (-0.35,  1.8),
        "Bullfinch/Tawny Owl":        ( 0.08, -2.5),
        "Chaffinch/Woodpecker":       ( 0.08,  1.5),
    }
    for x, y, lbl in zip(xs, ys, labels):
        dx_log, dy = OFFSETS.get(lbl, (0.08, 1.5))
        ax.annotate(
            lbl,
            xy=(x, y),
            xytext=(x * (10 ** dx_log), y + dy),
            fontsize=7.5, color=DARK_TEAL,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6),
            va="center",
        )


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure(records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    mya     = np.array([r["mya"]           for r in records])
    peak_ac = np.array([r["peak_accuracy"] for r in records])
    peak_ly = np.array([r["peak_layer"]    for r in records])
    labels  = [r["label"] for r in records]
    colors  = [LEVEL_COLORS[r["level"]] for r in records]
    log_mya = np.log10(mya)

    # ── Panel 1: Peak Accuracy vs MYA ──────────────────────────────────────
    _style_ax(
        ax1,
        xlabel="Phylogenetic Distance (MYA)",
        title="Peak Probe Accuracy vs Phylogenetic Distance",
        subtitle="Each point = one species pair",
    )
    ax1.set_ylabel("Peak LORO Accuracy (%)", fontsize=11, color=DARK_TEAL, fontweight="bold")
    ax1.set_ylim(50, 105)

    ax1.axhline(50, color=GRAY, linewidth=1, linestyle="--", alpha=0.6, zorder=1)
    ax1.text(0.6, 51.5, "Chance (50%)", fontsize=7.5, color=GRAY, transform=ax1.get_xaxis_transform())

    ax1.scatter(mya, peak_ac, c=colors, s=120, zorder=4, edgecolors="white", linewidths=0.8)
    _add_trend(ax1, log_mya, peak_ac)
    _label_points(ax1, mya, peak_ac, labels, log_mya)

    # ── Panel 2: Peak Layer vs MYA ──────────────────────────────────────────
    _style_ax(
        ax2,
        xlabel="Phylogenetic Distance (MYA)",
        title="Peak Layer Depth vs Phylogenetic Distance",
        subtitle="Deeper layer = more processing needed to separate species",
    )
    ax2.set_ylabel("Peak Layer Index", fontsize=11, color=DARK_TEAL, fontweight="bold")
    ax2.set_ylim(-0.5, 12.5)
    ax2.set_yticks(range(13))
    ax2.set_yticklabels(LAYER_LABELS, fontsize=8.5, color=DARK_TEAL)

    ax2.scatter(mya, peak_ly, c=colors, s=120, zorder=4, edgecolors="white", linewidths=0.8)
    _add_trend(ax2, log_mya, peak_ly)
    _label_points(ax2, mya, peak_ly, labels, log_mya)

    # ── Legend ──────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=col, label=lvl, edgecolor="white")
        for lvl, col in LEVEL_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        title="Taxonomic level",
        title_fontsize=9,
        fontsize=8.5,
        loc="center right",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        framealpha=0.9,
        edgecolor="#E0E0E0",
    )

    # ── Overall title ────────────────────────────────────────────────────────
    fig.suptitle(
        "Probe Accuracy Gradients Track Phylogenetic Distance",
        fontsize=16, fontweight="bold", color=DARK_TEAL, y=1.02,
    )

    # ── Footer ───────────────────────────────────────────────────────────────
    fig.text(
        0.5, -0.02,
        "Linear probes on AVEX ESP bioacoustic transformer (13 layers). "
        "LORO cross-validation. Phylogenetic distances from TimeTree.",
        fontsize=7, color=GRAY, ha="center",
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    png_path = out_dir / "phylogenetic_gradient.png"
    pdf_path = out_dir / "phylogenetic_gradient.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Saved PNG → {png_path}")
    print(f"Saved PDF → {pdf_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"

    records = build_records()

    print("=" * 50)
    print(f"{'Pair':<45} {'MYA':>6}  {'PeakAcc':>8}  {'PeakLayer':>10}")
    print("-" * 50)
    for r in sorted(records, key=lambda x: x["mya"]):
        print(f"{r['pair']:<45} {r['mya']:>6.1f}  {r['peak_accuracy']:>7.1f}%  "
              f"{LAYER_LABELS[r['peak_layer']]:>10}")
    print("=" * 50)

    save_csv(records, results_dir / "phylogenetic_gradient_data.csv")
    make_figure(records, results_dir)
