"""
probes/evaluate.py — Probe evaluation: accuracy curves and LDA projections.

Takes the per-layer accuracy dict from probes/train.py and produces:
  1. Accuracy curve — per-layer probe accuracy vs. layer index (PNG)
  2. LDA projection — 2D LinearDiscriminantAnalysis plot per selected layer (PNG)

Both outputs are saved to results/ with a filename prefix passed by the caller.

Layer index convention (matches data/loader.py):
  0      = CNN feature_projection output (embedding)
  1–12   = transformer layers 0–11
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Consistent colors for up to 8 classes; extend if needed
_CLASS_COLORS = [
    "#E53935",  # red
    "#1E88E5",  # blue
    "#43A047",  # green
    "#FB8C00",  # orange
    "#8E24AA",  # purple
    "#00ACC1",  # cyan
    "#F4511E",  # deep orange
    "#6D4C41",  # brown
]

# Default layers shown in the LDA panel (embedding + 4 transformer checkpoints)
DEFAULT_LDA_LAYERS = [0, 3, 6, 9, 12]


def _layer_label(layer: int) -> str:
    """Human-readable x-axis tick: 'emb' for index 0, 'T0'..'T11' for 1-12."""
    if layer == 0:
        return "emb"
    return f"T{layer - 1}"  


# ---------------------------------------------------------------------------
# Accuracy curve
# ---------------------------------------------------------------------------

def plot_accuracy_curve(
    accuracy_per_layer: dict[int, float],
    chance_level: float,
    label_names: list[str],
    title: str,
    save_path: str,
) -> None:
    """
    Plot per-layer LORO probe accuracy as a line chart and save to disk.

    Draws a dashed horizontal line at chance level for reference.
    Marks the peak layer with a gold dot.

    Parameters
    ----------
    accuracy_per_layer : { layer_index: accuracy } from probes/train.py
    chance_level       : 1 / n_classes, drawn as a reference line
    label_names        : list of class name strings (for the subtitle)
    title              : plot title (e.g. "Animals vs Music — AVES layer probe")
    save_path          : full path to output PNG (e.g. "results/animals_vs_music_accuracy.png")
    """
    layers = sorted(accuracy_per_layer)
    accs   = [accuracy_per_layer[l] for l in layers]
    labels = [_layer_label(l) for l in layers]

    peak_idx = int(np.argmax(accs))

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(layers, accs, "o-", color="#1976D2", linewidth=2, markersize=6, zorder=3)

    # Gold star at peak
    ax.scatter(
        [layers[peak_idx]], [accs[peak_idx]],
        color="gold", edgecolors="#333", s=120, zorder=4,
        label=f"Peak: {_layer_label(layers[peak_idx])} ({accs[peak_idx]:.1%})",
    )

    # Chance reference
    ax.axhline(
        chance_level, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
        label=f"Chance ({chance_level:.0%})",
    )

    ax.set_xticks(layers)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("AVES Layer", fontsize=11)
    ax.set_ylabel("LORO Accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"{title}\nClasses: {' vs '.join(label_names)}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy curve → {save_path}")


# ---------------------------------------------------------------------------
# LDA projection
# ---------------------------------------------------------------------------

def plot_lda_projection(
    dataset: dict[int, tuple[np.ndarray, np.ndarray]],
    layers_to_plot: list[int],
    label_names: list[str],
    title: str,
    save_path: str,
) -> None:
    """
    Fit LDA on each selected layer and plot the 2D discriminant projection.

    One subplot per layer. Points colored by class. Useful for visualizing
    whether class separation is linear and how it evolves across layers.

    Parameters
    ----------
    dataset       : { layer: (X, y) } from data.loader.build_dataset
    layers_to_plot: list of layer indices to include (e.g. [0, 3, 6, 9, 12])
    label_names   : list of class name strings for the legend
    title         : overall figure title
    save_path     : full path to output PNG (e.g. "results/animals_vs_music_lda.png")
    """
    n_panels = len(layers_to_plot)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, layer in zip(axes, layers_to_plot):
        X, y = dataset[layer]

        # Standardize before LDA (same convention as the probes)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_components = min(2, len(label_names) - 1)
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        coords = lda.fit_transform(X_scaled, y)

        for class_idx, name in enumerate(label_names):
            mask = y == class_idx
            color = _CLASS_COLORS[class_idx % len(_CLASS_COLORS)]
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1] if n_components > 1 else np.zeros(mask.sum()),
                c=color, alpha=0.3, s=3, label=name, rasterized=True,
            )

        ax.set_title(_layer_label(layer), fontsize=10, fontweight="bold")
        ax.set_xlabel("LD1", fontsize=8)
        ax.set_ylabel("LD2" if n_components > 1 else "", fontsize=8)
        ax.tick_params(labelsize=7)

    # Single legend on the last panel
    axes[-1].legend(fontsize=8, markerscale=4, loc="best")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved LDA projection → {save_path}")


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

def run_evaluation(
    accuracy_per_layer: dict[int, float],
    dataset: dict[int, tuple[np.ndarray, np.ndarray]],
    chance_level: float,
    label_names: list[str],
    experiment_name: str,
    results_dir: str = "results",
    lda_layers: list[int] | None = None,
) -> None:
    """
    Run the full evaluation suite for one experiment and save all outputs.

    Calls plot_accuracy_curve and plot_lda_projection, then prints a
    summary table of per-layer accuracy to stdout.

    Parameters
    ----------
    accuracy_per_layer : { layer: accuracy } from probes/train.py
    dataset            : { layer: (X, y) } from data.loader.build_dataset
    chance_level       : 1 / n_classes
    label_names        : class name strings
    experiment_name    : used as filename prefix, e.g. "animals_vs_music"
    results_dir        : directory to write PNGs into (default "results/")
    lda_layers         : which layers to panel in the LDA plot
                         (default: [0, 3, 6, 9, 12], clipped to available layers)
    """
    os.makedirs(results_dir, exist_ok=True)

    # Clip requested LDA layers to whatever is actually in the dataset
    available = set(dataset.keys())
    if lda_layers is None:
        lda_layers = [l for l in DEFAULT_LDA_LAYERS if l in available]
    else:
        lda_layers = [l for l in lda_layers if l in available]

    title = experiment_name.replace("_", " ").title()

    plot_accuracy_curve(
        accuracy_per_layer=accuracy_per_layer,
        chance_level=chance_level,
        label_names=label_names,
        title=f"{title} — AVES layer probe",
        save_path=os.path.join(results_dir, f"{experiment_name}_accuracy.png"),
    )

    plot_lda_projection(
        dataset=dataset,
        layers_to_plot=lda_layers,
        label_names=label_names,
        title=f"{title} — LDA projection",
        save_path=os.path.join(results_dir, f"{experiment_name}_lda.png"),
    )

    # Summary table to stdout
    layers = sorted(accuracy_per_layer)
    peak_layer = max(layers, key=lambda l: accuracy_per_layer[l])
    print(f"\n{'':=<45}")
    print(f"  {title} — results summary")
    print(f"{'':=<45}")
    print(f"  Classes : {' vs '.join(label_names)}")
    print(f"  Chance  : {chance_level:.1%}")
    print(f"  Peak    : {_layer_label(peak_layer)} ({accuracy_per_layer[peak_layer]:.1%})")
    print(f"{'':─<45}")
    print(f"  {'Layer':>6}  {'Accuracy':>9}  {'vs chance':>10}")
    for l in layers:
        acc = accuracy_per_layer[l]
        print(f"  {_layer_label(l):>6}  {acc:>8.1%}  {acc - chance_level:>+9.1%}")
    print(f"{'':=<45}\n")
