"""
probes/evaluate.py — Probe evaluation: accuracy curves and LDA projections.

Takes the per-layer accuracy dict from probes/train.py and produces:
  1. Accuracy curve — per-layer probe accuracy vs. layer index (PNG)
  2. LDA projection — 2D LinearDiscriminantAnalysis plot per selected layer (PNG)

Both outputs are saved to results/ with a filename prefix passed by the caller.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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
    ...


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
    layers_to_plot: list of layer indices to include (e.g. [0, 3, 6, 9, 11])
    label_names   : list of class name strings for the legend
    title         : overall figure title
    save_path     : full path to output PNG (e.g. "results/animals_vs_music_lda.png")
    """
    ...


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
    """
    ...
