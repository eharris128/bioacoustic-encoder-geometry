"""
probes/train.py — Per-layer logistic regression probe training.

Trains one logistic regression probe per AVES layer on the (X, y) datasets
produced by data/loader.py. Uses leave-one-recording-out (LORO) cross-
validation to measure per-layer accuracy without data leakage.

Conventions (match existing project scripts):
- PCA to 50 dims before logistic regression (768-dim is slow on CPU)
- Train/test split: LORO (temporal order preserved within each recording)
- Random seed: 42
- Suppress sklearn convergence warnings: python -W ignore probes/train.py
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Single-layer probe
# ---------------------------------------------------------------------------

def train_probe_single_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    pca_components: int = 50,
    random_state: int = 42,
) -> tuple:
    """
    Fit a PCA-reduced logistic regression probe on one layer's training data.

    Parameters
    ----------
    X_train       : (n_train_frames, 768) float32 array
    y_train       : (n_train_frames,) integer labels
    pca_components: number of PCA dims to reduce to before LR (default 50)
    random_state  : random seed for reproducibility

    Returns
    -------
    probe   : fitted LogisticRegression
    scaler  : fitted StandardScaler (applied before PCA)
    pca     : fitted PCA transformer
    """
    ...


# ---------------------------------------------------------------------------
# LORO cross-validation
# ---------------------------------------------------------------------------

def loro_cross_validate(
    dataset: dict[int, tuple[np.ndarray, np.ndarray]],
    recording_ids: list[str],
    frames_per_recording: dict[str, int],
    pca_components: int = 50,
    random_state: int = 42,
) -> dict[int, float]:
    """
    Leave-one-recording-out cross-validation across all 12 layers.

    For each fold: hold out all frames from one recording, train on the rest,
    evaluate on the held-out recording. Average accuracy across folds per layer.

    Parameters
    ----------
    dataset              : { layer: (X, y) } from data.loader.build_dataset
    recording_ids        : ordered list of recording IDs (same order as dataset rows)
    frames_per_recording : { rec_id: n_frames } to reconstruct fold boundaries
    pca_components       : PCA dimensionality before logistic regression
    random_state         : random seed

    Returns
    -------
    accuracy_per_layer : { layer_index: mean_loro_accuracy }
    """
    ...


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train_all_layers(
    dataset: dict[int, tuple[np.ndarray, np.ndarray]],
    recording_ids: list[str],
    frames_per_recording: dict[str, int],
    pca_components: int = 50,
    random_state: int = 42,
) -> dict:
    """
    Run LORO cross-validation across all layers and return per-layer results.

    Parameters
    ----------
    dataset              : { layer: (X, y) } from data.loader.build_dataset
    recording_ids        : ordered list of recording IDs
    frames_per_recording : { rec_id: n_frames }
    pca_components       : PCA dims before LR
    random_state         : random seed

    Returns
    -------
    results : {
        "accuracy_per_layer": { layer: float },   # LORO mean accuracy
        "chance_level": float,                    # 1 / n_classes
        "n_classes": int,
        "n_recordings": int,
    }
    """
    ...
