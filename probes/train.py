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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=pca_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    probe = LogisticRegression(max_iter=1000, random_state=random_state)
    probe.fit(X_pca, y_train)

    return probe, scaler, pca


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
    # Build cumulative frame offsets so we can slice out each recording's rows
    # from the stacked X/y arrays produced by build_dataset.
    offsets: list[int] = []
    cursor = 0
    for rec_id in recording_ids:
        offsets.append(cursor)
        cursor += frames_per_recording[rec_id]

    accuracy_per_layer: dict[int, float] = {}

    for layer, (X, y) in dataset.items():
        fold_accuracies: list[float] = []

        for i, rec_id in enumerate(recording_ids):
            start = offsets[i]
            end   = start + frames_per_recording[rec_id]

            # Hold out recording i; train on everything else
            X_test  = X[start:end]
            y_test  = y[start:end]
            X_train = np.concatenate([X[:start], X[end:]], axis=0)
            y_train = np.concatenate([y[:start], y[end:]], axis=0)

            probe, scaler, pca = train_probe_single_layer(
                X_train, y_train,
                pca_components=pca_components,
                random_state=random_state,
            )

            # Apply the same scaler and PCA fitted on train to the test set
            X_test_scaled = scaler.transform(X_test)
            X_test_pca    = pca.transform(X_test_scaled)

            fold_accuracies.append(probe.score(X_test_pca, y_test))

        accuracy_per_layer[layer] = float(np.mean(fold_accuracies))

    return accuracy_per_layer


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
    # Infer n_classes from the label set of any layer (all layers share the same y)
    any_y = next(iter(dataset.values()))[1]
    n_classes = int(len(np.unique(any_y)))
    chance_level = 1.0 / n_classes

    print(
        f"train_all_layers: {len(recording_ids)} recordings, "
        f"{n_classes} classes, chance={chance_level:.3f}",
        flush=True,
    )

    accuracy_per_layer = loro_cross_validate(
        dataset=dataset,
        recording_ids=recording_ids,
        frames_per_recording=frames_per_recording,
        pca_components=pca_components,
        random_state=random_state,
    )

    # Print a quick per-layer accuracy table for immediate inspection
    print(f"\n{'Layer':>6}  {'Accuracy':>9}  {'vs chance':>10}")
    print("-" * 30)
    for layer in sorted(accuracy_per_layer):
        acc = accuracy_per_layer[layer]
        delta = acc - chance_level
        label = "embedding" if layer == 0 else f"layer {layer:>2}"
        print(f"{label:>9}  {acc:>8.1%}  {delta:>+9.1%}")

    return {
        "accuracy_per_layer": accuracy_per_layer,
        "chance_level": chance_level,
        "n_classes": n_classes,
        "n_recordings": len(recording_ids),
    }
