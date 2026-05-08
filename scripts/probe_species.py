"""Linear probing: at which layer does AVES make species linearly separable?

Trains a logistic regression probe on each layer's frame embeddings to classify
Bullfinch vs Hawfinch. Uses leave-one-recording-out cross-validation so we test
on recordings the probe has never seen.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import torch
from aves import load_feature_extractor
from aves.utils import load_audio

# --- Config ---
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
NUM_LAYERS = 12
MAX_FRAMES_PER_RECORDING = 3000  # Cap long recordings to keep things balanced

RECORDINGS = {
    # Bullfinch (label=0)
    "bullfinch_XC1077468": ("audio/bullfinch/XC1077468.mp3", 0),
    "bullfinch_XC965743": ("audio/bullfinch/XC965743.mp3", 0),
    "bullfinch_XC938052": ("audio/bullfinch/XC938052.mp3", 0),
    "bullfinch_XC805629": ("audio/bullfinch/XC805629.mp3", 0),
    # Skip XC1086809 — it's 35MB, would dominate the dataset
    # Hawfinch (label=1)
    "hawfinch_XC944735": ("audio/hawfinch/XC944735.mp3", 1),
    "hawfinch_XC1087947": ("audio/hawfinch/XC1087947.mp3", 1),
    "hawfinch_XC1086752": ("audio/hawfinch/XC1086752.mp3", 1),
    "hawfinch_XC1084204": ("audio/hawfinch/XC1084204.mp3", 1),
    "hawfinch_XC1083076": ("audio/hawfinch/XC1083076.mp3", 1),
}

# --- Load model ---
print("Loading model...")
model = load_feature_extractor(
    config_path=CONFIG_PATH,
    model_path=MODEL_PATH,
    device="cpu",
    for_inference=True,
)

# --- Extract embeddings from all layers for all recordings ---
print(f"\nExtracting embeddings from {len(RECORDINGS)} recordings...")
# Structure: {recording_id: {layer_idx: np.array(n_frames, 768)}}
all_embeddings = {}
recording_labels = {}  # {recording_id: species_label}
recording_species = {}  # {recording_id: species_name}

for rec_id, (path, label) in RECORDINGS.items():
    species = "Bullfinch" if label == 0 else "Hawfinch"
    print(f"  {rec_id} ({species})...", end=" ", flush=True)

    audio = load_audio(path, mono=True, mono_avg=False)
    t0 = time.time()
    layer_outputs = model.extract_features(audio, layers=None)
    elapsed = time.time() - t0

    embeddings = {}
    for layer_idx, layer_out in enumerate(layer_outputs):
        emb = layer_out.squeeze(0).cpu().numpy()  # (n_frames, 768)
        # Cap frames
        if emb.shape[0] > MAX_FRAMES_PER_RECORDING:
            rng = np.random.default_rng(42)
            idx = rng.choice(emb.shape[0], MAX_FRAMES_PER_RECORDING, replace=False)
            idx.sort()
            emb = emb[idx]
        embeddings[layer_idx] = emb

    n_frames = embeddings[0].shape[0]
    print(f"{n_frames} frames, {elapsed:.1f}s")

    all_embeddings[rec_id] = embeddings
    recording_labels[rec_id] = label
    recording_species[rec_id] = species

# --- Leave-one-recording-out cross-validation per layer ---
rec_ids = list(RECORDINGS.keys())
n_recs = len(rec_ids)

print(f"\nRunning leave-one-recording-out CV ({n_recs} folds) across {NUM_LAYERS} layers...")

# Results: accuracy per layer per fold
layer_accuracies = np.zeros((NUM_LAYERS, n_recs))
layer_mean_acc = np.zeros(NUM_LAYERS)
layer_std_acc = np.zeros(NUM_LAYERS)

for layer_idx in range(NUM_LAYERS):
    fold_accs = []

    for fold, test_rec in enumerate(rec_ids):
        # Train on all other recordings
        train_recs = [r for r in rec_ids if r != test_rec]

        X_train = np.concatenate([all_embeddings[r][layer_idx] for r in train_recs])
        y_train = np.concatenate([
            np.full(all_embeddings[r][layer_idx].shape[0], recording_labels[r])
            for r in train_recs
        ])

        X_test = all_embeddings[test_rec][layer_idx]
        y_test = np.full(X_test.shape[0], recording_labels[test_rec])

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train logistic regression
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf.fit(X_train, y_train)

        acc = accuracy_score(y_test, clf.predict(X_test))
        layer_accuracies[layer_idx, fold] = acc
        fold_accs.append(acc)

    layer_mean_acc[layer_idx] = np.mean(fold_accs)
    layer_std_acc[layer_idx] = np.std(fold_accs)
    print(f"  Layer {layer_idx:2d}: {layer_mean_acc[layer_idx]:.1%} ± {layer_std_acc[layer_idx]:.1%}")

# --- Plot results ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Species Probe: Bullfinch vs Hawfinch\n"
             "(Linear probe accuracy per AVES layer, leave-one-recording-out CV)",
             fontsize=14, fontweight="bold")

# Bar chart with error bars
colors = plt.cm.viridis(np.linspace(0.2, 0.9, NUM_LAYERS))
ax1.bar(range(NUM_LAYERS), layer_mean_acc, yerr=layer_std_acc,
        color=colors, edgecolor="black", linewidth=0.5, capsize=3)
ax1.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Chance (50%)")
ax1.set_xlabel("Layer", fontsize=12)
ax1.set_ylabel("Accuracy", fontsize=12)
ax1.set_title("Mean Accuracy per Layer", fontsize=12)
ax1.set_xticks(range(NUM_LAYERS))
ax1.set_ylim(0.4, 1.05)
ax1.legend(fontsize=10)

# Heatmap of per-fold accuracies
im = ax2.imshow(layer_accuracies.T, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
ax2.set_xlabel("Layer", fontsize=12)
ax2.set_ylabel("Test Recording", fontsize=12)
ax2.set_title("Per-Fold Accuracy (each row = held-out recording)", fontsize=12)
ax2.set_xticks(range(NUM_LAYERS))
ax2.set_yticks(range(n_recs))
ax2.set_yticklabels([f"{recording_species[r]}\n{r.split('_')[1]}" for r in rec_ids], fontsize=8)
plt.colorbar(im, ax=ax2, label="Accuracy")

# Add accuracy text to heatmap cells
for i in range(NUM_LAYERS):
    for j in range(n_recs):
        ax2.text(i, j, f"{layer_accuracies[i,j]:.0%}",
                 ha="center", va="center", fontsize=7,
                 color="black" if layer_accuracies[i,j] > 0.6 else "white")

plt.tight_layout()
plt.savefig("probe_species.png", dpi=150, bbox_inches="tight")
print(f"\nSaved probe_species.png")

# --- Summary ---
best_layer = np.argmax(layer_mean_acc)
print(f"\nBest layer: {best_layer} ({layer_mean_acc[best_layer]:.1%})")
print(f"Worst layer: {np.argmin(layer_mean_acc)} ({layer_mean_acc[np.argmin(layer_mean_acc)]:.1%})")
