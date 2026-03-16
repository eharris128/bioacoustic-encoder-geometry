"""Temporal context probing: do later layers encode what's coming next?

For each layer, tests whether a frame's embedding can predict the cluster
identity of frames at various temporal offsets (t+1, t+5, t+10, t+25, t+50).

If layer 11 can predict further into the future than layer 0, the transformer
is building temporal predictions — which is exactly what HuBERT's masked
prediction objective should encourage.

Control: shuffle temporal order to verify we're measuring real sequential
structure, not just acoustic similarity between co-occurring sounds.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

import torch
from aves import load_feature_extractor
from aves.utils import load_audio

# --- Config ---
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
AUDIO_DIR = Path("./audio/bullfinch")
SKIP = {"XC1086809.mp3", "XC657517.mp3"}
NUM_LAYERS = 12
N_CLUSTERS = 8  # Cluster vocabulary size
OFFSETS = [1, 3, 5, 10, 25, 50]  # Frames into the future to predict (~20ms each)
SR = 16000

# --- Load model ---
print("Loading model...")
model = load_feature_extractor(
    config_path=CONFIG_PATH,
    model_path=MODEL_PATH,
    device="cpu",
    for_inference=True,
)

# --- Load recordings and extract all layers ---
audio_files = sorted([f for f in AUDIO_DIR.glob("*.mp3") if f.name not in SKIP])[:12]
print(f"Using {len(audio_files)} recordings\n")

# Store per-recording, per-layer embeddings (preserving temporal order)
recordings = []  # list of {layer_idx: (n_frames, 768)}

for i, path in enumerate(audio_files):
    print(f"  [{i+1:2d}/{len(audio_files)}] {path.stem}...", end=" ", flush=True)
    try:
        audio = load_audio(str(path), mono=True, mono_avg=False)
    except Exception:
        print("SKIP")
        continue

    t0 = time.time()
    layer_outputs = model.extract_features(audio, layers=None)
    elapsed = time.time() - t0

    rec = {}
    for layer_idx, layer_out in enumerate(layer_outputs):
        rec[layer_idx] = layer_out.squeeze(0).cpu().numpy()

    n_frames = rec[0].shape[0]
    print(f"{n_frames} frames, {elapsed:.1f}s")
    recordings.append(rec)

print(f"\nLoaded {len(recordings)} recordings")

# --- Cluster using layer 11 embeddings (our "vocabulary") ---
print(f"\nBuilding cluster vocabulary (k={N_CLUSTERS}) from layer 11...")
all_layer11 = np.concatenate([rec[11] for rec in recordings], axis=0)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
kmeans.fit(all_layer11)

# Assign cluster labels to each frame in each recording
recording_labels = []  # list of arrays, one per recording
for rec in recordings:
    labels = kmeans.predict(rec[11])
    recording_labels.append(labels)

print(f"Cluster distribution: {np.bincount(np.concatenate(recording_labels))}")

# --- Build train/test datasets for temporal prediction ---
# For each offset k: X = embedding of frame t, y = cluster of frame t+k
# Split: first 80% of each recording for train, last 20% for test

def build_dataset(layer_idx, offset, shuffle_control=False):
    """Build (X, y) pairs for predicting cluster at t+offset from embedding at t."""
    X_train, y_train, X_test, y_test = [], [], [], []

    for rec_idx, rec in enumerate(recordings):
        emb = rec[layer_idx]  # (n_frames, 768)
        labels = recording_labels[rec_idx]
        n = len(labels)

        if offset >= n:
            continue

        # Pairs: (frame_t embedding, cluster of frame t+offset)
        X = emb[:n - offset]
        y = labels[offset:]

        if shuffle_control:
            # Shuffle y to break temporal structure (keep marginal distribution)
            rng = np.random.default_rng(42 + rec_idx)
            y = rng.permutation(y)

        # Split 80/20 temporally
        split = int(0.8 * len(X))
        X_train.append(X[:split])
        y_train.append(y[:split])
        X_test.append(X[split:])
        y_test.append(y[split:])

    X_tr = np.concatenate(X_train)
    y_tr = np.concatenate(y_train)
    X_te = np.concatenate(X_test)
    y_te = np.concatenate(y_test)

    # Subsample training data to 10k for speed
    if len(X_tr) > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_tr), 10000, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    return X_tr, y_tr, X_te, y_te


# --- Run probes ---
print(f"\nRunning temporal probes across {NUM_LAYERS} layers x {len(OFFSETS)} offsets...")
results = np.zeros((NUM_LAYERS, len(OFFSETS)))
results_shuffled = np.zeros((NUM_LAYERS, len(OFFSETS)))

for oi, offset in enumerate(OFFSETS):
    offset_ms = offset * 20
    print(f"\n  Offset: t+{offset} ({offset_ms}ms into the future)")

    for layer_idx in range(NUM_LAYERS):
        # Real temporal order
        X_train, y_train, X_test, y_test = build_dataset(layer_idx, offset, shuffle_control=False)

        # PCA to 50 dims — makes logistic regression ~200x faster
        from sklearn.decomposition import PCA as PCAsk
        pca = PCAsk(n_components=50, random_state=42)
        scaler = StandardScaler()
        X_train_s = pca.fit_transform(scaler.fit_transform(X_train))
        X_test_s = pca.transform(scaler.transform(X_test))

        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", n_jobs=-1)
        clf.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_s))
        results[layer_idx, oi] = acc

        # Shuffled control
        X_train_sh, y_train_sh, X_test_sh, y_test_sh = build_dataset(layer_idx, offset, shuffle_control=True)
        X_train_sh_s = pca.fit_transform(scaler.fit_transform(X_train_sh))
        X_test_sh_s = pca.transform(scaler.transform(X_test_sh))

        clf_sh = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", n_jobs=-1)
        clf_sh.fit(X_train_sh_s, y_train_sh)
        acc_sh = accuracy_score(y_test_sh, clf_sh.predict(X_test_sh_s))
        results_shuffled[layer_idx, oi] = acc_sh

    print(f"    Layer  0: {results[0, oi]:.1%} (shuffled: {results_shuffled[0, oi]:.1%})")
    print(f"    Layer 11: {results[11, oi]:.1%} (shuffled: {results_shuffled[11, oi]:.1%})")

# Chance level
chance = 1.0 / N_CLUSTERS
print(f"\nChance level: {chance:.1%}")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle("Temporal Context Probing: Can a frame's embedding predict the future?\n"
             f"(Predict cluster identity at t+k from embedding at t, {N_CLUSTERS} clusters, "
             f"{len(recordings)} Bullfinch recordings)",
             fontsize=14, fontweight="bold")

# 1. Heatmap: accuracy by layer x offset
ax = axes[0]
im = ax.imshow(results, aspect="auto", cmap="YlOrRd", vmin=chance, vmax=results.max())
ax.set_xlabel("Temporal offset", fontsize=12)
ax.set_ylabel("Layer", fontsize=12)
ax.set_title("Prediction Accuracy\n(real temporal order)", fontsize=12)
ax.set_xticks(range(len(OFFSETS)))
ax.set_xticklabels([f"t+{o}\n({o*20}ms)" for o in OFFSETS], fontsize=9)
ax.set_yticks(range(NUM_LAYERS))
plt.colorbar(im, ax=ax, label="Accuracy")

for i in range(NUM_LAYERS):
    for j in range(len(OFFSETS)):
        ax.text(j, i, f"{results[i,j]:.0%}", ha="center", va="center",
                fontsize=7, color="black" if results[i,j] < 0.5 else "white")

# 2. Temporal context advantage: real - shuffled
advantage = results - results_shuffled
ax = axes[1]
im = ax.imshow(advantage, aspect="auto", cmap="RdBu_r", vmin=-0.05,
               vmax=max(0.05, advantage.max()))
ax.set_xlabel("Temporal offset", fontsize=12)
ax.set_ylabel("Layer", fontsize=12)
ax.set_title("Temporal Context Advantage\n(real accuracy - shuffled control)", fontsize=12)
ax.set_xticks(range(len(OFFSETS)))
ax.set_xticklabels([f"t+{o}\n({o*20}ms)" for o in OFFSETS], fontsize=9)
ax.set_yticks(range(NUM_LAYERS))
plt.colorbar(im, ax=ax, label="Advantage (pp)")

for i in range(NUM_LAYERS):
    for j in range(len(OFFSETS)):
        ax.text(j, i, f"{advantage[i,j]:+.1%}", ha="center", va="center",
                fontsize=6, color="black")

# 3. Line plot: accuracy vs offset for selected layers
ax = axes[2]
offset_ms = [o * 20 for o in OFFSETS]
for layer_idx in [0, 3, 6, 9, 11]:
    ax.plot(offset_ms, results[layer_idx], "o-", label=f"Layer {layer_idx}", linewidth=2)
ax.axhline(chance, color="gray", linestyle="--", alpha=0.5, label=f"Chance ({chance:.0%})")
ax.set_xlabel("Temporal offset (ms)", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Future Prediction by Layer\n(how far can each layer see?)", fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, results.max() + 0.05)

plt.tight_layout()
plt.savefig("temporal_context.png", dpi=150, bbox_inches="tight")
print("\nSaved temporal_context.png")
