"""Within-species embedding structure: how do 30 Bullfinch recordings organize
across layers?

For each layer, extracts frame embeddings from all 30 recordings, projects to 2D
with PCA, and colors each recording differently. Shows whether the model groups
frames by individual/recording or by something else (call type, spectral quality).
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from pathlib import Path
from sklearn.decomposition import PCA

import torch
from aves import load_feature_extractor
from aves.utils import load_audio

# --- Config ---
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
AUDIO_DIR = Path("./audio/bullfinch")
NUM_LAYERS = 12
MAX_FRAMES_PER_RECORDING = 500  # Keep manageable with 30 recordings
SR = 16000

# Skip the 35MB file — it would dominate everything
SKIP = {"XC1086809.mp3"}

# --- Load model ---
print("Loading model...")
model = load_feature_extractor(
    config_path=CONFIG_PATH,
    model_path=MODEL_PATH,
    device="cpu",
    for_inference=True,
)

# --- Load all recordings ---
audio_files = sorted([f for f in AUDIO_DIR.glob("*.mp3") if f.name not in SKIP])
print(f"\nFound {len(audio_files)} recordings (skipping {SKIP})")

all_embeddings = {}  # {filename: list of 12 arrays, each (n_frames, 768)}
recording_names = []

for i, path in enumerate(audio_files):
    name = path.stem
    print(f"  [{i+1:2d}/{len(audio_files)}] {name}...", end=" ", flush=True)

    try:
        audio = load_audio(str(path), mono=True, mono_avg=False)
    except Exception as e:
        print(f"SKIP (decode error)")
        continue

    t0 = time.time()
    layer_outputs = model.extract_features(audio, layers=None)
    elapsed = time.time() - t0

    embeddings = []
    for layer_out in layer_outputs:
        emb = layer_out.squeeze(0).cpu().numpy()
        if emb.shape[0] > MAX_FRAMES_PER_RECORDING:
            rng = np.random.default_rng(42)
            idx = rng.choice(emb.shape[0], MAX_FRAMES_PER_RECORDING, replace=False)
            idx.sort()
            emb = emb[idx]
        embeddings.append(emb)

    n_frames = embeddings[0].shape[0]
    print(f"{n_frames} frames, {elapsed:.1f}s")

    all_embeddings[name] = embeddings
    recording_names.append(name)

n_recordings = len(recording_names)
print(f"\nTotal: {n_recordings} recordings loaded")

# --- Generate distinct colors for each recording ---
colors = []
for i in range(n_recordings):
    hue = i / n_recordings
    colors.append(hsv_to_rgb([hue, 0.7, 0.9]))

# --- PCA per layer ---
fig, axes = plt.subplots(3, 4, figsize=(24, 18))
fig.suptitle(f"Within-Species Structure: {n_recordings} Bullfinch Recordings Across Layers\n"
             f"(PCA → 2D, each color = different recording, {MAX_FRAMES_PER_RECORDING} frames max each)",
             fontsize=15, fontweight="bold")

# Also track: how much does recording identity explain variance at each layer?
# Use silhouette-like metric: mean intra-recording distance vs mean inter-recording distance
from sklearn.metrics import silhouette_score

sil_scores = []

for layer_idx in range(NUM_LAYERS):
    ax = axes[layer_idx // 4, layer_idx % 4]

    # Concatenate all recordings
    all_frames = []
    all_labels = []
    for i, name in enumerate(recording_names):
        frames = all_embeddings[name][layer_idx]
        all_frames.append(frames)
        all_labels.extend([i] * len(frames))

    combined = np.concatenate(all_frames, axis=0)
    labels_arr = np.array(all_labels)

    # PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined)
    var_explained = pca.explained_variance_ratio_

    # Silhouette score (how well do recordings separate?)
    # Subsample for speed if needed
    if len(labels_arr) > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(labels_arr), 5000, replace=False)
        sil = silhouette_score(combined[idx], labels_arr[idx], sample_size=None)
    else:
        sil = silhouette_score(combined, labels_arr)
    sil_scores.append(sil)

    # Plot each recording
    offset = 0
    for i, name in enumerate(recording_names):
        n = len(all_embeddings[name][layer_idx])
        ax.scatter(
            coords[offset:offset+n, 0],
            coords[offset:offset+n, 1],
            c=[colors[i]],
            alpha=0.3,
            s=3,
            rasterized=True,
        )
        offset += n

    ax.set_title(f"Layer {layer_idx}  (sil={sil:.3f})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=8)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig("individuals_pca.png", dpi=150, bbox_inches="tight")
print("Saved individuals_pca.png")

# --- Silhouette score across layers ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(NUM_LAYERS), sil_scores, color=plt.cm.viridis(np.linspace(0.2, 0.9, NUM_LAYERS)),
       edgecolor="black", linewidth=0.5)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Silhouette Score", fontsize=12)
ax.set_title("Recording Separability Across Layers\n"
             "(higher = frames from same recording cluster together more tightly)",
             fontsize=13, fontweight="bold")
ax.set_xticks(range(NUM_LAYERS))
ax.axhline(0, color="red", linestyle="--", alpha=0.5, label="No structure")
ax.legend()

for i, s in enumerate(sil_scores):
    ax.text(i, s + 0.005, f"{s:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("individuals_silhouette.png", dpi=150, bbox_inches="tight")
print("Saved individuals_silhouette.png")

print(f"\nSilhouette scores by layer:")
for i, s in enumerate(sil_scores):
    print(f"  Layer {i:2d}: {s:.4f}")
print(f"\nBest: Layer {np.argmax(sil_scores)} ({max(sil_scores):.4f})")
print(f"Worst: Layer {np.argmin(sil_scores)} ({min(sil_scores):.4f})")
