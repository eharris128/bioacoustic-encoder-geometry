"""Exploratory: how does AVES represent piano music vs bird vocalizations?

Extracts frame-level embeddings from all 12 layers for multiple piano recordings
and bird audio, reduces to 2D with PCA, and plots each layer colored by source.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from aves import load_feature_extractor
from aves.utils import load_audio

# --- Config ---
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
BIRD_FILES = {
    "Bullfinch": "./aves/example_audios/XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.mp3",
}
PIANO_FILES = [
    "/home/evan/Downloads/paulyudin-piano-music-piano-485929.mp3",
    "/home/evan/Downloads/paulyudin-piano-music-piano-485929(1).mp3",
    "/home/evan/Downloads/sigmamusicart-piano-music-504007.mp3",
    "/home/evan/Downloads/solarflex-emotional-piano-music-499244.mp3",
    "/home/evan/Downloads/the_mountain-piano-background-music-487020.mp3",
]
NUM_LAYERS = 12
MAX_FRAMES_PER_SOURCE = 2000

# --- Load model ---
print("Loading model...")
model = load_feature_extractor(
    config_path=CONFIG_PATH,
    model_path=MODEL_PATH,
    device="cpu",
    for_inference=True,
)
print("Model loaded.\n")

# --- Extract embeddings from all layers ---
# Bird embeddings
bird_embeddings = []  # list of 12 arrays
for name, path in BIRD_FILES.items():
    print(f"Processing {name}...")
    audio = load_audio(path, mono=True, mono_avg=False)
    print(f"  Audio shape: {audio.shape} ({audio.shape[-1]/16000:.1f}s at 16kHz)")
    t0 = time.time()
    layer_outputs = model.extract_features(audio, layers=None)
    print(f"  Extracted {len(layer_outputs)} layers in {time.time()-t0:.1f}s")
    bird_embeddings = [layer.squeeze(0).cpu().numpy() for layer in layer_outputs]
    print(f"  Frames per layer: {bird_embeddings[0].shape[0]}")

# Piano embeddings — pool all files together
piano_layer_frames = [[] for _ in range(NUM_LAYERS)]
for i, path in enumerate(PIANO_FILES):
    short_name = path.split("/")[-1]
    print(f"Processing Piano {i+1}/{len(PIANO_FILES)}: {short_name}...")
    audio = load_audio(path, mono=True, mono_avg=False)
    print(f"  Audio shape: {audio.shape} ({audio.shape[-1]/16000:.1f}s at 16kHz)")
    t0 = time.time()
    layer_outputs = model.extract_features(audio, layers=None)
    print(f"  Extracted {len(layer_outputs)} layers in {time.time()-t0:.1f}s")
    for layer_idx in range(NUM_LAYERS):
        piano_layer_frames[layer_idx].append(layer_outputs[layer_idx].squeeze(0).cpu().numpy())

piano_embeddings = [np.concatenate(frames, axis=0) for frames in piano_layer_frames]
print(f"Total piano frames per layer: {piano_embeddings[0].shape[0]}")

# Subsample both sources
rng = np.random.default_rng(42)
for label, embs in [("Bird", bird_embeddings), ("Piano", piano_embeddings)]:
    n = embs[0].shape[0]
    if n > MAX_FRAMES_PER_SOURCE:
        idx = rng.choice(n, MAX_FRAMES_PER_SOURCE, replace=False)
        idx.sort()
        for i in range(NUM_LAYERS):
            embs[i] = embs[i][idx]
        print(f"Subsampled {label} to {MAX_FRAMES_PER_SOURCE} frames")

all_embeddings = {"Bullfinch": bird_embeddings, "Piano": piano_embeddings}

# --- PCA + plot for each layer ---
source_names = list(all_embeddings.keys())
colors = {"Bullfinch": "#FF5722", "Piano": "#9C27B0"}

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("AVES Layer Representations: Bullfinch vs Piano (5 tracks)\n(PCA → 2D, each dot = one 20ms frame)",
             fontsize=14, fontweight="bold")

for layer_idx in range(NUM_LAYERS):
    ax = axes[layer_idx // 4, layer_idx % 4]

    source_frames = []
    source_labels = []
    for source in source_names:
        frames = all_embeddings[source][layer_idx]
        source_frames.append(frames)
        source_labels.extend([source] * len(frames))

    combined = np.concatenate(source_frames, axis=0)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined)
    var_explained = pca.explained_variance_ratio_

    offset = 0
    for source in source_names:
        n = len(all_embeddings[source][layer_idx])
        ax.scatter(
            coords[offset:offset+n, 0],
            coords[offset:offset+n, 1],
            c=colors[source],
            alpha=0.3,
            s=3,
            label=source,
            rasterized=True,
        )
        offset += n

    ax.set_title(f"Layer {layer_idx}", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=8)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=8)
    ax.tick_params(labelsize=7)

    if layer_idx == 0:
        ax.legend(fontsize=8, markerscale=4)

plt.tight_layout()
plt.savefig("piano_vs_bird_pca.png", dpi=150, bbox_inches="tight")
print(f"\nSaved plot to piano_vs_bird_pca.png")
