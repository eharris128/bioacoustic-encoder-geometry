"""Exploratory interpretability: how do AVES layer representations differ across species?

Extracts frame-level embeddings from all 12 layers for both example audio files,
reduces to 2D with PCA, and plots each layer colored by species.
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
AUDIO_FILES = {
    "Guineafowl": "./aves/example_audios/XC936872 - Helmeted Guineafowl - Numida meleagris.wav",
    "Bullfinch": "./aves/example_audios/XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.mp3",
}
NUM_LAYERS = 12
# Subsample frames for plotting (both files together could be thousands of frames)
MAX_FRAMES_PER_SPECIES = 2000

# --- Load model ---
print("Loading model...")
model = load_feature_extractor(
    config_path=CONFIG_PATH,
    model_path=MODEL_PATH,
    device="cpu",
    for_inference=True,
)
print("Model loaded.\n")

# --- Extract embeddings from all layers for both files ---
all_embeddings = {}  # {species: list of 12 tensors, each (num_frames, 768)}

for species, path in AUDIO_FILES.items():
    print(f"Processing {species}...")
    audio = load_audio(path, mono=True, mono_avg=False)
    print(f"  Audio shape: {audio.shape} ({audio.shape[-1]/16000:.1f}s at 16kHz)")

    t0 = time.time()
    layer_outputs = model.extract_features(audio, layers=None)  # list of 12 tensors
    print(f"  Extracted {len(layer_outputs)} layers in {time.time()-t0:.1f}s")

    # Each layer output is (1, num_frames, 768) — squeeze batch dim
    embeddings = [layer.squeeze(0).cpu().numpy() for layer in layer_outputs]
    print(f"  Frames per layer: {embeddings[0].shape[0]}")

    # Subsample if too many frames
    n_frames = embeddings[0].shape[0]
    if n_frames > MAX_FRAMES_PER_SPECIES:
        idx = np.random.default_rng(42).choice(n_frames, MAX_FRAMES_PER_SPECIES, replace=False)
        idx.sort()
        embeddings = [e[idx] for e in embeddings]
        print(f"  Subsampled to {MAX_FRAMES_PER_SPECIES} frames")

    all_embeddings[species] = embeddings

# --- PCA + plot for each layer ---
species_names = list(all_embeddings.keys())
colors = {"Guineafowl": "#2196F3", "Bullfinch": "#FF5722"}

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("AVES Layer Representations: Guineafowl vs Bullfinch\n(PCA → 2D, each dot = one 20ms frame)",
             fontsize=14, fontweight="bold")

for layer_idx in range(NUM_LAYERS):
    ax = axes[layer_idx // 4, layer_idx % 4]

    # Concatenate both species for joint PCA
    species_frames = []
    species_labels = []
    for species in species_names:
        frames = all_embeddings[species][layer_idx]
        species_frames.append(frames)
        species_labels.extend([species] * len(frames))

    combined = np.concatenate(species_frames, axis=0)

    # PCA to 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined)
    var_explained = pca.explained_variance_ratio_

    # Plot each species
    offset = 0
    for species in species_names:
        n = len(all_embeddings[species][layer_idx])
        ax.scatter(
            coords[offset:offset+n, 0],
            coords[offset:offset+n, 1],
            c=colors[species],
            alpha=0.3,
            s=3,
            label=species,
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
plt.savefig("layer_exploration.png", dpi=150, bbox_inches="tight")
print(f"\nSaved plot to layer_exploration.png")
