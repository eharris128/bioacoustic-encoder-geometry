"""Exploratory: how does AVES represent mixtures of bird + violin audio?

Mixes bullfinch and violin waveforms at three ratios (75/25, 50/50, 25/75
bird/violin), extracts AVES embeddings, and plots where mixed frames land
relative to pure bird and pure violin clusters in PCA space.
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
BIRD_PATH = "./aves/example_audios/XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.mp3"
VIOLIN_PATH = "./audio/violin/solarflex-emotional-inspiring-violin-499245.mp3"
MIX_RATIOS = [(0.75, 0.25), (0.50, 0.50), (0.25, 0.75)]  # (bird_weight, violin_weight)
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

# --- Load and align audio ---
bird_audio = load_audio(BIRD_PATH, mono=True, mono_avg=False)
violin_audio = load_audio(VIOLIN_PATH, mono=True, mono_avg=False)
print(f"Bird audio:   {bird_audio.shape} ({bird_audio.shape[-1]/16000:.1f}s)")
print(f"Violin audio: {violin_audio.shape} ({violin_audio.shape[-1]/16000:.1f}s)")

# Trim both to the shorter length
min_len = min(bird_audio.shape[-1], violin_audio.shape[-1])
bird_audio = bird_audio[..., :min_len]
violin_audio = violin_audio[..., :min_len]
print(f"Trimmed both to {min_len} samples ({min_len/16000:.1f}s)\n")

# --- Extract pure source embeddings ---
def extract_layers(audio, label):
    print(f"Extracting {label}...")
    t0 = time.time()
    layer_outputs = model.extract_features(audio, layers=None)
    print(f"  {len(layer_outputs)} layers in {time.time()-t0:.1f}s, {layer_outputs[0].shape[1]} frames")
    return [layer.squeeze(0).cpu().numpy() for layer in layer_outputs]

bird_embs = extract_layers(bird_audio, "Bird (pure)")
violin_embs = extract_layers(violin_audio, "Violin (pure)")

# --- Create and extract mixed audio ---
mix_embs = {}
for bird_w, violin_w in MIX_RATIOS:
    mixed = bird_w * bird_audio + violin_w * violin_audio
    label = f"Mix {int(bird_w*100)}/{int(violin_w*100)}"
    mix_embs[label] = extract_layers(mixed, label)

# --- Subsample ---
rng = np.random.default_rng(42)
all_sources = {"Bullfinch": bird_embs, "Violin": violin_embs}
all_sources.update(mix_embs)

for label, embs in all_sources.items():
    n = embs[0].shape[0]
    if n > MAX_FRAMES_PER_SOURCE:
        idx = rng.choice(n, MAX_FRAMES_PER_SOURCE, replace=False)
        idx.sort()
        for i in range(NUM_LAYERS):
            embs[i] = embs[i][idx]
        print(f"Subsampled {label} to {MAX_FRAMES_PER_SOURCE} frames")

# --- PCA + plot ---
source_names = list(all_sources.keys())
colors = {
    "Bullfinch": "#FF5722",
    "Violin": "#4CAF50",
    "Mix 75/25": "#FF9800",   # bird-dominant — orange-ish
    "Mix 50/50": "#9E9E9E",   # neutral — gray
    "Mix 25/75": "#8BC34A",   # violin-dominant — light green
}

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("AVES Layer Representations: Bird/Violin Mixtures\n"
             "(PCA → 2D, pure sources + 3 mix ratios)",
             fontsize=14, fontweight="bold")

for layer_idx in range(NUM_LAYERS):
    ax = axes[layer_idx // 4, layer_idx % 4]

    # Fit PCA on pure sources only, then project mixtures into same space
    pure_combined = np.concatenate([
        all_sources["Bullfinch"][layer_idx],
        all_sources["Violin"][layer_idx],
    ], axis=0)

    pca = PCA(n_components=2)
    pca.fit(pure_combined)
    var_explained = pca.explained_variance_ratio_

    # Plot pure sources first (behind), then mixtures on top
    for source in ["Bullfinch", "Violin"]:
        coords = pca.transform(all_sources[source][layer_idx])
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=colors[source], alpha=0.2, s=3,
            label=source, rasterized=True,
        )

    for source in source_names:
        if source in ("Bullfinch", "Violin"):
            continue
        coords = pca.transform(all_sources[source][layer_idx])
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=colors[source], alpha=0.4, s=5,
            label=source, rasterized=True,
        )

    ax.set_title(f"Layer {layer_idx}", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})", fontsize=8)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})", fontsize=8)
    ax.tick_params(labelsize=7)

    if layer_idx == 0:
        ax.legend(fontsize=7, markerscale=4, loc="best")

plt.tight_layout()
plt.savefig("mixed_bird_violin_pca.png", dpi=150, bbox_inches="tight")
print(f"\nSaved plot to mixed_bird_violin_pca.png")
