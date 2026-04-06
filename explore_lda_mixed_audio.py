"""LDA: where do bird/violin mixtures land in discriminant space?

Fits LDA on pure bird vs violin frames, then projects waveform mixtures
at three ratios (75/25, 50/50, 25/75) into the same discriminant space.
With 2 classes, LDA gives 1 component — we add PCA-2 as the second axis
for a richer view.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from aves import load_feature_extractor
from aves.utils import load_audio

# --- Config ---
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
BIRD_PATH = "./aves/example_audios/XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.mp3"
VIOLIN_PATH = "./audio/violin/solarflex-emotional-inspiring-violin-499245.mp3"
MIX_RATIOS = [(0.75, 0.25), (0.50, 0.50), (0.25, 0.75)]
NUM_LAYERS = 12
MAX_FRAMES = 2000

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
min_len = min(bird_audio.shape[-1], violin_audio.shape[-1])
bird_audio = bird_audio[..., :min_len]
violin_audio = violin_audio[..., :min_len]
print(f"Both trimmed to {min_len} samples ({min_len/16000:.1f}s)\n")


def extract_layers(audio, label):
    print(f"Extracting {label}...")
    t0 = time.time()
    layer_outputs = model.extract_features(audio, layers=None)
    print(f"  {len(layer_outputs)} layers in {time.time()-t0:.1f}s, {layer_outputs[0].shape[1]} frames")
    return [layer.squeeze(0).cpu().numpy() for layer in layer_outputs]


# --- Extract pure + mixed ---
bird_embs = extract_layers(bird_audio, "Bird (pure)")
violin_embs = extract_layers(violin_audio, "Violin (pure)")

mix_embs = {}
for bird_w, violin_w in MIX_RATIOS:
    mixed = bird_w * bird_audio + violin_w * violin_audio
    label = f"Mix {int(bird_w*100)}/{int(violin_w*100)}"
    mix_embs[label] = extract_layers(mixed, label)

# --- LDA + plot ---
colors_pure = {"Bird": "#FF5722", "Violin": "#4CAF50"}
colors_mix = {
    "Mix 75/25": "#FF9800",
    "Mix 50/50": "#9E9E9E",
    "Mix 25/75": "#8BC34A",
}

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("AVES: Bird/Violin Mixtures in LDA Space\n"
             "(LD1 = max class separation, PC-residual = orthogonal variance)",
             fontsize=14, fontweight="bold")

mix_ld1_means = {label: [] for label in mix_embs}

for layer_idx in range(NUM_LAYERS):
    ax = axes[layer_idx // 4, layer_idx % 4]

    X_bird = bird_embs[layer_idx]
    X_violin = violin_embs[layer_idx]
    X_pure = np.concatenate([X_bird, X_violin], axis=0)
    y_pure = np.array([0]*len(X_bird) + [1]*len(X_violin))

    # Fit LDA (1 component for 2 classes)
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_pure, y_pure)
    ld1_all = lda.transform(X_pure)

    # Use PCA on the LDA residual for the second axis
    residual = X_pure - X_pure @ lda.scalings_ @ lda.scalings_.T
    pca = PCA(n_components=1)
    pc_resid = pca.fit_transform(residual)

    # Plot pure sources
    n_bird = len(X_bird)
    ax.scatter(ld1_all[:n_bird, 0], pc_resid[:n_bird, 0],
               c=colors_pure["Bird"], alpha=0.2, s=3, label="Bird", rasterized=True)
    ax.scatter(ld1_all[n_bird:, 0], pc_resid[n_bird:, 0],
               c=colors_pure["Violin"], alpha=0.2, s=3, label="Violin", rasterized=True)

    # Project and plot mixtures
    for label, embs in mix_embs.items():
        X_mix = embs[layer_idx]
        ld1_mix = lda.transform(X_mix)
        resid_mix = X_mix - X_mix @ lda.scalings_ @ lda.scalings_.T
        pc_resid_mix = pca.transform(resid_mix)
        ax.scatter(ld1_mix[:, 0], pc_resid_mix[:, 0],
                   c=colors_mix[label], alpha=0.4, s=5, label=label, rasterized=True)
        mix_ld1_means[label].append(float(np.mean(ld1_mix)))

    ax.set_title(f"Layer {layer_idx}", fontsize=11, fontweight="bold")
    ax.set_xlabel("LD1 (class separation)", fontsize=8)
    ax.set_ylabel("PC-residual", fontsize=8)
    ax.tick_params(labelsize=7)

    if layer_idx == 0:
        ax.legend(fontsize=6, markerscale=4, loc="best")

plt.tight_layout()
plt.savefig("lda_mixed_bird_violin.png", dpi=150, bbox_inches="tight")
print(f"\nSaved scatter plot to lda_mixed_bird_violin.png")

# --- LD1 position trajectory across layers ---
fig2, ax2 = plt.subplots(figsize=(10, 5))

# Get pure source LD1 means for reference
bird_ld1_means = []
violin_ld1_means = []
for layer_idx in range(NUM_LAYERS):
    X_bird = bird_embs[layer_idx]
    X_violin = violin_embs[layer_idx]
    X_pure = np.concatenate([X_bird, X_violin], axis=0)
    y_pure = np.array([0]*len(X_bird) + [1]*len(X_violin))
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_pure, y_pure)
    bird_ld1_means.append(float(np.mean(lda.transform(X_bird))))
    violin_ld1_means.append(float(np.mean(lda.transform(X_violin))))

ax2.plot(range(NUM_LAYERS), bird_ld1_means, "o--", color="#FF5722", alpha=0.5, label="Bird (pure)")
ax2.plot(range(NUM_LAYERS), violin_ld1_means, "o--", color="#4CAF50", alpha=0.5, label="Violin (pure)")
for label, means in mix_ld1_means.items():
    ax2.plot(range(NUM_LAYERS), means, "s-", color=colors_mix[label], linewidth=2, markersize=7, label=label)

ax2.set_xlabel("AVES Layer", fontsize=12)
ax2.set_ylabel("Mean LD1 Position", fontsize=12)
ax2.set_title("Where Do Mixtures Fall on the Bird↔Violin Discriminant Axis?",
              fontsize=13, fontweight="bold")
ax2.set_xticks(range(NUM_LAYERS))
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lda_mixed_bird_violin_trajectory.png", dpi=150, bbox_inches="tight")
print(f"Saved trajectory plot to lda_mixed_bird_violin_trajectory.png")
