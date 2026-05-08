"""LDA: how well does each AVES layer linearly separate bird vs piano vs violin?

Uses Linear Discriminant Analysis instead of PCA — finds the 2D projection that
maximally separates the three classes. Also reports per-layer classification
accuracy using LDA as a classifier.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from aves import load_feature_extractor
from aves.utils import load_audio

# --- Config ---
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
BIRD_FILES = [
    "./aves/example_audios/XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.mp3",
]
PIANO_FILES = [
    "/home/evan/Downloads/paulyudin-piano-music-piano-485929.mp3",
    "/home/evan/Downloads/sigmamusicart-piano-music-504007.mp3",
    "/home/evan/Downloads/solarflex-emotional-piano-music-499244.mp3",
    "/home/evan/Downloads/the_mountain-piano-background-music-487020.mp3",
]
VIOLIN_FILES = [
    "./audio/violin/good_b_music-romantic-violin-waltz-real-violin-497682.mp3",
    "./audio/violin/nickpanekaiassets-cinematic-baroque-violin-melody-287276.mp3",
    "./audio/violin/solarflex-emotional-inspiring-violin-499245.mp3",
    "./audio/violin/soulfuljamtracks-strings-violin-background-478146.mp3",
    "./audio/violin/vibehorn-violin-background-music-483067.mp3",
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


def extract_and_pool(file_list, label):
    """Extract embeddings from multiple files and pool frames across all."""
    layer_frames = [[] for _ in range(NUM_LAYERS)]
    for i, path in enumerate(file_list):
        short_name = path.split("/")[-1]
        print(f"Processing {label} {i+1}/{len(file_list)}: {short_name}...")
        audio = load_audio(path, mono=True, mono_avg=False)
        print(f"  Audio shape: {audio.shape} ({audio.shape[-1]/16000:.1f}s at 16kHz)")
        t0 = time.time()
        layer_outputs = model.extract_features(audio, layers=None)
        print(f"  Extracted {len(layer_outputs)} layers in {time.time()-t0:.1f}s")
        for layer_idx in range(NUM_LAYERS):
            layer_frames[layer_idx].append(layer_outputs[layer_idx].squeeze(0).cpu().numpy())
    embeddings = [np.concatenate(frames, axis=0) for frames in layer_frames]
    print(f"Total {label} frames per layer: {embeddings[0].shape[0]}")
    return embeddings


# --- Extract all sources ---
bird_embeddings = extract_and_pool(BIRD_FILES, "Bird")
piano_embeddings = extract_and_pool(PIANO_FILES, "Piano")
violin_embeddings = extract_and_pool(VIOLIN_FILES, "Violin")

# --- Subsample ---
rng = np.random.default_rng(42)
for label, embs in [("Bird", bird_embeddings), ("Piano", piano_embeddings), ("Violin", violin_embeddings)]:
    n = embs[0].shape[0]
    if n > MAX_FRAMES_PER_SOURCE:
        idx = rng.choice(n, MAX_FRAMES_PER_SOURCE, replace=False)
        idx.sort()
        for i in range(NUM_LAYERS):
            embs[i] = embs[i][idx]
        print(f"Subsampled {label} to {MAX_FRAMES_PER_SOURCE} frames")

all_embeddings = {"Bullfinch": bird_embeddings, "Piano": piano_embeddings, "Violin": violin_embeddings}
source_names = list(all_embeddings.keys())
colors = {"Bullfinch": "#FF5722", "Piano": "#9C27B0", "Violin": "#4CAF50"}

# --- LDA per layer: 2D projection + classification accuracy ---
layer_accuracies = []

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("AVES Layer Representations: Bird vs Piano vs Violin (LDA → 2D)\n"
             "Projection maximizes class separation",
             fontsize=14, fontweight="bold")

for layer_idx in range(NUM_LAYERS):
    ax = axes[layer_idx // 4, layer_idx % 4]

    # Build feature matrix and labels
    X_parts = [all_embeddings[s][layer_idx] for s in source_names]
    y_parts = [np.full(len(x), i) for i, x in enumerate(X_parts)]
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts)

    # Temporal train/test split (80/20)
    split = int(0.8 * len(X))
    idx = np.arange(len(X))
    # Shuffle with fixed seed for this layer
    layer_rng = np.random.default_rng(42 + layer_idx)
    layer_rng.shuffle(idx)
    train_idx, test_idx = idx[:split], idx[split:]

    # Fit LDA
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X[train_idx], y[train_idx])

    # Classification accuracy on test set
    acc = lda.score(X[test_idx], y[test_idx])
    layer_accuracies.append(acc)

    # Project all points for visualization
    coords = lda.transform(X)

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

    ax.set_title(f"Layer {layer_idx} (acc={acc:.0%})", fontsize=11, fontweight="bold")
    ax.set_xlabel("LD1", fontsize=8)
    ax.set_ylabel("LD2", fontsize=8)
    ax.tick_params(labelsize=7)

    if layer_idx == 0:
        ax.legend(fontsize=8, markerscale=4)

plt.tight_layout()
plt.savefig("lda_violin_piano_bird.png", dpi=150, bbox_inches="tight")
print(f"\nSaved scatter plot to lda_violin_piano_bird.png")

# --- Accuracy curve across layers ---
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(range(NUM_LAYERS), layer_accuracies, "o-", color="#1976D2", linewidth=2, markersize=8)
ax2.set_xlabel("AVES Layer", fontsize=12)
ax2.set_ylabel("LDA Classification Accuracy", fontsize=12)
ax2.set_title("3-Way LDA Accuracy Across AVES Layers\n(Bird vs Piano vs Violin)", fontsize=13, fontweight="bold")
ax2.set_xticks(range(NUM_LAYERS))
ax2.set_ylim(0, 1.05)
ax2.axhline(1/3, color="gray", linestyle="--", alpha=0.5, label="Chance (33%)")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lda_violin_piano_bird_accuracy.png", dpi=150, bbox_inches="tight")
print(f"Saved accuracy curve to lda_violin_piano_bird_accuracy.png")

print("\n--- Per-layer LDA accuracy ---")
for i, acc in enumerate(layer_accuracies):
    print(f"  Layer {i:2d}: {acc:.1%}")
