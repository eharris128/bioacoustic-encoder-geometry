"""CKA (Centered Kernel Alignment) between all pairs of AVES layers.

Produces a 12x12 similarity matrix showing which layers compute similar
representations. Reveals processing regimes/phase transitions in the network.

Uses linear CKA (Kornblith et al., 2019) — fast, no kernel hyperparameters.
Runs on a subset of Bullfinch recordings.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from aves import load_feature_extractor
from aves.utils import load_audio

# --- Config ---
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
AUDIO_DIR = Path("./audio/bullfinch")
NUM_LAYERS = 12
MAX_FRAMES_PER_RECORDING = 500
SKIP = {"XC1086809.mp3", "XC657517.mp3"}  # Too large / corrupted


def linear_cka(X, Y):
    """Compute linear CKA between two matrices X and Y.

    X: (n, p) — n samples, p features
    Y: (n, q) — n samples, q features

    Returns scalar CKA similarity in [0, 1].
    """
    # Center columns
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # HSIC with linear kernel: HSIC(K,L) = ||Y^T X||_F^2 / (n-1)^2
    # But for CKA we need: HSIC(K,L) / sqrt(HSIC(K,K) * HSIC(L,L))
    XtX = X.T @ X  # (p, p)
    YtY = Y.T @ Y  # (q, q)
    XtY = X.T @ Y  # (p, q)

    hsic_xy = np.sum(XtY ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)

    return hsic_xy / np.sqrt(hsic_xx * hsic_yy + 1e-10)


# --- Load model ---
print("Loading model...")
model = load_feature_extractor(
    config_path=CONFIG_PATH,
    model_path=MODEL_PATH,
    device="cpu",
    for_inference=True,
)

# --- Load recordings and extract all layers ---
audio_files = sorted([f for f in AUDIO_DIR.glob("*.mp3") if f.name not in SKIP])
# Use 10 recordings for speed
audio_files = audio_files[:10]
print(f"\nUsing {len(audio_files)} recordings for CKA")

# Collect frame embeddings per layer across all recordings
layer_embeddings = [[] for _ in range(NUM_LAYERS)]

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

    for layer_idx, layer_out in enumerate(layer_outputs):
        emb = layer_out.squeeze(0).cpu().numpy()
        if emb.shape[0] > MAX_FRAMES_PER_RECORDING:
            rng = np.random.default_rng(42)
            idx = rng.choice(emb.shape[0], MAX_FRAMES_PER_RECORDING, replace=False)
            idx.sort()
            emb = emb[idx]
        layer_embeddings[layer_idx].append(emb)

    print(f"{elapsed:.1f}s")

# Concatenate all frames per layer
for layer_idx in range(NUM_LAYERS):
    layer_embeddings[layer_idx] = np.concatenate(layer_embeddings[layer_idx], axis=0)

n_frames = layer_embeddings[0].shape[0]
print(f"\nTotal frames per layer: {n_frames}")

# --- Compute CKA matrix ---
print("Computing 12x12 CKA matrix...")
t0 = time.time()
cka_matrix = np.zeros((NUM_LAYERS, NUM_LAYERS))

for i in range(NUM_LAYERS):
    for j in range(i, NUM_LAYERS):
        cka = linear_cka(layer_embeddings[i], layer_embeddings[j])
        cka_matrix[i, j] = cka
        cka_matrix[j, i] = cka

print(f"Done in {time.time()-t0:.1f}s")

# --- Also compute CKA of each layer with the raw audio features ---
# Use mel spectrogram as a "layer 0 baseline" (what the input looks like)
print("Computing CKA with mel spectrogram baseline...")
import torchaudio

spec_embeddings = []
for path in audio_files:
    try:
        audio = load_audio(str(path), mono=True, mono_avg=False)
    except Exception:
        continue

    spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, n_mels=80,
    )
    if audio.ndim == 1:
        audio_for_spec = audio.unsqueeze(0)
    else:
        audio_for_spec = audio[:1]  # Take first channel if stereo
    spec = spec_transform(audio_for_spec).squeeze(0).permute(1, 0).numpy()  # (n_frames, 80)

    if spec.shape[0] > MAX_FRAMES_PER_RECORDING:
        rng = np.random.default_rng(42)
        idx = rng.choice(spec.shape[0], MAX_FRAMES_PER_RECORDING, replace=False)
        idx.sort()
        spec = spec[idx]
    spec_embeddings.append(spec)

spec_combined = np.concatenate(spec_embeddings, axis=0)
# Align frame counts (spec may differ slightly)
min_frames = min(n_frames, spec_combined.shape[0])
spec_combined = spec_combined[:min_frames]

cka_with_spec = []
for layer_idx in range(NUM_LAYERS):
    cka = linear_cka(layer_embeddings[layer_idx][:min_frames], spec_combined)
    cka_with_spec.append(cka)
    print(f"  Layer {layer_idx}: CKA with mel spec = {cka:.4f}")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(22, 7),
                          gridspec_kw={"width_ratios": [1, 1, 0.6]})
fig.suptitle("CKA Analysis: AVES Layer Similarity (28 Bullfinch recordings)",
             fontsize=14, fontweight="bold")

# 1. CKA heatmap
ax = axes[0]
im = ax.imshow(cka_matrix, cmap="magma", vmin=0, vmax=1)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Layer", fontsize=12)
ax.set_title("Linear CKA Between Layers", fontsize=12)
ax.set_xticks(range(NUM_LAYERS))
ax.set_yticks(range(NUM_LAYERS))
plt.colorbar(im, ax=ax, label="CKA similarity", shrink=0.8)

# Add values
for i in range(NUM_LAYERS):
    for j in range(NUM_LAYERS):
        color = "white" if cka_matrix[i, j] < 0.5 else "black"
        ax.text(j, i, f"{cka_matrix[i,j]:.2f}", ha="center", va="center",
                fontsize=6, color=color)

# 2. CKA with adjacent layers (how much does each layer change the representation?)
ax = axes[1]
adjacent_cka = [cka_matrix[i, i+1] for i in range(NUM_LAYERS - 1)]
ax.bar(range(NUM_LAYERS - 1), adjacent_cka,
       color=plt.cm.viridis(np.linspace(0.2, 0.9, NUM_LAYERS - 1)),
       edgecolor="black", linewidth=0.5)
ax.set_xlabel("Layer transition", fontsize=12)
ax.set_ylabel("CKA with next layer", fontsize=12)
ax.set_title("Adjacent Layer Similarity\n(low = big transformation)", fontsize=12)
ax.set_xticks(range(NUM_LAYERS - 1))
ax.set_xticklabels([f"{i}→{i+1}" for i in range(NUM_LAYERS - 1)], fontsize=8, rotation=45)
ax.set_ylim(0.5, 1.0)

for i, v in enumerate(adjacent_cka):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

# 3. CKA with mel spectrogram
ax = axes[2]
ax.barh(range(NUM_LAYERS), cka_with_spec,
        color=plt.cm.viridis(np.linspace(0.2, 0.9, NUM_LAYERS)),
        edgecolor="black", linewidth=0.5)
ax.set_ylabel("Layer", fontsize=12)
ax.set_xlabel("CKA with Mel Spectrogram", fontsize=12)
ax.set_title("Similarity to Input\n(acoustic grounding)", fontsize=12)
ax.set_yticks(range(NUM_LAYERS))
ax.invert_yaxis()

for i, v in enumerate(cka_with_spec):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("cka_analysis.png", dpi=150, bbox_inches="tight")
print("\nSaved cka_analysis.png")
