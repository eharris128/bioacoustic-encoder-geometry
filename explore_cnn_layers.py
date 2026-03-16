"""Explore the CNN feature extractor: the 7 convolutional layers that transform
raw audio BEFORE the transformer sees it.

Hooks into each CNN layer to extract intermediate representations, then:
1. CKA between all stages (raw audio → 7 CNN layers → feature projection → 12 transformer layers)
2. PCA visualization of CNN layer outputs
3. CKA with mel spectrogram at each stage to pinpoint where acoustic grounding is lost
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torchaudio
from aves import load_feature_extractor
from aves.utils import load_audio

# --- Config ---
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
AUDIO_DIR = Path("./audio/bullfinch")
SKIP = {"XC1086809.mp3", "XC657517.mp3"}
NUM_CNN_LAYERS = 7
NUM_TRANSFORMER_LAYERS = 12
MAX_FRAMES = 500
SR = 16000


def linear_cka(X, Y):
    """Linear CKA between two matrices."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XtY = X.T @ Y
    XtX = X.T @ X
    YtY = Y.T @ Y
    return np.sum(XtY ** 2) / np.sqrt(np.sum(XtX ** 2) * np.sum(YtY ** 2) + 1e-10)


# --- Load model ---
print("Loading model...")
model = load_feature_extractor(
    config_path=CONFIG_PATH,
    model_path=MODEL_PATH,
    device="cpu",
    for_inference=True,
)

# --- Register hooks on CNN layers ---
cnn_outputs = {}


def make_cnn_hook(layer_idx):
    def hook_fn(module, input, output):
        # CNN layers return (tensor, length) tuple
        # tensor shape: (batch, channels, time)
        out = output[0] if isinstance(output, tuple) else output
        cnn_outputs[layer_idx] = out.detach().squeeze(0).permute(1, 0).numpy()
    return hook_fn


hooks = []
for i in range(NUM_CNN_LAYERS):
    h = model.model.feature_extractor.conv_layers[i].register_forward_hook(make_cnn_hook(i))
    hooks.append(h)

# Also hook the feature projection (output of CNN → input to transformer)
def proj_hook(module, input, output):
    cnn_outputs["projection"] = output.detach().squeeze(0).numpy()

h = model.model.encoder.feature_projection.register_forward_hook(proj_hook)
hooks.append(h)

# --- Process recordings ---
audio_files = sorted([f for f in AUDIO_DIR.glob("*.mp3") if f.name not in SKIP])[:10]
print(f"Using {len(audio_files)} recordings")

# Collect outputs from all stages
all_stages = {}  # {stage_name: list of arrays}
stage_names = [f"CNN_{i}" for i in range(NUM_CNN_LAYERS)] + ["Projection"] + [f"Transformer_{i}" for i in range(NUM_TRANSFORMER_LAYERS)]
for name in stage_names:
    all_stages[name] = []

# Also collect mel spectrograms
all_mel = []

for i, path in enumerate(audio_files):
    print(f"  [{i+1:2d}/{len(audio_files)}] {path.stem}...", end=" ", flush=True)

    try:
        audio = load_audio(str(path), mono=True, mono_avg=False)
    except Exception:
        print("SKIP")
        continue

    # Clear CNN outputs
    cnn_outputs.clear()

    # Forward pass — triggers both CNN hooks and extracts transformer layers
    t0 = time.time()
    transformer_outputs = model.extract_features(audio, layers=None)
    elapsed = time.time() - t0

    # Collect CNN outputs
    for j in range(NUM_CNN_LAYERS):
        emb = cnn_outputs[j]
        if emb.shape[0] > MAX_FRAMES:
            rng = np.random.default_rng(42)
            idx = rng.choice(emb.shape[0], MAX_FRAMES, replace=False)
            idx.sort()
            emb = emb[idx]
        all_stages[f"CNN_{j}"].append(emb)

    # Feature projection
    emb = cnn_outputs["projection"]
    if emb.shape[0] > MAX_FRAMES:
        rng = np.random.default_rng(42)
        idx = rng.choice(emb.shape[0], MAX_FRAMES, replace=False)
        idx.sort()
        emb = emb[idx]
    all_stages["Projection"].append(emb)

    # Transformer outputs
    for j, layer_out in enumerate(transformer_outputs):
        emb = layer_out.squeeze(0).cpu().numpy()
        if emb.shape[0] > MAX_FRAMES:
            rng = np.random.default_rng(42)
            idx = rng.choice(emb.shape[0], MAX_FRAMES, replace=False)
            idx.sort()
            emb = emb[idx]
        all_stages[f"Transformer_{j}"].append(emb)

    # Mel spectrogram (matched to transformer frame rate: hop=320)
    if audio.ndim == 1:
        audio_for_spec = audio.unsqueeze(0)
    else:
        audio_for_spec = audio[:1]
    spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=1024, hop_length=320, n_mels=80,
    )
    spec = spec_transform(audio_for_spec).squeeze(0).permute(1, 0).numpy()
    if spec.shape[0] > MAX_FRAMES:
        rng = np.random.default_rng(42)
        idx = rng.choice(spec.shape[0], MAX_FRAMES, replace=False)
        idx.sort()
        spec = spec[idx]
    all_mel.append(spec)

    print(f"{elapsed:.1f}s")

# Clean up hooks
for h in hooks:
    h.remove()

# Concatenate per stage
for name in stage_names:
    all_stages[name] = np.concatenate(all_stages[name], axis=0)
mel_combined = np.concatenate(all_mel, axis=0)

# Align frame counts (CNN layers have different temporal resolutions)
# We'll compute CKA only between stages with the same frame count
# CNN layers progressively downsample, so they have different lengths

print(f"\nStage dimensions:")
for name in stage_names:
    print(f"  {name:20s}: {all_stages[name].shape}")
print(f"  {'Mel spectrogram':20s}: {mel_combined.shape}")

# --- CKA with mel spectrogram at each stage ---
# Only stages with same frame count as mel can be compared directly
# Transformer stages and projection should match; CNN stages won't
# For CNN stages, downsample mel to match their frame count

print(f"\nCKA with mel spectrogram at each stage:")
cka_with_mel = {}

for name in stage_names:
    n_frames_stage = all_stages[name].shape[0]
    n_frames_mel = mel_combined.shape[0]

    if n_frames_stage == n_frames_mel:
        cka = linear_cka(all_stages[name], mel_combined)
    else:
        # Resample mel to match stage frame count
        min_n = min(n_frames_stage, n_frames_mel)
        rng = np.random.default_rng(42)
        idx_stage = rng.choice(n_frames_stage, min_n, replace=False)
        idx_stage.sort()
        idx_mel = np.linspace(0, n_frames_mel - 1, min_n).astype(int)
        cka = linear_cka(all_stages[name][idx_stage], mel_combined[idx_mel])

    cka_with_mel[name] = cka
    print(f"  {name:20s}: {cka:.4f}")

# --- CKA between adjacent stages (full pipeline) ---
print(f"\nCKA between adjacent stages:")
adjacent_cka = []
for i in range(len(stage_names) - 1):
    name_a = stage_names[i]
    name_b = stage_names[i + 1]
    n_a = all_stages[name_a].shape[0]
    n_b = all_stages[name_b].shape[0]
    min_n = min(n_a, n_b)
    rng = np.random.default_rng(42)
    idx_a = np.linspace(0, n_a - 1, min_n).astype(int)
    idx_b = np.linspace(0, n_b - 1, min_n).astype(int)
    cka = linear_cka(all_stages[name_a][idx_a], all_stages[name_b][idx_b])
    adjacent_cka.append(cka)
    print(f"  {name_a:20s} → {name_b:20s}: {cka:.4f}")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
fig.suptitle("Full Pipeline: CNN Feature Extractor → Transformer\n"
             "(7 CNN layers + projection + 12 transformer layers)",
             fontsize=14, fontweight="bold")

# 1. CKA with mel spectrogram across full pipeline
x_labels = [f"C{i}" for i in range(NUM_CNN_LAYERS)] + ["Proj"] + [f"T{i}" for i in range(NUM_TRANSFORMER_LAYERS)]
cka_values = [cka_with_mel[name] for name in stage_names]

colors = (["#E57373"] * NUM_CNN_LAYERS +  # Red for CNN
          ["#FFB74D"] +  # Orange for projection
          ["#64B5F6"] * NUM_TRANSFORMER_LAYERS)  # Blue for transformer

ax1.bar(range(len(stage_names)), cka_values, color=colors, edgecolor="black", linewidth=0.5)
ax1.set_xticks(range(len(stage_names)))
ax1.set_xticklabels(x_labels, fontsize=9)
ax1.set_ylabel("CKA with Mel Spectrogram", fontsize=12)
ax1.set_title("Acoustic Grounding Across the Full Pipeline\n"
              "(where does the model diverge from raw spectral features?)", fontsize=12)

for i, v in enumerate(cka_values):
    ax1.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=7, rotation=45)

# Add region labels
ax1.axvspan(-0.5, NUM_CNN_LAYERS - 0.5, alpha=0.1, color="red", label="CNN layers")
ax1.axvspan(NUM_CNN_LAYERS - 0.5, NUM_CNN_LAYERS + 0.5, alpha=0.1, color="orange", label="Projection")
ax1.axvspan(NUM_CNN_LAYERS + 0.5, len(stage_names) - 0.5, alpha=0.1, color="blue", label="Transformer layers")
ax1.legend(fontsize=10)

# 2. Adjacent CKA across full pipeline
transition_labels = [f"{x_labels[i]}→{x_labels[i+1]}" for i in range(len(x_labels) - 1)]
transition_colors = (["#E57373"] * (NUM_CNN_LAYERS - 1) +
                     ["#FFB74D"] +  # CNN→Proj
                     ["#FFB74D"] +  # Proj→T0
                     ["#64B5F6"] * (NUM_TRANSFORMER_LAYERS - 1))

ax2.bar(range(len(adjacent_cka)), adjacent_cka, color=transition_colors, edgecolor="black", linewidth=0.5)
ax2.set_xticks(range(len(adjacent_cka)))
ax2.set_xticklabels(transition_labels, fontsize=7, rotation=45, ha="right")
ax2.set_ylabel("CKA with Next Stage", fontsize=12)
ax2.set_title("Transformation Magnitude at Each Step\n"
              "(lower = bigger representational change)", fontsize=12)
ax2.set_ylim(0, 1.05)

for i, v in enumerate(adjacent_cka):
    ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7, rotation=45)

plt.tight_layout()
plt.savefig("cnn_pipeline.png", dpi=150, bbox_inches="tight")
print("\nSaved cnn_pipeline.png")
