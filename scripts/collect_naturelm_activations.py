"""Collect AVES activation statistics from NatureLM-audio-training.

Streams samples, extracts all 13 residual stream activations (embedding + 12 layers),
computes per-layer statistics: L2 norms, singular values, PCA alignment, intrinsic
dimensionality. Saves raw activations and metadata for further analysis.

Usage:
    python collect_naturelm_activations.py [--n_samples 5] [--device cpu] [--model all]

Models: all, bio, nonbio, core (all share same architecture, different training data)
"""

import argparse
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.decomposition import PCA

from aves import load_feature_extractor

# --- Args ---
MODELS = {
    "all":    "aves-base-all",
    "bio":    "aves-base-bio",
    "nonbio": "aves-base-nonbio",
    "core":   "aves-base-core",
}

parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", type=int, default=5)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--model", type=str, default="all", choices=MODELS.keys(),
                    help="Which AVES model to use (all, bio, nonbio, core)")
args = parser.parse_args()

N_SAMPLES = args.n_samples
DEVICE = args.device
MODEL_NAME = MODELS[args.model]
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"  # same config for all 4
MODEL_PATH = f"./models/{MODEL_NAME}.torchaudio.pt"
NUM_LAYERS = 12
OUTPUT_PREFIX = f"naturelm_{args.model}"

# --- Load model ---
print(f"Loading {MODEL_NAME} on {DEVICE}...")
model = load_feature_extractor(
    config_path=CONFIG_PATH,
    model_path=MODEL_PATH,
    device=DEVICE,
    for_inference=True,
)
print("Model loaded.\n")

# --- Stream and extract ---
print(f"Streaming {N_SAMPLES} samples from NatureLM-audio-training...")
ds = load_dataset(
    "EarthSpeciesProject/NatureLM-audio-training",
    split="train",
    streaming=True,
)

# Storage: mean-pooled activations per sample per layer
# Shape will be (n_samples, 12, 768)
mean_pooled = []
metadata_list = []
t_total = time.time()

for i, item in enumerate(ds):
    if i >= N_SAMPLES:
        break

    t0 = time.time()

    # --- Audio ---
    audio_array = item["audio"]["array"]
    sr = item["audio"]["sampling_rate"]
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
    if sr != 16000:
        import torchaudio
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)
    duration = len(audio_tensor) / 16000

    # --- Extract ---
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    audio_tensor = audio_tensor.to(DEVICE)

    with torch.no_grad():
        layer_outputs = model.extract_features(audio_tensor, layers=None)

    # Mean-pool each layer: (1, n_frames, 768) -> (768,)
    pooled = torch.stack([layer.squeeze(0).mean(dim=0).cpu() for layer in layer_outputs])
    mean_pooled.append(pooled.numpy())

    # --- Metadata ---
    meta_raw = item.get("metadata", "{}")
    try:
        meta_parsed = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
    except json.JSONDecodeError:
        meta_parsed = {}

    meta = {
        "index": i,
        "file_name": item.get("file_name") or "",
        "source_dataset": item.get("source_dataset") or "",
        "task": item.get("task") or "",
        "output": item.get("output") or "",
        "duration_s": round(duration, 2),
        "n_frames": layer_outputs[0].shape[1],
        # Taxonomic info from metadata JSON
        "species": meta_parsed.get("species") or "",
        "genus": meta_parsed.get("genus") or "",
        "family": meta_parsed.get("family") or "",
        "order": meta_parsed.get("order") or "",
        "class": meta_parsed.get("class") or "",
    }
    metadata_list.append(meta)

    t_sample = time.time() - t0
    print(f"  [{i+1}/{N_SAMPLES}] {meta['source_dataset']:>12s} | "
          f"{duration:5.1f}s | {meta.get('species', '?'):>30s} | "
          f"{meta['order']:>20s} | {t_sample:.1f}s")

mean_pooled = np.stack(mean_pooled)  # (n_samples, 12, 768)
t_elapsed = time.time() - t_total
print(f"\nCollected {N_SAMPLES} samples in {t_elapsed:.1f}s")
print(f"Activation tensor shape: {mean_pooled.shape}")

# --- Save raw data ---
np.save(f"{OUTPUT_PREFIX}_activations.npy", mean_pooled)
with open(f"{OUTPUT_PREFIX}_metadata.json", "w") as f:
    json.dump(metadata_list, f, indent=2)
print(f"Saved {OUTPUT_PREFIX}_activations.npy and {OUTPUT_PREFIX}_metadata.json")

# --- Compute activation statistics per layer ---
print("\n=== Activation Statistics ===\n")

# L2 norms: per sample per layer
l2_norms = np.linalg.norm(mean_pooled, axis=2)  # (n_samples, 12)
print("L2 norms (mean across samples per layer):")
for layer_idx in range(NUM_LAYERS):
    norms = l2_norms[:, layer_idx]
    print(f"  Layer {layer_idx:2d}: mean={norms.mean():.2f}, std={norms.std():.2f}, "
          f"min={norms.min():.2f}, max={norms.max():.2f}")

# Singular values per layer
print("\nTop-5 singular values per layer:")
for layer_idx in range(NUM_LAYERS):
    layer_data = mean_pooled[:, layer_idx, :]  # (n_samples, 768)
    # Center the data
    centered = layer_data - layer_data.mean(axis=0)
    if centered.shape[0] > 1:
        svs = np.linalg.svd(centered, compute_uv=False)
        top5 = svs[:min(5, len(svs))]
        print(f"  Layer {layer_idx:2d}: {', '.join(f'{s:.2f}' for s in top5)}")
    else:
        print(f"  Layer {layer_idx:2d}: (need >1 sample for SVD)")

# PCA variance explained per layer
if mean_pooled.shape[0] > 2:
    print("\nPCA variance explained (first 3 components):")
    for layer_idx in range(NUM_LAYERS):
        layer_data = mean_pooled[:, layer_idx, :]
        n_components = min(3, layer_data.shape[0] - 1)
        pca = PCA(n_components=n_components)
        pca.fit(layer_data)
        var_exp = pca.explained_variance_ratio_
        print(f"  Layer {layer_idx:2d}: {', '.join(f'{v:.1%}' for v in var_exp)}")

# Intrinsic dimensionality estimate (MLE method)
print("\nIntrinsic dimensionality (participation ratio):")
for layer_idx in range(NUM_LAYERS):
    layer_data = mean_pooled[:, layer_idx, :]
    centered = layer_data - layer_data.mean(axis=0)
    if centered.shape[0] > 1:
        svs = np.linalg.svd(centered, compute_uv=False)
        eigenvalues = svs ** 2
        # Participation ratio: (sum(lambda))^2 / sum(lambda^2)
        pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        print(f"  Layer {layer_idx:2d}: PR = {pr:.2f}")
    else:
        print(f"  Layer {layer_idx:2d}: (need >1 sample)")

# --- Plot L2 norms across layers ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# L2 norm per layer
ax = axes[0]
ax.errorbar(range(NUM_LAYERS), l2_norms.mean(axis=0), yerr=l2_norms.std(axis=0),
            fmt="o-", color="#1976D2", capsize=4, linewidth=2, markersize=8)
ax.set_xlabel("AVES Layer", fontsize=12)
ax.set_ylabel("L2 Norm (mean-pooled)", fontsize=12)
ax.set_title(f"Activation L2 Norms — {MODEL_NAME} (n={N_SAMPLES})", fontsize=13, fontweight="bold")
ax.set_xticks(range(NUM_LAYERS))
ax.grid(True, alpha=0.3)

# Singular value spectrum per layer (heatmap)
ax = axes[1]
max_svs = min(N_SAMPLES, 10)
sv_matrix = np.zeros((NUM_LAYERS, max_svs))
for layer_idx in range(NUM_LAYERS):
    layer_data = mean_pooled[:, layer_idx, :]
    centered = layer_data - layer_data.mean(axis=0)
    if centered.shape[0] > 1:
        svs = np.linalg.svd(centered, compute_uv=False)
        sv_matrix[layer_idx, :min(max_svs, len(svs))] = svs[:max_svs]

im = ax.imshow(sv_matrix, aspect="auto", cmap="viridis")
ax.set_xlabel("Singular Value Index", fontsize=12)
ax.set_ylabel("AVES Layer", fontsize=12)
ax.set_title(f"Singular Value Spectrum — {MODEL_NAME} (n={N_SAMPLES})", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, label="Singular Value")

plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_activation_stats.png", dpi=150, bbox_inches="tight")
print(f"\nSaved plot to {OUTPUT_PREFIX}_activation_stats.png")

# --- Print metadata summary ---
print("\n=== Sample Summary ===")
for m in metadata_list:
    print(f"  {m['index']:3d} | {m['source_dataset']:>12s} | {m['duration_s']:5.1f}s | "
          f"{m.get('order', ''):>20s} | {m.get('species', '')}")
