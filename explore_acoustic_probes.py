"""Acoustic feature probing: what physical sound properties does each layer encode?

Extracts frame-level acoustic features (energy, spectral centroid, pitch,
spectral bandwidth, zero-crossing rate) directly from audio, then trains
linear probes per layer to predict each feature from embeddings.

Also characterizes the late-layer clusters by their acoustic profiles.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import torch
import torchaudio
from aves import load_feature_extractor
from aves.utils import load_audio

# --- Config ---
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
AUDIO_DIR = Path("./audio/bullfinch")
SKIP = {"XC1086809.mp3", "XC657517.mp3"}
NUM_LAYERS = 12
N_CLUSTERS = 8
SR = 16000
HOP = 320  # AVES downsampling factor — aligns acoustic features to model frames


def extract_acoustic_features(audio, sr=16000, hop_length=320):
    """Extract frame-level acoustic features aligned to AVES frame rate.

    Returns dict of {feature_name: (n_frames,)} arrays.
    """
    if isinstance(audio, torch.Tensor):
        audio_np = audio.numpy()
    else:
        audio_np = audio

    if audio_np.ndim > 1:
        audio_np = audio_np[0]

    n_frames = len(audio_np) // hop_length

    features = {}

    # 1. RMS Energy — per frame
    rms = np.array([
        np.sqrt(np.mean(audio_np[i * hop_length:(i + 1) * hop_length] ** 2))
        for i in range(n_frames)
    ])
    features["Energy (RMS)"] = rms

    # 2. Zero-crossing rate — per frame
    zcr = np.array([
        np.mean(np.abs(np.diff(np.sign(audio_np[i * hop_length:(i + 1) * hop_length])))) / 2
        for i in range(n_frames)
    ])
    features["Zero-Crossing Rate"] = zcr

    # 3. Spectral features via STFT
    # Use same hop_length to align with AVES frames
    stft = np.abs(np.fft.rfft(
        np.array([audio_np[i * hop_length:(i + 1) * hop_length] for i in range(n_frames)]),
        n=hop_length, axis=1
    ))
    freqs = np.fft.rfftfreq(hop_length, d=1.0 / sr)

    # Spectral centroid: weighted mean of frequencies
    power = stft ** 2
    power_sum = power.sum(axis=1, keepdims=True) + 1e-10
    spectral_centroid = (power * freqs[np.newaxis, :]).sum(axis=1) / power_sum.squeeze()
    features["Spectral Centroid (Hz)"] = spectral_centroid

    # Spectral bandwidth: weighted std of frequencies
    spectral_bw = np.sqrt(
        (power * (freqs[np.newaxis, :] - spectral_centroid[:, np.newaxis]) ** 2).sum(axis=1)
        / power_sum.squeeze()
    )
    features["Spectral Bandwidth (Hz)"] = spectral_bw

    # Spectral flatness: geometric mean / arithmetic mean (1 = noise, 0 = tonal)
    log_power = np.log(power + 1e-10)
    geo_mean = np.exp(log_power.mean(axis=1))
    arith_mean = power.mean(axis=1) + 1e-10
    features["Spectral Flatness"] = geo_mean / arith_mean

    # High-frequency energy ratio (above 4kHz vs total)
    freq_4k_idx = np.searchsorted(freqs, 4000)
    hf_ratio = power[:, freq_4k_idx:].sum(axis=1) / power_sum.squeeze()
    features["HF Energy Ratio (>4kHz)"] = hf_ratio

    return features


# --- Load model ---
print("Loading model...")
model = load_feature_extractor(
    config_path=CONFIG_PATH,
    model_path=MODEL_PATH,
    device="cpu",
    for_inference=True,
)

# --- Load recordings, extract embeddings + acoustic features ---
audio_files = sorted([f for f in AUDIO_DIR.glob("*.mp3") if f.name not in SKIP])[:12]
print(f"Using {len(audio_files)} recordings\n")

all_embeddings = {i: [] for i in range(NUM_LAYERS)}
all_acoustic = None  # Will init on first file
feature_names = None

for i, path in enumerate(audio_files):
    print(f"  [{i+1:2d}/{len(audio_files)}] {path.stem}...", end=" ", flush=True)
    try:
        audio = load_audio(str(path), mono=True, mono_avg=False)
    except Exception:
        print("SKIP")
        continue

    # Extract AVES embeddings
    t0 = time.time()
    layer_outputs = model.extract_features(audio, layers=None)
    elapsed = time.time() - t0

    # Extract acoustic features
    acoustic = extract_acoustic_features(audio, sr=SR, hop_length=HOP)

    if feature_names is None:
        feature_names = list(acoustic.keys())
        all_acoustic = {name: [] for name in feature_names}

    # Align frame counts (acoustic and AVES may differ by a few frames)
    n_aves = layer_outputs[0].shape[1]
    n_acoustic = len(acoustic[feature_names[0]])
    n_frames = min(n_aves, n_acoustic)

    for layer_idx in range(NUM_LAYERS):
        emb = layer_outputs[layer_idx].squeeze(0)[:n_frames].cpu().numpy()
        all_embeddings[layer_idx].append(emb)

    for name in feature_names:
        all_acoustic[name].append(acoustic[name][:n_frames])

    print(f"{n_frames} frames, {elapsed:.1f}s")

# Concatenate
for layer_idx in range(NUM_LAYERS):
    all_embeddings[layer_idx] = np.concatenate(all_embeddings[layer_idx], axis=0)
for name in feature_names:
    all_acoustic[name] = np.concatenate(all_acoustic[name], axis=0)

n_total = all_embeddings[0].shape[0]
print(f"\nTotal frames: {n_total}")

# --- Probe each layer for each acoustic feature ---
print(f"\nProbing {NUM_LAYERS} layers for {len(feature_names)} acoustic features...")
print("(Ridge regression with PCA to 50 dims, 80/20 temporal split)\n")

r2_matrix = np.zeros((NUM_LAYERS, len(feature_names)))

# Global train/test split (80/20)
split = int(0.8 * n_total)

for fi, feat_name in enumerate(feature_names):
    y = all_acoustic[feat_name]
    y_train, y_test = y[:split], y[split:]

    for layer_idx in range(NUM_LAYERS):
        X = all_embeddings[layer_idx]

        # PCA to 50 dims for speed
        pca = PCA(n_components=50, random_state=42)
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X_train = pca.fit_transform(scaler_x.fit_transform(X[:split]))
        X_test = pca.transform(scaler_x.transform(X[split:]))
        yt = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        yt_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        reg = Ridge(alpha=1.0)
        reg.fit(X_train, yt)
        r2 = r2_score(yt_test, reg.predict(X_test))
        r2_matrix[layer_idx, fi] = max(0, r2)  # Clip negative R2 to 0

    print(f"  {feat_name:30s}: best layer {np.argmax(r2_matrix[:, fi])} "
          f"(R²={r2_matrix[:, fi].max():.3f})")

# --- Cluster acoustic profiles ---
print(f"\nClustering layer 11 embeddings (k={N_CLUSTERS})...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(all_embeddings[11])

# --- Plot 1: R² heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                          gridspec_kw={"width_ratios": [2, 1.2]})
fig.suptitle("Acoustic Feature Probing: What does each layer encode?\n"
             f"(Ridge regression R², {n_total} Bullfinch frames, PCA→50 dims)",
             fontsize=14, fontweight="bold")

ax = axes[0]
im = ax.imshow(r2_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=r2_matrix.max())
ax.set_xlabel("Acoustic Feature", fontsize=12)
ax.set_ylabel("Layer", fontsize=12)
ax.set_title("Linear Probe R² per Layer", fontsize=12)
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels(feature_names, fontsize=9, rotation=30, ha="right")
ax.set_yticks(range(NUM_LAYERS))
plt.colorbar(im, ax=ax, label="R²")

for i in range(NUM_LAYERS):
    for j in range(len(feature_names)):
        ax.text(j, i, f"{r2_matrix[i,j]:.2f}", ha="center", va="center",
                fontsize=7, color="black" if r2_matrix[i,j] < 0.3 else "white")

# Line plot: R² vs layer for each feature
ax = axes[1]
for fi, feat_name in enumerate(feature_names):
    ax.plot(range(NUM_LAYERS), r2_matrix[:, fi], "o-", label=feat_name, linewidth=2, markersize=4)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("R²", fontsize=12)
ax.set_title("Feature Decodability Across Layers", fontsize=12)
ax.set_xticks(range(NUM_LAYERS))
ax.legend(fontsize=8, loc="best")

plt.tight_layout()
plt.savefig("acoustic_probes.png", dpi=150, bbox_inches="tight")
print("Saved acoustic_probes.png")

# --- Plot 2: Cluster acoustic profiles ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"Acoustic Profiles of Layer 11 Clusters (k={N_CLUSTERS})\n"
             f"What does each cluster sound like?",
             fontsize=14, fontweight="bold")

cluster_colors = plt.cm.Set2(np.linspace(0, 1, N_CLUSTERS))

for fi, feat_name in enumerate(feature_names):
    ax = axes[fi // 3, fi % 3]

    cluster_means = []
    cluster_stds = []
    for c in range(N_CLUSTERS):
        mask = cluster_labels == c
        vals = all_acoustic[feat_name][mask]
        cluster_means.append(vals.mean())
        cluster_stds.append(vals.std())

    x = np.arange(N_CLUSTERS)
    ax.bar(x, cluster_means, yerr=cluster_stds, color=cluster_colors,
           edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xlabel("Cluster", fontsize=10)
    ax.set_ylabel(feat_name, fontsize=10)
    ax.set_title(feat_name, fontsize=11, fontweight="bold")
    ax.set_xticks(x)

plt.tight_layout()
plt.savefig("cluster_acoustic_profiles.png", dpi=150, bbox_inches="tight")
print("Saved cluster_acoustic_profiles.png")
