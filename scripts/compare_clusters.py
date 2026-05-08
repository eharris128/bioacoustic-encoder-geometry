"""Compare two clusters side-by-side: spectrograms, centroid distance, and the specific
dimensions where they differ most.

Usage: python compare_clusters.py [species] [cluster_a] [cluster_b]
Default: guineafowl 1 2
"""

import sys
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from aves import load_feature_extractor
from aves.utils import load_audio

# --- Args ---
species = sys.argv[1] if len(sys.argv) > 1 else "guineafowl"
cluster_a = int(sys.argv[2]) if len(sys.argv) > 2 else 1
cluster_b = int(sys.argv[3]) if len(sys.argv) > 3 else 2

AUDIO_FILES = {
    "bullfinch": "./aves/example_audios/XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.mp3",
    "guineafowl": "./aves/example_audios/XC936872 - Helmeted Guineafowl - Numida meleagris.wav",
}

CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
SR = 16000
SAMPLES_PER_FRAME = 320
N_CLUSTERS = 6

# --- Load and extract ---
print("Loading model...")
model = load_feature_extractor(config_path=CONFIG_PATH, model_path=MODEL_PATH, device="cpu", for_inference=True)

audio = load_audio(AUDIO_FILES[species], mono=True, mono_avg=False)
embeddings = model.extract_features(audio, layers=-1).squeeze(0).cpu().numpy()

from sklearn.cluster import KMeans
labels = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit_predict(embeddings)

mask_a = labels == cluster_a
mask_b = labels == cluster_b
emb_a = embeddings[mask_a]
emb_b = embeddings[mask_b]

print(f"\nComparing {species} — Cluster {cluster_a} ({mask_a.sum()} frames) vs Cluster {cluster_b} ({mask_b.sum()} frames)")

# --- 1. Which embedding dimensions differ most? ---
mean_a = emb_a.mean(axis=0)
mean_b = emb_b.mean(axis=0)
diff = mean_a - mean_b
top_dims = np.argsort(np.abs(diff))[::-1][:20]

print(f"\nTop 20 dimensions with largest centroid difference:")
print(f"{'Dim':>5} {'C'+str(cluster_a)+' mean':>12} {'C'+str(cluster_b)+' mean':>12} {'Diff':>10}")
for d in top_dims:
    print(f"{d:>5} {mean_a[d]:>12.3f} {mean_b[d]:>12.3f} {diff[d]:>10.3f}")

# --- 2. Cosine similarity between centroids ---
cos_sim = np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b))
l2_dist = np.linalg.norm(mean_a - mean_b)
print(f"\nCentroid cosine similarity: {cos_sim:.4f}")
print(f"Centroid L2 distance: {l2_dist:.2f}")

# Compare to other cluster pairs for context
print(f"\nFor context — all pairwise centroid distances:")
centroids = []
for c in range(N_CLUSTERS):
    centroids.append(embeddings[labels == c].mean(axis=0))
for i in range(N_CLUSTERS):
    for j in range(i+1, N_CLUSTERS):
        d = np.linalg.norm(centroids[i] - centroids[j])
        cs = np.dot(centroids[i], centroids[j]) / (np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j]))
        marker = " <--- comparing these" if (i == cluster_a and j == cluster_b) or (i == cluster_b and j == cluster_a) else ""
        print(f"  C{i} vs C{j}: L2={d:.2f}, cos={cs:.4f}{marker}")

# --- 3. Spectrograms of example segments from each cluster ---
audio_np = audio.numpy()

def get_longest_segments(mask, n=3):
    """Find n longest contiguous runs."""
    runs = []
    in_run = False
    start = 0
    for i in range(len(mask)):
        if mask[i] and not in_run:
            start = i
            in_run = True
        elif not mask[i] and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(mask)))
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    return runs[:n]

segs_a = get_longest_segments(mask_a, 3)
segs_b = get_longest_segments(mask_b, 3)

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
fig.suptitle(f"{species.title()} — Cluster {cluster_a} vs Cluster {cluster_b} (longest segments)\n"
             f"Cosine similarity: {cos_sim:.4f}, L2 distance: {l2_dist:.2f}",
             fontsize=14, fontweight="bold")

spec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=512, hop_length=160, n_mels=80, f_min=50, f_max=8000,
)

for row, (cluster_id, segs, color) in enumerate([
    (cluster_a, segs_a, "#2196F3"),
    (cluster_b, segs_b, "#FF5722"),
]):
    for col, (seg_start, seg_end) in enumerate(segs):
        ax = axes[row, col]
        # Add 5-frame context on each side
        ctx = 5
        s_start = max(0, seg_start - ctx)
        s_end = min(len(audio_np) // SAMPLES_PER_FRAME, seg_end + ctx)

        sample_start = s_start * SAMPLES_PER_FRAME
        sample_end = min(s_end * SAMPLES_PER_FRAME, len(audio_np))
        segment_audio = torch.from_numpy(audio_np[sample_start:sample_end]).unsqueeze(0).float()

        spec = spec_transform(segment_audio).squeeze().numpy()
        spec_db = 10 * np.log10(spec + 1e-10)

        dur = (sample_end - sample_start) / SR
        time_ax = np.linspace(0, dur, spec_db.shape[1])
        freq_ax = np.linspace(50, 8000, spec_db.shape[0])

        ax.pcolormesh(time_ax, freq_ax, spec_db, cmap="magma", shading="auto")

        # Highlight the actual cluster region (not the context)
        cluster_start_t = (seg_start - s_start) * SAMPLES_PER_FRAME / SR
        cluster_end_t = (seg_end - s_start) * SAMPLES_PER_FRAME / SR
        ax.axvline(cluster_start_t, color=color, linewidth=2, linestyle="--", alpha=0.8)
        ax.axvline(cluster_end_t, color=color, linewidth=2, linestyle="--", alpha=0.8)

        seg_dur = (seg_end - seg_start) * SAMPLES_PER_FRAME / SR
        ax.set_title(f"C{cluster_id} seg {col+1} ({seg_dur*1000:.0f}ms, t={seg_start*SAMPLES_PER_FRAME/SR:.1f}s)",
                     fontsize=10, color=color, fontweight="bold")
        ax.set_ylabel("Hz" if col == 0 else "", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)

plt.tight_layout()
outpath = f"cluster_output/compare_c{cluster_a}_vs_c{cluster_b}_{species}.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\nSaved {outpath}")

# --- 4. Distribution of embedding values on the top-5 most different dimensions ---
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle(f"Embedding distributions on dimensions where C{cluster_a} and C{cluster_b} differ most",
             fontsize=13, fontweight="bold")

for i, dim in enumerate(top_dims[:5]):
    ax = axes[i]
    ax.hist(emb_a[:, dim], bins=40, alpha=0.6, color="#2196F3", label=f"C{cluster_a}", density=True)
    ax.hist(emb_b[:, dim], bins=40, alpha=0.6, color="#FF5722", label=f"C{cluster_b}", density=True)
    ax.set_title(f"Dim {dim} (Δ={diff[dim]:.2f})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Activation", fontsize=9)
    if i == 0:
        ax.legend(fontsize=9)

plt.tight_layout()
outpath2 = f"cluster_output/dim_diffs_c{cluster_a}_vs_c{cluster_b}_{species}.png"
plt.savefig(outpath2, dpi=150, bbox_inches="tight")
print(f"Saved {outpath2}")
