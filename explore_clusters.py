"""Exploratory interpretability: what do the late-layer sub-clusters correspond to acoustically?

Clusters frame embeddings from layer 11, maps cluster labels back onto a spectrogram,
and exports short audio clips per cluster so you can listen to what each one captures.
"""

import time
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path

from aves import load_feature_extractor
from aves.utils import load_audio

# --- Config ---
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
AUDIO_FILES = {
    "Bullfinch": "./aves/example_audios/XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.mp3",
    "Guineafowl": "./aves/example_audios/XC936872 - Helmeted Guineafowl - Numida meleagris.wav",
}
LAYER = 11  # Last layer — strongest sub-cluster structure
N_CLUSTERS = 6  # Start with 6, can adjust
SAMPLES_PER_FRAME = 320  # AVES downsampling factor: 16000Hz / 50fps
SR = 16000
OUTPUT_DIR = Path("./cluster_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Load model ---
print("Loading model...")
model = load_feature_extractor(
    config_path=CONFIG_PATH,
    model_path=MODEL_PATH,
    device="cpu",
    for_inference=True,
)

# --- Process each audio file ---
for species, audio_path in AUDIO_FILES.items():
    print(f"\n{'='*60}")
    print(f"Processing: {species}")
    print(f"{'='*60}")

    # Load audio
    audio = load_audio(audio_path, mono=True, mono_avg=False)
    n_samples = audio.shape[-1]
    duration = n_samples / SR
    print(f"Audio: {duration:.1f}s, {n_samples} samples")

    # Extract layer 11 embeddings
    t0 = time.time()
    embeddings = model.extract_features(audio, layers=LAYER)  # (1, n_frames, 768)
    embeddings = embeddings.squeeze(0).cpu().numpy()  # (n_frames, 768)
    n_frames = embeddings.shape[0]
    print(f"Extracted {n_frames} frames from layer {LAYER} in {time.time()-t0:.1f}s")

    # --- Cluster ---
    print(f"Clustering into {N_CLUSTERS} groups...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # --- PCA colored by cluster ---
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    # --- Compute spectrogram for the original audio ---
    # Use the raw audio (before AVES processing) for visualization
    audio_for_spec = audio.unsqueeze(0) if audio.ndim == 1 else audio.unsqueeze(0)
    spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=1024, hop_length=SAMPLES_PER_FRAME,
        n_mels=80, f_min=50, f_max=8000,
    )
    mel_spec = spec_transform(audio_for_spec).squeeze().numpy()  # (n_mels, n_spec_frames)
    mel_spec_db = 10 * np.log10(mel_spec + 1e-10)

    # Align spectrogram frames to AVES frames (should be close but may differ by a few)
    min_frames = min(n_frames, mel_spec_db.shape[1])
    mel_spec_db = mel_spec_db[:, :min_frames]
    labels_aligned = labels[:min_frames]

    # --- Colors ---
    cluster_colors = plt.cm.Set2(np.linspace(0, 1, N_CLUSTERS))
    cmap = ListedColormap(cluster_colors)

    # --- Figure: spectrogram + cluster overlay + PCA ---
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 0.5, 2], width_ratios=[3, 1], hspace=0.3, wspace=0.3)

    # Spectrogram
    ax_spec = fig.add_subplot(gs[0, 0])
    time_axis = np.arange(mel_spec_db.shape[1]) * SAMPLES_PER_FRAME / SR
    freq_axis = np.linspace(50, 8000, mel_spec_db.shape[0])
    ax_spec.pcolormesh(time_axis, freq_axis, mel_spec_db, cmap="magma", shading="auto")
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=11)
    ax_spec.set_title(f"{species} — Mel Spectrogram", fontsize=13, fontweight="bold")
    ax_spec.set_xlim(0, time_axis[-1])

    # Cluster strip below spectrogram
    ax_strip = fig.add_subplot(gs[1, 0], sharex=ax_spec)
    strip_time = np.arange(len(labels_aligned)) * SAMPLES_PER_FRAME / SR
    for i in range(len(labels_aligned) - 1):
        ax_strip.axvspan(strip_time[i], strip_time[i+1],
                         color=cluster_colors[labels_aligned[i]], alpha=0.9)
    ax_strip.set_ylabel("Cluster", fontsize=11)
    ax_strip.set_xlabel("Time (s)", fontsize=11)
    ax_strip.set_yticks([])
    ax_strip.set_xlim(0, time_axis[-1])

    # Add cluster legend
    for c in range(N_CLUSTERS):
        count = np.sum(labels_aligned == c)
        pct = count / len(labels_aligned) * 100
        ax_strip.plot([], [], 's', color=cluster_colors[c], markersize=10,
                      label=f"C{c} ({pct:.0f}%)")
    ax_strip.legend(loc="upper right", ncol=N_CLUSTERS, fontsize=8, framealpha=0.9)

    # PCA scatter colored by cluster
    ax_pca = fig.add_subplot(gs[0, 1])
    for c in range(N_CLUSTERS):
        mask = labels == c
        ax_pca.scatter(coords[mask, 0], coords[mask, 1],
                       c=[cluster_colors[c]], alpha=0.4, s=5, label=f"C{c}")
    ax_pca.set_title(f"Layer {LAYER} PCA — {N_CLUSTERS} clusters", fontsize=12, fontweight="bold")
    ax_pca.set_xlabel("PC1", fontsize=10)
    ax_pca.set_ylabel("PC2", fontsize=10)
    ax_pca.legend(fontsize=8, markerscale=3)

    # Cluster statistics
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis("off")

    stats_text = f"Cluster Summary — {species} (Layer {LAYER}, k={N_CLUSTERS})\n\n"
    for c in range(N_CLUSTERS):
        mask = labels_aligned == c
        count = np.sum(mask)
        pct = count / len(labels_aligned) * 100
        # Find contiguous segments for this cluster
        transitions = np.diff(mask.astype(int))
        n_segments = max(1, (np.sum(transitions == 1) + (1 if mask[0] else 0)))
        avg_segment_len = count / n_segments * SAMPLES_PER_FRAME / SR * 1000  # ms
        stats_text += (f"  Cluster {c}: {count} frames ({pct:.1f}%), "
                       f"{n_segments} segments, avg {avg_segment_len:.0f}ms each\n")

    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=11, fontfamily="monospace", verticalalignment="top")

    plt.savefig(OUTPUT_DIR / f"clusters_{species.lower()}.png", dpi=150, bbox_inches="tight")
    print(f"Saved clusters_{species.lower()}.png")

    # --- Export audio clips per cluster ---
    # For each cluster, concatenate the top-5 longest contiguous segments
    audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
    species_dir = OUTPUT_DIR / species.lower()
    species_dir.mkdir(exist_ok=True)

    for c in range(N_CLUSTERS):
        mask = labels_aligned == c

        # Find contiguous runs of this cluster
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

        # Sort by length, take top 5
        runs.sort(key=lambda r: r[1] - r[0], reverse=True)
        top_runs = runs[:5]

        # Concatenate audio for these runs (with 100ms silence gaps)
        silence = np.zeros(int(SR * 0.1))
        segments = []
        for run_start, run_end in top_runs:
            sample_start = run_start * SAMPLES_PER_FRAME
            sample_end = min(run_end * SAMPLES_PER_FRAME, len(audio_np))
            segments.append(audio_np[sample_start:sample_end])
            segments.append(silence)

        if segments:
            clip = np.concatenate(segments)
            clip_tensor = torch.from_numpy(clip).unsqueeze(0).float()
            clip_path = species_dir / f"cluster_{c}.wav"
            torchaudio.save(str(clip_path), clip_tensor, SR)

            total_dur = sum(r[1] - r[0] for r in top_runs) * SAMPLES_PER_FRAME / SR
            print(f"  Cluster {c}: exported {len(top_runs)} segments ({total_dur:.1f}s) → {clip_path}")

print(f"\nDone! Check {OUTPUT_DIR}/ for plots and audio clips.")
print("Listen to the cluster_*.wav files to hear what each cluster captures.")
