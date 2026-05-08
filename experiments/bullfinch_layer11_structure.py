"""Within-species structure: what does AVES layer 11 encode in Bullfinch?

Motivation
----------
Cross-species probing (RUN-000001) shows layer 11 distinguishes Bullfinch from
Hawfinch at 93.5%. But that is a between-species result. The question here is:

  What structure exists in layer-11 representations *within* a single species?

If AVES layer 11 encodes call type, syllable identity, or vocalization state,
we expect k-means clusters to align with acoustic features (energy, spectral
centroid, ZCR). If clusters are acoustically arbitrary, the within-species
representation is opaque to simple acoustic analysis.

Method
------
1. Extract layer-11 embeddings from all Bullfinch recordings.
2. K-means cluster them (k = 2, 4, 6, 8, 10) — sweep k to find natural structure.
3. For each cluster: compute mean acoustic features (energy, spectral centroid,
   zero-crossing rate, spectral rolloff) as proxies for call type.
4. Measure cluster-acoustic alignment: do acoustic features predict cluster
   membership (one-vs-rest logistic regression accuracy)?
5. Export max-activating audio segments for the top clusters (WAV snippets
   for manual listening / future annotation).

North Star
----------
Does AVES layer 11 organize Bullfinch frames by acoustically meaningful
call-type structure, or is its within-species representation opaque?

Proxy task
----------
Acoustic feature → cluster prediction accuracy (above chance = yes, it encodes
something acoustically interpretable; at chance = opaque).

Outputs
-------
  layer11_cluster_pca.png          — PCA of layer-11 embeddings, colored by cluster
  layer11_cluster_acoustics.png    — per-cluster acoustic feature profiles
  layer11_cluster_alignment.png    — acoustic prediction accuracy vs. k
  layer11_silhouette.png           — silhouette score vs. k (natural k selection)
  audio_snippets/cluster_N_top.wav — 10s montage of max-energy frames per cluster
  result.json                      — written by job wrapper
"""

from __future__ import annotations

import os
import struct
import time
import wave
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import torch
import torchaudio

from aves import load_feature_extractor
from aves.utils import load_audio

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
TARGET_LAYER = 10  # layer 10 is the causal bottleneck per RUN-000006
FRAME_HOP_MS = 20        # AVES frame stride ~20ms
SAMPLE_RATE = 16000
FRAME_SAMPLES = int(FRAME_HOP_MS / 1000 * SAMPLE_RATE)  # samples per frame
SNIPPET_FRAMES = 10      # frames per audio snippet
K_VALUES = [2, 4, 6, 8, 10]
MAX_FRAMES_PER_RECORDING = 3000

BULLFINCH_RECORDINGS = [
    "audio/bullfinch/XC1077468.mp3",
    "audio/bullfinch/XC965743.mp3",
    "audio/bullfinch/XC938052.mp3",
    "audio/bullfinch/XC805629.mp3",
]

# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_layer11_with_audio(
    model, paths: list[str]
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """
    Returns:
      embeddings: (N, 768) layer-11 embeddings, all recordings concatenated
      audio_frames: list of (n_frames_i, FRAME_SAMPLES) waveform arrays
      rec_ids: list of recording id per frame (length N)
    """
    all_embs: list[np.ndarray] = []
    all_audio_frames: list[np.ndarray] = []
    all_rec_ids: list[str] = []

    for path in paths:
        rec_id = Path(path).stem
        print(f"  {rec_id}...", end=" ", flush=True)

        # Load waveform for audio snippet extraction
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        wav = waveform.squeeze(0).numpy()  # (n_samples,)

        # Extract embeddings
        audio = load_audio(path, mono=True, mono_avg=False)
        t0 = time.time()
        layer_outputs = model.extract_features(audio, layers=None)
        elapsed = time.time() - t0

        emb = layer_outputs[TARGET_LAYER].squeeze(0).cpu().numpy()  # (n_frames, 768)
        n_frames = emb.shape[0]

        # Slice waveform into per-frame chunks aligned to embeddings
        # AVES uses a CNN front-end with hop ~20ms; approximate alignment
        frame_chunks = []
        for i in range(n_frames):
            start = i * FRAME_SAMPLES
            end = start + FRAME_SAMPLES
            chunk = wav[start:end] if end <= len(wav) else np.zeros(FRAME_SAMPLES)
            if len(chunk) < FRAME_SAMPLES:
                chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
            frame_chunks.append(chunk)
        audio_frames_arr = np.stack(frame_chunks, axis=0)  # (n_frames, FRAME_SAMPLES)

        # Cap frames
        if n_frames > MAX_FRAMES_PER_RECORDING:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_frames, MAX_FRAMES_PER_RECORDING, replace=False)
            idx.sort()
            emb = emb[idx]
            audio_frames_arr = audio_frames_arr[idx]
            n_frames = MAX_FRAMES_PER_RECORDING

        print(f"{n_frames} frames, {elapsed:.1f}s", flush=True)
        all_embs.append(emb)
        all_audio_frames.append(audio_frames_arr)
        all_rec_ids.extend([rec_id] * n_frames)

    embeddings = np.concatenate(all_embs, axis=0)
    audio_frames = np.concatenate(all_audio_frames, axis=0)
    return embeddings, audio_frames, all_rec_ids

# ---------------------------------------------------------------------------
# Acoustic features (per frame)
# ---------------------------------------------------------------------------

def compute_acoustic_features(audio_frames: np.ndarray) -> np.ndarray:
    """
    Compute per-frame acoustic features.
    Returns (N, 4): [log_energy, spectral_centroid, zcr, spectral_rolloff]
    """
    N = audio_frames.shape[0]
    features = np.zeros((N, 4))

    for i, frame in enumerate(audio_frames):
        # Log energy
        energy = np.sum(frame ** 2) + 1e-10
        features[i, 0] = np.log(energy)

        # Spectral centroid
        fft = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), d=1.0 / SAMPLE_RATE)
        fft_sum = fft.sum() + 1e-10
        features[i, 1] = np.sum(freqs * fft) / fft_sum

        # Zero-crossing rate
        features[i, 2] = float(np.sum(np.abs(np.diff(np.sign(frame))))) / len(frame)

        # Spectral rolloff (85th percentile of energy)
        cumsum = np.cumsum(fft)
        rolloff_thresh = 0.85 * cumsum[-1]
        rolloff_idx = np.searchsorted(cumsum, rolloff_thresh)
        features[i, 3] = freqs[min(rolloff_idx, len(freqs) - 1)]

    return features

# ---------------------------------------------------------------------------
# K-means sweep
# ---------------------------------------------------------------------------

def run_kmeans_sweep(
    embeddings: np.ndarray,
    acoustic_features: np.ndarray,
    k_values: list[int],
) -> dict:
    sc = StandardScaler()
    X = sc.fit_transform(embeddings)

    results = {}
    for k in k_values:
        print(f"  k={k}...", end=" ", flush=True)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        sil = silhouette_score(X, labels, sample_size=min(5000, len(X)))

        # Acoustic prediction: can acoustic features predict cluster labels?
        ac_sc = StandardScaler()
        X_ac = ac_sc.fit_transform(acoustic_features)
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                                 multi_class="auto")
        clf.fit(X_ac, labels)
        ac_acc = accuracy_score(labels, clf.predict(X_ac))
        chance = 1.0 / k

        print(f"sil={sil:.3f}, acoustic_acc={ac_acc:.3f} (chance={chance:.2f})", flush=True)
        results[k] = {
            "labels": labels,
            "km": km,
            "silhouette": sil,
            "acoustic_acc": ac_acc,
            "chance": chance,
        }

    return results, sc

# ---------------------------------------------------------------------------
# Audio snippet export
# ---------------------------------------------------------------------------

def export_snippets(
    audio_frames: np.ndarray,
    labels: np.ndarray,
    acoustic_features: np.ndarray,
    k: int,
    out_dir: Path,
) -> None:
    """For each cluster, export a WAV montage of the top-energy frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_energy = acoustic_features[:, 0]

    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster_idx = np.where(mask)[0]
        if len(cluster_idx) == 0:
            continue

        # Select top-energy frames from this cluster
        top_n = min(25, len(cluster_idx))
        energies = log_energy[cluster_idx]
        top_order = np.argsort(energies)[::-1][:top_n]
        selected_idx = cluster_idx[top_order]

        # Build montage: frames + short silence between them
        silence = np.zeros(FRAME_SAMPLES // 4, dtype=np.float32)
        segments = []
        for idx in selected_idx:
            seg = audio_frames[idx].astype(np.float32)
            # Normalize segment
            peak = np.abs(seg).max()
            if peak > 1e-6:
                seg = seg / peak * 0.8
            segments.append(seg)
            segments.append(silence)

        montage = np.concatenate(segments)

        # Save as WAV
        out_path = out_dir / f"cluster_{cluster_id:02d}_top_energy.wav"
        _write_wav(out_path, montage, SAMPLE_RATE)


def _write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    samples_int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_int16.tobytes())

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    embeddings: np.ndarray,
    acoustic_features: np.ndarray,
    sweep_results: dict,
    k_values: list[int],
    best_k: int,
) -> None:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(StandardScaler().fit_transform(embeddings))
    var = pca.explained_variance_ratio_

    best = sweep_results[best_k]
    labels = best["labels"]

    # ---- Figure 1: PCA colored by cluster ----
    fig, axes = plt.subplots(1, len(k_values), figsize=(5 * len(k_values), 5))
    fig.suptitle(
        f"AVES Layer-{TARGET_LAYER} Bullfinch Embeddings: K-Means Clusters\n"
        f"(PCA 2D, {embeddings.shape[0]} frames from {len(BULLFINCH_RECORDINGS)} recordings)",
        fontsize=13, fontweight="bold",
    )
    for ax, k in zip(axes, k_values):
        lbl = sweep_results[k]["labels"]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=lbl, cmap="tab10",
                        alpha=0.3, s=2, rasterized=True)
        sil = sweep_results[k]["silhouette"]
        ac = sweep_results[k]["acoustic_acc"]
        chance = sweep_results[k]["chance"]
        ax.set_title(f"k={k}\nsil={sil:.3f}, ac={ac:.2f} (ch={chance:.2f})", fontsize=10)
        ax.set_xlabel(f"PC1 ({var[0]:.0%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({var[1]:.0%})", fontsize=8)
        ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig("layer11_cluster_pca.png", dpi=150, bbox_inches="tight")
    print("Saved layer11_cluster_pca.png")

    # ---- Figure 2: Acoustic profiles per cluster (best k) ----
    feat_names = ["Log energy", "Spectral centroid (Hz)", "ZCR", "Spectral rolloff (Hz)"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"Per-Cluster Acoustic Profiles (k={best_k}, best silhouette)\n"
        f"Error bars = ±1 std across frames",
        fontsize=13, fontweight="bold",
    )
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    for feat_idx, (ax, name) in enumerate(zip(axes, feat_names)):
        for cluster_id in range(best_k):
            mask = labels == cluster_id
            vals = acoustic_features[mask, feat_idx]
            ax.bar(cluster_id, vals.mean(), yerr=vals.std(),
                   color=colors[cluster_id], edgecolor="black",
                   linewidth=0.5, capsize=4,
                   label=f"C{cluster_id} (n={mask.sum()})")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Cluster", fontsize=10)
        ax.set_xticks(range(best_k))
        if feat_idx == 0:
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("layer11_cluster_acoustics.png", dpi=150, bbox_inches="tight")
    print("Saved layer11_cluster_acoustics.png")

    # ---- Figure 3: Silhouette + acoustic accuracy vs k ----
    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        "Cluster Quality vs. k\n"
        "Silhouette: embedding coherence | Acoustic acc: acoustic interpretability",
        fontsize=12, fontweight="bold",
    )
    sils = [sweep_results[k]["silhouette"] for k in k_values]
    ac_accs = [sweep_results[k]["acoustic_acc"] for k in k_values]
    chances = [sweep_results[k]["chance"] for k in k_values]

    ax1.plot(k_values, sils, "o-", color="steelblue", linewidth=2,
             markersize=8, label="Silhouette score")
    ax1.set_xlabel("k (number of clusters)", fontsize=12)
    ax1.set_ylabel("Silhouette score", fontsize=12, color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(k_values, ac_accs, "s--", color="darkorange", linewidth=2,
             markersize=8, label="Acoustic pred. accuracy")
    ax2.plot(k_values, chances, ":+", color="gray", linewidth=1,
             markersize=6, label="Chance baseline")
    ax2.set_ylabel("Acoustic → cluster accuracy", fontsize=12, color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper right")
    ax1.set_xticks(k_values)
    plt.tight_layout()
    plt.savefig("layer11_silhouette.png", dpi=150, bbox_inches="tight")
    print("Saved layer11_silhouette.png")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading AVES model on {device}...", flush=True)
    model = load_feature_extractor(
        config_path=CONFIG_PATH,
        model_path=MODEL_PATH,
        device=device,
        for_inference=True,
    )

    print(f"\nExtracting layer-{TARGET_LAYER} embeddings + audio frames...", flush=True)
    embeddings, audio_frames, rec_ids = extract_layer11_with_audio(
        model, BULLFINCH_RECORDINGS
    )
    print(f"Total: {embeddings.shape[0]} frames, dim {embeddings.shape[1]}", flush=True)

    print("\nComputing acoustic features...", flush=True)
    t0 = time.time()
    acoustic_features = compute_acoustic_features(audio_frames)
    print(f"Done in {time.time()-t0:.1f}s", flush=True)

    print(f"\nRunning k-means sweep: k={K_VALUES}...", flush=True)
    sweep_results, scaler = run_kmeans_sweep(embeddings, acoustic_features, K_VALUES)

    # Best k by silhouette
    best_k = max(K_VALUES, key=lambda k: sweep_results[k]["silhouette"])
    print(f"\nBest k by silhouette: {best_k} "
          f"(sil={sweep_results[best_k]['silhouette']:.3f})", flush=True)

    print("\nExporting audio snippets...", flush=True)
    export_snippets(
        audio_frames,
        sweep_results[best_k]["labels"],
        acoustic_features,
        best_k,
        Path("audio_snippets"),
    )
    print(f"Saved audio_snippets/ ({best_k} clusters)", flush=True)

    print("\nPlotting...", flush=True)
    plot_results(embeddings, acoustic_features, sweep_results, K_VALUES, best_k)

    best_sil = sweep_results[best_k]["silhouette"]
    best_ac = sweep_results[best_k]["acoustic_acc"]
    best_chance = sweep_results[best_k]["chance"]
    ac_lift = best_ac - best_chance

    print(f"\nLayer-11 Within-Species Structure Summary:")
    print(f"  Best k:              {best_k}")
    print(f"  Silhouette score:    {best_sil:.4f}")
    print(f"  Acoustic accuracy:   {best_ac:.4f} (chance = {best_chance:.3f}, lift = {ac_lift:.4f})")
    print(f"  Total frames:        {embeddings.shape[0]}")


if __name__ == "__main__":
    main()
