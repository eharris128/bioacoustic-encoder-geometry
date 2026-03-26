"""Noise-in-audio experiment: where does recording noise live in AVES activation space?

Motivation
----------
Prior experiments (contrastive_patch_species.py) identified the per-layer species direction
by comparing mean activations across species. This experiment asks an analogous question
for acoustic background noise: is there a consistent linear direction in activation space
that encodes recording noise level?

Method
------
For each recording in RECORDINGS and each SNR level in SNR_LEVELS_DB:
  1. Add calibrated white noise to the raw audio waveform at that SNR
  2. Run a clean forward pass through AVES (no activation hooks)
  3. Extract and subsample frame-level activations at all 12 layers
  4. Compute per-layer mean activation across frames

Per layer, fit PCA to the (n_snr × n_recordings, 768) matrix of SNR-indexed mean
activations. The first PC is the noise direction at that layer — the axis of maximum
variance explained by noise level.

Orthogonality check: if SPECIES_RECORDINGS is populated, compute the per-layer species
direction (normalize(mean_species1 - mean_species0)) and report |cosine similarity| with
the noise direction. Low cosine similarity → noise and species are geometrically separable.

Outputs
-------
  noiselevelexperiment/noise_snr_curves.png         — L2 activation shift vs SNR, per layer
  noiselevelexperiment/noise_direction_variance.png — variance explained by noise PC1, per layer
  noiselevelexperiment/noise_species_ortho.png      — |cos sim| noise dir vs species dir (optional)
  result.json                                       — written by job wrapper
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import soundfile as sf
from scipy import signal as scipy_signal
from sklearn.decomposition import PCA

from aves import load_feature_extractor

# ---------------------------------------------------------------------------
# Audio loading (bypasses torchaudio/torchcodec entirely)
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load WAV via soundfile, convert to mono, resample to target_sr.
    Returns (1, n_samples) float32 tensor."""
    data, sr = sf.read(path, always_2d=True)  # (n_samples, n_channels)
    data = data.mean(axis=1)                   # mono
    if sr != target_sr:
        n_out = int(round(len(data) * target_sr / sr))
        data = scipy_signal.resample(data, n_out)
    return torch.from_numpy(data.astype(np.float32)).unsqueeze(0)  # (1, n_samples)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
NUM_LAYERS = 12
MAX_FRAMES_PER_RECORDING = 1000

SNR_LEVELS_DB = [40.0, 30.0, 20.0, 15.0, 10.0, 7.0, 5.0, 3.0, 1.0, 0.0]

RECORDINGS: dict[str, str] = {
     "guineafowl_01": "audio/helmeted-guinea-fowl/XC280506 - Helmeted Guineafowl - Numida meleagris.wav",                                           
      "guineafowl_02": "audio/helmeted-guinea-fowl/XC364521 - Helmeted Guineafowl - Numida meleagris.wav",                                           
      "guineafowl_03": "audio/helmeted-guinea-fowl/XC709655 - Helmeted Guineafowl - Numida meleagris.wav",                                           
  }       


SPECIES_RECORDINGS: dict[str, tuple[str, int]] = {
    # "bullfinch_XC1077468": ("audio/bullfinch/XC1077468.mp3", 0),
    # "hawfinch_XC944735":   ("audio/hawfinch/XC944735.mp3",   1),
}

# ---------------------------------------------------------------------------
# Audio noise addition
# ---------------------------------------------------------------------------

def add_white_noise(audio: torch.Tensor, snr_db: float, rng: np.random.Generator) -> torch.Tensor:
    """
    Add white Gaussian noise to audio at a target SNR.

    audio  : (1, n_samples) float32 tensor at 16kHz
    snr_db : target signal-to-noise ratio in dB
    Returns noisy audio tensor of same shape.
    """
    signal = audio.numpy().astype(np.float64)
    signal_power = np.mean(signal ** 2)
    if signal_power < 1e-10:
        return audio  # silent input — can't calibrate SNR
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), signal.shape).astype(np.float32)
    return torch.from_numpy((signal + noise).astype(np.float32))

# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_layer_means(
    model,
    audio: torch.Tensor,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Run a single clean forward pass and return per-layer mean activation.
    Returns (NUM_LAYERS, 768) array.
    """
    layer_outputs = model.extract_features(audio, layers=None)
    means = []
    for lo in layer_outputs:
        frames = lo.squeeze(0).cpu().numpy()  # (n_frames, 768)
        n = frames.shape[0]
        if n > MAX_FRAMES_PER_RECORDING:
            idx = rng.choice(n, MAX_FRAMES_PER_RECORDING, replace=False)
            idx.sort()
            frames = frames[idx]
        means.append(frames.mean(axis=0))  # (768,)
    return np.stack(means, axis=0)  # (NUM_LAYERS, 768)

# ---------------------------------------------------------------------------
# SNR sweep
# ---------------------------------------------------------------------------

def run_snr_sweep(model, recordings: dict[str, str]) -> dict:
    """
    For each recording × SNR level, extract per-layer mean activations.
    Returns {rec_id: {"snr_means": (n_snr, NUM_LAYERS, 768), "audio_path": str}}.
    """
    rng = np.random.default_rng(42)
    results = {}
    for rec_id, path in recordings.items():
        print(f"  {rec_id}...", flush=True)
        audio_clean = load_audio(path)
        snr_means = []
        for snr_db in SNR_LEVELS_DB:
            noisy = add_white_noise(audio_clean, snr_db, rng)
            t0 = time.time()
            means = extract_layer_means(model, noisy, rng)  # (NUM_LAYERS, 768)
            elapsed = time.time() - t0
            snr_means.append(means)
            print(f"    SNR={snr_db:5.1f}dB  {elapsed:.1f}s", flush=True)
        results[rec_id] = {
            "snr_means": np.stack(snr_means, axis=0),  # (n_snr, NUM_LAYERS, 768)
            "audio_path": path,
        }
    return results

# ---------------------------------------------------------------------------
# Noise direction: PCA over SNR-indexed mean activations per layer
# ---------------------------------------------------------------------------

def compute_noise_directions(sweep: dict) -> dict[int, dict]:
    """
    For each layer, stack all (rec × snr) mean activations and fit PCA.
    The first PC is the noise direction.
    Returns {layer: {"direction": (768,), "variance_explained": float}}.
    """
    rec_ids = list(sweep.keys())
    directions = {}
    for layer in range(NUM_LAYERS):
        rows = []
        for rec_id in rec_ids:
            rows.append(sweep[rec_id]["snr_means"][:, layer, :])  # (n_snr, 768)
        X = np.concatenate(rows, axis=0)  # (n_rec * n_snr, 768)
        pca = PCA(n_components=1)
        pca.fit(X)
        directions[layer] = {
            "direction": pca.components_[0],  # (768,) unit-norm
            "variance_explained": float(pca.explained_variance_ratio_[0]),
        }
    return directions

# ---------------------------------------------------------------------------
# PC elbow: minimum components to explain threshold variance per layer
# ---------------------------------------------------------------------------

def find_num_components(sweep: dict, threshold: float = 0.80) -> dict[int, int]:
    """
    For each layer, find the minimum number of PCs needed to explain
    `threshold` fraction of variance in the (n_rec × n_snr, 768) activation matrix.
    Returns {layer: n_components}.
    """
    rec_ids = list(sweep.keys())
    max_components = min(10, len(rec_ids) * len(SNR_LEVELS_DB))
    components_needed = {}
    for layer in range(NUM_LAYERS):
        rows = []
        for rec_id in rec_ids:
            rows.append(sweep[rec_id]["snr_means"][:, layer, :])
        X = np.concatenate(rows, axis=0)
        pca = PCA(n_components=max_components)
        pca.fit(X)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        hits = np.where(cumulative >= threshold)[0]
        n_components = int(hits[0] + 1) if len(hits) > 0 else max_components
        components_needed[layer] = n_components
        print(f"    Layer {layer:2d}: {n_components} components to explain {threshold:.0%} variance "
              f"(PC1={pca.explained_variance_ratio_[0]:.3f})", flush=True)
    return components_needed

# ---------------------------------------------------------------------------
# Species direction (optional, for orthogonality check)
# ---------------------------------------------------------------------------

def compute_species_directions(model) -> dict[int, np.ndarray] | None:
    """
    Compute normalize(mean_species1 - mean_species0) per layer.
    Returns None if SPECIES_RECORDINGS is empty or any file is missing.
    """
    if not SPECIES_RECORDINGS:
        return None
    rng = np.random.default_rng(42)
    layer_means: dict[int, dict[int, list[np.ndarray]]] = {
        layer: {0: [], 1: []} for layer in range(NUM_LAYERS)
    }
    for rec_id, (path, label) in SPECIES_RECORDINGS.items():
        if not Path(path).exists():
            print(f"  Warning: {path} not found — skipping orthogonality analysis", flush=True)
            return None
        audio = load_audio(path)
        means = extract_layer_means(model, audio, rng)  # (NUM_LAYERS, 768)
        for layer in range(NUM_LAYERS):
            layer_means[layer][label].append(means[layer])
    directions = {}
    for layer in range(NUM_LAYERS):
        m0 = np.mean(layer_means[layer][0], axis=0)
        m1 = np.mean(layer_means[layer][1], axis=0)
        diff = m1 - m0
        norm = np.linalg.norm(diff)
        directions[layer] = diff / norm if norm > 1e-8 else diff
    return directions

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    sweep: dict,
    noise_dirs: dict[int, dict],
    species_dirs: dict[int, np.ndarray] | None,
) -> dict:
    layers = list(range(NUM_LAYERS))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, NUM_LAYERS))
    rec_ids = list(sweep.keys())

    # ---- Figure 1: L2 shift vs SNR per layer ----
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        "Activation shift vs. recording SNR\n"
        "(mean L2 distance from clean-audio baseline, per layer)",
        fontsize=13, fontweight="bold",
    )
    for layer in layers:
        shifts = []
        for snr_idx in range(len(SNR_LEVELS_DB)):
            layer_means = np.array([
                sweep[r]["snr_means"][snr_idx, layer, :] for r in rec_ids
            ])  # (n_rec, 768)
            baseline = np.array([
                sweep[r]["snr_means"][0, layer, :] for r in rec_ids  # index 0 = 40dB (cleanest)
            ])
            shift = float(np.mean(np.linalg.norm(layer_means - baseline, axis=1)))
            shifts.append(shift)
        # Plot with x-axis as SNR descending (left = noisier)
        ax.plot(list(range(len(SNR_LEVELS_DB))), shifts, "o-", color=colors[layer],
                linewidth=1.5, markersize=4, label=f"L{layer}")
    ax.set_xticks(range(len(SNR_LEVELS_DB)))
    ax.set_xticklabels([f"{s:.0f}" for s in SNR_LEVELS_DB], fontsize=9)
    ax.set_xlabel("SNR (dB) — right = noisier", fontsize=12)
    ax.set_ylabel("Mean L2 shift from 40dB baseline", fontsize=12)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig("noiselevelexperiment/noise_snr_curves.png", dpi=150, bbox_inches="tight")
    print("Saved noiselevelexperiment/noise_snr_curves.png")

    # ---- Figure 2: variance explained by noise PC1 per layer ----
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Variance explained by noise direction (PC1) per layer\n"
        "(higher = noise defines a consistent linear direction at this layer)",
        fontsize=13, fontweight="bold",
    )
    var_exp = [noise_dirs[layer]["variance_explained"] for layer in layers]
    bar_colors = plt.cm.Blues(np.array(var_exp) / max(var_exp))
    bars = ax.bar(layers, var_exp, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Fraction of variance explained (PC1)", fontsize=12)
    ax.set_xticks(layers)
    for bar, val in zip(bars, var_exp):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig("noiselevelexperiment/noise_direction_variance.png", dpi=150, bbox_inches="tight")
    print("Saved noiselevelexperiment/noise_direction_variance.png")

    # ---- Figure 3: orthogonality with species direction (optional) ----
    ortho_per_layer = None
    if species_dirs is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(
            "|Cosine similarity| between noise direction and species direction per layer\n"
            "(lower = noise and species are geometrically separable)",
            fontsize=13, fontweight="bold",
        )
        ortho_per_layer = {}
        cosines = []
        for layer in layers:
            nd = noise_dirs[layer]["direction"]
            sd = species_dirs[layer]
            cos_sim = float(abs(np.dot(nd, sd) / (np.linalg.norm(nd) * np.linalg.norm(sd) + 1e-10)))
            ortho_per_layer[layer] = cos_sim
            cosines.append(cos_sim)
        bar_colors = plt.cm.RdYlGn_r(np.array(cosines))
        bars = ax.bar(layers, cosines, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0.1, color="gray", linestyle=":", alpha=0.5, label="|cos|=0.1 reference")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("|Cosine similarity|", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(layers)
        ax.legend(fontsize=9)
        for bar, val in zip(bars, cosines):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        plt.tight_layout()
        plt.savefig("noiselevelexperiment/noise_species_ortho.png", dpi=150, bbox_inches="tight")
        print("Saved noiselevelexperiment/noise_species_ortho.png")

    return {
        "variance_explained_per_layer": {
            str(k): noise_dirs[k]["variance_explained"] for k in layers
        },
        "best_noise_layer": int(max(layers, key=lambda k: noise_dirs[k]["variance_explained"])),
        "ortho_per_layer": (
            {str(k): v for k, v in ortho_per_layer.items()}
            if ortho_per_layer is not None else None
        ),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not RECORDINGS:
        print("RECORDINGS is empty — populate it with audio file paths before running.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading AVES model on {device}...", flush=True)
    model = load_feature_extractor(
        config_path=CONFIG_PATH,
        model_path=MODEL_PATH,
        device=device,
        for_inference=True,
    )

    print(
        f"\nRunning SNR sweep ({len(SNR_LEVELS_DB)} levels × {len(RECORDINGS)} recordings)...",
        flush=True,
    )
    sweep = run_snr_sweep(model, RECORDINGS)

    print("\nFitting noise directions (PCA per layer)...", flush=True)
    noise_dirs = compute_noise_directions(sweep)

    print("\nFinding PC elbow (components needed for 80% variance)...", flush=True)
    elbow = find_num_components(sweep, threshold=0.80)

    print("\nComputing species directions (if available)...", flush=True)
    species_dirs = compute_species_directions(model)
    if species_dirs is None:
        print("  Skipping orthogonality analysis (SPECIES_RECORDINGS not populated or files missing).")

    print("\nPlotting...", flush=True)
    summary = plot_results(sweep, noise_dirs, species_dirs)

    best = summary["best_noise_layer"]
    print(f"\nNoise Direction Summary:")
    print(f"  Layer with highest noise variance explained: {best} "
          f"({noise_dirs[best]['variance_explained']:.3f})")
    print("  Variance explained per layer:")
    for k, v in summary["variance_explained_per_layer"].items():
        print(f"    Layer {int(k):2d}: {v:.3f}")
    print("\n  Components for 80% variance per layer:")
    for k, v in elbow.items():
        print(f"    Layer {k:2d}: {v}")
    if summary["ortho_per_layer"]:
        print("  |Cosine similarity| noise vs species:")
        for k, v in summary["ortho_per_layer"].items():
            print(f"    Layer {int(k):2d}: {v:.4f}")


if __name__ == "__main__":
    main()
