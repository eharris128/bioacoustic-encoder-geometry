"""

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
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import soundfile as sf
from scipy import signal as scipy_signal
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from aves import load_feature_extractor


# Audio loading (bypasses torchaudio/torchcodec entirely)
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio (WAV or MP3), convert to mono, resample to target_sr.
    Tries soundfile first (WAV/FLAC), falls back to torchaudio for MP3.
    Returns (1, n_samples) float32 tensor."""
    try:
        data, sr = sf.read(path, always_2d=True)  # (n_samples, n_channels)
        data = data.mean(axis=1)                   # mono
        if sr != target_sr:
            n_out = int(round(len(data) * target_sr / sr))
            data = scipy_signal.resample(data, n_out)
        return torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
    except Exception:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdim=True)  # mono (1, n_samples)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform.float()


# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
NUM_LAYERS = 12
MAX_FRAMES_PER_RECORDING = 1000

SNR_LEVELS_DB = [40.0, 30.0, 20.0, 15.0, 10.0, 7.0, 5.0, 3.0, 1.0, 0.0]

RECORDINGS: dict[str, str] = {
    # --- Helmeted Guineafowl (3) --- add more by copying pattern below
    "guineafowl_01": "audio/helmeted-guinea-fowl/XC280506 - Helmeted Guineafowl - Numida meleagris.wav",
    "guineafowl_02": "audio/helmeted-guinea-fowl/XC364521 - Helmeted Guineafowl - Numida meleagris.wav",
    "guineafowl_03": "audio/helmeted-guinea-fowl/XC709655 - Helmeted Guineafowl - Numida meleagris.wav",
    # --- Eurasian Bullfinch (4) ---
    "bullfinch_01": "audio/bullfinch/XC1077468 - Eurasian Bullfinch - Pyrrhula pyrrhula.wav",
    "bullfinch_02": "audio/bullfinch/XC965743 - Eurasian Bullfinch - Pyrrhula pyrrhula.wav",
    "bullfinch_03": "audio/bullfinch/XC938052 - Eurasian Bullfinch - Pyrrhula pyrrhula.wav",
    "bullfinch_04": "audio/bullfinch/XC805629 - Eurasian Bullfinch - Pyrrhula pyrrhula rosacea.wav",
    # --- Hawfinch (4) ---
    "hawfinch_01": "audio/hawfinch/XC944735 - Hawfinch - Coccothraustes coccothraustes.wav",
    "hawfinch_02": "audio/hawfinch/XC1087947 - Hawfinch - Coccothraustes coccothraustes.wav",
    "hawfinch_03": "audio/hawfinch/XC1086752 - Hawfinch - Coccothraustes coccothraustes.wav",
    "hawfinch_04": "audio/hawfinch/XC1083076 - Hawfinch - Coccothraustes coccothraustes.wav",
}

SPECIES_RECORDINGS: dict[str, tuple[str, int]] = {
    "bullfinch_XC1077468": ("audio/bullfinch/XC1077468 - Eurasian Bullfinch - Pyrrhula pyrrhula.wav",        0),
    "bullfinch_XC965743":  ("audio/bullfinch/XC965743 - Eurasian Bullfinch - Pyrrhula pyrrhula.wav",         0),
    "bullfinch_XC938052":  ("audio/bullfinch/XC938052 - Eurasian Bullfinch - Pyrrhula pyrrhula.wav",         0),
    "bullfinch_XC805629":  ("audio/bullfinch/XC805629 - Eurasian Bullfinch - Pyrrhula pyrrhula rosacea.wav", 0),
    "hawfinch_XC944735":   ("audio/hawfinch/XC944735 - Hawfinch - Coccothraustes coccothraustes.wav",        1),
    "hawfinch_XC1087947":  ("audio/hawfinch/XC1087947 - Hawfinch - Coccothraustes coccothraustes.wav",       1),
    "hawfinch_XC1086752":  ("audio/hawfinch/XC1086752 - Hawfinch - Coccothraustes coccothraustes.wav",       1),
    "hawfinch_XC1084204":  ("audio/hawfinch/XC1084204 - Hawfinch - Coccothraustes coccothraustes.wav",       1),
    "hawfinch_XC1083076":  ("audio/hawfinch/XC1083076 - Hawfinch - Coccothraustes coccothraustes.wav",       1),
}


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

# Noise direction: PCA over SNR-indexed mean activations per layer
# ---------------------------------------------------------------------------

def compute_noise_directions(sweep: dict) -> dict[int, dict]:
    """
    For each layer, stack all (rec × snr) mean activations and fit PCA.
    Returns top 3 PCs (the noise subspace) plus variance explained.
    Returns {layer: {"direction": (768,), "directions_3d": (3, 768),
                     "variance_explained": float, "variance_explained_3d": float}}.
    """
    rec_ids = list(sweep.keys())
    n_components = min(3, len(rec_ids) * len(SNR_LEVELS_DB) - 1)
    directions = {}
    for layer in range(NUM_LAYERS):
        rows = []
        for rec_id in rec_ids:
            rows.append(sweep[rec_id]["snr_means"][:, layer, :])  # (n_snr, 768)
        X = np.concatenate(rows, axis=0)  # (n_rec * n_snr, 768)
        pca = PCA(n_components=n_components)
        pca.fit(X)
        directions[layer] = {
            "direction": pca.components_[0],             # (768,) PC1 unit-norm
            "directions_3d": pca.components_,            # (3, 768) noise subspace
            "variance_explained": float(pca.explained_variance_ratio_[0]),
            "variance_explained_3d": float(pca.explained_variance_ratio_.sum()),
        }
    return directions

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

# 3D subspace overlap utility
# ---------------------------------------------------------------------------

def full_subspace_overlap(noise_pcs: np.ndarray, species_dir: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute how much of species_dir lies in the noise subspace spanned by noise_pcs.

    noise_pcs   : (k, 768) — k orthonormal noise PCs
    species_dir : (768,)   — unit-norm species direction

    Returns:
        total_overlap : scalar in [0, 1] — ||proj of species_dir onto noise subspace||
        per_pc_cos    : (k,) — |cos(theta)| between species_dir and each PC individually
    """
    projections = noise_pcs @ species_dir          # (k,) coordinates in noise subspace
    total_overlap = float(np.linalg.norm(projections))
    per_pc_cos = np.abs(projections)               # (k,) individual cosines
    return total_overlap, per_pc_cos

# Monotonicity check via Spearman correlation
# ---------------------------------------------------------------------------

def compute_monotonicity(sweep: dict, noise_dirs: dict[int, dict]) -> dict[int, dict]:
    """
    For each layer, project each (recording × SNR) mean activation onto noise PC1
    and compute Spearman ρ between the projection and descending SNR level.
    A high |ρ| (negative, since lower SNR = higher noise = larger projection)
    confirms that the noise direction captures a monotonic noise response.

    Returns {layer: {"mean_rho": float, "per_rec_rho": list[float]}}.
    """
    rec_ids = list(sweep.keys())
    # SNR_LEVELS_DB is ordered high→low; noise increases as index increases
    snr_indices = list(range(len(SNR_LEVELS_DB)))  # 0=cleanest, 9=noisiest
    results = {}
    for layer in range(NUM_LAYERS):
        pc1 = noise_dirs[layer]["direction"]  # (768,)
        per_rec_rho = []
        for rec_id in rec_ids:
            means = sweep[rec_id]["snr_means"][:, layer, :]  # (n_snr, 768)
            projections = means @ pc1                         # (n_snr,) scalar projection per SNR
            rho, _ = spearmanr(snr_indices, projections)
            per_rec_rho.append(float(rho))
        results[layer] = {
            "mean_rho": float(np.mean(per_rec_rho)),
            "per_rec_rho": per_rec_rho,
        }
        print(f"    Layer {layer:2d}: mean Spearman ρ = {results[layer]['mean_rho']:+.3f}", flush=True)
    return results

# Species direction (for orthogonality check)
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


# UMAP visualization
# ---------------------------------------------------------------------------

def plot_umap(sweep: dict, layers_to_plot: list[int] = [0, 3, 6, 9, 11]) -> None:
    """
    For selected layers, project all (rec × SNR) mean activations to 2D with UMAP.
    Color = SNR level (clean → noisy). Saves noise_umap.png.
    """
    import umap as umap_lib
    rec_ids = list(sweep.keys())
    n_snr = len(SNR_LEVELS_DB)
    snr_vals = np.array(SNR_LEVELS_DB)

    fig, axes = plt.subplots(1, len(layers_to_plot), figsize=(5 * len(layers_to_plot), 5))
    fig.suptitle(
        "UMAP of noise sweep activations per layer\n"
        "(color = SNR level: yellow=clean, purple=noisy)",
        fontsize=13, fontweight="bold",
    )

    sc = None
    for ax, layer in zip(axes, layers_to_plot):
        X = np.concatenate(
            [sweep[r]["snr_means"][:, layer, :] for r in rec_ids], axis=0
        )  # (n_rec * n_snr, 768)
        labels_snr = np.tile(snr_vals, len(rec_ids))

        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        emb = reducer.fit_transform(X)  # (n_rec * n_snr, 2)

        sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels_snr, cmap="plasma_r",
                        s=40, alpha=0.85, vmin=0, vmax=40)
        ax.set_title(f"Layer {layer}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    if sc is not None:
        plt.colorbar(sc, ax=axes[-1], label="SNR (dB)")
    plt.tight_layout()
    plt.savefig("noiselevelexperiment/noise_umap.png", dpi=150, bbox_inches="tight")
    print("Saved noiselevelexperiment/noise_umap.png")


# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    sweep: dict,
    noise_dirs: dict[int, dict],
    species_dirs: dict[int, np.ndarray] | None,
    monotonicity: dict[int, dict],
) -> dict:
    layers = list(range(NUM_LAYERS))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, NUM_LAYERS))
    rec_ids = list(sweep.keys())

    # Precompute per-layer L2 shifts (used in two figures)
    all_shifts = {}  # layer -> list of shifts, one per SNR index
    for layer in layers:
        shifts = []
        for snr_idx in range(len(SNR_LEVELS_DB)):
            layer_means = np.array([
                sweep[r]["snr_means"][snr_idx, layer, :] for r in rec_ids
            ])
            baseline = np.array([
                sweep[r]["snr_means"][0, layer, :] for r in rec_ids  # 40dB = cleanest
            ])
            shift = float(np.mean(np.linalg.norm(layer_means - baseline, axis=1)))
            shifts.append(shift)
        all_shifts[layer] = shifts

    # ---- Figure 1: L2 shift vs SNR per layer ----
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        "Activation shift vs. recording SNR\n"
        "(mean L2 distance from clean-audio baseline, per layer)",
        fontsize=13, fontweight="bold",
    )
    for layer in layers:
        ax.plot(list(range(len(SNR_LEVELS_DB))), all_shifts[layer], "o-", color=colors[layer],
                linewidth=1.5, markersize=4, label=f"L{layer}")
    ax.set_xticks(range(len(SNR_LEVELS_DB)))
    ax.set_xticklabels([f"{s:.0f}" for s in SNR_LEVELS_DB], fontsize=9)
    ax.set_xlabel("SNR (dB) — right = noisier", fontsize=12)
    ax.set_ylabel("Mean L2 shift from 40dB baseline", fontsize=12)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig("noiselevelexperiment/noise_snr_curves.png", dpi=150, bbox_inches="tight")
    print("Saved noiselevelexperiment/noise_snr_curves.png")

    # ---- Figure 2: variance explained by noise PC1 + monotonicity vs SNR curve ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Monotonicity confirmation: noise PC1 variance explained vs. Spearman ρ (SNR curve)",
        fontsize=13, fontweight="bold",
    )

    var_exp = [noise_dirs[layer]["variance_explained"] for layer in layers]
    bar_colors = plt.cm.Blues(np.array(var_exp) / max(var_exp))
    bars = ax1.bar(layers, var_exp, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Fraction of variance explained (PC1)", fontsize=12)
    ax1.set_title("Noise direction PC1 variance explained", fontsize=11)
    ax1.set_xticks(layers)
    for bar, val in zip(bars, var_exp):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    rhos = [monotonicity[layer]["mean_rho"] for layer in layers]
    rho_colors = plt.cm.RdYlGn(np.array(rhos))  # green = positive (monotonic with noise increase)
    bars2 = ax2.bar(layers, rhos, color=rho_colors, edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(0.9, color="gray", linestyle=":", alpha=0.5, label="ρ=0.9 reference")
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Mean Spearman ρ (noise index vs PC1 projection)", fontsize=12)
    ax2.set_title("Monotonicity: SNR index vs PC1 projection per layer", fontsize=11)
    ax2.set_xticks(layers)
    ax2.set_ylim(-1.05, 1.05)
    ax2.legend(fontsize=9)
    for bar, val in zip(bars2, rhos):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.02 if val >= 0 else -0.07),
                 f"{val:+.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig("noiselevelexperiment/noise_direction_variance.png", dpi=150, bbox_inches="tight")
    print("Saved noiselevelexperiment/noise_direction_variance.png")

    # ---- Figure 3: 3D subspace orthogonality with species direction ----
    ortho_per_layer = None
    if species_dirs is not None:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(
            "Noise subspace vs. species direction orthogonality per layer\n"
            "(3D noise subspace: PC1 + PC2 + PC3)",
            fontsize=13, fontweight="bold",
        )

        ortho_per_layer = {}
        total_overlaps, pc1_cos, pc2_cos, pc3_cos = [], [], [], []

        for layer in layers:
            noise_pcs = noise_dirs[layer]["directions_3d"]   # (3, 768)
            sd = species_dirs[layer]                          # (768,) unit-norm
            total, per_pc = full_subspace_overlap(noise_pcs, sd)
            ortho_per_layer[layer] = {
                "total_overlap": total,
                "pc1_cos": float(per_pc[0]),
                "pc2_cos": float(per_pc[1]) if len(per_pc) > 1 else 0.0,
                "pc3_cos": float(per_pc[2]) if len(per_pc) > 2 else 0.0,
            }
            total_overlaps.append(total)
            pc1_cos.append(float(per_pc[0]))
            pc2_cos.append(float(per_pc[1]) if len(per_pc) > 1 else 0.0)
            pc3_cos.append(float(per_pc[2]) if len(per_pc) > 2 else 0.0)

        # Top panel: total 3D subspace overlap
        top_colors = plt.cm.RdYlGn_r(np.array(total_overlaps))
        bars_top = ax_top.bar(layers, total_overlaps, color=top_colors, edgecolor="black", linewidth=0.5)
        ax_top.axhline(0.1, color="gray", linestyle=":", alpha=0.5, label="0.1 reference")
        ax_top.set_ylabel("||proj(species → noise subspace)||", fontsize=11)
        ax_top.set_title("Total species overlap with 3D noise subspace (lower = more separable)", fontsize=11)
        ax_top.set_ylim(0, 1.05)
        ax_top.legend(fontsize=9)
        for bar, val in zip(bars_top, total_overlaps):
            ax_top.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        # Bottom panel: per-PC breakdown
        x = np.array(layers)
        w = 0.25
        ax_bot.bar(x - w, pc1_cos, width=w, label="PC1 |cos|", color="#4C72B0", edgecolor="black", linewidth=0.5)
        ax_bot.bar(x,     pc2_cos, width=w, label="PC2 |cos|", color="#DD8452", edgecolor="black", linewidth=0.5)
        ax_bot.bar(x + w, pc3_cos, width=w, label="PC3 |cos|", color="#55A868", edgecolor="black", linewidth=0.5)
        ax_bot.set_xlabel("Layer", fontsize=12)
        ax_bot.set_ylabel("|Cosine similarity|", fontsize=11)
        ax_bot.set_title("Per-PC breakdown: species direction vs each noise PC", fontsize=11)
        ax_bot.set_ylim(0, 1.05)
        ax_bot.set_xticks(layers)
        ax_bot.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig("noiselevelexperiment/noise_species_ortho.png", dpi=150, bbox_inches="tight")
        print("Saved noiselevelexperiment/noise_species_ortho.png")

    return {
        "variance_explained_per_layer": {
            str(k): noise_dirs[k]["variance_explained"] for k in layers
        },
        "best_noise_layer": int(max(layers, key=lambda k: noise_dirs[k]["variance_explained"])),
        "monotonicity_rho_per_layer": {str(k): monotonicity[k]["mean_rho"] for k in layers},
        "ortho_per_layer": (
            {str(k): v for k, v in ortho_per_layer.items()}
            if ortho_per_layer is not None else None
        ),
    }


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

    print("\nFitting noise directions (PCA per layer, 3 components)...", flush=True)
    noise_dirs = compute_noise_directions(sweep)

    print("\nFinding PC elbow (components needed for 80% variance)...", flush=True)
    elbow = find_num_components(sweep, threshold=0.80)

    print("\nChecking monotonicity (Spearman ρ: SNR index vs PC1 projection)...", flush=True)
    monotonicity = compute_monotonicity(sweep, noise_dirs)

    print("\nComputing species directions (if available)...", flush=True)
    species_dirs = compute_species_directions(model)
    if species_dirs is None:
        print("  Skipping orthogonality analysis (SPECIES_RECORDINGS not populated or files missing).")

    print("\nPlotting...", flush=True)
    summary = plot_results(sweep, noise_dirs, species_dirs, monotonicity)

    print("\nRunning UMAP...", flush=True)
    plot_umap(sweep)

    best = summary["best_noise_layer"]
    print(f"\nNoise Direction Summary:")
    print(f"  Layer with highest noise variance explained: {best} "
          f"({noise_dirs[best]['variance_explained']:.3f})")
    print("  Variance explained per layer (PC1 / 3D subspace):")
    for k in range(NUM_LAYERS):
        v1 = noise_dirs[k]["variance_explained"]
        v3 = noise_dirs[k]["variance_explained_3d"]
        print(f"    Layer {k:2d}: PC1={v1:.3f}  3D={v3:.3f}")
    print("\n  Components for 80% variance per layer:")
    for k, v in elbow.items():
        print(f"    Layer {k:2d}: {v}")
    print("\n  Monotonicity (mean Spearman ρ per layer):")
    for k, v in summary["monotonicity_rho_per_layer"].items():
        print(f"    Layer {int(k):2d}: {float(v):+.3f}")
    if summary["ortho_per_layer"]:
        print("\n  3D noise subspace overlap with species direction:")
        for k, v in summary["ortho_per_layer"].items():
            print(f"    Layer {int(k):2d}: total={v['total_overlap']:.3f}  "
                  f"PC1={v['pc1_cos']:.3f}  PC2={v['pc2_cos']:.3f}  PC3={v['pc3_cos']:.3f}")


if __name__ == "__main__":
    main()
