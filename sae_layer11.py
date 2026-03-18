"""Sparse Autoencoder on AVES layer-11 embeddings (Bullfinch).

Trains a sparse autoencoder (SAE) to decompose the 768-dim AVES layer-11
representation into a larger set of sparse, interpretable directions.

Architecture:
  - Encoder: Linear(768, hidden_dim) + ReLU
  - Decoder: Linear(hidden_dim, 768, bias=False)  — columns normalised to unit norm
  - Loss:    MSE reconstruction  +  lambda_l1 * L1 sparsity on hidden activations

Outputs:
  sae_layer11_training.png  — loss and sparsity curves
  sae_layer11_features.png  — per-feature mean activation heat-map
  sae_layer11_weights.npz   — encoder/decoder weights for downstream analysis
  result.json               — written by the job wrapper
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from aves import load_feature_extractor
from aves.utils import load_audio

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
TARGET_LAYER = 11           # 0-indexed; layer 11 is the last transformer layer
EMBED_DIM = 768
HIDDEN_DIM = 3072           # 4× expansion factor
LAMBDA_L1 = 1e-3            # sparsity coefficient
LR = 1e-3
BATCH_SIZE = 512
EPOCHS = 100
MAX_FRAMES_PER_RECORDING = 3000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BULLFINCH_RECORDINGS = [
    "audio/bullfinch/XC1077468.mp3",
    "audio/bullfinch/XC965743.mp3",
    "audio/bullfinch/XC938052.mp3",
    "audio/bullfinch/XC805629.mp3",
]

# ---------------------------------------------------------------------------
# SAE model
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        # No bias on decoder; unit-norm columns enforced each step
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.encoder(x))
        x_hat = self.decoder(h)
        return h, x_hat

    def normalise_decoder(self) -> None:
        """Clamp decoder column norms to ≤1 (prevents trivial unit-scaling)."""
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1.0)
            self.decoder.weight.div_(norms)

# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_layer11_embeddings(model, paths: list[str]) -> np.ndarray:
    all_frames: list[np.ndarray] = []
    for path in paths:
        print(f"  Extracting: {path}", flush=True)
        audio = load_audio(path, mono=True, mono_avg=False)
        t0 = time.time()
        layer_outputs = model.extract_features(audio, layers=None)
        elapsed = time.time() - t0
        emb = layer_outputs[TARGET_LAYER].squeeze(0).cpu().numpy()  # (n_frames, 768)
        if emb.shape[0] > MAX_FRAMES_PER_RECORDING:
            rng = np.random.default_rng(42)
            idx = rng.choice(emb.shape[0], MAX_FRAMES_PER_RECORDING, replace=False)
            idx.sort()
            emb = emb[idx]
        print(f"    {emb.shape[0]} frames in {elapsed:.1f}s", flush=True)
        all_frames.append(emb)
    return np.concatenate(all_frames, axis=0)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sae(
    data: np.ndarray,
) -> tuple[SparseAutoencoder, list[float], list[float], list[float]]:
    X = torch.tensor(data, dtype=torch.float32).to(DEVICE)

    # Normalise inputs to zero mean, unit variance per-feature
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp(min=1e-6)
    X = (X - mean) / std

    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    sae = SparseAutoencoder(EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    optim = torch.optim.Adam(sae.parameters(), lr=LR)

    recon_losses: list[float] = []
    l1_losses: list[float] = []
    sparsities: list[float] = []  # mean fraction of active features per sample

    print(f"\nTraining SAE: {EMBED_DIM} → {HIDDEN_DIM} → {EMBED_DIM}", flush=True)
    print(f"  Device: {DEVICE}, Epochs: {EPOCHS}, Batch: {BATCH_SIZE}", flush=True)
    print(f"  Data: {X.shape[0]} frames", flush=True)

    for epoch in range(EPOCHS):
        epoch_recon = 0.0
        epoch_l1 = 0.0
        epoch_sparsity = 0.0
        n_batches = 0

        for (xb,) in loader:
            h, x_hat = sae(xb)
            recon = F.mse_loss(x_hat, xb)
            l1 = LAMBDA_L1 * h.abs().mean()
            loss = recon + l1

            optim.zero_grad()
            loss.backward()
            optim.step()
            sae.normalise_decoder()

            epoch_recon += recon.item()
            epoch_l1 += l1.item()
            # Fraction of active features (h > 0) per sample, averaged over batch
            epoch_sparsity += (h > 0).float().mean(dim=1).mean().item()
            n_batches += 1

        epoch_recon /= n_batches
        epoch_l1 /= n_batches
        epoch_sparsity /= n_batches

        recon_losses.append(epoch_recon)
        l1_losses.append(epoch_l1)
        sparsities.append(epoch_sparsity)

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{EPOCHS}: "
                f"recon={epoch_recon:.4f}, l1={epoch_l1:.4f}, "
                f"sparsity={epoch_sparsity:.3f}",
                flush=True,
            )

    return sae, recon_losses, l1_losses, sparsities

# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_feature_stats(sae: SparseAutoencoder, data: np.ndarray) -> np.ndarray:
    """Return mean activation of each hidden feature across all frames."""
    X = torch.tensor(data, dtype=torch.float32).to(DEVICE)
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp(min=1e-6)
    X = (X - mean) / std

    all_h: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, X.shape[0], BATCH_SIZE):
            xb = X[i : i + BATCH_SIZE]
            h, _ = sae(xb)
            all_h.append(h.cpu())
    H = torch.cat(all_h, dim=0)  # (N, hidden_dim)
    return H.numpy()

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training(
    recon_losses: list[float],
    l1_losses: list[float],
    sparsities: list[float],
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"SAE Training: AVES Layer {TARGET_LAYER} Bullfinch Embeddings\n"
        f"({EMBED_DIM}→{HIDDEN_DIM}, λ={LAMBDA_L1})",
        fontsize=13,
        fontweight="bold",
    )

    epochs = range(1, EPOCHS + 1)
    ax1.plot(epochs, recon_losses, label="Reconstruction MSE", color="steelblue")
    ax1.plot(epochs, l1_losses, label="L1 penalty", color="darkorange", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.set_yscale("log")

    ax2.plot(epochs, sparsities, color="seagreen")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean active fraction")
    ax2.set_title("Mean Sparsity (fraction of features > 0)")
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("sae_layer11_training.png", dpi=150, bbox_inches="tight")
    print("Saved sae_layer11_training.png")


def plot_features(H: np.ndarray) -> dict[str, float]:
    """Plot feature mean activations and return summary statistics."""
    mean_act = H.mean(axis=0)          # (hidden_dim,)
    frac_active = (H > 0).mean(axis=0) # (hidden_dim,)
    dead_frac = float((frac_active == 0).mean())

    # Sort features by mean activation
    order = np.argsort(mean_act)[::-1]
    top_n = 50

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"SAE Feature Analysis: AVES Layer {TARGET_LAYER} Bullfinch\n"
        f"({HIDDEN_DIM} features, {H.shape[0]} frames)",
        fontsize=13,
        fontweight="bold",
    )

    # Top-N mean activations
    ax = axes[0, 0]
    ax.bar(range(top_n), mean_act[order[:top_n]], color="steelblue")
    ax.set_title(f"Top {top_n} features by mean activation")
    ax.set_xlabel("Feature rank")
    ax.set_ylabel("Mean activation")

    # Histogram of mean activations (all features)
    ax = axes[0, 1]
    ax.hist(mean_act, bins=100, color="darkorange", log=True)
    ax.set_title("Distribution of mean activations (all features)")
    ax.set_xlabel("Mean activation")
    ax.set_ylabel("Count (log scale)")

    # Fraction of frames each feature is active
    ax = axes[1, 0]
    ax.hist(frac_active, bins=50, color="seagreen", log=True)
    ax.set_title(f"Feature activation frequency\n(dead features: {dead_frac:.1%})")
    ax.set_xlabel("Fraction of frames active")
    ax.set_ylabel("Count (log scale)")

    # Heatmap of top-20 features across a sample of frames
    ax = axes[1, 1]
    sample_frames = min(200, H.shape[0])
    rng = np.random.default_rng(42)
    frame_idx = rng.choice(H.shape[0], sample_frames, replace=False)
    frame_idx.sort()
    H_sample = H[frame_idx][:, order[:20]].T  # (20, sample_frames)
    im = ax.imshow(H_sample, aspect="auto", cmap="viridis")
    ax.set_title("Top-20 feature activations across sampled frames")
    ax.set_xlabel("Frame (sampled)")
    ax.set_ylabel("Feature rank")
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"F{order[i]}" for i in range(20)], fontsize=7)
    plt.colorbar(im, ax=ax, label="Activation")

    plt.tight_layout()
    plt.savefig("sae_layer11_features.png", dpi=150, bbox_inches="tight")
    print("Saved sae_layer11_features.png")

    return {
        "dead_feature_frac": dead_frac,
        "mean_active_frac": float(frac_active.mean()),
        "max_mean_activation": float(mean_act.max()),
        "n_features_above_mean": int((mean_act > mean_act.mean()).sum()),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading AVES model...", flush=True)
    model = load_feature_extractor(
        config_path=CONFIG_PATH,
        model_path=MODEL_PATH,
        device="cpu",
        for_inference=True,
    )

    print(f"\nExtracting layer-{TARGET_LAYER} embeddings from Bullfinch recordings...", flush=True)
    embeddings = extract_layer11_embeddings(model, BULLFINCH_RECORDINGS)
    print(f"Total frames: {embeddings.shape[0]}, dim: {embeddings.shape[1]}", flush=True)

    sae, recon_losses, l1_losses, sparsities = train_sae(embeddings)

    print("\nComputing feature statistics...", flush=True)
    H = compute_feature_stats(sae, embeddings)

    print("\nPlotting...", flush=True)
    plot_training(recon_losses, l1_losses, sparsities)
    feature_stats = plot_features(H)

    print("\nSaving weights...", flush=True)
    np.savez(
        "sae_layer11_weights.npz",
        encoder_weight=sae.encoder.weight.detach().cpu().numpy(),
        encoder_bias=sae.encoder.bias.detach().cpu().numpy(),
        decoder_weight=sae.decoder.weight.detach().cpu().numpy(),
    )
    print("Saved sae_layer11_weights.npz")

    # Summary line for result.json parsing
    final_recon = recon_losses[-1]
    final_sparsity = sparsities[-1]
    dead_frac = feature_stats["dead_feature_frac"]

    print(f"\nSAE Summary:")
    print(f"  Final reconstruction loss: {final_recon:.6f}")
    print(f"  Final sparsity (active fraction): {final_sparsity:.4f}")
    print(f"  Dead features: {dead_frac:.2%}")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Target layer: {TARGET_LAYER}")


if __name__ == "__main__":
    main()
