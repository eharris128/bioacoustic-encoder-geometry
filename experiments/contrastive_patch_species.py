"""Contrastive direction patching: find the causal bottleneck layer for species.

Motivation
----------
Mean-replacement patching (RUN-000003) showed transfer_acc=1.0 at every layer,
because replacing all activations with the opposite species mean is an overwhelming
intervention regardless of layer. This tells us each layer's mean *contains* species
information, not which layer is causally *load-bearing*.

This experiment uses additive direction patching with a scale sweep:

  patched_activation = activation + alpha * d_k

where d_k = normalize(mean_B_k - mean_A_k) is the unit-norm species direction at layer k,
and alpha is swept over a range from 0 (no intervention) to large (strong intervention).

The layer requiring the *smallest* alpha to flip 50% of frames to the opposite species
at layer 11 is the minimal causal bottleneck — the layer where species information is
most directly load-bearing for the final representation.

Outputs
-------
  contrastive_patch_alpha_sweep.png  — flip rate vs alpha, one curve per layer
  contrastive_patch_summary.png      — alpha_50 (half-flip scale) per layer
  result.json                        — written by job wrapper
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from aves import load_feature_extractor
from aves.utils import load_audio

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
NUM_LAYERS = 12
MAX_FRAMES_PER_RECORDING = 1000  # smaller cap: we run many forward passes

# Alpha sweep: from gentle nudge to strong push
# Units are multiples of the unit-norm species direction
ALPHA_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

RECORDINGS = {
    "bullfinch_XC1077468": ("audio/bullfinch/XC1077468.mp3", 0),
    "bullfinch_XC965743":  ("audio/bullfinch/XC965743.mp3",  0),
    "bullfinch_XC938052":  ("audio/bullfinch/XC938052.mp3",  0),
    "bullfinch_XC805629":  ("audio/bullfinch/XC805629.mp3",  0),
    "hawfinch_XC944735":   ("audio/hawfinch/XC944735.mp3",   1),
    "hawfinch_XC1087947":  ("audio/hawfinch/XC1087947.mp3",  1),
    "hawfinch_XC1086752":  ("audio/hawfinch/XC1086752.mp3",  1),
    "hawfinch_XC1084204":  ("audio/hawfinch/XC1084204.mp3",  1),
    "hawfinch_XC1083076":  ("audio/hawfinch/XC1083076.mp3",  1),
}

# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_all_layers(model, paths_labels: dict) -> dict[str, dict]:
    results = {}
    for rec_id, (path, label) in paths_labels.items():
        print(f"  {rec_id}...", end=" ", flush=True)
        audio = load_audio(path, mono=True, mono_avg=False)
        t0 = time.time()
        layer_outputs = model.extract_features(audio, layers=None)
        elapsed = time.time() - t0
        stacked = np.stack([lo.squeeze(0).cpu().numpy() for lo in layer_outputs], axis=0)
        stacked = stacked.transpose(1, 0, 2)  # (n_frames, n_layers, 768)
        n_frames = stacked.shape[0]
        if n_frames > MAX_FRAMES_PER_RECORDING:
            idx = np.random.default_rng(42).choice(n_frames, MAX_FRAMES_PER_RECORDING, replace=False)
            idx.sort()
            stacked = stacked[idx]
        print(f"{stacked.shape[0]} frames, {elapsed:.1f}s", flush=True)
        results[rec_id] = {"embs": stacked, "label": label, "audio_path": path}
    return results

# ---------------------------------------------------------------------------
# Probe training (all data, one probe per layer — used for layer-11 evaluation)
# ---------------------------------------------------------------------------

def train_layer11_probe_loro(
    data: dict[str, dict],
) -> tuple[LogisticRegression, StandardScaler, float]:
    """Train final layer-11 probe on all data; report LORO accuracy."""
    rec_ids = list(data.keys())
    fold_accs = []
    for test_rec in rec_ids:
        train_recs = [r for r in rec_ids if r != test_rec]
        X_train = np.concatenate([data[r]["embs"][:, 11, :] for r in train_recs])
        y_train = np.concatenate([
            np.full(data[r]["embs"].shape[0], data[r]["label"]) for r in train_recs
        ])
        X_test = data[test_rec]["embs"][:, 11, :]
        y_test = np.full(X_test.shape[0], data[test_rec]["label"])
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf.fit(sc.fit_transform(X_train), y_train)
        fold_accs.append(accuracy_score(y_test, clf.predict(sc.transform(X_test))))

    X_all = np.concatenate([data[r]["embs"][:, 11, :] for r in rec_ids])
    y_all = np.concatenate([
        np.full(data[r]["embs"].shape[0], data[r]["label"]) for r in rec_ids
    ])
    sc_final = StandardScaler()
    clf_final = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    clf_final.fit(sc_final.fit_transform(X_all), y_all)
    loro_acc = float(np.mean(fold_accs))
    print(f"  Layer-11 probe LORO accuracy: {loro_acc:.1%}", flush=True)
    return clf_final, sc_final, loro_acc


def compute_species_directions(
    data: dict[str, dict], train_recs: list[str]
) -> dict[int, np.ndarray]:
    """
    Return {layer: unit_norm_direction} where direction points from species 0 → species 1.
    """
    directions = {}
    for layer in range(NUM_LAYERS):
        frames_0 = np.concatenate([
            data[r]["embs"][:, layer, :] for r in train_recs if data[r]["label"] == 0
        ])
        frames_1 = np.concatenate([
            data[r]["embs"][:, layer, :] for r in train_recs if data[r]["label"] == 1
        ])
        diff = frames_1.mean(axis=0) - frames_0.mean(axis=0)
        norm = np.linalg.norm(diff)
        directions[layer] = diff / norm if norm > 1e-8 else diff
    return directions

# ---------------------------------------------------------------------------
# Contrastive direction patching
# ---------------------------------------------------------------------------

def contrastive_patch_run(
    model,
    audio,
    patch_layer: int,
    direction: np.ndarray,
    alpha: float,
    sign: float,  # +1 to push toward species 1, -1 to push toward species 0
) -> np.ndarray:
    """
    Run forward pass with additive hook at layer k:
      output += alpha * sign * direction
    Returns layer-11 embeddings: (n_frames, 768).
    """
    patch_vec = torch.from_numpy(alpha * sign * direction).float()
    try:
        device = next(model.model.parameters()).device
    except Exception:
        device = torch.device("cpu")
    patch_vec = patch_vec.to(device)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            return (h + patch_vec.unsqueeze(0).unsqueeze(0),) + output[1:]
        return output + patch_vec.unsqueeze(0).unsqueeze(0)

    layer_module = model.model.encoder.transformer.layers[patch_layer]
    hook = layer_module.register_forward_hook(hook_fn)
    try:
        layer_outputs = model.extract_features(audio, layers=None)
    finally:
        hook.remove()
    return layer_outputs[11].squeeze(0).cpu().numpy()


def run_alpha_sweep(
    model,
    data: dict[str, dict],
    probe: LogisticRegression,
    scaler: StandardScaler,
) -> dict[int, dict[float, float]]:
    """
    For each (patch_layer, alpha), compute mean flip rate across all test recordings.
    Returns {patch_layer: {alpha: flip_rate}}.
    """
    rec_ids = list(data.keys())
    # results[layer][alpha] = list of flip rates (one per fold)
    results: dict[int, dict[float, list[float]]] = {
        k: {a: [] for a in ALPHA_VALUES} for k in range(NUM_LAYERS)
    }

    for test_rec in rec_ids:
        train_recs = [r for r in rec_ids if r != test_rec]
        test_label = data[test_rec]["label"]
        opposite_label = 1 - test_label
        # sign: push toward opposite species
        sign = 1.0 if test_label == 0 else -1.0
        species_dirs = compute_species_directions(data, train_recs)

        print(
            f"  test={test_rec} ({'Bullfinch' if test_label == 0 else 'Hawfinch'} → "
            f"{'Hawfinch' if test_label == 0 else 'Bullfinch'})",
            flush=True,
        )

        audio_path = data[test_rec]["audio_path"]
        audio = load_audio(audio_path, mono=True, mono_avg=False)

        for patch_layer in range(NUM_LAYERS):
            direction = species_dirs[patch_layer]
            layer_results = []
            for alpha in ALPHA_VALUES:
                l11 = contrastive_patch_run(
                    model, audio, patch_layer, direction, alpha, sign
                )
                # Cap frames
                n = min(l11.shape[0], MAX_FRAMES_PER_RECORDING)
                if l11.shape[0] > n:
                    idx = np.random.default_rng(42).choice(l11.shape[0], n, replace=False)
                    idx.sort()
                    l11 = l11[idx]
                preds = probe.predict(scaler.transform(l11))
                flip_rate = float((preds == opposite_label).mean())
                results[patch_layer][alpha].append(flip_rate)
                layer_results.append(f"α={alpha:.0f}→{flip_rate:.2f}")
            print(f"    layer {patch_layer:2d}: {' | '.join(layer_results)}", flush=True)

    return {
        k: {a: float(np.mean(v)) for a, v in alphas.items()}
        for k, alphas in results.items()
    }

# ---------------------------------------------------------------------------
# Alpha_50: the alpha at which flip rate crosses 50%
# ---------------------------------------------------------------------------

def compute_alpha50(flip_rates: dict[float, float]) -> float:
    """
    Interpolate the alpha at which flip rate = 0.50.
    Returns inf if 50% is never reached.
    """
    alphas = sorted(flip_rates.keys())
    for i in range(len(alphas) - 1):
        a0, a1 = alphas[i], alphas[i + 1]
        r0, r1 = flip_rates[a0], flip_rates[a1]
        if r0 <= 0.5 <= r1:
            # Linear interpolation
            if r1 == r0:
                return a0
            t = (0.5 - r0) / (r1 - r0)
            return float(a0 + t * (a1 - a0))
    return float("inf")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    sweep: dict[int, dict[float, float]],
) -> dict:
    alphas = sorted(ALPHA_VALUES)
    layers = list(range(NUM_LAYERS))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, NUM_LAYERS))

    alpha50_per_layer = {k: compute_alpha50(sweep[k]) for k in layers}
    finite_alpha50 = [v for v in alpha50_per_layer.values() if not np.isinf(v)]
    best_layer = int(min(alpha50_per_layer, key=lambda k: alpha50_per_layer[k]))

    # ---- Figure 1: alpha sweep curves ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Contrastive Direction Patching: Causal Bottleneck for Species\n"
        "(Bullfinch ↔ Hawfinch, LORO, additive patch along species direction)",
        fontsize=13, fontweight="bold",
    )

    ax = axes[0]
    for layer in layers:
        flip_rates = [sweep[layer][a] for a in alphas]
        ax.plot(alphas, flip_rates, "o-", color=colors[layer],
                linewidth=1.5, markersize=4, label=f"L{layer}")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5, label="50% flip")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("Patch scale α (log scale)", fontsize=12)
    ax.set_ylabel("Flip rate at layer 11", fontsize=12)
    ax.set_title("Flip rate vs. patch scale, by patch layer", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7, ncol=2, loc="upper left")

    # ---- Figure 2: alpha_50 bar chart ----
    ax = axes[1]
    a50_vals = [
        alpha50_per_layer[k] if not np.isinf(alpha50_per_layer[k]) else max(alphas) * 2
        for k in layers
    ]
    bar_colors = plt.cm.RdYlGn_r(np.array(a50_vals) / (max(alphas) * 2))
    bars = ax.bar(layers, a50_vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.axhline(max(alphas), color="gray", linestyle=":", alpha=0.5, label="Max α tested")
    ax.set_xlabel("Patch layer", fontsize=12)
    ax.set_ylabel("α₅₀ (scale to flip 50% of frames)", fontsize=12)
    ax.set_title(
        "Causal sensitivity: α needed to flip 50% of frames\n(lower = more causally load-bearing)",
        fontsize=11,
    )
    ax.set_xticks(layers)
    ax.legend(fontsize=9)

    # Annotate best layer
    ax.bar(best_layer, a50_vals[best_layer], color="gold", edgecolor="black",
           linewidth=1.5, label=f"Best: L{best_layer}")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, a50_vals):
        label = f"{val:.1f}" if val < max(alphas) * 1.5 else ">max"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(alphas) * 0.02,
            label,
            ha="center", va="bottom", fontsize=7,
        )

    plt.tight_layout()
    plt.savefig("contrastive_patch_alpha_sweep.png", dpi=150, bbox_inches="tight")
    print("Saved contrastive_patch_alpha_sweep.png")

    # ---- Figure 3: summary heatmap (layer × alpha → flip rate) ----
    flip_matrix = np.array([
        [sweep[k][a] for a in alphas] for k in layers
    ])  # (n_layers, n_alphas)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        "Causal Sensitivity Heatmap\n"
        "(flip rate at layer 11 vs. patch layer and scale α)",
        fontsize=13, fontweight="bold",
    )
    im = ax.imshow(flip_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   origin="lower")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([str(a) for a in alphas], fontsize=9)
    ax.set_yticks(layers)
    ax.set_yticklabels([f"Layer {k}" for k in layers], fontsize=9)
    ax.set_xlabel("Patch scale α", fontsize=12)
    ax.set_ylabel("Patch layer", fontsize=12)
    plt.colorbar(im, ax=ax, label="Flip rate at layer 11")
    # Annotate cells
    for i in range(NUM_LAYERS):
        for j in range(len(alphas)):
            ax.text(j, i, f"{flip_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if 0.2 < flip_matrix[i, j] < 0.8 else "white")
    plt.tight_layout()
    plt.savefig("contrastive_patch_summary.png", dpi=150, bbox_inches="tight")
    print("Saved contrastive_patch_summary.png")

    return {
        "best_causal_layer": best_layer,
        "best_alpha50": float(alpha50_per_layer[best_layer])
        if not np.isinf(alpha50_per_layer[best_layer]) else None,
        "alpha50_per_layer": {
            str(k): (float(v) if not np.isinf(v) else None)
            for k, v in alpha50_per_layer.items()
        },
    }

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

    print(f"\nExtracting all-layer embeddings...", flush=True)
    data = extract_all_layers(model, RECORDINGS)

    print("\nTraining layer-11 probe...", flush=True)
    probe, scaler, loro_acc = train_layer11_probe_loro(data)

    print("\nRunning contrastive direction patching sweep...", flush=True)
    sweep = run_alpha_sweep(model, data, probe, scaler)

    print("\nPlotting...", flush=True)
    summary = plot_results(sweep)

    best = summary["best_causal_layer"]
    a50 = summary["best_alpha50"]
    print(f"\nContrastive Patch Summary:")
    print(f"  Best causal layer: {best} (alpha_50 = {a50})")
    print(f"  Alpha_50 per layer:", flush=True)
    for k, v in summary["alpha50_per_layer"].items():
        print(f"    Layer {int(k):2d}: {v if v is not None else '>max'}")


if __name__ == "__main__":
    main()
