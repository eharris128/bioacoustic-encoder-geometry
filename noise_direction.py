"""Noise direction patching: control experiment for contrastive direction patching.

Motivation
----------
Contrastive direction patching (contrastive_patch_species.py) found that patching
along the unit-norm species direction d_k = normalize(mean_B - mean_A) at layer k
flips species classification at layer 11. But does this require the *specific* species
direction, or does any sufficiently large perturbation work?

This experiment patches with random unit-norm vectors instead of the species direction:

  patched_activation = activation + alpha * r_k

where r_k is a random unit vector in R^768, drawn fresh for each (layer, trial).
We average flip rates over NUM_NOISE_TRIALS random directions per layer.

If the noise-level direction is specifically load-bearing, random directions should
require substantially larger alpha to achieve the same level shift — the alpha_1
curve should sit far above the contrastive-patch baseline.

Outputs
-------
  noise_direction_alpha_sweep.png  — mean abs level shift vs alpha, one curve per layer
  noise_direction_summary.png      — alpha_1 per layer (noise vs noise-level direction)
  result.json                      — written by job wrapper
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
MAX_FRAMES_PER_RECORDING = 1000

# Alpha sweep — same range as contrastive_patch_species.py for direct comparison
ALPHA_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

# Number of independent random directions to average over per (layer, fold)
NUM_NOISE_TRIALS = 20

RECORDINGS = {
    "bullfinch_XC1077468": ("audio/bullfinch/XC1077468.mp3", 0),
    
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
# Probe training (layer-11, LORO — identical to contrastive_patch_species.py)
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

# ---------------------------------------------------------------------------
# Random unit vector generation
# ---------------------------------------------------------------------------

def random_unit_vectors(dim: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return (n, dim) array of unit-norm random vectors."""
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms > 1e-8, norms, 1.0)

# ---------------------------------------------------------------------------
# Noise direction patching (single forward pass)
# ---------------------------------------------------------------------------

def noise_patch_run(
    model,
    audio,
    patch_layer: int,
    direction: np.ndarray,  # unit-norm random vector, shape (768,)
    alpha: float,
) -> np.ndarray:
    """
    Run forward pass with additive hook at layer k:
      output += alpha * direction
    Returns layer-11 embeddings: (n_frames, 768).
    """
    patch_vec = torch.from_numpy(alpha * direction).float()
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

# ---------------------------------------------------------------------------
# Alpha sweep over noise directions
# ---------------------------------------------------------------------------

def run_noise_alpha_sweep(
    model,
    data: dict[str, dict],
    probe: LogisticRegression,
    scaler: StandardScaler,
) -> dict[int, dict[float, float]]:
    """
    For each (patch_layer, alpha), average mean absolute level shift over
    NUM_NOISE_TRIALS random directions and all LORO test recordings.
    Returns {patch_layer: {alpha: mean_abs_shift}}.
    """
    rec_ids = list(data.keys())
    rng = np.random.default_rng(42)

    # Pre-draw noise directions: (NUM_LAYERS, NUM_NOISE_TRIALS, 768)
    noise_dirs = np.stack([
        random_unit_vectors(768, NUM_NOISE_TRIALS, rng) for _ in range(NUM_LAYERS)
    ])  # (NUM_LAYERS, NUM_NOISE_TRIALS, 768)

    # results[layer][alpha] = list of mean abs shifts (one per fold × trial)
    results: dict[int, dict[float, list[float]]] = {
        k: {a: [] for a in ALPHA_VALUES} for k in range(NUM_LAYERS)
    }

    for test_rec in rec_ids:
        test_label = data[test_rec]["label"]
        audio_path = data[test_rec]["audio_path"]
        audio = load_audio(audio_path, mono=True, mono_avg=False)

        print(
            f"  test={test_rec} (noise level {test_label})",
            flush=True,
        )

        for patch_layer in range(NUM_LAYERS):
            layer_results = []
            for alpha in ALPHA_VALUES:
                if alpha == 0.0:
                    # No patch: baseline shift (should be ~0 if probe is accurate)
                    l11 = model.extract_features(audio, layers=None)[11].squeeze(0).cpu().numpy()
                    n = min(l11.shape[0], MAX_FRAMES_PER_RECORDING)
                    if l11.shape[0] > n:
                        idx = np.random.default_rng(42).choice(l11.shape[0], n, replace=False)
                        idx.sort()
                        l11 = l11[idx]
                    preds = probe.predict(scaler.transform(l11))
                    shift = float(np.abs(preds - test_label).mean())
                    results[patch_layer][alpha].append(shift)
                    layer_results.append(f"α=0→{shift:.2f}")
                    continue

                trial_shifts = []
                for trial_idx in range(NUM_NOISE_TRIALS):
                    direction = noise_dirs[patch_layer, trial_idx]
                    l11 = noise_patch_run(model, audio, patch_layer, direction, alpha)
                    n = min(l11.shape[0], MAX_FRAMES_PER_RECORDING)
                    if l11.shape[0] > n:
                        idx = np.random.default_rng(42).choice(l11.shape[0], n, replace=False)
                        idx.sort()
                        l11 = l11[idx]
                    preds = probe.predict(scaler.transform(l11))
                    trial_shifts.append(float(np.abs(preds - test_label).mean()))

                shift = float(np.mean(trial_shifts))
                results[patch_layer][alpha].append(shift)
                layer_results.append(f"α={alpha:.0f}→{shift:.2f}")

            print(f"    layer {patch_layer:2d}: {' | '.join(layer_results)}", flush=True)

    return {
        k: {a: float(np.mean(v)) for a, v in alphas.items()}
        for k, alphas in results.items()
    }

# ---------------------------------------------------------------------------
# Alpha_50 interpolation (identical to contrastive_patch_species.py)
# ---------------------------------------------------------------------------

def compute_alpha1(shifts: dict[float, float]) -> float:
    """Interpolate alpha at which mean abs level shift = 1.0. Returns inf if never reached."""
    alphas = sorted(shifts.keys())
    for i in range(len(alphas) - 1):
        a0, a1 = alphas[i], alphas[i + 1]
        r0, r1 = shifts[a0], shifts[a1]
        if r0 <= 1.0 <= r1:
            if r1 == r0:
                return a0
            t = (1.0 - r0) / (r1 - r0)
            return float(a0 + t * (a1 - a0))
    return float("inf")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(sweep: dict[int, dict[float, float]]) -> dict:
    alphas = sorted(ALPHA_VALUES)
    layers = list(range(NUM_LAYERS))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, NUM_LAYERS))

    alpha1_per_layer = {k: compute_alpha1(sweep[k]) for k in layers}
    finite_alpha1 = [v for v in alpha1_per_layer.values() if not np.isinf(v)]
    best_layer = int(min(alpha1_per_layer, key=lambda k: alpha1_per_layer[k]))

    # ---- Figure 1: alpha sweep curves ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Noise Direction Patching: Control Experiment\n"
        f"(random unit vectors, {NUM_NOISE_TRIALS} trials/layer, LORO, mean abs level shift)",
        fontsize=13, fontweight="bold",
    )

    ax = axes[0]
    for layer in layers:
        shifts = [sweep[layer][a] for a in alphas]
        ax.plot(alphas, shifts, "o-", color=colors[layer],
                linewidth=1.5, markersize=4, label=f"L{layer}")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="shift=1 level")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("Patch scale α (log scale)", fontsize=12)
    ax.set_ylabel("Mean abs level shift at layer 11", fontsize=12)
    ax.set_title("Mean abs level shift vs. patch scale (noise directions)", fontsize=11)
    ax.set_ylim(-0.1, 4.1)
    ax.legend(fontsize=7, ncol=2, loc="upper left")

    # ---- Figure 2: alpha_1 bar chart ----
    ax = axes[1]
    a1_vals = [
        alpha1_per_layer[k] if not np.isinf(alpha1_per_layer[k]) else max(alphas) * 2
        for k in layers
    ]
    bar_colors = plt.cm.RdYlGn_r(np.array(a1_vals) / (max(alphas) * 2))
    bars = ax.bar(layers, a1_vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.axhline(max(alphas), color="gray", linestyle=":", alpha=0.5, label="Max α tested")
    ax.set_xlabel("Patch layer", fontsize=12)
    ax.set_ylabel("α₁ (scale to shift 1 noise level)", fontsize=12)
    ax.set_title(
        "Noise direction α₁ per layer\n(compare to noise-level direction — higher = less specific)",
        fontsize=11,
    )
    ax.set_xticks(layers)
    ax.legend(fontsize=9)

    ax.bar(best_layer, a1_vals[best_layer], color="gold", edgecolor="black",
           linewidth=1.5, label=f"Best: L{best_layer}")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, a1_vals):
        label = f"{val:.1f}" if val < max(alphas) * 1.5 else ">max"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(alphas) * 0.02,
            label,
            ha="center", va="bottom", fontsize=7,
        )

    plt.tight_layout()
    plt.savefig("noise_direction_alpha_sweep.png", dpi=150, bbox_inches="tight")
    print("Saved noise_direction_alpha_sweep.png")

    # ---- Figure 3: heatmap (layer × alpha → flip rate) ----
    flip_matrix = np.array([
        [sweep[k][a] for a in alphas] for k in layers
    ])  # (n_layers, n_alphas)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        "Noise Direction Patching Heatmap\n"
        "(mean abs level shift at layer 11 vs. patch layer and scale α, random directions)",
        fontsize=13, fontweight="bold",
    )
    im = ax.imshow(flip_matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=4,
                   origin="lower")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([str(a) for a in alphas], fontsize=9)
    ax.set_yticks(layers)
    ax.set_yticklabels([f"Layer {k}" for k in layers], fontsize=9)
    ax.set_xlabel("Patch scale α", fontsize=12)
    ax.set_ylabel("Patch layer", fontsize=12)
    plt.colorbar(im, ax=ax, label="Mean abs level shift at layer 11")
    for i in range(NUM_LAYERS):
        for j in range(len(alphas)):
            ax.text(j, i, f"{flip_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if 0.2 < flip_matrix[i, j] < 0.8 else "white")
    plt.tight_layout()
    plt.savefig("noise_direction_summary.png", dpi=150, bbox_inches="tight")
    print("Saved noise_direction_summary.png")

    return {
        "best_layer_noise": best_layer,
        "best_alpha1_noise": float(alpha1_per_layer[best_layer])
        if not np.isinf(alpha1_per_layer[best_layer]) else None,
        "alpha1_per_layer_noise": {
            str(k): (float(v) if not np.isinf(v) else None)
            for k, v in alpha1_per_layer.items()
        },
        "num_noise_trials": NUM_NOISE_TRIALS,
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

    print("\nExtracting all-layer embeddings...", flush=True)
    data = extract_all_layers(model, RECORDINGS)

    print("\nTraining layer-11 probe...", flush=True)
    probe, scaler, loro_acc = train_layer11_probe_loro(data)

    print(f"\nRunning noise direction sweep ({NUM_NOISE_TRIALS} trials/layer)...", flush=True)
    sweep = run_noise_alpha_sweep(model, data, probe, scaler)

    print("\nPlotting...", flush=True)
    summary = plot_results(sweep)

    best = summary["best_layer_noise"]
    a1 = summary["best_alpha1_noise"]
    print(f"\nNoise Direction Summary:")
    print(f"  Best layer (noise): {best} (alpha_1 = {a1})")
    print(f"  Alpha_1 per layer (noise):", flush=True)
    for k, v in summary["alpha1_per_layer_noise"].items():
        print(f"    Layer {int(k):2d}: {v if v is not None else '>max'}")


if __name__ == "__main__":
    main()
