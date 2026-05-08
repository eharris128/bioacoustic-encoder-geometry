"""Causal tracing: which AVES layer causally carries species information?

Motivation
----------
The species probe shows layer 1 (93.6%) beats layer 11 (88%) — unexpected for a
deep transformer. Two hypotheses:
  A. Layer 1 encodes raw spectral features that are already species-diagnostic;
     later layers transform them for other purposes and reduce linear separability.
  B. Species information is re-encoded or routed at multiple layers; the
     layer-1 advantage is a probe artefact, not a causal one.

Experiment
----------
For each "patch layer" k in 0..11:
  1. Take a held-out recording of species A.
  2. Register a hook on transformer layer k that replaces its entire output
     with the mean activation of species B at layer k (computed from training
     recordings, leave-one-recording-out).
  3. Run the full forward pass. Layers k+1..11 process the patched signal.
  4. Apply the layer-11 species probe to the resulting layer-11 embeddings.
  5. Report the "transfer accuracy": what fraction of frames now predict species B?

If layer k is the causal bottleneck for species → transfer accuracy ~1.0.
If layer k's information is not load-bearing → transfer accuracy stays ~0.5.

Outputs
-------
  causal_trace_species.png    — transfer accuracy by patch layer (main result)
  species_separation.png      — probe accuracy + mean-distance per layer
  result.json                 — written by the job wrapper
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
MAX_FRAMES_PER_RECORDING = 2000

RECORDINGS = {
    # Bullfinch (label=0)
    "bullfinch_XC1077468": ("audio/bullfinch/XC1077468.mp3", 0),
    "bullfinch_XC965743":  ("audio/bullfinch/XC965743.mp3",  0),
    "bullfinch_XC938052":  ("audio/bullfinch/XC938052.mp3",  0),
    "bullfinch_XC805629":  ("audio/bullfinch/XC805629.mp3",  0),
    # Hawfinch (label=1)
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
    """Return {rec_id: {"embs": (n_frames, n_layers, 768), "label": int}}."""
    results = {}
    for rec_id, (path, label) in paths_labels.items():
        print(f"  {rec_id}...", end=" ", flush=True)
        audio = load_audio(path, mono=True, mono_avg=False)
        t0 = time.time()
        layer_outputs = model.extract_features(audio, layers=None)
        elapsed = time.time() - t0

        # Stack: (n_layers, n_frames, 768)
        stacked = np.stack([lo.squeeze(0).cpu().numpy() for lo in layer_outputs], axis=0)
        # Transpose to (n_frames, n_layers, 768)
        stacked = stacked.transpose(1, 0, 2)

        n_frames = stacked.shape[0]
        if n_frames > MAX_FRAMES_PER_RECORDING:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_frames, MAX_FRAMES_PER_RECORDING, replace=False)
            idx.sort()
            stacked = stacked[idx]

        print(f"{stacked.shape[0]} frames, {elapsed:.1f}s", flush=True)
        results[rec_id] = {"embs": stacked, "label": label}
    return results

# ---------------------------------------------------------------------------
# Probe training (leave-one-recording-out)
# ---------------------------------------------------------------------------

def train_probes_loro(
    data: dict[str, dict],
) -> tuple[list[LogisticRegression], list[StandardScaler], dict]:
    """Train one probe per layer using LORO CV. Return probes, scalers, CV accuracies."""
    rec_ids = list(data.keys())
    layer_probes: list[LogisticRegression] = []
    layer_scalers: list[StandardScaler] = []
    layer_accs: dict[int, float] = {}

    for layer in range(NUM_LAYERS):
        fold_accs = []
        for test_rec in rec_ids:
            train_recs = [r for r in rec_ids if r != test_rec]

            X_train = np.concatenate([data[r]["embs"][:, layer, :] for r in train_recs])
            y_train = np.concatenate([
                np.full(data[r]["embs"].shape[0], data[r]["label"]) for r in train_recs
            ])
            X_test = data[test_rec]["embs"][:, layer, :]
            y_test = np.full(X_test.shape[0], data[test_rec]["label"])

            sc = StandardScaler()
            X_train_s = sc.fit_transform(X_train)
            X_test_s = sc.transform(X_test)
            clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
            clf.fit(X_train_s, y_train)
            fold_accs.append(accuracy_score(y_test, clf.predict(X_test_s)))

        # Fit final probe on all data
        X_all = np.concatenate([data[r]["embs"][:, layer, :] for r in rec_ids])
        y_all = np.concatenate([
            np.full(data[r]["embs"].shape[0], data[r]["label"]) for r in rec_ids
        ])
        sc_final = StandardScaler()
        X_all_s = sc_final.fit_transform(X_all)
        clf_final = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf_final.fit(X_all_s, y_all)

        layer_probes.append(clf_final)
        layer_scalers.append(sc_final)
        layer_accs[layer] = float(np.mean(fold_accs))
        print(f"  Layer {layer:2d} probe: LORO accuracy = {layer_accs[layer]:.1%}", flush=True)

    return layer_probes, layer_scalers, layer_accs


def compute_species_means(
    data: dict[str, dict], train_recs: list[str]
) -> dict[int, dict[int, np.ndarray]]:
    """Return {layer: {label: mean_embedding}} computed over train_recs."""
    means: dict[int, dict[int, np.ndarray]] = {}
    for layer in range(NUM_LAYERS):
        means[layer] = {}
        for label in [0, 1]:
            frames = np.concatenate([
                data[r]["embs"][:, layer, :]
                for r in train_recs if data[r]["label"] == label
            ])
            means[layer][label] = frames.mean(axis=0)
    return means

# ---------------------------------------------------------------------------
# Causal patching
# ---------------------------------------------------------------------------

def causal_patch_run(
    model,
    audio,
    patch_layer: int,
    patch_vector: np.ndarray,
) -> np.ndarray:
    """
    Run model.extract_features with a hook that replaces layer `patch_layer`'s
    output with `patch_vector` (broadcast across all sequence positions).
    Returns layer-11 embeddings: (n_frames, 768).
    """
    patch_tensor = torch.from_numpy(patch_vector).float()

    # Determine device from model parameters
    try:
        device = next(model.model.parameters()).device
    except Exception:
        device = torch.device("cpu")
    patch_tensor = patch_tensor.to(device)

    def hook_fn(module, input, output):
        # output may be a tensor or a tuple; handle both
        if isinstance(output, tuple):
            h = output[0]  # (batch, seq, 768)
            patched = patch_tensor.unsqueeze(0).unsqueeze(0).expand_as(h).clone()
            return (patched,) + output[1:]
        else:
            patched = patch_tensor.unsqueeze(0).unsqueeze(0).expand_as(output).clone()
            return patched

    layer_module = model.model.encoder.transformer.layers[patch_layer]
    hook = layer_module.register_forward_hook(hook_fn)
    try:
        layer_outputs = model.extract_features(audio, layers=None)
    finally:
        hook.remove()

    return layer_outputs[11].squeeze(0).cpu().numpy()  # (n_frames, 768)


def run_causal_patching(
    model,
    data: dict[str, dict],
    probes: list[LogisticRegression],
    scalers: list[StandardScaler],
) -> dict[int, dict]:
    """
    For each patch layer k, for each test recording (LORO):
      - Patch layer k output with opposite-species mean (from training recordings)
      - Run forward pass to layer 11
      - Apply layer-11 probe
      - Record transfer accuracy (fraction predicting opposite species)
    Returns {patch_layer: {"transfer_acc": float, "n_frames": int}}.
    """
    rec_ids = list(data.keys())
    results: dict[int, list[float]] = {k: [] for k in range(NUM_LAYERS)}
    total_frames: dict[int, int] = {k: 0 for k in range(NUM_LAYERS)}

    for test_rec in rec_ids:
        train_recs = [r for r in rec_ids if r != test_rec]
        test_label = data[test_rec]["label"]
        opposite_label = 1 - test_label
        species_name = "Bullfinch" if test_label == 0 else "Hawfinch"
        opposite_name = "Hawfinch" if test_label == 0 else "Bullfinch"

        print(f"  Patching test={test_rec} ({species_name} → {opposite_name})", flush=True)

        # Compute opposite-species means from training recordings
        species_means = compute_species_means(data, train_recs)

        # Load audio for this test recording
        audio_path = RECORDINGS[test_rec][0]
        audio = load_audio(audio_path, mono=True, mono_avg=False)

        for patch_layer in range(NUM_LAYERS):
            patch_vec = species_means[patch_layer][opposite_label]

            try:
                l11_embs = causal_patch_run(model, audio, patch_layer, patch_vec)
            except Exception as e:
                print(f"    Layer {patch_layer}: hook failed ({e}), skipping", flush=True)
                continue

            # Cap frames to match data
            n_frames = min(l11_embs.shape[0], MAX_FRAMES_PER_RECORDING)
            rng = np.random.default_rng(42)
            if l11_embs.shape[0] > n_frames:
                idx = rng.choice(l11_embs.shape[0], n_frames, replace=False)
                idx.sort()
                l11_embs = l11_embs[idx]

            # Apply layer-11 probe
            X = scalers[11].transform(l11_embs)
            preds = probes[11].predict(X)
            transfer_acc = float((preds == opposite_label).mean())
            results[patch_layer].append(transfer_acc)
            total_frames[patch_layer] += l11_embs.shape[0]
            print(f"    Layer {patch_layer:2d}: transfer_acc={transfer_acc:.3f}", flush=True)

    return {
        k: {
            "transfer_acc": float(np.mean(v)) if v else float("nan"),
            "n_folds": len(v),
        }
        for k, v in results.items()
    }

# ---------------------------------------------------------------------------
# Species separation: mean L2 distance between species at each layer
# ---------------------------------------------------------------------------

def compute_separation(data: dict[str, dict]) -> dict[int, float]:
    separation = {}
    for layer in range(NUM_LAYERS):
        frames_0 = np.concatenate([
            data[r]["embs"][:, layer, :] for r in data if data[r]["label"] == 0
        ])
        frames_1 = np.concatenate([
            data[r]["embs"][:, layer, :] for r in data if data[r]["label"] == 1
        ])
        mean_0 = frames_0.mean(axis=0)
        mean_1 = frames_1.mean(axis=0)
        separation[layer] = float(np.linalg.norm(mean_0 - mean_1))
    return separation

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_causal_trace(
    patch_results: dict[int, dict],
    probe_accs: dict[int, float],
    separation: dict[int, float],
) -> dict:
    layers = list(range(NUM_LAYERS))
    transfer_accs = [patch_results[k]["transfer_acc"] for k in layers]
    accs = [probe_accs[k] for k in layers]
    seps = [separation[k] for k in layers]

    # ---- Figure 1: causal trace ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Causal Tracing: Which AVES Layer Carries Species Information?\n"
        "(Bullfinch ↔ Hawfinch, leave-one-recording-out)",
        fontsize=13, fontweight="bold",
    )

    ax = axes[0]
    bar_colors = plt.cm.plasma(np.array(transfer_accs))
    bars = ax.bar(layers, transfer_accs, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Chance (50%)")
    ax.axhline(1.0, color="green", linestyle=":", alpha=0.4, label="Perfect transfer")
    ax.set_xlabel("Patch layer", fontsize=12)
    ax.set_ylabel("Transfer accuracy at layer 11", fontsize=12)
    ax.set_title(
        "Causal Transfer Accuracy\n"
        "(fraction of frames predicting opposite species at layer 11)",
        fontsize=11,
    )
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)

    # Annotate bars
    for bar, val in zip(bars, transfer_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=8,
        )

    # ---- Probe accuracy overlay ----
    ax2 = ax.twinx()
    ax2.plot(layers, accs, "s--", color="steelblue", linewidth=1.5,
             markersize=6, label="Probe accuracy (LORO)")
    ax2.set_ylabel("Probe accuracy", fontsize=11, color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right", fontsize=9)

    # ---- Figure 1 panel 2: separation ----
    ax = axes[1]
    ax.plot(layers, seps, "o-", color="darkorange", linewidth=2, markersize=7)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("L2 distance between species means", fontsize=12)
    ax.set_title(
        "Species Mean Separation per Layer\n"
        "(L2 distance: Bullfinch mean vs Hawfinch mean)",
        fontsize=11,
    )
    ax.set_xticks(layers)

    plt.tight_layout()
    plt.savefig("causal_trace_species.png", dpi=150, bbox_inches="tight")
    print("Saved causal_trace_species.png")

    # ---- Figure 2: separation + probe side by side ----
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Species Information Across Layers", fontsize=13, fontweight="bold")
    ax.bar(layers, accs, color="steelblue", alpha=0.6, label="Probe accuracy (LORO)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Probe accuracy", fontsize=12)
    ax.set_xticks(layers)
    ax.set_ylim(0.4, 1.05)
    ax3 = ax.twinx()
    ax3.plot(layers, seps, "o-", color="darkorange", linewidth=2, markersize=7, label="Mean separation (L2)")
    ax3.set_ylabel("L2 separation", fontsize=11, color="darkorange")
    ax3.tick_params(axis="y", labelcolor="darkorange")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    plt.tight_layout()
    plt.savefig("species_separation.png", dpi=150, bbox_inches="tight")
    print("Saved species_separation.png")

    best_patch_layer = int(np.argmax(transfer_accs))
    return {
        "best_patch_layer": best_patch_layer,
        "best_transfer_acc": float(transfer_accs[best_patch_layer]),
        "layer1_transfer_acc": float(transfer_accs[1]),
        "layer11_transfer_acc": float(transfer_accs[11]),
        "layer1_probe_acc": float(accs[1]),
        "layer11_probe_acc": float(accs[11]),
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

    print(f"\nExtracting all-layer embeddings for {len(RECORDINGS)} recordings...", flush=True)
    data = extract_all_layers(model, RECORDINGS)

    print("\nTraining layer probes (LORO)...", flush=True)
    probes, scalers, probe_accs = train_probes_loro(data)

    print("\nComputing species separation...", flush=True)
    separation = compute_separation(data)

    print("\nRunning causal patching (this takes a while)...", flush=True)
    patch_results = run_causal_patching(model, data, probes, scalers)

    print("\nPlotting...", flush=True)
    summary = plot_causal_trace(patch_results, probe_accs, separation)

    print("\nCausal Trace Summary:")
    print(f"  Best patch layer:       {summary['best_patch_layer']} "
          f"(transfer acc = {summary['best_transfer_acc']:.3f})")
    print(f"  Layer 1 transfer acc:   {summary['layer1_transfer_acc']:.3f}  "
          f"(probe acc = {summary['layer1_probe_acc']:.3f})")
    print(f"  Layer 11 transfer acc:  {summary['layer11_transfer_acc']:.3f}  "
          f"(probe acc = {summary['layer11_probe_acc']:.3f})")


if __name__ == "__main__":
    main()
