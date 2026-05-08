"""Compare AVES (animal sounds) vs HuBERT (human speech): same architecture, different training data.

Runs the same analyses on both models using the same Bullfinch audio:
1. Recording identity erasure (silhouette scores across layers)
2. Adjacent layer CKA (transformation magnitude)
3. Attention locality (local vs global attention per layer)
4. CKA with mel spectrogram (acoustic grounding)

Tests whether the layer hierarchy is architecture-driven or data-driven.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.models import wav2vec2_model
from aves import load_feature_extractor
from aves.utils import load_audio

# --- Config ---
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
AUDIO_DIR = Path("./audio/bullfinch")
SKIP = {"XC1086809.mp3", "XC657517.mp3"}
NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64
MAX_FRAMES = 500
SR = 16000


def linear_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XtY = X.T @ Y
    return np.sum(XtY ** 2) / np.sqrt(np.sum((X.T @ X) ** 2) * np.sum((Y.T @ Y) ** 2) + 1e-10)


# --- Load both models ---
print("Loading AVES (animal sounds)...")
aves_model = load_feature_extractor(
    config_path=CONFIG_PATH,
    model_path=MODEL_PATH,
    device="cpu",
    for_inference=True,
)

print("Loading HuBERT (human speech)...")
bundle = torchaudio.pipelines.HUBERT_BASE
hubert_model = bundle.get_model().eval()
print("Both models loaded.\n")

# --- Load audio files ---
audio_files = sorted([f for f in AUDIO_DIR.glob("*.mp3") if f.name not in SKIP])[:15]
print(f"Using {len(audio_files)} Bullfinch recordings\n")

audios = []
for path in audio_files:
    try:
        audio = load_audio(str(path), mono=True, mono_avg=False)
        audios.append(audio)
    except Exception:
        pass
print(f"Loaded {len(audios)} audio files\n")


def extract_all_layers(model, audio, is_aves=True):
    """Extract embeddings from all 12 transformer layers."""
    with torch.no_grad():
        if is_aves:
            outputs = model.extract_features(audio, layers=None)
            return [o.squeeze(0).cpu().numpy() for o in outputs]
        else:
            # HuBERT via torchaudio: use extract_features directly
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            features, _ = model.extract_features(audio)
            return [f.squeeze(0).cpu().numpy() for f in features]


def get_attention_locality(model, audio, is_aves=True):
    """Hook into attention Q/K and compute locality per layer per head."""
    attention_locality = {}

    def make_hook(layer_idx):
        def hook_fn(module, args, output):
            x = args[0]
            batch, seq_len, embed_dim = x.shape
            q = module.q_proj(x).view(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            k = module.k_proj(x).view(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            scale = HEAD_DIM ** 0.5
            attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1)
            attn = attn[0].detach().numpy()  # (heads, seq, seq)

            # Compute locality: mean attention within ±5 frames
            window = 5
            locality_per_head = []
            for head in range(NUM_HEADS):
                local_weight = 0
                for i in range(seq_len):
                    lo = max(0, i - window)
                    hi = min(seq_len, i + window + 1)
                    local_weight += attn[head, i, lo:hi].sum()
                locality_per_head.append(local_weight / seq_len)
            attention_locality[layer_idx] = np.mean(locality_per_head)
        return hook_fn

    # Get the transformer layers
    if is_aves:
        layers = model.model.encoder.transformer.layers
    else:
        layers = model.encoder.transformer.layers

    hooks = []
    for i in range(NUM_LAYERS):
        h = layers[i].attention.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Forward pass
    with torch.no_grad():
        if is_aves:
            model.extract_features(audio, layers=None)
        else:
            inp = audio.unsqueeze(0) if audio.ndim == 1 else audio
            model.extract_features(inp)

    for h in hooks:
        h.remove()

    return [attention_locality[i] for i in range(NUM_LAYERS)]


# --- Run analyses for both models ---
models = {
    "AVES (animal sounds)": (aves_model, True),
    "HuBERT (human speech)": (hubert_model, False),
}

results = {}

for model_name, (model, is_aves) in models.items():
    print(f"{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")

    # 1. Extract embeddings from all layers for all recordings
    all_embeddings = {i: [] for i in range(NUM_LAYERS)}
    recording_labels = []

    for rec_idx, audio in enumerate(audios):
        layer_outputs = extract_all_layers(model, audio, is_aves)
        for layer_idx, emb in enumerate(layer_outputs):
            if emb.shape[0] > MAX_FRAMES:
                rng = np.random.default_rng(42)
                idx = rng.choice(emb.shape[0], MAX_FRAMES, replace=False)
                idx.sort()
                emb = emb[idx]
            all_embeddings[layer_idx].append(emb)
            if layer_idx == 0:
                recording_labels.extend([rec_idx] * emb.shape[0])

    for layer_idx in range(NUM_LAYERS):
        all_embeddings[layer_idx] = np.concatenate(all_embeddings[layer_idx], axis=0)

    labels_arr = np.array(recording_labels)
    n_total = all_embeddings[0].shape[0]

    # 2. Silhouette scores (recording separability)
    print("  Computing silhouette scores...")
    sil_scores = []
    for layer_idx in range(NUM_LAYERS):
        if n_total > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_total, 5000, replace=False)
            sil = silhouette_score(all_embeddings[layer_idx][idx], labels_arr[idx])
        else:
            sil = silhouette_score(all_embeddings[layer_idx], labels_arr)
        sil_scores.append(sil)
    print(f"    Range: {min(sil_scores):.4f} to {max(sil_scores):.4f}")

    # 3. Adjacent layer CKA
    print("  Computing adjacent CKA...")
    adj_cka = []
    for i in range(NUM_LAYERS - 1):
        cka = linear_cka(all_embeddings[i], all_embeddings[i + 1])
        adj_cka.append(cka)

    # 4. CKA with mel spectrogram
    print("  Computing CKA with mel spec...")
    mel_parts = []
    for audio in audios:
        if audio.ndim == 1:
            a = audio.unsqueeze(0)
        else:
            a = audio[:1]
        spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=1024, hop_length=320, n_mels=80,
        )(a).squeeze(0).permute(1, 0).numpy()
        if spec.shape[0] > MAX_FRAMES:
            rng = np.random.default_rng(42)
            idx = rng.choice(spec.shape[0], MAX_FRAMES, replace=False)
            idx.sort()
            spec = spec[idx]
        mel_parts.append(spec)
    mel_combined = np.concatenate(mel_parts, axis=0)
    min_n = min(n_total, mel_combined.shape[0])

    cka_mel = []
    for layer_idx in range(NUM_LAYERS):
        cka = linear_cka(all_embeddings[layer_idx][:min_n], mel_combined[:min_n])
        cka_mel.append(cka)

    # 5. Attention locality (use first audio only for speed)
    print("  Computing attention locality...")
    attn_locality = get_attention_locality(model, audios[0], is_aves)

    results[model_name] = {
        "silhouette": sil_scores,
        "adjacent_cka": adj_cka,
        "cka_mel": cka_mel,
        "attention_locality": attn_locality,
    }

    print(f"  Done.\n")

# --- Plot comparison ---
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("AVES (animal sounds) vs HuBERT (human speech)\n"
             "Same architecture, same audio input, different training data",
             fontsize=15, fontweight="bold")

colors = {"AVES (animal sounds)": "#2196F3", "HuBERT (human speech)": "#FF5722"}

# 1. Recording identity erasure
ax = axes[0, 0]
for model_name, res in results.items():
    ax.plot(range(NUM_LAYERS), res["silhouette"], "o-",
            color=colors[model_name], label=model_name, linewidth=2, markersize=6)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Silhouette Score", fontsize=12)
ax.set_title("Recording Identity Erasure\n(lower = more abstract, recording-invariant)", fontsize=12)
ax.set_xticks(range(NUM_LAYERS))
ax.legend(fontsize=10)

# 2. Adjacent CKA
ax = axes[0, 1]
x = np.arange(NUM_LAYERS - 1)
width = 0.35
for i, (model_name, res) in enumerate(results.items()):
    ax.bar(x + i * width, res["adjacent_cka"], width,
           color=colors[model_name], label=model_name, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Layer transition", fontsize=12)
ax.set_ylabel("CKA with next layer", fontsize=12)
ax.set_title("Transformation Magnitude\n(lower = bigger change per layer)", fontsize=12)
ax.set_xticks(x + width / 2)
ax.set_xticklabels([f"{i}→{i+1}" for i in range(NUM_LAYERS - 1)], fontsize=8)
ax.set_ylim(0.95, 1.0)
ax.legend(fontsize=10)

# 3. CKA with mel spectrogram
ax = axes[1, 0]
for model_name, res in results.items():
    ax.plot(range(NUM_LAYERS), res["cka_mel"], "o-",
            color=colors[model_name], label=model_name, linewidth=2, markersize=6)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("CKA with Mel Spectrogram", fontsize=12)
ax.set_title("Acoustic Grounding\n(higher = more similar to raw spectral features)", fontsize=12)
ax.set_xticks(range(NUM_LAYERS))
ax.legend(fontsize=10)

# 4. Attention locality
ax = axes[1, 1]
for model_name, res in results.items():
    ax.plot(range(NUM_LAYERS), res["attention_locality"], "o-",
            color=colors[model_name], label=model_name, linewidth=2, markersize=6)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Mean local attention (±100ms)", fontsize=12)
ax.set_title("Attention Locality\n(higher = attends nearby, lower = attends globally)", fontsize=12)
ax.set_xticks(range(NUM_LAYERS))
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("compare_hubert.png", dpi=150, bbox_inches="tight")
print("Saved compare_hubert.png")
