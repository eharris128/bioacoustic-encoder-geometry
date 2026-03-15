"""Exploratory interpretability: visualize attention heads across AVES layers.

Hooks into Q/K projections to compute attention weights manually,
since torchaudio's wav2vec2 doesn't expose them directly.

Uses the shorter Bullfinch file (~21s, 1049 frames) to keep matrices manageable.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from aves import load_feature_extractor
from aves.utils import load_audio

# --- Config ---
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
# Use the shorter file — attention matrix is (num_frames x num_frames) per head
AUDIO_PATH = "./aves/example_audios/XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.mp3"

NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 768 // NUM_HEADS  # 64

# --- Load model ---
print("Loading model...")
model = load_feature_extractor(
    config_path=CONFIG_PATH,
    model_path=MODEL_PATH,
    device="cpu",
    for_inference=True,
)
print("Model loaded.\n")

# --- Register hooks to capture Q and K after projection ---
attention_weights = {}  # {layer_idx: (num_heads, seq_len, seq_len)}


def make_hook(layer_idx):
    """Create a hook that captures the attention module's input, computes Q/K, and stores attention weights."""
    def hook_fn(module, args, output):
        # The input to SelfAttention.forward is (x, attention_mask, position_bias, key_padding_mask)
        x = args[0]  # (batch, seq_len, embed_dim)
        batch, seq_len, embed_dim = x.shape

        # Compute Q and K using the module's projection layers
        q = module.q_proj(x)  # (batch, seq_len, embed_dim)
        k = module.k_proj(x)  # (batch, seq_len, embed_dim)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        # Compute attention weights: softmax(Q @ K^T / sqrt(d_k))
        scale = HEAD_DIM ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (batch, heads, seq, seq)
        attn = F.softmax(attn, dim=-1)

        # Store (squeeze batch dim)
        attention_weights[layer_idx] = attn[0].detach().numpy()  # (num_heads, seq_len, seq_len)

    return hook_fn


# Register hooks on each layer's attention module
hooks = []
for i in range(NUM_LAYERS):
    attn_module = model.model.encoder.transformer.layers[i].attention
    h = attn_module.register_forward_hook(make_hook(i))
    hooks.append(h)

# --- Run inference to trigger hooks ---
print(f"Loading audio: {AUDIO_PATH}")
audio = load_audio(AUDIO_PATH, mono=True, mono_avg=False)
print(f"Audio: {audio.shape[-1]/16000:.1f}s, {audio.shape[-1]} samples")

print("Running forward pass to capture attention weights...")
t0 = time.time()
_ = model.extract_features(audio, layers=None)
print(f"Done in {time.time()-t0:.1f}s")
print(f"Captured attention for {len(attention_weights)} layers")

# Clean up hooks
for h in hooks:
    h.remove()

seq_len = attention_weights[0].shape[-1]
print(f"Sequence length: {seq_len} frames ({seq_len * 0.02:.1f}s)")

# --- Plot 1: All 12 heads for a selected layer (last layer) ---
PLOT_LAYER = 11
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle(f"Attention Heads — Layer {PLOT_LAYER} (Bullfinch, {seq_len} frames)\n"
             f"Each plot: where each frame (y-axis) attends to (x-axis)",
             fontsize=14, fontweight="bold")

for head_idx in range(NUM_HEADS):
    ax = axes[head_idx // 4, head_idx % 4]
    attn = attention_weights[PLOT_LAYER][head_idx]  # (seq_len, seq_len)

    im = ax.imshow(attn, aspect="auto", cmap="viridis",
                   interpolation="none")
    ax.set_title(f"Head {head_idx}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Key position (frame)", fontsize=8)
    ax.set_ylabel("Query position (frame)", fontsize=8)
    ax.tick_params(labelsize=7)

plt.colorbar(im, ax=axes, shrink=0.6, label="Attention weight")
plt.tight_layout()
plt.savefig("attention_heads_layer11.png", dpi=150, bbox_inches="tight")
print("Saved attention_heads_layer11.png")

# --- Plot 2: One head per layer — show how attention evolves across depth ---
# Pick the head with the most interesting (least uniform) pattern per layer
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("Attention Patterns Across Layers (Bullfinch)\n"
             "Showing the head with highest entropy variance per layer",
             fontsize=14, fontweight="bold")

for layer_idx in range(NUM_LAYERS):
    ax = axes[layer_idx // 4, layer_idx % 4]

    # Find the most "structured" head (lowest mean entropy = most peaked attention)
    entropies = []
    for head_idx in range(NUM_HEADS):
        attn = attention_weights[layer_idx][head_idx]
        # Compute mean entropy across query positions
        # Add small epsilon to avoid log(0)
        ent = -np.sum(attn * np.log(attn + 1e-10), axis=-1).mean()
        entropies.append(ent)
    best_head = int(np.argmin(entropies))

    attn = attention_weights[layer_idx][best_head]
    im = ax.imshow(attn, aspect="auto", cmap="viridis",
                   interpolation="none")
    ax.set_title(f"Layer {layer_idx}, Head {best_head} (entropy={entropies[best_head]:.2f})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Key (frame)", fontsize=8)
    ax.set_ylabel("Query (frame)", fontsize=8)
    ax.tick_params(labelsize=7)

plt.colorbar(im, ax=axes, shrink=0.6, label="Attention weight")
plt.tight_layout()
plt.savefig("attention_across_layers.png", dpi=150, bbox_inches="tight")
print("Saved attention_across_layers.png")

# --- Plot 3: Attention head specialization summary ---
# For each head in each layer, compute: mean entropy (uniform vs peaked)
# and mean diagonal weight (local vs global attention)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Attention Head Specialization Summary (Bullfinch)",
             fontsize=14, fontweight="bold")

entropy_matrix = np.zeros((NUM_LAYERS, NUM_HEADS))
locality_matrix = np.zeros((NUM_LAYERS, NUM_HEADS))

for layer_idx in range(NUM_LAYERS):
    for head_idx in range(NUM_HEADS):
        attn = attention_weights[layer_idx][head_idx]

        # Mean entropy (high = uniform/diffuse, low = peaked/specialized)
        ent = -np.sum(attn * np.log(attn + 1e-10), axis=-1).mean()
        entropy_matrix[layer_idx, head_idx] = ent

        # Locality: mean attention within a local window (±5 frames = ±100ms)
        window = 5
        local_weight = 0
        for i in range(seq_len):
            lo = max(0, i - window)
            hi = min(seq_len, i + window + 1)
            local_weight += attn[i, lo:hi].sum()
        locality_matrix[layer_idx, head_idx] = local_weight / seq_len

im1 = ax1.imshow(entropy_matrix, aspect="auto", cmap="RdYlBu_r")
ax1.set_xlabel("Head", fontsize=11)
ax1.set_ylabel("Layer", fontsize=11)
ax1.set_title("Mean Entropy per Head\n(low = peaked/specialized, high = diffuse)", fontsize=11)
ax1.set_xticks(range(NUM_HEADS))
ax1.set_yticks(range(NUM_LAYERS))
plt.colorbar(im1, ax=ax1, label="Entropy (nats)")

im2 = ax2.imshow(locality_matrix, aspect="auto", cmap="RdYlBu_r")
ax2.set_xlabel("Head", fontsize=11)
ax2.set_ylabel("Layer", fontsize=11)
ax2.set_title("Local Attention (±100ms window)\n(high = attends nearby, low = attends globally)", fontsize=11)
ax2.set_xticks(range(NUM_HEADS))
ax2.set_yticks(range(NUM_LAYERS))
plt.colorbar(im2, ax=ax2, label="Fraction of attention in local window")

plt.tight_layout()
plt.savefig("attention_specialization.png", dpi=150, bbox_inches="tight")
print("Saved attention_specialization.png")
