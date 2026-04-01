"""Streamlit app: explore how AVES represents mixtures of audio sources.

Run with: streamlit run app_mixed_pca.py
"""

import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from aves import load_feature_extractor
from aves.utils import load_audio

# --- Audio catalog ---
AUDIO_FILES = {
    "Bullfinch": "./aves/example_audios/XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.mp3",
    "Piano – Paulyudin": "/home/evan/Downloads/paulyudin-piano-music-piano-485929.mp3",
    "Piano – Sigmamusicart": "/home/evan/Downloads/sigmamusicart-piano-music-504007.mp3",
    "Piano – Solarflex": "/home/evan/Downloads/solarflex-emotional-piano-music-499244.mp3",
    "Piano – The Mountain": "/home/evan/Downloads/the_mountain-piano-background-music-487020.mp3",
    "Violin – Romantic Waltz": "./audio/violin/good_b_music-romantic-violin-waltz-real-violin-497682.mp3",
    "Violin – Baroque Melody": "./audio/violin/nickpanekaiassets-cinematic-baroque-violin-melody-287276.mp3",
    "Violin – Inspiring": "./audio/violin/solarflex-emotional-inspiring-violin-499245.mp3",
    "Violin – Background": "./audio/violin/soulfuljamtracks-strings-violin-background-478146.mp3",
    "Violin – Vibehorn": "./audio/violin/vibehorn-violin-background-music-483067.mp3",
}
NUM_LAYERS = 12
MAX_FRAMES = 2000

st.set_page_config(page_title="AVES Audio Mixing PCA", layout="wide")
st.title("AVES Audio Mixing PCA Explorer")
st.markdown("Mix two audio sources and see where the mixture lands in AVES representation space.")


@st.cache_resource
def load_model():
    return load_feature_extractor(
        config_path="./aves/config/default_cfg_aves-base-all.json",
        model_path="./models/aves-base-all.torchaudio.pt",
        device="cpu",
        for_inference=True,
    )


@st.cache_data
def load_and_extract(path):
    """Load audio and extract all 12 layer embeddings."""
    model = load_model()
    audio = load_audio(path, mono=True, mono_avg=False)
    layer_outputs = model.extract_features(audio, layers=None)
    embeddings = [layer.squeeze(0).cpu().numpy() for layer in layer_outputs]
    return audio.numpy(), embeddings


def subsample(embs, max_frames, rng):
    n = embs[0].shape[0]
    if n > max_frames:
        idx = rng.choice(n, max_frames, replace=False)
        idx.sort()
        return [e[idx] for e in embs]
    return embs


# --- Sidebar controls ---
st.sidebar.header("Audio Sources")
source_a_name = st.sidebar.selectbox("Source A", list(AUDIO_FILES.keys()), index=0)
source_b_name = st.sidebar.selectbox("Source B", list(AUDIO_FILES.keys()), index=5)

st.sidebar.header("Mix Ratio")
mix_weight = st.sidebar.slider(
    "Source A weight", 0.0, 1.0, 0.5, 0.05,
    help="Source B weight = 1 - Source A weight"
)
st.sidebar.write(f"**{source_a_name}**: {mix_weight:.0%} / **{source_b_name}**: {1-mix_weight:.0%}")

st.sidebar.header("Display")
layer_idx = st.sidebar.slider("Layer", 0, 11, 6)
show_all_layers = st.sidebar.checkbox("Show all 12 layers", value=False)

# --- Load data ---
with st.spinner(f"Loading {source_a_name}..."):
    audio_a, embs_a = load_and_extract(AUDIO_FILES[source_a_name])
with st.spinner(f"Loading {source_b_name}..."):
    audio_b, embs_b = load_and_extract(AUDIO_FILES[source_b_name])

# --- Mix audio and extract ---
min_len = min(audio_a.shape[-1], audio_b.shape[-1])
audio_a_trimmed = audio_a[..., :min_len]
audio_b_trimmed = audio_b[..., :min_len]
mixed_audio = mix_weight * audio_a_trimmed + (1 - mix_weight) * audio_b_trimmed

mixed_key = f"{source_a_name}|{source_b_name}|{mix_weight:.2f}"


@st.cache_data
def extract_mixed(_mixed_audio_bytes, _key):
    """Extract embeddings for mixed audio (keyed by content hash)."""
    model = load_model()
    audio_tensor = torch.from_numpy(_mixed_audio_bytes)
    layer_outputs = model.extract_features(audio_tensor, layers=None)
    return [layer.squeeze(0).cpu().numpy() for layer in layer_outputs]


embs_mix = extract_mixed(mixed_audio, mixed_key)

# --- Subsample ---
rng = np.random.default_rng(42)
embs_a_sub = subsample(embs_a, MAX_FRAMES, rng)
embs_b_sub = subsample(embs_b, MAX_FRAMES, rng)
embs_mix_sub = subsample(embs_mix, MAX_FRAMES, rng)

# --- Audio playback ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**{source_a_name}** (pure)")
    st.audio(audio_a_trimmed, sample_rate=16000)
with col2:
    st.markdown(f"**Mix** ({mix_weight:.0%} / {1-mix_weight:.0%})")
    st.audio(mixed_audio, sample_rate=16000)
with col3:
    st.markdown(f"**{source_b_name}** (pure)")
    st.audio(audio_b_trimmed, sample_rate=16000)


# --- Plotting ---
def plot_layer(ax, layer_i, embs_a_sub, embs_b_sub, embs_mix_sub, source_a_name, source_b_name):
    """PCA fit on pure sources, project mixture into same space."""
    pure_combined = np.concatenate([embs_a_sub[layer_i], embs_b_sub[layer_i]], axis=0)
    pca = PCA(n_components=2)
    pca.fit(pure_combined)
    var_exp = pca.explained_variance_ratio_

    coords_a = pca.transform(embs_a_sub[layer_i])
    coords_b = pca.transform(embs_b_sub[layer_i])
    coords_mix = pca.transform(embs_mix_sub[layer_i])

    ax.scatter(coords_a[:, 0], coords_a[:, 1], c="#FF5722", alpha=0.2, s=3,
               label=source_a_name, rasterized=True)
    ax.scatter(coords_b[:, 0], coords_b[:, 1], c="#4CAF50", alpha=0.2, s=3,
               label=source_b_name, rasterized=True)
    ax.scatter(coords_mix[:, 0], coords_mix[:, 1], c="#2196F3", alpha=0.4, s=5,
               label="Mixture", rasterized=True)

    ax.set_title(f"Layer {layer_i}", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var_exp[0]:.0%})", fontsize=8)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.0%})", fontsize=8)
    ax.tick_params(labelsize=7)
    return ax


if show_all_layers:
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f"{source_a_name} ({mix_weight:.0%}) + {source_b_name} ({1-mix_weight:.0%}) in AVES",
                 fontsize=14, fontweight="bold")
    for i in range(NUM_LAYERS):
        ax = axes[i // 4, i % 4]
        plot_layer(ax, i, embs_a_sub, embs_b_sub, embs_mix_sub, source_a_name, source_b_name)
        if i == 0:
            ax.legend(fontsize=7, markerscale=4)
    plt.tight_layout()
    st.pyplot(fig)
else:
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_layer(ax, layer_idx, embs_a_sub, embs_b_sub, embs_mix_sub, source_a_name, source_b_name)
    ax.legend(fontsize=9, markerscale=4)
    fig.suptitle(f"Layer {layer_idx}: {source_a_name} ({mix_weight:.0%}) + {source_b_name} ({1-mix_weight:.0%})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

# --- Stats ---
st.markdown("---")
st.markdown("**Frame counts:** "
            f"{source_a_name}: {embs_a_sub[0].shape[0]}, "
            f"{source_b_name}: {embs_b_sub[0].shape[0]}, "
            f"Mix: {embs_mix_sub[0].shape[0]}")
