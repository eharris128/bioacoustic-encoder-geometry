"""Streamlit app: inspect ESP-AVES2 comparison artifacts.

Run with:
    streamlit run app_esp_aves2_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA


DEFAULT_COMPARE_DIR = Path(
    "artifacts/comparisons/naturelm_by_source_100each_20260418T171459Z/"
    "sl_eat_bio_ssl_all_vs_sl_eat_all_ssl_all"
)
SOURCE_ORDER = [
    "All",
    "Xeno-canto",
    "WavCaps",
    "NatureLM",
    "Watkins",
    "iNaturalist",
    "Animal Sound Archive",
]
METRIC_LABELS = {
    "cosine_distance": "Cosine Distance",
    "l2_distance": "L2 Distance",
    "norm_delta_b_minus_a": "Norm Delta (model_b - model_a)",
}


st.set_page_config(page_title="ESP-AVES2 Comparison", layout="wide")
st.title("ESP-AVES2 Roadmap Comparison")
st.markdown("Compare pooled residual activations across the two completed `sl_*` ESP-AVES2 runs.")


@st.cache_data
def load_artifacts(compare_dir_str: str):
    compare_dir = Path(compare_dir_str)
    if not compare_dir.exists():
        raise FileNotFoundError(f"Comparison directory does not exist: {compare_dir}")

    metadata = pd.read_csv(compare_dir / "sample_metadata.csv")
    per_sample = pd.read_csv(compare_dir / "per_sample_layer_metrics.csv")
    summary = pd.read_csv(compare_dir / "source_layer_summary.csv")
    with (compare_dir / "comparison_config.json").open() as f:
        config = json.load(f)
    arrays = np.load(compare_dir / "pooled_embeddings.npz", allow_pickle=True)

    embeddings_a = arrays["embeddings_a"].astype(np.float32)
    embeddings_b = arrays["embeddings_b"].astype(np.float32)
    layer_names = [str(name) for name in arrays["layer_names"].tolist()]
    model_a = str(arrays["model_a"][0])
    model_b = str(arrays["model_b"][0])
    return metadata, per_sample, summary, config, embeddings_a, embeddings_b, layer_names, model_a, model_b


def source_colors(source_values: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {
        source_name: cmap(idx % 10)
        for idx, source_name in enumerate(source_values)
    }


def plot_heatmap(summary_df: pd.DataFrame, metric_name: str, source_order: list[str]) -> plt.Figure:
    pivot = summary_df.pivot_table(
        index="source_dataset",
        columns="layer_idx",
        values=f"{metric_name}_mean",
        aggfunc="first",
    ).reindex(source_order)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
    ax.set_title(METRIC_LABELS[metric_name], fontsize=12, fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Source Dataset")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(column) for column in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(f"{metric_name}_mean")
    plt.tight_layout()
    return fig


def plot_metric_profiles(summary_df: pd.DataFrame, metric_name: str, selected_sources: list[str]) -> plt.Figure:
    value_col = f"{metric_name}_mean"
    colors = source_colors(selected_sources)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for source_name in selected_sources:
        rows = summary_df[summary_df["source_dataset"] == source_name].sort_values("layer_idx")
        if rows.empty:
            continue
        ax.plot(
            rows["layer_idx"],
            rows[value_col],
            label=source_name,
            color=colors[source_name],
            linewidth=2.0 if source_name == "All" else 1.5,
        )
    ax.set_title(f"{METRIC_LABELS[metric_name]} by layer", fontsize=12, fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel(value_col)
    ax.set_xticks(range(13))
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    return fig


def plot_embedding_pca(
    metadata: pd.DataFrame,
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    model_a: str,
    model_b: str,
    layer_idx: int,
    selected_sources: list[str],
) -> plt.Figure:
    mask = metadata["source_dataset"].isin(selected_sources)
    metadata = metadata.loc[mask].reset_index(drop=True)
    vectors_a = embeddings_a[mask.to_numpy(), layer_idx, :]
    vectors_b = embeddings_b[mask.to_numpy(), layer_idx, :]

    combined = np.concatenate([vectors_a, vectors_b], axis=0)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(combined)
    coords_a = coords[: len(vectors_a)]
    coords_b = coords[len(vectors_a) :]

    colors = source_colors(selected_sources)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for source_name in selected_sources:
        source_mask = metadata["source_dataset"] == source_name
        ax.scatter(
            coords_a[source_mask, 0],
            coords_a[source_mask, 1],
            color=colors[source_name],
            marker="o",
            alpha=0.45,
            s=26,
            label=f"{source_name} | {model_a}",
        )
        ax.scatter(
            coords_b[source_mask, 0],
            coords_b[source_mask, 1],
            color=colors[source_name],
            marker="x",
            alpha=0.7,
            s=28,
            label=f"{source_name} | {model_b}",
        )

    ax.set_title(
        f"Layer {layer_idx}: pooled embedding PCA\n"
        f"PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    return fig


def plot_delta_pca(
    metadata: pd.DataFrame,
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    model_a: str,
    model_b: str,
    layer_idx: int,
    selected_sources: list[str],
) -> plt.Figure:
    mask = metadata["source_dataset"].isin(selected_sources)
    metadata = metadata.loc[mask].reset_index(drop=True)
    delta = embeddings_b[mask.to_numpy(), layer_idx, :] - embeddings_a[mask.to_numpy(), layer_idx, :]
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(delta)

    colors = source_colors(selected_sources)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for source_name in selected_sources:
        source_mask = metadata["source_dataset"] == source_name
        ax.scatter(
            coords[source_mask, 0],
            coords[source_mask, 1],
            color=colors[source_name],
            alpha=0.75,
            s=28,
            label=source_name,
        )

    ax.set_title(
        f"Layer {layer_idx}: {model_b} - {model_a}\n"
        f"PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    return fig


compare_dir = st.sidebar.text_input("Comparison Dir", str(DEFAULT_COMPARE_DIR))

try:
    metadata, per_sample, summary, config, embeddings_a, embeddings_b, layer_names, model_a, model_b = load_artifacts(compare_dir)
except Exception as exc:  # noqa: BLE001
    st.error(str(exc))
    st.stop()

metric_name = st.sidebar.selectbox("Metric", list(METRIC_LABELS.keys()), index=0, format_func=METRIC_LABELS.get)
layer_idx = st.sidebar.slider("Layer", 0, len(layer_names) - 1, int(config.get("pca_layer", 11)))
selected_sources = st.sidebar.multiselect(
    "Source Datasets",
    [value for value in SOURCE_ORDER if value != "All"],
    default=[value for value in SOURCE_ORDER if value != "All"],
)
top_k = st.sidebar.slider("Top Examples", 5, 50, 15, 5)

st.markdown(f"**Comparison:** `{model_b}` vs `{model_a}`")
st.markdown(f"**Manifest:** `{config['run_dir']}`")

if not selected_sources:
    st.warning("Select at least one source dataset.")
    st.stop()

left, right = st.columns(2)
with left:
    heatmap_df = summary[summary["source_dataset"].isin(["All", *selected_sources])]
    st.pyplot(plot_heatmap(heatmap_df, metric_name, ["All", *selected_sources]))
with right:
    profile_df = summary[summary["source_dataset"].isin(["All", *selected_sources])]
    st.pyplot(plot_metric_profiles(profile_df, metric_name, ["All", *selected_sources]))

pca_left, pca_right = st.columns(2)
with pca_left:
    st.pyplot(plot_embedding_pca(metadata, embeddings_a, embeddings_b, model_a, model_b, layer_idx, selected_sources))
with pca_right:
    st.pyplot(plot_delta_pca(metadata, embeddings_a, embeddings_b, model_a, model_b, layer_idx, selected_sources))

filtered = per_sample[
    (per_sample["layer_idx"] == layer_idx)
    & (per_sample["source_dataset"].isin(selected_sources))
].copy()
filtered = filtered.sort_values(metric_name, ascending=False).head(top_k)

st.markdown(f"**Top {top_k} examples at layer {layer_idx} by {METRIC_LABELS[metric_name]}**")
st.dataframe(
    filtered[
        [
            "file_name",
            "source_dataset",
            "task",
            "output",
            "valid_token_count",
            "cosine_distance",
            "l2_distance",
            "norm_model_a",
            "norm_model_b",
            "norm_delta_b_minus_a",
        ]
    ],
    use_container_width=True,
)
