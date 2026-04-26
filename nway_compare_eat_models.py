"""4-way comparison of ESP-AVES2 EAT-family models on the NatureLM activation pilot.

Loads pooled embeddings for all four extracted models on the same frozen
manifest, then writes:
- consolidated pooled embeddings (one .npz with all four models, sample-aligned)
- cross-model x cross-layer linear-CKA matrix + heatmap
- per-model L2-norm distribution table per (layer, source_dataset)

Usage:
    python nway_compare_eat_models.py
    python nway_compare_eat_models.py --models eat_all,eat_bio
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_esp_aves2_models import load_pooled_embeddings


DEFAULT_RUN_DIR = Path("artifacts/roadmap_part1/naturelm_by_source_100each_20260418T171459Z")
DEFAULT_MODELS = ["eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all"]
PLOT_SOURCE_ORDER = [
    "Xeno-canto",
    "WavCaps",
    "NatureLM",
    "Watkins",
    "iNaturalist",
    "Animal Sound Archive",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model keys (must all share the same manifest).",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    # Centered linear CKA: HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y)).
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xtx = x @ x.T
    yty = y @ y.T
    num = float((xtx * yty).sum())
    denom = float(np.sqrt((xtx * xtx).sum() * (yty * yty).sum()))
    if denom == 0.0:
        return float("nan")
    return num / denom


def load_all_models(
    run_dir: Path, model_keys: list[str]
) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[str]]:
    pooled_by_model: dict[str, np.ndarray] = {}
    layer_names: list[str] | None = None
    metadata_ref: pd.DataFrame | None = None

    for model_key in model_keys:
        model_dir = run_dir / model_key
        pooled, metadata, layers = load_pooled_embeddings(model_dir)
        pooled_by_model[model_key] = pooled

        if layer_names is None:
            layer_names = layers
            metadata_ref = metadata
            continue

        if layers != layer_names:
            raise RuntimeError(f"Layer-name mismatch for {model_key}")
        if not np.array_equal(
            metadata["row_index"].to_numpy(), metadata_ref["row_index"].to_numpy()
        ):
            raise RuntimeError(f"row_index alignment mismatch for {model_key}")
        for column in ("id", "file_name", "source_dataset"):
            if not np.array_equal(
                metadata[column].to_numpy(), metadata_ref[column].to_numpy()
            ):
                raise RuntimeError(f"Metadata mismatch on column={column} for {model_key}")

    assert layer_names is not None and metadata_ref is not None
    return pooled_by_model, metadata_ref, layer_names


def compute_cka_matrix(
    pooled_by_model: dict[str, np.ndarray], model_keys: list[str], n_layers: int
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    flat_keys = [(model_key, layer_idx) for model_key in model_keys for layer_idx in range(n_layers)]
    matrix = np.zeros((len(flat_keys), len(flat_keys)), dtype=np.float64)
    for i, (model_i, layer_i) in enumerate(flat_keys):
        x_i = pooled_by_model[model_i][:, layer_i, :].astype(np.float64)
        for j, (model_j, layer_j) in enumerate(flat_keys):
            if j < i:
                matrix[i, j] = matrix[j, i]
                continue
            if i == j:
                matrix[i, j] = 1.0
                continue
            x_j = pooled_by_model[model_j][:, layer_j, :].astype(np.float64)
            matrix[i, j] = linear_cka(x_i, x_j)
    return matrix, flat_keys


def plot_cka_heatmap(
    matrix: np.ndarray,
    model_keys: list[str],
    n_layers: int,
    output_path: Path,
    n_samples: int,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")

    for i in range(1, len(model_keys)):
        ax.axhline(i * n_layers - 0.5, color="white", linewidth=1.2)
        ax.axvline(i * n_layers - 0.5, color="white", linewidth=1.2)

    midpoints = [(i + 0.5) * n_layers - 0.5 for i in range(len(model_keys))]
    ax.set_xticks(midpoints)
    ax.set_xticklabels(model_keys, rotation=20, ha="right")
    ax.set_yticks(midpoints)
    ax.set_yticklabels(model_keys)

    for i, model_key in enumerate(model_keys):
        for layer_idx in range(n_layers):
            offset = i * n_layers + layer_idx
            if layer_idx % 3 == 0:
                ax.text(
                    offset,
                    -0.5,
                    f"L{layer_idx}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="dimgray",
                )

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Linear CKA")
    ax.set_title(
        f"Cross-model x cross-layer linear CKA (n={n_samples} samples, mean-pooled)"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_norm_summary(
    pooled_by_model: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    model_keys: list[str],
    layer_names: list[str],
    output_path: Path,
) -> pd.DataFrame:
    records: list[dict] = []
    for model_key in model_keys:
        pooled = pooled_by_model[model_key]
        norms = np.linalg.norm(pooled, axis=-1)
        for layer_idx, layer_name in enumerate(layer_names):
            for source_dataset, group in metadata.groupby("source_dataset"):
                indices = group.index.to_numpy()
                values = norms[indices, layer_idx]
                records.append(
                    {
                        "model": model_key,
                        "layer_idx": layer_idx,
                        "layer_name": layer_name,
                        "source_dataset": source_dataset,
                        "n_samples": int(len(values)),
                        "norm_mean": float(values.mean()),
                        "norm_std": float(values.std()),
                        "norm_median": float(np.median(values)),
                    }
                )
            values_all = norms[:, layer_idx]
            records.append(
                {
                    "model": model_key,
                    "layer_idx": layer_idx,
                    "layer_name": layer_name,
                    "source_dataset": "All",
                    "n_samples": int(len(values_all)),
                    "norm_mean": float(values_all.mean()),
                    "norm_std": float(values_all.std()),
                    "norm_median": float(np.median(values_all)),
                }
            )

    norm_df = pd.DataFrame.from_records(records)
    norm_df.to_csv(output_path, index=False)
    return norm_df


def plot_norm_profiles(
    norm_df: pd.DataFrame,
    model_keys: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(model_keys), figsize=(4.2 * len(model_keys), 4.5), sharey=True)
    if len(model_keys) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    color_by_source = {source: cmap(idx) for idx, source in enumerate(PLOT_SOURCE_ORDER)}

    for ax, model_key in zip(axes, model_keys):
        sub = norm_df[norm_df["model"] == model_key]
        for source in PLOT_SOURCE_ORDER + ["All"]:
            entries = sub[sub["source_dataset"] == source].sort_values("layer_idx")
            if entries.empty:
                continue
            line_style = "-" if source != "All" else "--"
            line_width = 1.2 if source != "All" else 2.2
            color = color_by_source.get(source, "black")
            ax.plot(
                entries["layer_idx"].to_numpy(),
                entries["norm_mean"].to_numpy(),
                line_style,
                linewidth=line_width,
                color=color,
                label=source,
            )
        ax.set_title(model_key)
        ax.set_xlabel("layer index")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean L2 norm of pooled embedding")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    fig.suptitle("Per-model mean L2 norm by layer and source dataset")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_cka(
    matrix: np.ndarray,
    flat_keys: list[tuple[str, int]],
    model_keys: list[str],
    n_layers: int,
) -> pd.DataFrame:
    records: list[dict] = []
    for i, model_a in enumerate(model_keys):
        for j, model_b in enumerate(model_keys):
            if j <= i:
                continue
            diag = []
            for layer_idx in range(n_layers):
                a_idx = i * n_layers + layer_idx
                b_idx = j * n_layers + layer_idx
                diag.append(matrix[a_idx, b_idx])
            records.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "same_layer_cka_mean": float(np.mean(diag)),
                    "same_layer_cka_min": float(np.min(diag)),
                    "same_layer_cka_min_layer": int(np.argmin(diag)),
                    "same_layer_cka_max": float(np.max(diag)),
                    "same_layer_cka_max_layer": int(np.argmax(diag)),
                }
            )
    return pd.DataFrame.from_records(records)


def main() -> None:
    args = parse_args()
    model_keys = [key.strip() for key in args.models.split(",") if key.strip()]
    output_dir = args.output_dir or (
        Path("artifacts/comparisons") / args.run_dir.name / "nway_eat_all4"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pooled_by_model, metadata, layer_names = load_all_models(args.run_dir, model_keys)
    n_samples = len(metadata)
    n_layers = len(layer_names)
    print(f"Loaded {len(model_keys)} models, {n_samples} samples, {n_layers} layers", flush=True)

    np.savez_compressed(
        output_dir / "pooled_embeddings_all4.npz",
        **{f"embeddings_{k}": v.astype(np.float16) for k, v in pooled_by_model.items()},
        row_index=metadata["row_index"].to_numpy(np.int64),
        source_dataset=metadata["source_dataset"].to_numpy(),
        file_name=metadata["file_name"].to_numpy(),
        layer_names=np.asarray(layer_names),
        models=np.asarray(model_keys),
    )
    metadata.to_csv(output_dir / "sample_metadata.csv", index=False)

    print("Computing CKA matrix...", flush=True)
    cka_matrix, flat_keys = compute_cka_matrix(pooled_by_model, model_keys, n_layers)

    np.savez_compressed(
        output_dir / "cka_matrix.npz",
        cka=cka_matrix.astype(np.float32),
        models=np.asarray(model_keys),
        layer_names=np.asarray(layer_names),
    )

    cka_df = pd.DataFrame(
        cka_matrix,
        index=pd.MultiIndex.from_tuples(flat_keys, names=["model", "layer_idx"]),
        columns=pd.MultiIndex.from_tuples(flat_keys, names=["model", "layer_idx"]),
    )
    cka_df.to_csv(output_dir / "cka_matrix.csv")

    plot_cka_heatmap(cka_matrix, model_keys, n_layers, output_dir / "cka_heatmap.png", n_samples)

    summary_df = summarize_cka(cka_matrix, flat_keys, model_keys, n_layers)
    summary_df.to_csv(output_dir / "cka_pairwise_summary.csv", index=False)
    print("\nSame-layer CKA summary (model_a vs model_b):", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    print("\nComputing per-model L2-norm distributions...", flush=True)
    norm_df = write_norm_summary(
        pooled_by_model, metadata, model_keys, layer_names, output_dir / "norm_by_layer_source.csv"
    )
    plot_norm_profiles(norm_df, model_keys, output_dir / "norm_profiles.png")

    print(f"\nSaved artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
