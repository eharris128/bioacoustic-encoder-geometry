"""Compare two ESP-AVES2 activation runs and save summary plots/artifacts.

This script aligns samples across two completed activation extraction runs,
mean-pools valid patch tokens for each layer, and writes:
- compact pooled embeddings for both models
- per-sample, per-layer comparison metrics
- source-dataset/layer summary tables
- a few first-pass comparison plots

Usage:
    python compare_esp_aves2_models.py
    python compare_esp_aves2_models.py \
        --run_dir artifacts/roadmap_part1/naturelm_by_source_100each_20260418T171459Z \
        --model_a sl_eat_all_ssl_all \
        --model_b sl_eat_bio_ssl_all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


DEFAULT_RUN_DIR = Path("artifacts/roadmap_part1/naturelm_by_source_100each_20260418T171459Z")
DEFAULT_MODEL_A = "sl_eat_all_ssl_all"
DEFAULT_MODEL_B = "sl_eat_bio_ssl_all"
DEFAULT_SOURCE_ORDER = [
    "All",
    "Xeno-canto",
    "WavCaps",
    "NatureLM",
    "Watkins",
    "iNaturalist",
    "Animal Sound Archive",
]
PLOT_SOURCE_ORDER = DEFAULT_SOURCE_ORDER[1:]
METRIC_COLUMNS = [
    "cosine_similarity",
    "cosine_distance",
    "l2_distance",
    "norm_model_a",
    "norm_model_b",
    "norm_delta_b_minus_a",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model_a", type=str, default=DEFAULT_MODEL_A)
    parser.add_argument("--model_b", type=str, default=DEFAULT_MODEL_B)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to artifacts/comparisons/<run_id>/<model_b>_vs_<model_a>",
    )
    parser.add_argument(
        "--pca_layer",
        type=int,
        default=11,
        help="Layer index for the static PCA plot.",
    )
    return parser.parse_args()


def default_output_dir(args: argparse.Namespace) -> Path:
    return Path("artifacts/comparisons") / args.run_dir.name / f"{args.model_b}_vs_{args.model_a}"


def load_run_summary(model_dir: Path) -> dict:
    summary_path = model_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    with summary_path.open() as f:
        return json.load(f)


def pooled_layer_vectors(activations: torch.Tensor, valid_token_count: int) -> np.ndarray:
    valid_token_count = max(1, min(valid_token_count, activations.shape[1]))
    if valid_token_count > 1:
        pooled = activations[:, 1:valid_token_count, :].float().mean(dim=1)
    else:
        pooled = activations[:, 0, :].float()
    return pooled.numpy().astype(np.float32)


def load_pooled_embeddings(model_dir: Path) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    summary = load_run_summary(model_dir)
    layer_names = summary["layer_names"]
    shard_paths = sorted((model_dir / "shards").glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {model_dir / 'shards'}")

    pooled_by_row: dict[int, np.ndarray] = {}
    metadata_by_row: dict[int, dict] = {}

    for shard_idx, shard_path in enumerate(shard_paths, start=1):
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        activations = payload["activations"]
        samples = payload["samples"]
        print(
            f"Loading {model_dir.name} shard {shard_idx}/{len(shard_paths)}: {shard_path.name}",
            flush=True,
        )
        for sample_idx, sample in enumerate(samples):
            row_index = int(sample["row_index"])
            if row_index in pooled_by_row:
                raise RuntimeError(f"Duplicate row_index detected in {model_dir.name}: {row_index}")

            pooled_by_row[row_index] = pooled_layer_vectors(
                activations[sample_idx],
                valid_token_count=int(sample.get("valid_token_count") or activations.shape[2]),
            )
            metadata_by_row[row_index] = {
                "row_index": row_index,
                "id": sample.get("id") or "",
                "file_name": sample.get("file_name") or "",
                "source_dataset": sample.get("source_dataset") or "",
                "task": sample.get("task") or "",
                "output": sample.get("output") or "",
                "valid_token_count": int(sample.get("valid_token_count") or 0),
                "fbank_frames_before_pad": int(sample.get("fbank_frames_before_pad") or 0),
            }

    row_indices = sorted(pooled_by_row)
    pooled = np.stack([pooled_by_row[row_index] for row_index in row_indices], axis=0)
    metadata = pd.DataFrame([metadata_by_row[row_index] for row_index in row_indices])
    metadata = metadata.sort_values("row_index").reset_index(drop=True)
    return pooled, metadata, layer_names


def align_models(
    pooled_a: np.ndarray,
    meta_a: pd.DataFrame,
    pooled_b: np.ndarray,
    meta_b: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows_a = meta_a["row_index"].to_numpy()
    rows_b = meta_b["row_index"].to_numpy()
    if not np.array_equal(rows_a, rows_b):
        raise RuntimeError("Model row orders do not match; cannot compare runs safely.")

    merged = meta_a.copy()
    mismatch_cols = ["id", "file_name", "source_dataset", "task", "output"]
    for column in mismatch_cols:
        if not np.array_equal(meta_a[column].to_numpy(), meta_b[column].to_numpy()):
            raise RuntimeError(f"Metadata mismatch between runs for column: {column}")
    return pooled_a, pooled_b, merged


def compute_metric_arrays(
    pooled_a: np.ndarray,
    pooled_b: np.ndarray,
) -> dict[str, np.ndarray]:
    eps = 1e-8
    norm_a = np.linalg.norm(pooled_a, axis=-1)
    norm_b = np.linalg.norm(pooled_b, axis=-1)
    dot = np.sum(pooled_a * pooled_b, axis=-1)
    cosine_similarity = dot / np.clip(norm_a * norm_b, eps, None)
    cosine_distance = 1.0 - cosine_similarity
    l2_distance = np.linalg.norm(pooled_b - pooled_a, axis=-1)

    return {
        "cosine_similarity": cosine_similarity,
        "cosine_distance": cosine_distance,
        "l2_distance": l2_distance,
        "norm_model_a": norm_a,
        "norm_model_b": norm_b,
        "norm_delta_b_minus_a": norm_b - norm_a,
    }


def build_per_sample_metrics(
    metadata: pd.DataFrame,
    metrics: dict[str, np.ndarray],
    layer_names: list[str],
) -> pd.DataFrame:
    records: list[dict] = []
    row_count = len(metadata)
    for row_idx in range(row_count):
        meta_row = metadata.iloc[row_idx]
        for layer_idx, layer_name in enumerate(layer_names):
            record = {
                "row_index": int(meta_row["row_index"]),
                "id": meta_row["id"],
                "file_name": meta_row["file_name"],
                "source_dataset": meta_row["source_dataset"],
                "task": meta_row["task"],
                "output": meta_row["output"],
                "valid_token_count": int(meta_row["valid_token_count"]),
                "fbank_frames_before_pad": int(meta_row["fbank_frames_before_pad"]),
                "layer_idx": layer_idx,
                "layer_name": layer_name,
            }
            for metric_name, values in metrics.items():
                record[metric_name] = float(values[row_idx, layer_idx])
            records.append(record)
    return pd.DataFrame.from_records(records)


def build_summary(per_sample_df: pd.DataFrame) -> pd.DataFrame:
    summary_frames = [per_sample_df]
    overall = per_sample_df.copy()
    overall["source_dataset"] = "All"
    summary_frames.append(overall)
    combined = pd.concat(summary_frames, ignore_index=True)

    grouped = (
        combined.groupby(["source_dataset", "layer_idx", "layer_name"], sort=False)[METRIC_COLUMNS]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in column if part)
        for column in grouped.columns.to_flat_index()
    ]
    grouped = grouped.rename(
        columns={
            "source_dataset": "source_dataset",
            "layer_idx": "layer_idx",
            "layer_name": "layer_name",
        }
    )
    return grouped


def save_npz(
    output_path: Path,
    pooled_a: np.ndarray,
    pooled_b: np.ndarray,
    metadata: pd.DataFrame,
    layer_names: list[str],
    args: argparse.Namespace,
) -> None:
    np.savez_compressed(
        output_path,
        embeddings_a=pooled_a.astype(np.float16),
        embeddings_b=pooled_b.astype(np.float16),
        row_index=metadata["row_index"].to_numpy(np.int64),
        source_dataset=metadata["source_dataset"].to_numpy(),
        file_name=metadata["file_name"].to_numpy(),
        layer_names=np.asarray(layer_names),
        model_a=np.asarray([args.model_a]),
        model_b=np.asarray([args.model_b]),
    )


def get_source_colors(source_values: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {
        source_name: cmap(idx % 10)
        for idx, source_name in enumerate(source_values)
    }


def plot_heatmap(
    summary_df: pd.DataFrame,
    metric_name: str,
    output_path: Path,
    title: str,
    source_order: list[str],
) -> None:
    value_col = f"{metric_name}_mean"
    pivot = summary_df.pivot_table(
        index="source_dataset",
        columns="layer_idx",
        values=value_col,
        aggfunc="first",
    )
    pivot = pivot.reindex(source_order)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Source Dataset")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(column) for column in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(value_col)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_profiles(
    summary_df: pd.DataFrame,
    metric_name: str,
    output_path: Path,
    title: str,
    source_order: list[str],
) -> None:
    value_col = f"{metric_name}_mean"
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = get_source_colors(source_order)
    for source_name in source_order:
        source_rows = summary_df[summary_df["source_dataset"] == source_name].sort_values("layer_idx")
        if source_rows.empty:
            continue
        linewidth = 2.5 if source_name == "All" else 1.5
        alpha = 1.0 if source_name == "All" else 0.85
        ax.plot(
            source_rows["layer_idx"],
            source_rows[value_col],
            label=source_name,
            color=colors[source_name],
            linewidth=linewidth,
            alpha=alpha,
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel(value_col)
    ax.set_xticks(range(13))
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca(
    pooled_a: np.ndarray,
    pooled_b: np.ndarray,
    metadata: pd.DataFrame,
    layer_idx: int,
    output_path: Path,
    model_a: str,
    model_b: str,
) -> None:
    source_values = [value for value in DEFAULT_SOURCE_ORDER if value != "All" and value in metadata["source_dataset"].unique()]
    colors = get_source_colors(source_values)

    vectors_a = pooled_a[:, layer_idx, :]
    vectors_b = pooled_b[:, layer_idx, :]
    combined = np.concatenate([vectors_a, vectors_b], axis=0)
    model_pca = PCA(n_components=2, random_state=42)
    model_coords = model_pca.fit_transform(combined)
    coords_a = model_coords[: len(vectors_a)]
    coords_b = model_coords[len(vectors_a) :]

    delta_vectors = vectors_b - vectors_a
    delta_pca = PCA(n_components=2, random_state=42)
    delta_coords = delta_pca.fit_transform(delta_vectors)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for source_name in source_values:
        mask = metadata["source_dataset"] == source_name
        axes[0].scatter(
            coords_a[mask, 0],
            coords_a[mask, 1],
            label=f"{source_name} | {model_a}",
            color=colors[source_name],
            marker="o",
            alpha=0.55,
            s=28,
        )
        axes[0].scatter(
            coords_b[mask, 0],
            coords_b[mask, 1],
            label=f"{source_name} | {model_b}",
            color=colors[source_name],
            marker="x",
            alpha=0.75,
            s=28,
        )
        axes[1].scatter(
            delta_coords[mask, 0],
            delta_coords[mask, 1],
            label=source_name,
            color=colors[source_name],
            alpha=0.75,
            s=28,
        )

    axes[0].set_title(
        f"Layer {layer_idx}: pooled embeddings\nPC1={model_pca.explained_variance_ratio_[0]:.1%}, "
        f"PC2={model_pca.explained_variance_ratio_[1]:.1%}",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_title(
        f"Layer {layer_idx}: {model_b} - {model_a}\nPC1={delta_pca.explained_variance_ratio_[0]:.1%}, "
        f"PC2={delta_pca.explained_variance_ratio_[1]:.1%}",
        fontsize=11,
        fontweight="bold",
    )

    for axis in axes:
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=0.2)

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_a_dir = args.run_dir / args.model_a
    model_b_dir = args.run_dir / args.model_b
    print(f"Comparing {args.model_a} vs {args.model_b}", flush=True)
    print(f"Run dir: {args.run_dir}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)

    pooled_a, meta_a, layer_names_a = load_pooled_embeddings(model_a_dir)
    pooled_b, meta_b, layer_names_b = load_pooled_embeddings(model_b_dir)
    if layer_names_a != layer_names_b:
        raise RuntimeError("Layer names differ between runs.")

    pooled_a, pooled_b, metadata = align_models(pooled_a, meta_a, pooled_b, meta_b)
    metrics = compute_metric_arrays(pooled_a, pooled_b)
    per_sample_df = build_per_sample_metrics(metadata, metrics, layer_names_a)
    summary_df = build_summary(per_sample_df)

    metadata.to_csv(output_dir / "sample_metadata.csv", index=False)
    per_sample_df.to_csv(output_dir / "per_sample_layer_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "source_layer_summary.csv", index=False)
    save_npz(
        output_path=output_dir / "pooled_embeddings.npz",
        pooled_a=pooled_a,
        pooled_b=pooled_b,
        metadata=metadata,
        layer_names=layer_names_a,
        args=args,
    )

    config = {
        "run_dir": str(args.run_dir),
        "model_a": args.model_a,
        "model_b": args.model_b,
        "layer_names": layer_names_a,
        "pca_layer": args.pca_layer,
    }
    with (output_dir / "comparison_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    plot_heatmap(
        summary_df=summary_df,
        metric_name="cosine_distance",
        output_path=output_dir / "mean_cosine_distance_heatmap.png",
        title=f"{args.model_b} vs {args.model_a}: mean cosine distance by layer/source",
        source_order=DEFAULT_SOURCE_ORDER,
    )
    plot_heatmap(
        summary_df=summary_df,
        metric_name="norm_delta_b_minus_a",
        output_path=output_dir / "mean_norm_delta_heatmap.png",
        title=f"{args.model_b} minus {args.model_a}: mean pooled norm delta by layer/source",
        source_order=DEFAULT_SOURCE_ORDER,
    )
    plot_layer_profiles(
        summary_df=summary_df,
        metric_name="cosine_distance",
        output_path=output_dir / "mean_cosine_distance_profiles.png",
        title=f"{args.model_b} vs {args.model_a}: cosine distance profile by layer",
        source_order=PLOT_SOURCE_ORDER,
    )
    plot_pca(
        pooled_a=pooled_a,
        pooled_b=pooled_b,
        metadata=metadata,
        layer_idx=args.pca_layer,
        output_path=output_dir / f"layer_{args.pca_layer:02d}_pca.png",
        model_a=args.model_a,
        model_b=args.model_b,
    )
    print("Saved comparison artifacts.", flush=True)


if __name__ == "__main__":
    main()
