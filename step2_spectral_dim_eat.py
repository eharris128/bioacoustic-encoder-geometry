"""Step 2 of the ESP-AVES2 roadmap: spectral and intrinsic-dim statistics.

Reads the consolidated pooled embeddings produced by `nway_compare_eat_models.py`
and computes per-(model, layer):
- singular-value spectrum of the centered pooled-embedding matrix
- effective rank (exp of Shannon entropy of normalized eigenvalues)
- participation ratio (Sum(lambda)^2 / Sum(lambda^2))
- TwoNN intrinsic dimensionality estimate (Facco et al. 2017)

Also slices effective rank by source_dataset to address the roadmap's
"compare nature sounds to other sound" question.

Usage:
    python step2_spectral_dim_eat.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


DEFAULT_NWAY_DIR = Path(
    "artifacts/comparisons/naturelm_by_source_100each_20260418T171459Z/nway_eat_all4"
)
PLOT_SOURCE_ORDER = [
    "Xeno-canto",
    "WavCaps",
    "NatureLM",
    "Watkins",
    "iNaturalist",
    "Animal Sound Archive",
]
NATURE_SOURCES = {"Xeno-canto", "iNaturalist", "Animal Sound Archive", "Watkins"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nway_dir", type=Path, default=DEFAULT_NWAY_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--twonn_sample_size",
        type=int,
        default=None,
        help="If set, subsample to this many points before running TwoNN.",
    )
    return parser.parse_args()


def effective_rank(eigenvalues: np.ndarray) -> float:
    eigenvalues = eigenvalues[eigenvalues > 0]
    if eigenvalues.size == 0:
        return float("nan")
    p = eigenvalues / eigenvalues.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(eigenvalues: np.ndarray) -> float:
    eigenvalues = eigenvalues[eigenvalues > 0]
    if eigenvalues.size == 0:
        return float("nan")
    s = eigenvalues.sum()
    s2 = (eigenvalues ** 2).sum()
    return float(s * s / s2)


def twonn_intrinsic_dim(matrix: np.ndarray, sample_size: int | None = None) -> float:
    if sample_size is not None and matrix.shape[0] > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(matrix.shape[0], sample_size, replace=False)
        matrix = matrix[idx]
    if matrix.shape[0] < 3:
        return float("nan")
    nbrs = NearestNeighbors(n_neighbors=3).fit(matrix)
    dists, _ = nbrs.kneighbors(matrix)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    valid = (r1 > 0) & (r2 > r1)
    if valid.sum() < 3:
        return float("nan")
    mu = r2[valid] / r1[valid]
    log_mu = np.log(mu)
    log_mu_sorted = np.sort(log_mu)
    n = log_mu_sorted.size
    f_emp = np.arange(1, n + 1) / (n + 1)
    y = -np.log(1.0 - f_emp)
    return float((y * log_mu_sorted).sum() / (log_mu_sorted ** 2).sum())


def per_layer_spectrum(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = (s ** 2) / max(centered.shape[0] - 1, 1)
    return eigenvalues.astype(np.float64)


def load_consolidated(nway_dir: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[str], list[str]]:
    archive = np.load(nway_dir / "pooled_embeddings_all4.npz", allow_pickle=True)
    models = [str(m) for m in archive["models"].tolist()]
    layer_names = [str(name) for name in archive["layer_names"].tolist()]
    pooled_by_model = {
        model_key: archive[f"embeddings_{model_key}"].astype(np.float32) for model_key in models
    }
    metadata = pd.DataFrame(
        {
            "row_index": archive["row_index"].astype(np.int64),
            "source_dataset": [str(s) for s in archive["source_dataset"].tolist()],
            "file_name": [str(f) for f in archive["file_name"].tolist()],
        }
    )
    return pooled_by_model, metadata, models, layer_names


def compute_per_layer_stats(
    pooled_by_model: dict[str, np.ndarray],
    models: list[str],
    layer_names: list[str],
    twonn_sample_size: int | None,
) -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    records: list[dict] = []
    spectra: dict[tuple[str, int], np.ndarray] = {}
    n_layers = len(layer_names)
    for model_key in models:
        pooled = pooled_by_model[model_key]
        for layer_idx in range(n_layers):
            matrix = pooled[:, layer_idx, :].astype(np.float64)
            eigenvalues = per_layer_spectrum(matrix)
            spectra[(model_key, layer_idx)] = eigenvalues
            er = effective_rank(eigenvalues)
            pr = participation_ratio(eigenvalues)
            twonn = twonn_intrinsic_dim(matrix, sample_size=twonn_sample_size)
            top1 = float(eigenvalues[0] / eigenvalues.sum())
            top10 = float(eigenvalues[:10].sum() / eigenvalues.sum())
            records.append(
                {
                    "model": model_key,
                    "layer_idx": layer_idx,
                    "layer_name": layer_names[layer_idx],
                    "n_samples": int(matrix.shape[0]),
                    "embedding_dim": int(matrix.shape[1]),
                    "effective_rank": er,
                    "participation_ratio": pr,
                    "twonn_id": twonn,
                    "var_explained_top1": top1,
                    "var_explained_top10": top10,
                }
            )
    return pd.DataFrame.from_records(records), spectra


def compute_effective_rank_by_source(
    pooled_by_model: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    models: list[str],
    layer_names: list[str],
) -> pd.DataFrame:
    records: list[dict] = []
    sources_in_data = sorted(metadata["source_dataset"].unique().tolist())
    nature_mask = metadata["source_dataset"].isin(NATURE_SOURCES).to_numpy()
    music_speech_mask = ~nature_mask

    for model_key in models:
        pooled = pooled_by_model[model_key]
        for layer_idx in range(len(layer_names)):
            matrix = pooled[:, layer_idx, :].astype(np.float64)
            for source in sources_in_data:
                mask = (metadata["source_dataset"] == source).to_numpy()
                if mask.sum() < 5:
                    continue
                er = effective_rank(per_layer_spectrum(matrix[mask]))
                records.append(
                    {
                        "model": model_key,
                        "layer_idx": layer_idx,
                        "source_dataset": source,
                        "n_samples": int(mask.sum()),
                        "effective_rank": er,
                    }
                )
            for label, mask in (("nature_only", nature_mask), ("nonnature_only", music_speech_mask)):
                if mask.sum() < 5:
                    continue
                er = effective_rank(per_layer_spectrum(matrix[mask]))
                records.append(
                    {
                        "model": model_key,
                        "layer_idx": layer_idx,
                        "source_dataset": label,
                        "n_samples": int(mask.sum()),
                        "effective_rank": er,
                    }
                )
    return pd.DataFrame.from_records(records)


def plot_eigenvalue_spectra(
    spectra: dict[tuple[str, int], np.ndarray],
    models: list[str],
    n_layers: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis", n_layers)
    for ax, model_key in zip(axes, models):
        for layer_idx in range(n_layers):
            eigs = spectra[(model_key, layer_idx)]
            eigs = eigs[eigs > 0]
            x = np.arange(1, eigs.size + 1)
            ax.loglog(x, eigs, color=cmap(layer_idx), linewidth=0.9, alpha=0.85)
        ax.set_title(model_key)
        ax.set_xlabel("eigenvalue index")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("eigenvalue (centered cov, log)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_layers - 1))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.02)
    cbar.set_label("layer index")
    fig.suptitle("Eigenvalue spectra of mean-pooled embeddings (per model, all layers overlaid)")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for idx, model_key in enumerate(df["model"].unique()):
        sub = df[df["model"] == model_key].sort_values("layer_idx")
        ax.plot(
            sub["layer_idx"].to_numpy(),
            sub[metric].to_numpy(),
            marker="o",
            color=cmap(idx),
            label=model_key,
        )
    ax.set_xlabel("layer index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_nature_vs_other(
    source_df: pd.DataFrame,
    models: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model_key in zip(axes, models):
        for label, color in (("nature_only", "tab:green"), ("nonnature_only", "tab:red")):
            sub = source_df[
                (source_df["model"] == model_key) & (source_df["source_dataset"] == label)
            ].sort_values("layer_idx")
            if sub.empty:
                continue
            ax.plot(
                sub["layer_idx"].to_numpy(),
                sub["effective_rank"].to_numpy(),
                marker="o",
                color=color,
                label=label,
            )
        ax.set_title(model_key)
        ax.set_xlabel("layer index")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("effective rank")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.suptitle("Effective rank: nature sources vs non-nature (WavCaps + NatureLM)")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.nway_dir / "step2_spectral_dim")
    output_dir.mkdir(parents=True, exist_ok=True)

    pooled_by_model, metadata, models, layer_names = load_consolidated(args.nway_dir)
    print(f"Loaded {len(models)} models, {len(metadata)} samples, {len(layer_names)} layers", flush=True)

    print("Computing per-(model, layer) spectra + intrinsic dim...", flush=True)
    stats_df, spectra = compute_per_layer_stats(
        pooled_by_model, models, layer_names, twonn_sample_size=args.twonn_sample_size
    )
    stats_df.to_csv(output_dir / "per_layer_stats.csv", index=False)

    eigvals_archive = {
        f"{model_key}_layer{layer_idx:02d}": eigs.astype(np.float32)
        for (model_key, layer_idx), eigs in spectra.items()
    }
    np.savez_compressed(
        output_dir / "eigenvalues.npz",
        models=np.asarray(models),
        layer_names=np.asarray(layer_names),
        **eigvals_archive,
    )

    print("Computing per-source effective rank...", flush=True)
    source_df = compute_effective_rank_by_source(pooled_by_model, metadata, models, layer_names)
    source_df.to_csv(output_dir / "effective_rank_by_source.csv", index=False)

    print("Plotting...", flush=True)
    plot_eigenvalue_spectra(spectra, models, len(layer_names), output_dir / "eigenvalue_spectra.png")
    plot_layer_metric(
        stats_df,
        "effective_rank",
        "Effective rank (exp Shannon entropy of normalized eigenvalues)",
        "effective rank",
        output_dir / "effective_rank_by_layer.png",
    )
    plot_layer_metric(
        stats_df,
        "participation_ratio",
        "Participation ratio of mean-pooled embedding spectrum",
        "participation ratio",
        output_dir / "participation_ratio_by_layer.png",
    )
    plot_layer_metric(
        stats_df,
        "twonn_id",
        "TwoNN intrinsic-dimensionality estimate (mean-pooled embeddings)",
        "intrinsic dim",
        output_dir / "twonn_id_by_layer.png",
    )
    plot_nature_vs_other(source_df, models, output_dir / "effective_rank_nature_vs_other.png")

    print("\nPer-(model, layer) statistics:")
    print(stats_df.to_string(index=False), flush=True)

    print(f"\nSaved artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
