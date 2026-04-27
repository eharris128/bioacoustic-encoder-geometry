"""Step 2 (continued): subspace angles + L2-norm histograms for the EAT-family.

Reads the consolidated pooled embeddings produced by `nway_compare_eat_models.py`
and computes:

1. **L2-norm histograms** per `(model, layer)` — within-model distributions so the
   cross-model norm deltas in `norm_by_layer_source.csv` are interpretable.
2. **PCA / subspace alignment**:
   - Across layers within a model: principal angles between top-k subspaces of
     consecutive layer pairs (and a full layer-x-layer overlap heatmap).
   - Across models within a layer: principal angles between top-k subspaces of
     all 6 model pairs, per layer.
3. **Bio vs non-bio subspace angles** per `(model, layer)` — directly tests
   whether `sl_eat_bio_ssl_all`'s wider linear subspace is bio-specific.
4. **L0 shared-tokenizer confirmation** — across-model L0 angles printed
   separately as a sanity check that L0 is essentially the same subspace
   across all four models (motivated by the L0 effective rank ≈ 3 finding).

Subspace overlap summary metric: mean(cos(principal_angles)) where the angles
come from `scipy.linalg.subspace_angles`. 1.0 = identical subspaces, 0.0 =
fully orthogonal. Top-k = 10 (captures the dominant variance in every layer
based on the existing eigenvalue spectra).

Usage:
    python step2_subspace_angles_eat.py
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles


DEFAULT_NWAY_DIR = Path(
    "artifacts/comparisons/naturelm_by_source_100each_20260418T171459Z/nway_eat_all4"
)
NATURE_SOURCES = {"Xeno-canto", "iNaturalist", "Animal Sound Archive", "Watkins"}
TOP_K = 10
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nway_dir", type=Path, default=DEFAULT_NWAY_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    return parser.parse_args()


def load_consolidated(nway_dir: Path):
    archive = np.load(nway_dir / "pooled_embeddings_all4.npz", allow_pickle=True)
    models = [str(m) for m in archive["models"].tolist()]
    layer_names = [str(name) for name in archive["layer_names"].tolist()]
    pooled_by_model = {m: archive[f"embeddings_{m}"].astype(np.float32) for m in models}
    metadata = pd.DataFrame(
        {
            "row_index": archive["row_index"].astype(np.int64),
            "source_dataset": [str(s) for s in archive["source_dataset"].tolist()],
            "file_name": [str(f) for f in archive["file_name"].tolist()],
        }
    )
    return pooled_by_model, metadata, models, layer_names


def top_k_basis(matrix: np.ndarray, k: int) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:k].T  # columns are basis vectors, shape (D, k)


def overlap(basis_a: np.ndarray, basis_b: np.ndarray) -> tuple[float, np.ndarray]:
    angles = subspace_angles(basis_a, basis_b)
    cos_angles = np.cos(angles)
    return float(cos_angles.mean()), cos_angles


# ---------------------------------------------------------------------------
# 1. L2-norm histograms
# ---------------------------------------------------------------------------

def plot_l2_norm_histograms(
    pooled_by_model: dict[str, np.ndarray],
    models: list[str],
    layer_names: list[str],
    output_path: Path,
) -> pd.DataFrame:
    n_layers = len(layer_names)
    fig, axes = plt.subplots(len(models), n_layers, figsize=(2.0 * n_layers, 2.2 * len(models)),
                              sharex="col", sharey="row")
    records: list[dict] = []
    for row, model_key in enumerate(models):
        pooled = pooled_by_model[model_key]
        for col in range(n_layers):
            ax = axes[row, col] if len(models) > 1 else axes[col]
            norms = np.linalg.norm(pooled[:, col, :], axis=1)
            ax.hist(norms, bins=30, color="tab:blue", alpha=0.85, edgecolor="white", linewidth=0.3)
            if row == 0:
                ax.set_title(f"L{col}", fontsize=8)
            if col == 0:
                ax.set_ylabel(model_key, fontsize=8)
            ax.tick_params(labelsize=6)
            records.append({
                "model": model_key,
                "layer_idx": col,
                "layer_name": layer_names[col],
                "n": int(norms.size),
                "norm_mean": float(norms.mean()),
                "norm_std": float(norms.std()),
                "norm_p05": float(np.percentile(norms, 5)),
                "norm_p50": float(np.percentile(norms, 50)),
                "norm_p95": float(np.percentile(norms, 95)),
                "norm_min": float(norms.min()),
                "norm_max": float(norms.max()),
            })
    fig.suptitle("Within-model L2-norm distributions of mean-pooled embeddings (per layer)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# 2a. Across-layer subspace alignment (within model)
# ---------------------------------------------------------------------------

def across_layer_alignment(
    pooled_by_model: dict[str, np.ndarray],
    models: list[str],
    layer_names: list[str],
    k: int,
    output_dir: Path,
) -> pd.DataFrame:
    n_layers = len(layer_names)
    records: list[dict] = []
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4.0), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model_key in zip(axes, models):
        pooled = pooled_by_model[model_key]
        bases = [top_k_basis(pooled[:, layer_idx, :].astype(np.float64), k) for layer_idx in range(n_layers)]
        heat = np.zeros((n_layers, n_layers))
        for i in range(n_layers):
            for j in range(n_layers):
                mean_cos, _ = overlap(bases[i], bases[j])
                heat[i, j] = mean_cos
                if j > i:
                    records.append({
                        "model": model_key,
                        "layer_a": i,
                        "layer_b": j,
                        "mean_cos_principal_angles": mean_cos,
                        "k": k,
                    })
        im = ax.imshow(heat, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
        ax.set_title(model_key)
        ax.set_xlabel("layer b")
        ax.set_ylabel("layer a")
    fig.suptitle(f"Across-layer subspace overlap (mean cos principal angles, top-{k})")
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02, label="mean cos angle")
    fig.savefig(output_dir / "across_layer_subspace_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# 2b. Across-model subspace alignment (within layer)
# ---------------------------------------------------------------------------

def across_model_alignment(
    pooled_by_model: dict[str, np.ndarray],
    models: list[str],
    layer_names: list[str],
    k: int,
    output_dir: Path,
) -> pd.DataFrame:
    n_layers = len(layer_names)
    records: list[dict] = []
    pairs = list(itertools.combinations(models, 2))
    pair_curves: dict[tuple[str, str], list[float]] = {pair: [] for pair in pairs}
    for layer_idx in range(n_layers):
        bases = {
            m: top_k_basis(pooled_by_model[m][:, layer_idx, :].astype(np.float64), k) for m in models
        }
        for a, b in pairs:
            mean_cos, _ = overlap(bases[a], bases[b])
            pair_curves[(a, b)].append(mean_cos)
            records.append({
                "layer_idx": layer_idx,
                "layer_name": layer_names[layer_idx],
                "model_a": a,
                "model_b": b,
                "mean_cos_principal_angles": mean_cos,
                "k": k,
            })
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for idx, ((a, b), curve) in enumerate(pair_curves.items()):
        ax.plot(range(n_layers), curve, marker="o", color=cmap(idx), label=f"{a} vs {b}")
    ax.set_xlabel("layer index")
    ax.set_ylabel(f"mean cos principal angles (top-{k})")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Across-model subspace overlap per layer (all 6 pairs)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.savefig(output_dir / "across_model_subspace_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# 3. Bio vs non-bio subspace angles per (model, layer)
# ---------------------------------------------------------------------------

def bio_vs_nonbio_alignment(
    pooled_by_model: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    models: list[str],
    layer_names: list[str],
    k: int,
    output_dir: Path,
) -> pd.DataFrame:
    n_layers = len(layer_names)
    nature_mask = metadata["source_dataset"].isin(NATURE_SOURCES).to_numpy()
    nonnature_mask = ~nature_mask
    n_bio = int(nature_mask.sum())
    n_nonbio = int(nonnature_mask.sum())
    print(f"  bio samples: {n_bio}, non-bio samples: {n_nonbio}", flush=True)
    records: list[dict] = []
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for idx, model_key in enumerate(models):
        pooled = pooled_by_model[model_key]
        curve = []
        for layer_idx in range(n_layers):
            mat = pooled[:, layer_idx, :].astype(np.float64)
            basis_bio = top_k_basis(mat[nature_mask], k)
            basis_nonbio = top_k_basis(mat[nonnature_mask], k)
            mean_cos, cos_angles = overlap(basis_bio, basis_nonbio)
            curve.append(mean_cos)
            records.append({
                "model": model_key,
                "layer_idx": layer_idx,
                "layer_name": layer_names[layer_idx],
                "mean_cos_principal_angles": mean_cos,
                "min_cos_principal_angles": float(cos_angles.min()),
                "max_cos_principal_angles": float(cos_angles.max()),
                "k": k,
                "n_bio": n_bio,
                "n_nonbio": n_nonbio,
            })
        ax.plot(range(n_layers), curve, marker="o", color=cmap(idx), label=model_key)
    ax.set_xlabel("layer index")
    ax.set_ylabel(f"mean cos principal angles (top-{k})")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Bio-only vs non-bio-only subspace overlap per (model, layer)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.savefig(output_dir / "bio_vs_nonbio_subspace_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# 4. L0 shared-tokenizer printout
# ---------------------------------------------------------------------------

def print_l0_pairwise(across_model_df: pd.DataFrame, models: list[str]) -> None:
    l0 = across_model_df[across_model_df["layer_idx"] == 0]
    print("\nL0 across-model subspace overlap (testing shared tokenizer hypothesis):")
    for _, row in l0.iterrows():
        print(f"  {row['model_a']:>22s}  vs  {row['model_b']:<22s}  mean cos = {row['mean_cos_principal_angles']:.4f}")
    overall = float(l0["mean_cos_principal_angles"].mean())
    print(f"  overall L0 mean cos across {len(l0)} pairs = {overall:.4f}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.nway_dir / "step2_subspace_angles")
    output_dir.mkdir(parents=True, exist_ok=True)

    pooled_by_model, metadata, models, layer_names = load_consolidated(args.nway_dir)
    print(f"Loaded {len(models)} models, {len(metadata)} samples, {len(layer_names)} layers", flush=True)

    print("\n[1/4] L2-norm histograms...", flush=True)
    norms_df = plot_l2_norm_histograms(
        pooled_by_model, models, layer_names, output_dir / "l2_norm_histograms.png"
    )
    norms_df.to_csv(output_dir / "l2_norm_per_layer.csv", index=False)

    print("\n[2/4] Across-layer subspace alignment (within each model)...", flush=True)
    across_layer_df = across_layer_alignment(
        pooled_by_model, models, layer_names, args.top_k, output_dir
    )
    across_layer_df.to_csv(output_dir / "across_layer_subspace_overlap.csv", index=False)

    print("\n[3/4] Across-model subspace alignment (within each layer)...", flush=True)
    across_model_df = across_model_alignment(
        pooled_by_model, models, layer_names, args.top_k, output_dir
    )
    across_model_df.to_csv(output_dir / "across_model_subspace_overlap.csv", index=False)

    print("\n[4/4] Bio vs non-bio subspace angles per (model, layer)...", flush=True)
    bio_df = bio_vs_nonbio_alignment(
        pooled_by_model, metadata, models, layer_names, args.top_k, output_dir
    )
    bio_df.to_csv(output_dir / "bio_vs_nonbio_subspace_overlap.csv", index=False)

    print_l0_pairwise(across_model_df, models)

    print(f"\nSaved artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
