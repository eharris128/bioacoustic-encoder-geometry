"""
scripts/export_web_projection.py — Export a supervised 2D embedding projection
for the personal-website "Embedding projector" explorer exhibit.

For each (model, layer) we fit a 2-component LDA on the pooled embeddings using
the source dataset as the class label, giving the best 2D *separating* view of
that layer. Consecutive layers are aligned with an orthogonal Procrustes rotation
so the layer slider morphs smoothly instead of teleporting (LDA axes are
otherwise arbitrary per layer). We also record the silhouette of each view so the
UI can show how separation rises with depth and recedes at the final layer.

Unsupervised PCA does NOT separate these sources (silhouette ~0 at every layer);
the supervised view is what makes the layerwise story legible. The exhibit copy
labels this honestly as "best 2D separating view at this layer."

Usage:
    python -W ignore scripts/export_web_projection.py \
        --out ~/projects/personal-website/public/explorer/embedding_projection.json
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score

NPZ = (
    "artifacts/comparisons/naturelm_by_source_100each_20260418T171459Z/"
    "nway_eat_all4/pooled_embeddings_all4.npz"
)

LAYER_NAMES = ["emb", "T0", "T1", "T2", "T3", "T4",
               "T5", "T6", "T7", "T8", "T9", "T10", "T11"]

# Friendly, recognizable labels for the six source datasets.
SOURCE_LABELS = {
    "Xeno-canto":           "Bird song (Xeno-canto)",
    "iNaturalist":          "Wildlife (iNaturalist)",
    "Animal Sound Archive": "Animal calls (ASA)",
    "Watkins":              "Marine mammals (Watkins)",
    "WavCaps":              "General audio (WavCaps)",
    "NatureLM":             "Instruments/synth (NatureLM)",
}

# Friendly model labels, in a deliberate order (base -> fine-tuned).
MODEL_ORDER = ["eat_all", "eat_bio", "sl_eat_all_ssl_all", "sl_eat_bio_ssl_all"]
MODEL_LABELS = {
    "eat_all":            "EAT (all audio)",
    "eat_bio":            "EAT (bio only)",
    "sl_eat_all_ssl_all": "EAT-all + SSL",
    "sl_eat_bio_ssl_all": "EAT-bio + SSL",
}


def clean_name(raw: str) -> str:
    """URL-decode a manifest file_name and trim the extension for display."""
    name = urllib.parse.unquote(str(raw))
    for ext in (".flac", ".wav", ".mp3", ".ogg"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    return name[:60]


def project_model(E: np.ndarray, y: np.ndarray) -> tuple[list, list]:
    """LDA(2) per layer + Procrustes alignment across layers + silhouette."""
    proj_layers: list = []
    sep_layers: list = []
    prev: np.ndarray | None = None
    for li in range(E.shape[1]):
        X = E[:, li, :].astype(np.float64)
        Z = LDA(n_components=2).fit_transform(X, y)
        # Standardize: center, scale to unit RMS radius for a stable viewport.
        Z = Z - Z.mean(0)
        rms = np.sqrt((Z ** 2).sum(1).mean())
        Z = Z / (rms if rms > 0 else 1.0)
        # Align orientation to the previous layer so the slider morphs smoothly.
        if prev is not None:
            R, _ = orthogonal_procrustes(Z, prev)
            Z = Z @ R
        prev = Z
        sep_layers.append(round(float(silhouette_score(Z, y)), 3))
        proj_layers.append([[round(float(x), 2), round(float(v), 2)] for x, v in Z])
    return proj_layers, sep_layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path,
                    help="Destination JSON path (e.g. site public/explorer/...).")
    ap.add_argument("--npz", default=NPZ, type=Path)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    src = d["source_dataset"].astype(str)
    files = d["file_name"].astype(str)

    present = [s for s in SOURCE_LABELS if s in set(src)]
    source_names = [SOURCE_LABELS[s] for s in present]
    code = {s: i for i, s in enumerate(present)}
    y = np.array([code[s] for s in src])

    points = [{"s": int(code[s]), "f": clean_name(f)} for s, f in zip(src, files)]

    models = [m for m in MODEL_ORDER if f"embeddings_{m}" in d.files]
    proj: dict = {}
    sep: dict = {}
    for m in models:
        E = d[f"embeddings_{m}"]
        proj[m], sep[m] = project_model(E, y)
        print(f"  {MODEL_LABELS[m]:<22} silhouette by layer: "
              + " ".join(f"{v:+.2f}" for v in sep[m]))

    out = {
        "sources": source_names,
        "modelKeys": models,
        "modelLabels": [MODEL_LABELS[m] for m in models],
        "layers": LAYER_NAMES,
        "points": points,
        "proj": proj,
        "sep": sep,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    kb = args.out.stat().st_size / 1024
    print(f"\nWrote {args.out}  ({kb:.0f} KB, {len(points)} points, "
          f"{len(models)} models x {len(LAYER_NAMES)} layers)")


if __name__ == "__main__":
    main()
