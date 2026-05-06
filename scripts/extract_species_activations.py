"""
scripts/extract_species_activations.py — Pre-extract AVES activations per species from NatureLM.

Streams EarthSpeciesProject/NatureLM-audio-training shard-by-shard, filters by species name
and sample_rate=16000, runs inference through all 13 EAT layers, and saves one .npy + .json
sidecar per recording. This is the offline extraction step that makes large-scale species
probing feasible without re-running inference every experiment.

Activation shape per recording: (13, n_frames, 768) float32
  - Layer 0: CNN local_encoder output (embedding)
  - Layers 1–12: transformer blocks 0–11 (CLS token stripped)

Output structure:
  <out-dir>/
    rec_00000.npy       # shape (13, n_frames, 768) float32
    rec_00000.json      # metadata sidecar
    rec_00001.npy
    rec_00001.json
    ...
    manifest.json       # run summary
    errors.jsonl        # skipped recordings

Example usage:
  python scripts/extract_species_activations.py \\
    --species "Parus major" \\
    --rows 512 \\
    --out-dir ./activations/species-parus-major-eat-all

  python scripts/extract_species_activations.py \\
    --species "Parus caeruleus" \\
    --rows 512 \\
    --out-dir ./activations/species-parus-caeruleus-eat-all

  # With a different model:
  python scripts/extract_species_activations.py \\
    --species "Turdus merula" \\
    --rows 256 \\
    --model esp_aves2_eat_bio \\
    --out-dir ./activations/species-turdus-merula-eat-bio
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

# Allow imports from project root regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import (
    _audio_from_hf_item,
    extract_all_layers,
    parse_metadata,
    load_model,
    DATASET_REPO,
    NUM_LAYERS_TOTAL,
    DEFAULT_MAX_FRAMES,
    EAT_MODELS,
)

# HuggingFace repo ID → avex model name (for --model convenience)
_HF_TO_AVEX: dict[str, str] = {
    "EarthSpeciesProject/esp-aves2-eat-all":            "esp_aves2_eat_all",
    "EarthSpeciesProject/esp-aves2-eat-bio":            "esp_aves2_eat_bio",
    "EarthSpeciesProject/esp-aves2-sl-eat-all-ssl-all": "esp_aves2_sl_eat_all_ssl_all",
    "EarthSpeciesProject/esp-aves2-sl-eat-bio-ssl-all": "esp_aves2_sl_eat_bio_ssl_all",
}
DEFAULT_MODEL = "EarthSpeciesProject/esp-aves2-eat-all"


def _resolve_model_name(model_arg: str) -> str:
    """Accept either an HF repo ID or an avex model name."""
    if model_arg in _HF_TO_AVEX:
        return _HF_TO_AVEX[model_arg]
    if model_arg in EAT_MODELS:
        return model_arg
    raise ValueError(
        f"Unknown model '{model_arg}'. "
        f"Use an avex name ({sorted(EAT_MODELS)}) "
        f"or an HF repo ID ({sorted(_HF_TO_AVEX)})."
    )


def _log_error(errors_path: Path, record: dict) -> None:
    with open(errors_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def extract_species(
    species: str,
    rows: int,
    out_dir: Path,
    model_name: str = "esp_aves2_eat_all",
    device: str = "cpu",
    max_frames: int = DEFAULT_MAX_FRAMES,
    seed: int = 42,
) -> None:
    """
    Stream NatureLM, filter to `species`, extract activations, save to `out_dir`.

    Scans shards in shuffled order until `rows` valid recordings are saved.
    Skipped recordings (decode errors, wrong shape, wrong sample rate) are logged
    to errors.jsonl — they do not count toward the `rows` target.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_path = out_dir / "errors.jsonl"
    manifest_path = out_dir / "manifest.json"

    print(f"Loading model '{model_name}' on {device}...", flush=True)
    model = load_model(model_name, device=device)

    from huggingface_hub import hf_hub_download, list_repo_tree
    import pandas as pd

    naturelm_cache = ".naturelm_cache"
    os.makedirs(naturelm_cache, exist_ok=True)

    # Shard list is cached after first fetch (~10,577 parquet files)
    shard_list_path = os.path.join(naturelm_cache, "shard_list.txt")
    if os.path.exists(shard_list_path):
        with open(shard_list_path) as f:
            all_shards = [line.strip() for line in f if line.strip()]
    else:
        print("Fetching shard list from HuggingFace (one-time)...", flush=True)
        items = list(list_repo_tree(DATASET_REPO, repo_type="dataset", recursive=True))
        all_shards = sorted(
            getattr(it, "path", "") for it in items
            if getattr(it, "path", "").endswith(".parquet")
        )
        with open(shard_list_path, "w") as f:
            f.write("\n".join(all_shards))
        print(f"  Found {len(all_shards)} shards.", flush=True)

    # Shuffle for diverse species/source coverage
    random.seed(seed)
    random.shuffle(all_shards)

    rng = np.random.default_rng(seed)
    species_lower = species.strip().lower()

    saved = 0          # recordings written to disk
    rows_scanned = 0   # total rows examined across all shards

    for shard_path in all_shards:
        if saved >= rows:
            break

        try:
            local_path = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=shard_path,
                repo_type="dataset",
                local_dir=naturelm_cache,
            )
            df = pd.read_parquet(local_path)
        except Exception as e:
            _log_error(errors_path, {"shard": shard_path, "error": str(e)})
            continue

        for _, row in df.iterrows():
            if saved >= rows:
                break

            rows_scanned += 1
            if rows_scanned % 50 == 0:
                print(
                    f"  Scanned {rows_scanned} rows, saved {saved}/{rows} ...",
                    flush=True,
                )

            item = row.to_dict()
            meta = parse_metadata(item.get("metadata"))

            # Species filter (case-insensitive exact match)
            if meta.get("species", "").strip().lower() != species_lower:
                continue

            # Sample rate filter
            sr = meta.get("sample_rate")
            if sr is not None and int(sr) != 16000:
                continue

            # Decode audio and extract activations
            try:
                audio = _audio_from_hf_item(item)
                acts = extract_all_layers(
                    model, audio,
                    max_frames=max_frames,
                    rng=rng,
                    mode="raw",
                )  # (13, n_frames, 768)
            except Exception as e:
                _log_error(errors_path, {
                    "row_index": rows_scanned,
                    "file_name": item.get("file_name", ""),
                    "error": str(e),
                })
                continue

            # Shape guard: must be (13, n_frames, 768)
            if acts.ndim != 3 or acts.shape[0] != NUM_LAYERS_TOTAL or acts.shape[2] != 768:
                _log_error(errors_path, {
                    "row_index": rows_scanned,
                    "file_name": item.get("file_name", ""),
                    "error": f"unexpected activation shape {acts.shape}",
                })
                continue

            # Save .npy (float32, shape (13, n_frames, 768))
            rec_stem = f"rec_{saved:05d}"
            np.save(out_dir / f"{rec_stem}.npy", acts.astype(np.float32))

            # Save .json sidecar
            sidecar = {
                "species":           meta.get("species", species),
                "source_dataset":    item.get("source_dataset", ""),
                "file_name":         item.get("file_name", ""),
                "id":                item.get("id", ""),
                "sample_rate":       meta.get("sample_rate"),
                "row_index":         rows_scanned,
                "activation_shape":  list(acts.shape),
            }
            with open(out_dir / f"{rec_stem}.json", "w") as f:
                json.dump(sidecar, f, indent=2)

            saved += 1

    # Write manifest
    manifest = {
        "model":             model_name,
        "species":           species,
        "rows_requested":    rows,
        "rows_saved":        saved,
        "rows_scanned":      rows_scanned,
        "activation_shape":  "(13, n_frames, 768)",
        "max_frames":        max_frames,
        "seed":              seed,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Saved {saved}/{rows} recordings to {out_dir}/", flush=True)
    print(f"  Rows scanned: {rows_scanned}", flush=True)
    if saved < rows:
        print(
            f"  Warning: only found {saved} recordings matching '{species}' "
            f"(requested {rows}). NatureLM may have limited coverage for this species.",
            flush=True,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-extract AVES activations per species from NatureLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--species", required=True,
        help='Scientific species name, e.g. "Parus major"',
    )
    p.add_argument(
        "--rows", type=int, default=512,
        help="Number of recordings to save (stop early if NatureLM is exhausted).",
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=(
            "avex model name (e.g. esp_aves2_eat_all) "
            "or HF repo ID (e.g. EarthSpeciesProject/esp-aves2-eat-all)."
        ),
    )
    p.add_argument(
        "--out-dir", required=True,
        help="Output directory for .npy files, .json sidecars, and manifest.json.",
    )
    p.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Torch device for inference.",
    )
    p.add_argument(
        "--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
        help="Max patch tokens per recording (subsampled if audio is longer).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shard shuffling and frame subsampling.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model_name = _resolve_model_name(args.model)
    extract_species(
        species=args.species,
        rows=args.rows,
        out_dir=Path(args.out_dir),
        model_name=model_name,
        device=args.device,
        max_frames=args.max_frames,
        seed=args.seed,
    )
