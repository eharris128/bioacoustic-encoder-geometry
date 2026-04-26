"""Collect ESP-AVES2 EAT residual-stream activations for a fixed NatureLM manifest.

This is the pilot extractor for roadmap part 1:
1. Freeze a stratified sample manifest with `sample_naturelm_by_source.py`.
2. Run this script over that manifest for one or more ESP-AVES2 EAT checkpoints.
3. Save activation shards plus sample metadata under `artifacts/roadmap_part1/`.

The extractor captures:
- embedding state after `model.pos_drop`
- post-layer outputs after each transformer block `model.blocks.{0..11}`

Each sample therefore yields a tensor of shape `(13, 513, 768)` for the official
10-second EAT setup.

Usage:
    python collect_esp_aves2_activations.py \
        --manifest artifacts/manifests/naturelm_by_source_100each_*.jsonl \
        --models sl_eat_all_ssl_all,sl_eat_bio_ssl_all
"""

from __future__ import annotations

import argparse
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pyarrow.dataset as ds
import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel


DATASET_ID = "EarthSpeciesProject/NatureLM-audio-training"
DATASET_CONFIG = "NatureLM-audio-training"
DATASET_SPLIT = "train"

BASE_EAT_MODEL_ID = "worstchan/EAT-base_epoch30_pretrain"
TARGET_SAMPLE_RATE = 16_000
TARGET_FBANK_FRAMES = 1024
NUM_MELS = 128
PATCH_SIZE = 16
WINDOW_DURATION_S = 10.0
TOKENS_PER_SAMPLE = 513
EMBED_DIM = 768
NUM_BLOCKS = 12
DEFAULT_LAYER_NAMES = ["model.pos_drop"] + [f"model.blocks.{i}" for i in range(NUM_BLOCKS)]


@dataclass(frozen=True)
class ModelSpec:
    hf_repo: str
    checkpoint_filename: str
    norm_mean: float
    norm_std: float


MODEL_SPECS: Dict[str, ModelSpec] = {
    "eat_all": ModelSpec(
        hf_repo="EarthSpeciesProject/esp-aves2-eat-all",
        checkpoint_filename="esp-aves2-eat-all.safetensors",
        norm_mean=-5.553,
        norm_std=4.606,
    ),
    "eat_bio": ModelSpec(
        hf_repo="EarthSpeciesProject/esp-aves2-eat-bio",
        checkpoint_filename="esp-aves2-eat-bio.safetensors",
        norm_mean=-5.553,
        norm_std=4.606,
    ),
    "sl_eat_all_ssl_all": ModelSpec(
        hf_repo="EarthSpeciesProject/esp-aves2-sl-eat-all-ssl-all",
        checkpoint_filename="esp-aves2-sl-eat-all-ssl-all.safetensors",
        norm_mean=-5.553,
        norm_std=4.606,
    ),
    "sl_eat_bio_ssl_all": ModelSpec(
        hf_repo="EarthSpeciesProject/esp-aves2-sl-eat-bio-ssl-all",
        checkpoint_filename="esp-aves2-sl-eat-bio-ssl-all.safetensors",
        norm_mean=-5.553,
        norm_std=4.606,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--models",
        type=str,
        default="sl_eat_all_ssl_all,sl_eat_bio_ssl_all",
        help="Comma-separated model keys.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/roadmap_part1"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--shard_size", type=int, default=25)
    parser.add_argument("--window_selection", type=str, default="center", choices=["start", "center"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path, max_samples: int | None = None) -> List[dict]:
    records: List[dict] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    records.sort(key=lambda x: x["row_index"])
    return records


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim != 2:
        raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")
    if waveform.shape[0] == 1:
        return waveform.squeeze(0)
    return waveform.mean(dim=0)


def crop_window(audio: torch.Tensor, target_num_samples: int, selection: str) -> torch.Tensor:
    if audio.numel() <= target_num_samples:
        return audio
    if selection == "start":
        start = 0
    elif selection == "center":
        start = (audio.numel() - target_num_samples) // 2
    else:
        raise ValueError(f"Unsupported window selection: {selection}")
    return audio[start : start + target_num_samples]


def waveform_to_eat_input(
    audio: torch.Tensor,
    *,
    norm_mean: float,
    norm_std: float,
) -> tuple[torch.Tensor, int]:
    audio = audio - audio.mean()
    mel = torchaudio.compliance.kaldi.fbank(
        audio.unsqueeze(0),
        htk_compat=True,
        sample_frequency=TARGET_SAMPLE_RATE,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=NUM_MELS,
        dither=0.0,
        frame_shift=10,
    ).unsqueeze(0)

    original_frames = int(mel.shape[1])
    if original_frames < TARGET_FBANK_FRAMES:
        mel = F.pad(mel, (0, 0, 0, TARGET_FBANK_FRAMES - original_frames))
    else:
        mel = mel[:, :TARGET_FBANK_FRAMES, :]
        original_frames = TARGET_FBANK_FRAMES

    mel = (mel - norm_mean) / (norm_std * 2.0)
    mel = mel.unsqueeze(0)
    return mel, original_frames


def valid_token_count_from_frames(num_frames: int) -> int:
    valid_time_patches = math.ceil(min(num_frames, TARGET_FBANK_FRAMES) / PATCH_SIZE)
    valid_freq_patches = NUM_MELS // PATCH_SIZE
    return 1 + (valid_time_patches * valid_freq_patches)


class ManifestParquetAudioResolver:
    """Resolve audio bytes from the exact manifest parquet shard and row id."""

    def __init__(self, records: List[dict]) -> None:
        expected_ids_by_relpath: Dict[str, set[str]] = {}
        for record in records:
            relpath = record.get("parquet_relpath") or ""
            sample_id = record.get("id") or ""
            if not relpath or not sample_id:
                continue
            expected_ids_by_relpath.setdefault(relpath, set()).add(sample_id)

        self.expected_ids_by_relpath = expected_ids_by_relpath
        self.current_relpath: str | None = None
        self.current_local_path: Path | None = None
        self.current_rows: Dict[str, dict] = {}

    def _load_relpath(self, record: dict) -> None:
        relpath = record.get("parquet_relpath") or ""
        revision = record.get("parquet_revision") or ""
        sample_id = record.get("id") or ""
        if not relpath or not revision or not sample_id:
            raise RuntimeError(f"Manifest record is missing parquet metadata: {record}")

        expected_ids = sorted(self.expected_ids_by_relpath.get(relpath) or [])
        if not expected_ids:
            raise RuntimeError(f"No expected ids registered for parquet shard: {relpath}")

        local_path = Path(
            hf_hub_download(
                repo_id=record.get("dataset") or DATASET_ID,
                repo_type="dataset",
                filename=relpath,
                revision=revision,
            )
        )
        print(
            f"  loading manifest parquet {relpath} locally for {len(expected_ids)} expected samples",
            flush=True,
        )

        dataset = ds.dataset(str(local_path), format="parquet")
        table = dataset.to_table(
            columns=["id", "file_name", "source_dataset", "audio"],
            filter=ds.field("id").isin(expected_ids),
        )
        rows = {item["id"]: item for item in table.to_pylist()}
        missing_ids = [item_id for item_id in expected_ids if item_id not in rows]
        if missing_ids:
            raise RuntimeError(
                f"Parquet shard {relpath} is missing {len(missing_ids)} manifest ids; "
                f"first missing id={missing_ids[0]}"
            )

        self.current_relpath = relpath
        self.current_local_path = local_path
        self.current_rows = rows

    def fetch_audio(self, record: dict) -> tuple[torch.Tensor, int, str, str]:
        relpath = record.get("parquet_relpath") or ""
        sample_id = record.get("id") or ""
        if not relpath or not sample_id:
            raise RuntimeError(f"Manifest record is missing parquet lookup fields: {record}")

        if relpath != self.current_relpath:
            self.current_relpath = None
            self.current_local_path = None
            self.current_rows = {}
            self._load_relpath(record)

        row = self.current_rows.get(sample_id)
        if row is None:
            raise RuntimeError(f"Could not find sample id={sample_id} in cached parquet rows")
        if record.get("file_name") and row.get("file_name") != record["file_name"]:
            raise RuntimeError(
                f"Parquet id mismatch for {sample_id}: expected file_name={record['file_name']} "
                f"got {row.get('file_name')}"
            )
        if record.get("source_dataset") and row.get("source_dataset") != record["source_dataset"]:
            raise RuntimeError(
                f"Parquet source mismatch for {sample_id}: expected source_dataset="
                f"{record['source_dataset']} got {row.get('source_dataset')}"
            )

        audio = row.get("audio") or {}
        audio_bytes = audio.get("bytes")
        if isinstance(audio_bytes, memoryview):
            audio_bytes = audio_bytes.tobytes()
        elif isinstance(audio_bytes, bytearray):
            audio_bytes = bytes(audio_bytes)
        if not audio_bytes:
            raise RuntimeError(f"Parquet row for id={sample_id} does not contain audio bytes")

        audio_path = audio.get("path") or ""
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        local_path = str(self.current_local_path or "")
        return ensure_mono(waveform).to(torch.float32), int(sample_rate), local_path, audio_path


def _remap_checkpoint_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # ESP-AVES2 ships the EAT backbone in two checkpoint families that need
    # different remapping to the AutoModel-flat namespace:
    #   sl-eat-*-ssl-all: keys live under `backbone.model.*` plus a
    #     `classifier.*` head from the downstream fine-tune.
    #   eat-* (data2vec/EAT pretraining): bare `blocks.*` keys for the shared
    #     transformer body, plus a fairseq-style `modality_encoders.IMAGE.*`
    #     sub-tree that holds the audio (mel-as-2D-patches) input pathway.
    #     The `decoder.*` branch is reconstruction-only and unused for
    #     activation extraction.
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("classifier."):
            continue
        cleaned = key.removeprefix("backbone.")
        if cleaned.startswith("modality_encoders.IMAGE."):
            sub = cleaned[len("modality_encoders.IMAGE.") :]
            if sub.startswith("decoder."):
                continue
            if sub.startswith("context_encoder.norm."):
                cleaned = "pre_norm." + sub[len("context_encoder.norm.") :]
            else:
                cleaned = sub
        if not cleaned.startswith("model."):
            cleaned = "model." + cleaned
        out[cleaned] = value
    return out


def load_eat_model(model_key: str, device: str) -> torch.nn.Module:
    spec = MODEL_SPECS[model_key]
    model = AutoModel.from_pretrained(BASE_EAT_MODEL_ID, trust_remote_code=True).to(device)
    checkpoint_path = Path(hf_hub_download(spec.hf_repo, spec.checkpoint_filename))

    if checkpoint_path.stat().st_size < 1024:
        raise RuntimeError(
            f"{model_key} checkpoint is a placeholder file ({checkpoint_path.stat().st_size} bytes). "
            f"The upstream Hugging Face repo does not currently expose usable weights."
        )

    state = load_file(str(checkpoint_path), device="cpu")
    if not state:
        raise RuntimeError(
            f"{model_key} checkpoint contains zero tensors. The upstream Hugging Face repo "
            "appears to have an empty safetensors export."
        )

    cleaned_state = _remap_checkpoint_keys(state)

    incompatible = model.load_state_dict(cleaned_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys for {model_key}: {incompatible.unexpected_keys[:10]}"
        )
    if incompatible.missing_keys:
        raise RuntimeError(
            f"Missing checkpoint keys for {model_key}: {incompatible.missing_keys[:10]}"
        )

    model.eval()
    return model


def register_hooks(model: torch.nn.Module, layer_names: Iterable[str]) -> tuple[dict, dict]:
    outputs: Dict[str, torch.Tensor] = {}
    hooks = {}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                outputs[name] = output[0]
            elif isinstance(output, dict):
                outputs[name] = output["x"]
            else:
                outputs[name] = output

        return hook

    for name in layer_names:
        hooks[name] = model.get_submodule(name).register_forward_hook(make_hook(name))
    return hooks, outputs


def flush_shard(
    shard_dir: Path,
    shard_index: int,
    activation_batch: List[torch.Tensor],
    sample_batch: List[dict],
    model_key: str,
    dtype: torch.dtype,
) -> None:
    shard_path = shard_dir / f"shard_{shard_index:05d}.pt"
    activations = torch.stack(activation_batch).to(dtype=dtype, device="cpu")
    payload = {
        "model_key": model_key,
        "activation_shape": list(activations.shape),
        "activations": activations,
        "samples": sample_batch,
    }
    torch.save(payload, shard_path)
    print(f"  wrote {shard_path} | samples={len(sample_batch)} | shape={tuple(activations.shape)}", flush=True)


def scan_existing_shards(shard_dir: Path) -> tuple[set[int], Dict[str, int], int, int]:
    processed_row_indices: set[int] = set()
    counts_by_source: Dict[str, int] = {}
    next_shard_index = 0
    total_samples = 0

    if not shard_dir.exists():
        return processed_row_indices, counts_by_source, next_shard_index, total_samples

    shard_paths = sorted(shard_dir.glob("shard_*.pt"))
    for shard_path in shard_paths:
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        samples = payload.get("samples") or []
        for sample in samples:
            row_index = int(sample["row_index"])
            processed_row_indices.add(row_index)
            source_dataset = sample.get("source_dataset") or ""
            counts_by_source[source_dataset] = counts_by_source.get(source_dataset, 0) + 1
            total_samples += 1

        try:
            shard_num = int(shard_path.stem.split("_")[-1])
            next_shard_index = max(next_shard_index, shard_num + 1)
        except ValueError:
            continue

    return processed_row_indices, counts_by_source, next_shard_index, total_samples


def write_summary(
    *,
    summary_path: Path,
    args: argparse.Namespace,
    manifest_id: str,
    spec: ModelSpec,
    model_key: str,
    total_samples: int,
    counts_by_source: Dict[str, int],
) -> None:
    summary = {
        "manifest_path": str(args.manifest),
        "manifest_id": manifest_id,
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "data_access": "manifest parquet shards via hf_hub_download + pyarrow dataset filter",
        "model_key": model_key,
        "hf_repo": spec.hf_repo,
        "checkpoint_filename": spec.checkpoint_filename,
        "total_samples": total_samples,
        "counts_by_source": counts_by_source,
        "layer_names": DEFAULT_LAYER_NAMES,
        "activation_shape_per_sample": [13, TOKENS_PER_SAMPLE, EMBED_DIM],
        "dtype": args.dtype,
        "window_selection": args.window_selection,
        "window_duration_s": WINDOW_DURATION_S,
        "shard_size": args.shard_size,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}", flush=True)


def collect_for_model(
    model_key: str,
    records: List[dict],
    args: argparse.Namespace,
    manifest_id: str,
) -> None:
    spec = MODEL_SPECS[model_key]
    model_dir = args.output_dir / manifest_id / model_key
    shard_dir = model_dir / "shards"
    summary_path = model_dir / "summary.json"

    if args.skip_existing and summary_path.exists():
        print(f"Skipping {model_key}: {summary_path} already exists.", flush=True)
        return

    shard_dir.mkdir(parents=True, exist_ok=True)
    target_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    target_num_samples = int(TARGET_SAMPLE_RATE * WINDOW_DURATION_S)

    processed_row_indices, counts_by_source, shard_index, total_samples = scan_existing_shards(shard_dir)
    if processed_row_indices:
        print(
            (
                f"Resuming {model_key}: found {total_samples} saved samples across "
                f"{shard_index} shard(s)."
            ),
            flush=True,
        )
    records_to_process = [record for record in records if int(record["row_index"]) not in processed_row_indices]
    if not records_to_process:
        print(f"No remaining records for {model_key}; all manifest rows already have saved shards.", flush=True)
        write_summary(
            summary_path=summary_path,
            args=args,
            manifest_id=manifest_id,
            spec=spec,
            model_key=model_key,
            total_samples=total_samples,
            counts_by_source=counts_by_source,
        )
        return

    audio_resolver = ManifestParquetAudioResolver(records_to_process)
    model = load_eat_model(model_key, args.device)
    hooks, hook_outputs = register_hooks(model, DEFAULT_LAYER_NAMES)

    activation_batch: List[torch.Tensor] = []
    sample_batch: List[dict] = []

    try:
        for manifest_record in records_to_process:
            waveform, sample_rate, parquet_local_path, audio_asset_path = audio_resolver.fetch_audio(
                manifest_record
            )
            if sample_rate != TARGET_SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
            waveform = crop_window(
                waveform,
                target_num_samples=target_num_samples,
                selection=args.window_selection,
            )

            eat_input, original_frames = waveform_to_eat_input(
                waveform,
                norm_mean=spec.norm_mean,
                norm_std=spec.norm_std,
            )
            eat_input = eat_input.to(args.device)

            hook_outputs.clear()
            with torch.no_grad():
                _ = model.extract_features(eat_input)

            activations = []
            for layer_name in DEFAULT_LAYER_NAMES:
                tensor = hook_outputs[layer_name]
                tensor = tensor.detach().to("cpu")
                if tensor.shape != (1, TOKENS_PER_SAMPLE, EMBED_DIM):
                    raise RuntimeError(
                        f"{model_key} {layer_name} produced {tuple(tensor.shape)}, "
                        f"expected (1, {TOKENS_PER_SAMPLE}, {EMBED_DIM})"
                    )
                activations.append(tensor.squeeze(0))

            stacked = torch.stack(activations)
            if stacked.shape != (13, TOKENS_PER_SAMPLE, EMBED_DIM):
                raise RuntimeError(f"Unexpected stacked shape: {tuple(stacked.shape)}")

            sample_record = dict(manifest_record)
            sample_record.update(
                {
                    "audio_data_access": "manifest_parquet",
                    "audio_asset_path": audio_asset_path,
                    "audio_parquet_local_path": parquet_local_path,
                    "window_selection": args.window_selection,
                    "window_duration_s": WINDOW_DURATION_S,
                    "effective_sample_rate": TARGET_SAMPLE_RATE,
                    "fbank_frames_before_pad": original_frames,
                    "valid_token_count": valid_token_count_from_frames(original_frames),
                    "activation_shape": [13, TOKENS_PER_SAMPLE, EMBED_DIM],
                }
            )

            activation_batch.append(stacked)
            sample_batch.append(sample_record)
            total_samples += 1
            source_dataset = sample_record.get("source_dataset") or ""
            counts_by_source[source_dataset] = counts_by_source.get(source_dataset, 0) + 1

            print(
                f"  [{total_samples:4d}/{len(records)}] {model_key:>20s} | "
                f"{source_dataset:>20s} | row={sample_record['row_index']:>8d} | "
                f"valid_tokens={sample_record['valid_token_count']:3d} | "
                f"{sample_record['file_name']}",
                flush=True,
            )

            if len(activation_batch) >= args.shard_size:
                flush_shard(
                    shard_dir=shard_dir,
                    shard_index=shard_index,
                    activation_batch=activation_batch,
                    sample_batch=sample_batch,
                    model_key=model_key,
                    dtype=target_dtype,
                )
                shard_index += 1
                activation_batch = []
                sample_batch = []

    except Exception:
        if activation_batch:
            flush_shard(
                shard_dir=shard_dir,
                shard_index=shard_index,
                activation_batch=activation_batch,
                sample_batch=sample_batch,
                model_key=model_key,
                dtype=target_dtype,
            )
            print(f"Saved partial shard before aborting {model_key}.", flush=True)
        raise
    else:
        if activation_batch:
            flush_shard(
                shard_dir=shard_dir,
                shard_index=shard_index,
                activation_batch=activation_batch,
                sample_batch=sample_batch,
                model_key=model_key,
                dtype=target_dtype,
            )

        write_summary(
            summary_path=summary_path,
            args=args,
            manifest_id=manifest_id,
            spec=spec,
            model_key=model_key,
            total_samples=total_samples,
            counts_by_source=counts_by_source,
        )
    finally:
        for hook in hooks.values():
            hook.remove()


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest, max_samples=args.max_samples)
    if not records:
        raise ValueError(f"No records found in manifest: {args.manifest}")

    model_keys = [key.strip() for key in args.models.split(",") if key.strip()]
    invalid = [key for key in model_keys if key not in MODEL_SPECS]
    if invalid:
        raise ValueError(f"Unknown model keys: {invalid}. Valid keys: {sorted(MODEL_SPECS)}")

    manifest_id = args.manifest.stem
    print(f"Loaded {len(records)} manifest records from {args.manifest}", flush=True)
    print(f"Models: {', '.join(model_keys)}", flush=True)
    print(f"Output dir: {args.output_dir / manifest_id}", flush=True)

    for model_key in model_keys:
        print(f"\n=== Collecting {model_key} ===", flush=True)
        collect_for_model(model_key=model_key, records=records, args=args, manifest_id=manifest_id)


if __name__ == "__main__":
    main()
