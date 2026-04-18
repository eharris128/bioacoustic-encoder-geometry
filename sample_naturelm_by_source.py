"""Build a stratified NatureLM-audio-training sample manifest by source_dataset.

This manifest builder avoids streaming the full dataset through `datasets`. Instead
it scans the published Parquet shards with DuckDB, records exact global row indices,
and writes a frozen JSONL manifest for later activation extraction.

Usage:
    python sample_naturelm_by_source.py --samples_per_source 100
    python sample_naturelm_by_source.py --samples_per_source 100 \
        --source_datasets Xeno-canto,WavCaps,NatureLM,Watkins
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from huggingface_hub import HfApi, list_repo_files


DATASET_ID = "EarthSpeciesProject/NatureLM-audio-training"
DATASET_CONFIG = "NatureLM-audio-training"
DATASET_SPLIT = "train"

DEFAULT_SOURCE_DATASETS = [
    "Animal Sound Archive",
    "NatureLM",
    "WavCaps",
    "Watkins",
    "Xeno-canto",
    "iNaturalist",
]

FLAT_METADATA_KEYS = [
    "source",
    "url",
    "recordist",
    "duration",
    "sample_rate",
    "data_category",
    "common_name",
    "scientific_name",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "subspecies",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_per_source", type=int, default=100)
    parser.add_argument(
        "--source_datasets",
        type=str,
        default=",".join(DEFAULT_SOURCE_DATASETS),
        help="Comma-separated source_dataset values to target.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts/manifests"),
        help="Directory for the JSONL manifest and summary JSON.",
    )
    parser.add_argument(
        "--max_rows_to_scan",
        type=int,
        default=250_000,
        help="Stop after scanning this many dataset rows, even if some buckets are unfilled.",
    )
    parser.add_argument(
        "--max_parquet_files",
        type=int,
        default=None,
        help="Optional cap on the number of remote parquet files to scan.",
    )
    return parser.parse_args()


def parse_metadata(raw: object) -> dict[str, object]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def flatten_metadata(metadata: dict[str, object]) -> dict[str, object]:
    flat: dict[str, object] = {}
    for key in FLAT_METADATA_KEYS:
        value = metadata.get(key)
        if value is None:
            value = ""
        flat[key] = value
    return flat


def build_record(
    *,
    row_index: int,
    parquet_relpath: str,
    parquet_url: str,
    parquet_revision: str,
    item: dict[str, object],
) -> dict[str, object]:
    metadata_raw = item.get("metadata") or ""
    metadata = parse_metadata(metadata_raw)
    duration_s = metadata.get("duration")
    if duration_s in (None, ""):
        duration_s = ""

    record = {
        "row_index": row_index,
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "parquet_revision": parquet_revision,
        "parquet_relpath": parquet_relpath,
        "parquet_url": parquet_url,
        "id": item.get("id") or "",
        "file_name": item.get("file_name") or "",
        "source_dataset": item.get("source_dataset") or "",
        "task": item.get("task") or "",
        "output": item.get("output") or "",
        "license": item.get("license") or "",
        "instruction_text": item.get("instruction_text") or "",
        "duration_s": duration_s,
        "sampling_rate": metadata.get("sample_rate") or "",
        "metadata_raw": metadata_raw,
    }
    record.update(flatten_metadata(metadata))
    return record


def fetch_dataset_revision() -> str:
    info = HfApi().dataset_info(DATASET_ID)
    if not info.sha:
        raise RuntimeError(f"Could not resolve a dataset revision for {DATASET_ID}")
    return info.sha


def list_parquet_relpaths() -> list[str]:
    files = list_repo_files(DATASET_ID, repo_type="dataset")
    parquet_paths = [path for path in files if path.endswith(".parquet")]
    parquet_paths.sort()
    return parquet_paths


def build_parquet_url(*, relpath: str, revision: str) -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{revision}/{relpath}?download=1"
    )


def read_parquet_num_rows(con: duckdb.DuckDBPyConnection, parquet_url: str) -> int:
    result = con.execute("SELECT num_rows FROM parquet_file_metadata(?)", [parquet_url]).fetchone()
    if result is None:
        raise RuntimeError(f"Could not read Parquet metadata for {parquet_url}")
    return int(result[0])


def read_candidate_rows(
    con: duckdb.DuckDBPyConnection,
    *,
    parquet_url: str,
    target_sources: list[str],
) -> list[dict[str, object]]:
    placeholders = ",".join(["?"] * len(target_sources))
    sql = f"""
        SELECT
            file_name,
            metadata,
            source_dataset,
            id,
            license,
            instruction_text,
            output,
            task,
            row_number() OVER () - 1 AS local_row_index
        FROM read_parquet(?)
        WHERE source_dataset IN ({placeholders})
        ORDER BY local_row_index
    """
    rows = con.execute(sql, [parquet_url, *target_sources]).fetchall()
    return [
        {
            "file_name": row[0],
            "metadata": row[1],
            "source_dataset": row[2],
            "id": row[3],
            "license": row[4],
            "instruction_text": row[5],
            "output": row[6],
            "task": row[7],
            "local_row_index": int(row[8]),
        }
        for row in rows
    ]


def main() -> None:
    args = parse_args()
    target_sources = [s.strip() for s in args.source_datasets.split(",") if s.strip()]
    target_set = set(target_sources)
    if not target_sources:
        raise ValueError("No source_dataset values were provided.")

    parquet_revision = fetch_dataset_revision()
    parquet_relpaths = list_parquet_relpaths()
    if args.max_parquet_files is not None:
        parquet_relpaths = parquet_relpaths[: args.max_parquet_files]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"naturelm_by_source_{args.samples_per_source}each_{timestamp}"
    manifest_path = args.output_dir / f"{prefix}.jsonl"
    summary_path = args.output_dir / f"{prefix}_summary.json"

    print("Building stratified NatureLM sample manifest...")
    print(f"  dataset: {DATASET_ID}", flush=True)
    print(f"  parquet_revision: {parquet_revision}", flush=True)
    print(f"  samples_per_source: {args.samples_per_source}", flush=True)
    print(f"  target_sources: {', '.join(target_sources)}", flush=True)
    print(f"  max_rows_to_scan: {args.max_rows_to_scan}", flush=True)
    print(f"  manifest: {manifest_path}", flush=True)

    con = duckdb.connect()
    con.execute("LOAD httpfs")

    counts: Counter[str] = Counter()
    records: list[dict[str, object]] = []
    rows_scanned = 0
    parquet_files_scanned = 0

    for parquet_relpath in parquet_relpaths:
        if rows_scanned >= args.max_rows_to_scan:
            break
        if all(counts[source_name] >= args.samples_per_source for source_name in target_sources):
            break

        parquet_url = build_parquet_url(relpath=parquet_relpath, revision=parquet_revision)
        num_rows = read_parquet_num_rows(con, parquet_url)
        file_start_row_index = rows_scanned
        parquet_files_scanned += 1

        candidate_rows = read_candidate_rows(
            con,
            parquet_url=parquet_url,
            target_sources=target_sources,
        )

        print(
            (
                f"  scan {parquet_files_scanned:4d} | rows "
                f"{file_start_row_index:>8d}-{file_start_row_index + num_rows - 1:<8d} | "
                f"{parquet_relpath}"
            ),
            flush=True,
        )

        for item in candidate_rows:
            source = item["source_dataset"] or ""
            if source not in target_set:
                continue
            if counts[source] >= args.samples_per_source:
                continue

            global_row_index = file_start_row_index + int(item["local_row_index"])
            record = build_record(
                row_index=global_row_index,
                parquet_relpath=parquet_relpath,
                parquet_url=parquet_url,
                parquet_revision=parquet_revision,
                item=item,
            )
            records.append(record)
            counts[source] += 1

            print(
                (
                    f"    [{len(records):4d}] {source:>20s} "
                    f"{counts[source]:3d}/{args.samples_per_source} | "
                    f"row={global_row_index:>8d} | {record['file_name']}"
                ),
                flush=True,
            )

            if all(counts[source_name] >= args.samples_per_source for source_name in target_sources):
                break

        rows_scanned += num_rows

    missing = {
        source: args.samples_per_source - counts[source]
        for source in target_sources
        if counts[source] < args.samples_per_source
    }

    with manifest_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    summary = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "parquet_revision": parquet_revision,
        "samples_per_source": args.samples_per_source,
        "max_rows_to_scan": args.max_rows_to_scan,
        "max_parquet_files": args.max_parquet_files,
        "target_sources": target_sources,
        "counts_by_source": {source: counts[source] for source in target_sources},
        "total_records": len(records),
        "rows_scanned": rows_scanned,
        "parquet_files_scanned": parquet_files_scanned,
        "missing_by_source": missing,
        "manifest_path": str(manifest_path),
        "created_at_utc": timestamp,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nFinished.")
    print(f"  total_records: {len(records)}")
    print(f"  rows_scanned: {rows_scanned}")
    print(f"  parquet_files_scanned: {parquet_files_scanned}")
    print(f"  summary: {summary_path}")
    if missing:
        print("  missing_by_source:")
        for source, shortfall in missing.items():
            print(f"    {source}: {shortfall}")
    else:
        print("  All requested source buckets filled.")


if __name__ == "__main__":
    main()
