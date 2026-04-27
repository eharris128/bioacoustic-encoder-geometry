"""Build a taxonomy-aware NatureLM-audio-training manifest stratified by
bird Order, with anchors for Class (Mammalia) and §4 control (non-bio).

This is the per-Order companion to `sample_naturelm_by_source.py`. It
walks the published Parquet shards with DuckDB, filters by metadata
fields (class / order), and emits a frozen JSONL manifest compatible
with `collect_esp_aves2_activations.py` and the downstream taxonomic
analysis scripts.

Default targets (matches the teammate's bird-Order probe panel):
  Aves / Passeriformes      n=100
  Aves / Charadriiformes    n=100
  Aves / Piciformes         n=100
  Aves / Strigiformes       n=100
  Mammalia                  n=200  (mixed Orders, Class-level anchor)
  non-bio                   n=200  (WavCaps + NatureLM, §4 control)
                            ====
                            n=800

Usage:
    python sample_naturelm_by_order.py
    python sample_naturelm_by_order.py --samples_per_order 200 --max_rows_to_scan 1500000
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
from huggingface_hub import HfApi


DATASET_ID = "EarthSpeciesProject/NatureLM-audio-training"
DATASET_CONFIG = "NatureLM-audio-training"
DATASET_SPLIT = "train"

DEFAULT_TARGET_ORDERS = [
    "Passeriformes",
    "Charadriiformes",
    "Piciformes",
    "Strigiformes",
]
DEFAULT_NONBIO_SOURCES = ["WavCaps", "NatureLM"]

FLAT_METADATA_KEYS = [
    "source", "url", "recordist", "duration", "sample_rate",
    "data_category", "common_name", "scientific_name",
    "class", "order", "family", "genus", "species", "subspecies",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--samples_per_order", type=int, default=100,
                   help="Samples per bird Order in --target_orders.")
    p.add_argument("--target_orders", type=str,
                   default=",".join(DEFAULT_TARGET_ORDERS),
                   help="Comma-separated Aves Orders to stratify on.")
    p.add_argument("--samples_mammalia", type=int, default=200,
                   help="Mammalia samples (pooled across Orders).")
    p.add_argument("--samples_nonbio", type=int, default=200,
                   help="Non-bio samples (pooled across non-bio sources).")
    p.add_argument("--nonbio_sources", type=str,
                   default=",".join(DEFAULT_NONBIO_SOURCES))
    p.add_argument("--output_dir", type=Path,
                   default=Path("artifacts/manifests"))
    p.add_argument("--max_rows_to_scan", type=int, default=1_000_000,
                   help="Stop after scanning this many rows even if buckets unfilled.")
    p.add_argument("--max_parquet_files", type=int, default=None,
                   help="Optional cap on parquet files scanned. Default: all.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for within-bucket reservoir.")
    return p.parse_args()


def list_parquet_files() -> list[str]:
    api = HfApi()
    info = api.dataset_info(DATASET_ID)
    paths = []
    for sib in info.siblings:
        rfn = sib.rfilename
        if rfn.endswith(".parquet") and rfn.startswith(f"{DATASET_SPLIT}/"):
            paths.append(rfn)
    paths.sort()
    return paths


def parse_metadata(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def flatten_metadata(metadata: dict) -> dict:
    flat: dict = {}
    for k in FLAT_METADATA_KEYS:
        v = metadata.get(k, "")
        flat[k] = v if v is not None else ""
    return flat


def bucket_for(row: dict, target_orders: set[str], nonbio_sources: set[str]) -> str | None:
    meta = parse_metadata(row.get("metadata"))
    cls = (meta.get("class") or "").strip()
    order = (meta.get("order") or "").strip()
    src = (row.get("source_dataset") or "").strip()
    if cls == "Aves" and order in target_orders:
        return f"order:{order}"
    if cls == "Mammalia":
        return "class:Mammalia"
    if src in nonbio_sources:
        return "nonbio"
    return None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_orders = {x.strip() for x in args.target_orders.split(",") if x.strip()}
    nonbio_sources = {x.strip() for x in args.nonbio_sources.split(",") if x.strip()}

    # Bucket capacities
    capacities: dict[str, int] = {}
    for o in target_orders:
        capacities[f"order:{o}"] = args.samples_per_order
    capacities["class:Mammalia"] = args.samples_mammalia
    capacities["nonbio"] = args.samples_nonbio
    print(f"Target buckets: {capacities}", flush=True)

    parquet_files = list_parquet_files()
    print(f"Discovered {len(parquet_files)} parquet shards in dataset", flush=True)
    if args.max_parquet_files is not None:
        parquet_files = parquet_files[: args.max_parquet_files]

    api = HfApi()
    info = api.dataset_info(DATASET_ID)
    revision = info.sha or "main"

    con = duckdb.connect(":memory:")
    con.execute("INSTALL httpfs; LOAD httpfs;")

    selected: dict[str, list[dict]] = {b: [] for b in capacities}
    rows_scanned = 0
    t0 = time.time()
    rng_seed_offset = 0

    for parquet_relpath in parquet_files:
        if all(len(selected[b]) >= capacities[b] for b in capacities):
            print("All buckets filled; stopping shard scan.", flush=True)
            break
        if rows_scanned >= args.max_rows_to_scan:
            print(f"Hit --max_rows_to_scan ({args.max_rows_to_scan}); stopping.", flush=True)
            break

        url = (
            f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{revision}/"
            f"{parquet_relpath}?download=1"
        )
        # SELECT only the columns we need; avoid pulling audio bytes here.
        sql = f"""
            SELECT
                row_number() OVER () - 1 AS shard_row_idx,
                id, file_name, source_dataset, task, output, license,
                instruction_text, metadata
            FROM read_parquet('{url}')
        """
        try:
            cursor = con.execute(sql)
        except Exception as e:
            print(f"WARN: failed to read {parquet_relpath}: {e}", flush=True)
            continue

        cols = [d[0] for d in cursor.description]
        for tup in cursor.fetchall():
            row = dict(zip(cols, tup))
            rows_scanned += 1
            if rows_scanned >= args.max_rows_to_scan:
                break
            bucket = bucket_for(row, target_orders, nonbio_sources)
            if bucket is None or len(selected[bucket]) >= capacities[bucket]:
                continue
            meta = parse_metadata(row.get("metadata"))
            flat = flatten_metadata(meta)
            duration = meta.get("duration")
            sample_rate = meta.get("sample_rate") or 16000
            entry = {
                "row_index": int(row["shard_row_idx"]),
                "dataset": DATASET_ID,
                "config": DATASET_CONFIG,
                "split": DATASET_SPLIT,
                "parquet_revision": revision,
                "parquet_relpath": parquet_relpath,
                "parquet_url": url,
                "id": row["id"],
                "file_name": row["file_name"],
                "source_dataset": row["source_dataset"],
                "task": row.get("task"),
                "output": row.get("output"),
                "license": row.get("license"),
                "instruction_text": row.get("instruction_text"),
                "duration_s": float(duration) if isinstance(duration, (int, float)) else None,
                "sampling_rate": int(sample_rate) if isinstance(sample_rate, (int, float)) else 16000,
                "metadata_raw": row.get("metadata"),
                "bucket": bucket,
                **flat,
            }
            selected[bucket].append(entry)

        bucket_state = ", ".join(f"{b}={len(selected[b])}/{capacities[b]}" for b in capacities)
        elapsed = time.time() - t0
        print(f"  {parquet_relpath}: scanned={rows_scanned}, {bucket_state}, "
              f"{elapsed:.0f}s elapsed", flush=True)

    # Concatenate buckets in a deterministic order
    flat_records: list[dict] = []
    for bucket in sorted(selected):
        flat_records.extend(selected[bucket])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = {
        "timestamp": timestamp,
        "rows_scanned": rows_scanned,
        "samples_per_order": args.samples_per_order,
        "samples_mammalia": args.samples_mammalia,
        "samples_nonbio": args.samples_nonbio,
        "target_orders": sorted(target_orders),
        "nonbio_sources": sorted(nonbio_sources),
        "buckets_filled": {b: len(selected[b]) for b in capacities},
        "buckets_unfilled": {b: max(capacities[b] - len(selected[b]), 0) for b in capacities},
        "total_samples": len(flat_records),
        "parquet_revision": revision,
    }
    base = (f"naturelm_by_order_p{args.samples_per_order}"
            f"_m{args.samples_mammalia}_n{args.samples_nonbio}_{timestamp}")
    manifest_path = args.output_dir / f"{base}.jsonl"
    summary_path = args.output_dir / f"{base}_summary.json"

    with manifest_path.open("w") as f:
        for r in flat_records:
            f.write(json.dumps(r) + "\n")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Sampling complete ===")
    print(f"  scanned {rows_scanned} rows in {time.time() - t0:.0f}s")
    print(f"  wrote {len(flat_records)} samples to {manifest_path}")
    print(f"  buckets:")
    for b in capacities:
        n = len(selected[b])
        status = "OK" if n >= capacities[b] else f"SHORT ({capacities[b] - n})"
        print(f"    {b:<28} {n:>4} / {capacities[b]:<4}  {status}")


if __name__ == "__main__":
    main()
