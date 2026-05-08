"""Walk the cached NatureLM-audio-training parquet shards, pull the
`metadata` JSON column for each manifest sample, and emit an enriched
manifest with explicit phylum / class / order / family / genus /
species / subspecies fields.

The base manifest (`naturelm_by_source_100each_*.jsonl`) has only
`id`, `file_name`, `source_dataset`, `parquet_relpath`, `row_index`.
This adds the taxonomic columns so downstream scripts (per-Class /
per-Order analysis, species barycenters, Veitch hierarchy) don't have
to re-open the parquet.

Usage:
    python enrich_manifest_taxonomy.py
    python enrich_manifest_taxonomy.py --manifest <path> --output <path>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds


MANIFEST_ID = "naturelm_by_source_100each_20260418T171459Z"
DEFAULT_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}.jsonl")
DEFAULT_OUTPUT = Path(f"artifacts/manifests/{MANIFEST_ID}_taxonomic.jsonl")
DEFAULT_PARQUET_GLOB = str(
    Path.home() / ".cache" / "huggingface" / "hub" /
    "datasets--EarthSpeciesProject--NatureLM-audio-training" /
    "snapshots" / "*" / "train" / "part0" / "shard_*.parquet"
)

TAXONOMIC_FIELDS = (
    "phylum", "class", "order", "family", "genus", "species", "subspecies",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--parquet-glob", default=DEFAULT_PARQUET_GLOB)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    records: list[dict] = []
    with args.manifest.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    by_id = {r["id"]: r for r in records}
    print(f"Loaded {len(records)} manifest records", flush=True)

    import glob
    parquets = sorted(glob.glob(args.parquet_glob))
    print(f"Found {len(parquets)} parquet shards to scan", flush=True)
    if not parquets:
        raise SystemExit(f"No parquet shards matched {args.parquet_glob!r}")

    found = 0
    for path in parquets:
        if found == len(records):
            break
        d = ds.dataset(path, format="parquet")
        table = d.to_table(columns=["id", "metadata"]).to_pylist()
        for row in table:
            if row["id"] in by_id and "phylum" not in by_id[row["id"]]:
                try:
                    meta = json.loads(row["metadata"])
                except (TypeError, json.JSONDecodeError):
                    meta = {}
                target = by_id[row["id"]]
                for field in TAXONOMIC_FIELDS:
                    target[field] = meta.get(field, "") or ""
                found += 1
    print(f"Enriched {found}/{len(records)} records from parquet metadata", flush=True)
    if found < len(records):
        print(
            f"WARN: {len(records) - found} records missing from cached parquet shards",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote enriched manifest to {args.output}", flush=True)

    # Coverage report
    print("\n=== Coverage per source × taxonomic field (non-empty count) ===")
    sources = sorted({r.get("source_dataset", "") for r in records})
    header = f"{'source':<24} {'n':>5}  " + "  ".join(f"{f[:8]:>8}" for f in TAXONOMIC_FIELDS)
    print(header)
    for src in sources:
        sub = [r for r in records if r.get("source_dataset") == src]
        counts = [sum(bool(r.get(f, "")) for r in sub) for f in TAXONOMIC_FIELDS]
        line = f"{src:<24} {len(sub):>5}  " + "  ".join(f"{c:>8}" for c in counts)
        print(line)

    # Class distribution among the bio sources
    print("\n=== Class distribution (across all sources) ===")
    class_counts = Counter(r.get("class", "") for r in records)
    for cls, c in class_counts.most_common():
        label = cls if cls else "(empty)"
        print(f"  {label:<20} {c:>5}")

    # Order distribution within Aves
    aves_records = [r for r in records if r.get("class") == "Aves"]
    print(f"\n=== Order distribution within Aves (n={len(aves_records)}) ===")
    order_counts = Counter(r.get("order", "") for r in aves_records)
    for order, c in order_counts.most_common():
        label = order if order else "(empty)"
        print(f"  {label:<24} {c:>5}")

    # Top species (across all sources)
    print("\n=== Top species (across bio records, by sample count) ===")
    species_counts = Counter(
        r.get("species", "") for r in records if r.get("species")
    )
    for sp, c in species_counts.most_common(20):
        print(f"  {sp:<40} {c:>5}")
    print(f"  ... ({len(species_counts)} unique species total)")


if __name__ == "__main__":
    main()
