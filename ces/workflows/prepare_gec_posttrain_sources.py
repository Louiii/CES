"""Prepare clean CoEdit targets as sharded sources for GEC corruptions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--min-chars", type=int, default=32)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit < args.num_shards or args.num_shards < 2:
        raise ValueError("Require limit >= num_shards >= 2")
    existing = list(args.output_dir.glob("shard_*.parquet"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Source shards already exist under {args.output_dir}; pass --overwrite"
        )

    rows = read_jsonl(args.manifest)
    seen: set[str] = set()
    eligible: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("target_text", "")).strip()
        if not args.min_chars <= len(text) <= args.max_chars or text in seen:
            continue
        seen.add(text)
        eligible.append({
            "text": text,
            "source_text": str(row.get("source_text", "")),
            "benchmark_id": str(row.get("benchmark_id", "")),
            "source_example_id": str(row.get("source_example_id", "")),
            "dataset": str(row.get("dataset", "coedit")),
            "task": str(row.get("task", "gec")),
        })
    random.Random(args.seed).shuffle(eligible)
    selected = eligible[: args.limit]
    if len(selected) < args.limit:
        raise ValueError(
            f"Only {len(selected)} eligible unique targets for limit {args.limit}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_sizes = [len(selected) // args.num_shards] * args.num_shards
    for index in range(len(selected) % args.num_shards):
        shard_sizes[index] += 1
    offset = 0
    shard_records = []
    for shard_index, size in enumerate(shard_sizes):
        path = args.output_dir / f"shard_{shard_index:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(selected[offset : offset + size]), path)
        shard_records.append({
            "path": str(path.resolve()),
            "rows": size,
            "role": "validation" if shard_index == args.num_shards - 1 else "train",
        })
        offset += size
    manifest = {
        "input_manifest": str(args.manifest.resolve()),
        "selection_seed": args.seed,
        "eligible_rows": len(eligible),
        "selected_rows": len(selected),
        "train_rows": sum(row["rows"] for row in shard_records[:-1]),
        "validation_rows": shard_records[-1]["rows"],
        "shards": shard_records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
