"""Freeze deterministic source-only CoEdit GEC development and test manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


BENCHMARK_VERSION = "coedit_gec_v1"
DEFAULT_SPLIT_SEED = "ces-rho-gec-pareto-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_text(text: str) -> str:
    return " ".join(text.split())


def _pair_key(row: dict[str, Any]) -> tuple[str, str]:
    return _normalise_text(row["source_text"]), _normalise_text(row["target_text"])


def _row_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "dataset": "coedit",
            "task": "gec",
            "source_text": _normalise_text(row["source_text"]),
            "target_text": _normalise_text(row["target_text"]),
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_gec(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in pq.read_table(path).to_pylist():
        if row.get("task") != "gec":
            continue
        source = row.get("source_text")
        target = row.get("target_text")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if not source.strip() or not target.strip() or source == target:
            continue
        rows.append(row)
    return rows


def _deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _pair_key(row)
        current = unique.get(key)
        if current is None or str(row.get("example_id", "")) < str(current.get("example_id", "")):
            unique[key] = row
    return list(unique.values()), len(rows) - len(unique)


def _manifest_row(row: dict[str, Any], role: str) -> dict[str, Any]:
    row_hash = _row_hash(row)
    return {
        "benchmark_id": f"{BENCHMARK_VERSION}:{role}:{row_hash[:20]}",
        "benchmark_version": BENCHMARK_VERSION,
        "role": role,
        "dataset": "coedit",
        "source_split": str(row.get("split", "")),
        "source_example_id": str(row.get("example_id", "")),
        "task": "gec",
        "source_text": row["source_text"],
        "target_text": row["target_text"],
        "source_only_inference": True,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return _sha256_file(path)


def build_manifests(
    *,
    train_parquet: Path,
    validation_parquet: Path,
    output_dir: Path,
    sweep_size: int,
    confirmation_size: int,
    split_seed: str,
) -> dict[str, Any]:
    train_raw = _load_gec(train_parquet)
    validation_raw = _load_gec(validation_parquet)
    train, train_duplicates = _deduplicate(train_raw)
    validation, validation_duplicates = _deduplicate(validation_raw)

    validation_keys = {_pair_key(row) for row in validation}
    train = [row for row in train if _pair_key(row) not in validation_keys]
    required = sweep_size + confirmation_size
    if required > len(train):
        raise ValueError(f"Requested {required} train-pool rows, only {len(train)} are available")

    def split_order(row: dict[str, Any]) -> str:
        return hashlib.sha256(f"{split_seed}:{_row_hash(row)}".encode("utf-8")).hexdigest()

    train.sort(key=split_order)
    split_rows = {
        "development": [_manifest_row(row, "development") for row in validation],
        "sweep": [_manifest_row(row, "sweep") for row in train[:sweep_size]],
        "confirmation": [
            _manifest_row(row, "confirmation")
            for row in train[sweep_size : sweep_size + confirmation_size]
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for role, rows in split_rows.items():
        path = output_dir / f"{role}.jsonl"
        manifests[role] = {
            "path": path.name,
            "examples": len(rows),
            "sha256": _write_jsonl(path, rows),
        }

    selected_keys = {
        row["benchmark_id"]
        for rows in split_rows.values()
        for row in rows
    }
    metadata = {
        "benchmark_version": BENCHMARK_VERSION,
        "task_definition": (
            "Source-only grammatical error correction. The model receives source text "
            "and an edit-program opener; target text, gold spans and gold edits are hidden."
        ),
        "invalid_output_policy": "Use unchanged source text as the final prediction.",
        "split_seed": split_seed,
        "source_files": {
            "train": {
                "path": f"coedit/{train_parquet.name}",
                "sha256": _sha256_file(train_parquet),
                "gec_rows_before_deduplication": len(train_raw),
            },
            "validation": {
                "path": f"coedit/{validation_parquet.name}",
                "sha256": _sha256_file(validation_parquet),
                "gec_rows_before_deduplication": len(validation_raw),
            },
        },
        "deduplication": {
            "normalisation": "Collapse Unicode whitespace; require exact normalized source+target pair equality.",
            "train_duplicates_removed": train_duplicates,
            "validation_duplicates_removed": validation_duplicates,
            "cross_split_overlaps_removed": len(train_raw) - train_duplicates - len(train),
        },
        "manifests": manifests,
        "unused_unique_train_pool_rows": len(train) - required,
        "selected_unique_ids": len(selected_keys),
        "warning": (
            "The official CoEdit validation GEC rows have already informed decoding and gating "
            "and are development-only. Sweep rows may be used to compare the rho curve. "
            "Confirmation rows must remain untouched until choices are frozen."
        ),
    }
    meta_path = output_dir / "manifest.meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coedit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data/evals/coedit_gec_v1"), type=Path)
    parser.add_argument("--sweep-size", type=int, default=2000)
    parser.add_argument("--confirmation-size", type=int, default=5000)
    parser.add_argument("--split-seed", default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args()
    if args.sweep_size < 1 or args.confirmation_size < 1:
        parser.error("split sizes must be positive")
    metadata = build_manifests(
        train_parquet=args.coedit_dir / "train.parquet",
        validation_parquet=args.coedit_dir / "validation.parquet",
        output_dir=args.output_dir,
        sweep_size=args.sweep_size,
        confirmation_size=args.confirmation_size,
        split_seed=args.split_seed,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
