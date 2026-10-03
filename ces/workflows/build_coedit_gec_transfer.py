"""Freeze the CoEdit GEC rows reserved for supervised transfer training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ces.workflows.build_coedit_gec_benchmark import (
    _deduplicate,
    _load_gec,
    _manifest_row,
    _pair_key,
    _sha256_file,
    _write_jsonl,
)


TRANSFER_VERSION = "coedit_gec_transfer_v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_transfer_manifest(
    *,
    train_parquet: Path,
    validation_parquet: Path,
    benchmark_dir: Path,
    output_dir: Path,
) -> dict:
    train, train_duplicates = _deduplicate(_load_gec(train_parquet))
    validation, _validation_duplicates = _deduplicate(
        _load_gec(validation_parquet)
    )
    validation_keys = {_pair_key(row) for row in validation}
    train = [row for row in train if _pair_key(row) not in validation_keys]

    development_path = benchmark_dir / "sweep.jsonl"
    test_path = benchmark_dir / "confirmation.jsonl"
    development = _read_jsonl(development_path)
    test = _read_jsonl(test_path)
    held_out_keys = {
        _pair_key(row)
        for row in [*development, *test]
    }
    transfer_train = [
        row for row in train if _pair_key(row) not in held_out_keys
    ]
    transfer_train.sort(key=lambda row: str(row.get("example_id", "")))

    manifest_rows = []
    for row in transfer_train:
        item = _manifest_row(row, "finetune_train")
        item["benchmark_version"] = TRANSFER_VERSION
        item["benchmark_id"] = item["benchmark_id"].replace(
            "coedit_gec_v1:", f"{TRANSFER_VERSION}:", 1
        )
        manifest_rows.append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    train_sha = _write_jsonl(train_path, manifest_rows)
    metadata = {
        "version": TRANSFER_VERSION,
        "normalisation": (
            "Exact normalized source+target pair exclusion using the frozen "
            "coedit_gec_v1 development and confirmation manifests."
        ),
        "source_train_parquet": {
            "path": str(train_parquet),
            "sha256": _sha256_file(train_parquet),
            "duplicates_removed": train_duplicates,
        },
        "splits": {
            "train": {
                "path": train_path.name,
                "examples": len(manifest_rows),
                "sha256": train_sha,
            },
            "development": {
                "path": str(development_path),
                "examples": len(development),
                "sha256": _sha256_file(development_path),
            },
            "test": {
                "path": str(test_path),
                "examples": len(test),
                "sha256": _sha256_file(test_path),
            },
        },
        "disjoint": (
            len({_pair_key(row) for row in transfer_train} & held_out_keys) == 0
        ),
        "test_access_policy": (
            "Do not evaluate confirmation/test until phase-3 choices are frozen."
        ),
    }
    (output_dir / "manifest.meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coedit-dir", required=True, type=Path)
    parser.add_argument(
        "--benchmark-dir",
        default=Path("data/evals/coedit_gec_v1"),
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data/evals/coedit_gec_transfer_v1"),
        type=Path,
    )
    args = parser.parse_args()
    metadata = build_transfer_manifest(
        train_parquet=args.coedit_dir / "train.parquet",
        validation_parquet=args.coedit_dir / "validation.parquet",
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
