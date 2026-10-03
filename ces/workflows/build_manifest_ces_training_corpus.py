"""Build CES raw/candidate shards from frozen source-target JSONL manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

from nanochat.causal_serialisation_extended import (
    Delete,
    Insert,
    Replace,
    SequenceEdit,
    complement,
    resolve,
)
from nanochat.edit_dataset import edit_counts, sequence_edit_to_json
from nanochat.oracle_edit_likelihood import canonical_sequence_edit_from_tokens
from nanochat.tokenizer import get_extended_indel_tokeniser
from nanochat.tokenizer_contract import verify_canonical_tokenizer


def _read(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pair_to_character_edit(
    source: str, target: str
) -> tuple[SequenceEdit, SequenceEdit]:
    edits = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(
        list(source), list(target)
    ).as_list():
        if tag == "equal":
            continue
        if tag == "insert":
            edits.append(Insert(i1, np.asarray(list(target[j1:j2]))))
        elif tag == "delete":
            edits.append(Delete(i1, i2))
        elif tag == "replace":
            edits.append(
                Replace(i1, i2, np.asarray(list(target[j1:j2])))
            )
    forward = SequenceEdit(np.asarray(list(source)), edits)
    inverse = complement(forward)
    if "".join(resolve(forward).tolist()) != target:
        raise ValueError("Forward edit did not reconstruct target")
    if "".join(resolve(inverse).tolist()) != source:
        raise ValueError("Inverse edit did not reconstruct source")
    return forward, inverse


def token_sequence_edit_to_json(sequence_edit: SequenceEdit) -> str:
    """Serialize a token-coordinate CES program without its source sequence."""

    payload = []
    for edit in sequence_edit.edits:
        if isinstance(edit, Insert):
            payload.append(
                {
                    "type": "insert",
                    "pos": int(edit.pos),
                    "content": [int(token) for token in edit.content],
                }
            )
        elif isinstance(edit, Delete):
            payload.append(
                {
                    "type": "delete",
                    "start": int(edit.start),
                    "end": int(edit.end),
                }
            )
        elif isinstance(edit, Replace):
            payload.append(
                {
                    "type": "replace",
                    "start": int(edit.start),
                    "end": int(edit.end),
                    "content": [int(token) for token in edit.content],
                }
            )
        else:  # pragma: no cover - canonical alignment only emits these types
            raise TypeError(f"Unsupported canonical token edit: {type(edit)!r}")
    return json.dumps(payload, separators=(",", ":"))


def _candidate(
    row: dict,
    row_index: int,
    shard_id: str,
    tokenizer,
) -> tuple[dict, dict]:
    source = row["source_text"]
    target = row["target_text"]
    forward, inverse = pair_to_character_edit(source, target)
    source_tokens = [int(token) for token in tokenizer.encode(source)]
    target_tokens = [int(token) for token in tokenizer.encode(target)]
    forward_tokens = canonical_sequence_edit_from_tokens(
        source_tokens, target_tokens
    )
    inverse_tokens = complement(forward_tokens)
    if [int(token) for token in resolve(inverse_tokens)] != source_tokens:
        raise ValueError("Inverse token edit did not reconstruct source")
    counts = edit_counts(forward)
    common = {
        "dataset": str(row.get("dataset") or "coedit"),
        "task": str(row.get("task") or "gec"),
        "example_id": str(
            row.get("benchmark_id") or row.get("example_id") or row_index
        ),
    }
    raw = {"text": source, **common}
    candidate = {
        "row_index": row_index,
        "source_shard_id": shard_id,
        "source_text": source,
        "target_text": target,
        "forward_edits_json": sequence_edit_to_json(forward),
        "inverse_edits_json": sequence_edit_to_json(inverse),
        "forward_token_edits_json": token_sequence_edit_to_json(forward_tokens),
        "inverse_token_edits_json": token_sequence_edit_to_json(inverse_tokens),
        "forward_edit_counts_json": json.dumps(counts, sort_keys=True),
        "has_valid_edit_script": True,
        **common,
    }
    return raw, candidate


def _write_shard(
    rows: list[dict],
    *,
    shard_id: str,
    raw_dir: Path,
    candidate_dir: Path,
    tokenizer,
) -> int:
    raw_rows = []
    candidates = []
    for row in rows:
        raw, candidate = _candidate(
            row, len(raw_rows), shard_id, tokenizer
        )
        raw_rows.append(raw)
        candidates.append(candidate)
    pq.write_table(
        pa.Table.from_pylist(raw_rows),
        raw_dir / f"{shard_id}.parquet",
        row_group_size=1024,
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(candidates),
        candidate_dir / f"{shard_id}.parquet",
        row_group_size=1024,
        compression="zstd",
    )
    return len(raw_rows)


def build_corpus(
    *,
    train_manifest: Path,
    validation_manifest: Path,
    output_dir: Path,
    train_shards: int,
    seed: int,
) -> dict:
    train = _read(train_manifest)
    validation = _read(validation_manifest)
    random.Random(seed).shuffle(train)
    raw_dir = output_dir / "raw"
    candidate_dir = output_dir / "candidates"
    raw_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = get_extended_indel_tokeniser()
    tokenizer_contract = verify_canonical_tokenizer(tokenizer=tokenizer)

    train_examples = 0
    for shard_index in range(train_shards):
        shard_id = f"shard_{shard_index:05d}"
        count = _write_shard(
            train[shard_index::train_shards],
            shard_id=shard_id,
            raw_dir=raw_dir,
            candidate_dir=candidate_dir,
            tokenizer=tokenizer,
        )
        train_examples += count
        print(f"{shard_id}: {count:,}")
    validation_examples = _write_shard(
        validation,
        shard_id="shard_99999",
        raw_dir=raw_dir,
        candidate_dir=candidate_dir,
        tokenizer=tokenizer,
    )
    metadata = {
        "version": 2,
        "canonicalizer": {
            "character": "minimum unit-cost character Levenshtein",
            "token": "minimum unit-cost token Levenshtein",
        },
        "tokenizer_contract": tokenizer_contract,
        "direction": "source_to_target",
        "seed": seed,
        "train_manifest": str(train_manifest.resolve()),
        "validation_manifest": str(validation_manifest.resolve()),
        "train_examples": train_examples,
        "validation_examples": validation_examples,
        "train_shards": train_shards,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-shards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    if args.train_shards < 1:
        parser.error("--train-shards must be positive")
    print(
        json.dumps(
            build_corpus(
                train_manifest=args.train_manifest,
                validation_manifest=args.validation_manifest,
                output_dir=args.output_dir,
                train_shards=args.train_shards,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
