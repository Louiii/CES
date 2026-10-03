"""Make frozen text-pair benchmarks portable across tokenizer checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Iterable

import numpy as np

from nanochat.causal_serialisation_extended import (
    Copy,
    Delete,
    Insert,
    Move,
    Replace,
    SequenceEdit,
    TokensOperations,
)


def _tokens_hash(tokens: Iterable[int]) -> str:
    payload = json.dumps([int(token) for token in tokens], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decode_without_bos(tokenizer, tokens: Iterable[int]) -> str:
    bos = tokenizer.get_bos_token_id()
    return tokenizer.decode([int(token) for token in tokens if int(token) != bos])


def _token_edit_from_text_pair(tokenizer, source_text: str, target_text: str) -> SequenceEdit:
    """Build a tokenizer-native minimal edit program for a frozen text pair."""

    bos = tokenizer.get_bos_token_id()
    source_tokens = [bos, *[int(token) for token in tokenizer.encode(source_text)]]
    target_tokens = [bos, *[int(token) for token in tokenizer.encode(target_text)]]
    matcher = SequenceMatcher(a=source_tokens, b=target_tokens, autojunk=False)
    edits = []
    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            edits.append(Insert(
                source_start,
                np.asarray(target_tokens[target_start:target_end], dtype=np.int64),
            ))
        elif tag == "delete":
            edits.append(Delete(source_start, source_end))
        elif tag == "replace":
            edits.append(Replace(
                source_start,
                source_end,
                np.asarray(target_tokens[target_start:target_end], dtype=np.int64),
            ))
        else:  # pragma: no cover
            raise ValueError(f"Unsupported diff opcode: {tag}")
    return SequenceEdit(
        sequence=np.asarray(source_tokens, dtype=np.int64),
        edits=edits,
    )


def _edit_record(edit) -> dict[str, Any]:
    if isinstance(edit, Insert):
        return {
            "op": "insert",
            "pos": int(edit.pos),
            "content": [int(token) for token in edit.content],
        }
    if isinstance(edit, Delete):
        return {"op": "delete", "start": int(edit.start), "end": int(edit.end)}
    if isinstance(edit, Replace):
        return {
            "op": "replace",
            "start": int(edit.start),
            "end": int(edit.end),
            "content": [int(token) for token in edit.content],
        }
    if isinstance(edit, (Copy, Move)):
        return {
            "op": "copy" if isinstance(edit, Copy) else "move",
            "start": int(edit.start),
            "end": int(edit.end),
            "dest": int(edit.dest),
        }
    raise TypeError(f"Unsupported edit type: {type(edit).__name__}")


def prepare_benchmark_for_tokenizer(
    benchmark_rows: list[dict[str, Any]],
    *,
    tokenizer,
    token_ops: TokensOperations,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use compatible manifest IDs, or rebuild derived IDs from frozen text.

    Source/target strings and construction metadata are tokenizer-independent.
    Token IDs, pointer coordinates, and exact gold programs are derived fields.
    """

    bos = int(tokenizer.get_bos_token_id())
    incompatibilities: Counter[str] = Counter()
    max_manifest_token = -1
    for row in benchmark_rows:
        source_tokens = [int(token) for token in row["source_tokens"]]
        target_tokens = [int(token) for token in row["target_tokens"]]
        prompt_tokens = [int(token) for token in row["prompt_tokens"]]
        for values in (source_tokens, target_tokens, prompt_tokens):
            if values:
                max_manifest_token = max(max_manifest_token, max(values))
        if not source_tokens or source_tokens[0] != bos:
            incompatibilities["source_bos"] += 1
        if not target_tokens or target_tokens[0] != bos:
            incompatibilities["target_bos"] += 1
        if not prompt_tokens or prompt_tokens[-1] != int(token_ops.edit_start_token):
            incompatibilities["edit_start"] += 1
        try:
            if _decode_without_bos(tokenizer, source_tokens) != str(row["source_text"]):
                incompatibilities["source_decode"] += 1
            if _decode_without_bos(tokenizer, target_tokens) != str(row["target_text"]):
                incompatibilities["target_decode"] += 1
        except Exception:
            incompatibilities["decode_error"] += 1

    metadata = {
        "mode": "manifest_tokens" if not incompatibilities else "retokenized_text",
        "tokenizer_vocab_size": int(tokenizer.get_vocab_size()),
        "tokenizer_bos_token": bos,
        "tokenizer_edit_start_token": int(token_ops.edit_start_token),
        "tokenizer_edit_end_token": int(token_ops.edit_end_token),
        "manifest_max_token_id": max_manifest_token,
        "incompatibility_counts": dict(sorted(incompatibilities.items())),
    }
    if not incompatibilities:
        return benchmark_rows, metadata

    prepared: list[dict[str, Any]] = []
    for row in benchmark_rows:
        sequence_edit = _token_edit_from_text_pair(
            tokenizer,
            str(row["source_text"]),
            str(row["target_text"]),
        )
        serialised = token_ops.serialise(sequence_edit, indel_bos_pos=0)
        edit_start_indices = np.where(
            np.asarray(serialised.tokens) == int(token_ops.edit_start_token)
        )[0]
        if len(edit_start_indices) != 1:
            raise ValueError(
                f"Retokenized benchmark row {row['benchmark_id']} has "
                f"{len(edit_start_indices)} edit-start tokens"
            )
        edit_start_index = int(edit_start_indices[0])
        source_tokens = [int(token) for token in sequence_edit.sequence]
        target_tokens = [
            bos,
            *[int(token) for token in tokenizer.encode(str(row["target_text"]))],
        ]
        gold_edits = [_edit_record(edit) for edit in sequence_edit.edits]
        gold_op_counts = dict(sorted(Counter(edit["op"] for edit in gold_edits).items()))
        rebuilt = dict(row)
        rebuilt.update({
            "construction_gold_edits": row["gold_edits"],
            "construction_gold_op_counts": row["gold_op_counts"],
            "source_num_tokens": len(source_tokens),
            "target_num_tokens": len(target_tokens),
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
            "source_token_sha256": _tokens_hash(source_tokens),
            "target_token_sha256": _tokens_hash(target_tokens),
            "gold_edits": gold_edits,
            "gold_op_counts": gold_op_counts,
            "prompt_tokens": [
                int(token) for token in serialised.tokens[:edit_start_index + 1]
            ],
            "prompt_positions": [
                int(position) for position in serialised.positions[:edit_start_index + 1]
            ],
            "gold_program_tokens": [
                int(token) for token in serialised.tokens[edit_start_index + 1:]
            ],
            "gold_program_positions": [
                int(position) for position in serialised.positions[edit_start_index + 1:]
            ],
            "benchmark_tokenization_mode": "retokenized_text",
            "retokenized_repair_num_ops": len(gold_edits),
        })
        if _decode_without_bos(tokenizer, source_tokens) != str(row["source_text"]):
            raise AssertionError(f"Source text did not survive retokenization: {row['benchmark_id']}")
        if _decode_without_bos(tokenizer, target_tokens) != str(row["target_text"]):
            raise AssertionError(f"Target text did not survive retokenization: {row['benchmark_id']}")
        prepared.append(rebuilt)
    return prepared, metadata
