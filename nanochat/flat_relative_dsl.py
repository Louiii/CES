"""Autoregressive edit DSL with single-token backwards-relative coordinates."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from nanochat.causal_serialisation_extended import (
    Copy,
    Delete,
    Insert,
    Move,
    Replace,
    SequenceEdit,
    _edit_key,
)
from nanochat.edit_dsl import (
    EDIT_SCRIPT_CLOSE,
    EDIT_SCRIPT_OPEN,
    ar_dsl_to_sequence_edit,
)
from nanochat.tokenizer import (
    FLAT_RELATIVE_CONTEXT_LENGTH,
    FLAT_RELATIVE_SPECIAL_TOKENS,
    flat_relative_token,
)


def flat_relative_token_maps(tokenizer) -> tuple[dict[int, int], dict[int, int]]:
    """Return distance-to-token-id and token-id-to-distance maps."""

    specials = tokenizer.enc._special_tokens
    distance_to_id = {
        distance: int(specials[flat_relative_token(distance)])
        for distance in range(1, FLAT_RELATIVE_CONTEXT_LENGTH)
    }
    return distance_to_id, {token_id: distance for distance, token_id in distance_to_id.items()}


def _payload_text(tokenizer, values: np.ndarray) -> str:
    return tokenizer.decode([int(token) for token in np.asarray(values).tolist()])


def sequence_edit_to_flat_relative_tokens(seq_edit: SequenceEdit, tokenizer) -> list[int]:
    """Serialize an edit using one ordinary vocabulary token per coordinate.

    If a coordinate token is written at serialized index p and represents
    source coordinate c, its vocabulary value is BACK_(p-c). Coordinates
    remain the same zero-based token/boundary coordinates used by AR-DSL.
    """

    source_len = len(seq_edit.sequence)
    distance_to_id, _ = flat_relative_token_maps(tokenizer)
    output: list[int] = []

    def text(value: str) -> None:
        output.extend(int(token) for token in tokenizer.encode(value))

    def coordinate(value: int) -> None:
        if not 0 <= value <= source_len:
            raise ValueError(f"Coordinate {value} lies outside source length {source_len}")
        serialized_index = source_len + len(output)
        distance = serialized_index - value
        if distance not in distance_to_id:
            raise ValueError(
                f"Coordinate at serialized index {serialized_index} requires BACK_{distance}, "
                f"outside [1, {FLAT_RELATIVE_CONTEXT_LENGTH - 1}]"
            )
        output.append(distance_to_id[distance])

    text(f"\n{EDIT_SCRIPT_OPEN}\n")
    for edit in sorted(seq_edit.edits, key=_edit_key):
        if isinstance(edit, Delete):
            text("D ")
            coordinate(int(edit.start))
            text(" ")
            coordinate(int(edit.end))
        elif isinstance(edit, Insert):
            text("I ")
            coordinate(int(edit.pos))
            text(" ")
            text(json.dumps(_payload_text(tokenizer, edit.content), ensure_ascii=False))
        elif isinstance(edit, Replace):
            text("R ")
            coordinate(int(edit.start))
            text(" ")
            coordinate(int(edit.end))
            text(" ")
            text(json.dumps(_payload_text(tokenizer, edit.content), ensure_ascii=False))
        elif isinstance(edit, Copy):
            text("C ")
            coordinate(int(edit.start))
            text(" ")
            coordinate(int(edit.end))
            text(" -> ")
            coordinate(int(edit.dest))
        elif isinstance(edit, Move):
            text("M ")
            coordinate(int(edit.start))
            text(" ")
            coordinate(int(edit.end))
            text(" -> ")
            coordinate(int(edit.dest))
        else:
            raise TypeError(f"Unsupported edit type: {type(edit)!r}")
        text("\n")
    text(f"{EDIT_SCRIPT_CLOSE}\n")
    return output


def flat_relative_tokens_to_text(
    source_token_count: int,
    program_tokens: list[int] | np.ndarray,
    tokenizer,
) -> str:
    """Resolve BACK tokens to decimal coordinates and decode the textual DSL."""

    _, id_to_distance = flat_relative_token_maps(tokenizer)
    rewritten: list[int] = []
    for local_index, token in enumerate(np.asarray(program_tokens).tolist()):
        token = int(token)
        distance = id_to_distance.get(token)
        if distance is None:
            rewritten.append(token)
            continue
        serialized_index = source_token_count + local_index
        coordinate = serialized_index - distance
        if not 0 <= coordinate <= source_token_count:
            raise ValueError(
                f"BACK_{distance} at serialized index {serialized_index} resolves to "
                f"out-of-range coordinate {coordinate}"
            )
        rewritten.extend(int(value) for value in tokenizer.encode(str(coordinate)))
    return tokenizer.decode(rewritten)


def flat_relative_tokens_to_sequence_edit(
    source_tokens: list[int] | np.ndarray,
    program_tokens: list[int] | np.ndarray,
    tokenizer,
) -> SequenceEdit:
    """Parse a generated flat-relative program into a source-coordinate edit."""

    text = flat_relative_tokens_to_text(len(source_tokens), program_tokens, tokenizer)
    start = text.find(EDIT_SCRIPT_OPEN)
    if start < 0:
        raise ValueError("Generated program is missing <edit-script>")
    close = text.find(EDIT_SCRIPT_CLOSE, start + len(EDIT_SCRIPT_OPEN))
    if close < 0:
        raise ValueError("Generated program is missing </edit-script>")
    dsl = text[start:close + len(EDIT_SCRIPT_CLOSE)]
    return ar_dsl_to_sequence_edit(source_tokens, dsl, tokenizer)


def flat_relative_vocabulary_metadata(tokenizer) -> dict[str, Any]:
    """Small auditable description stored with evaluation output."""

    distance_to_id, _ = flat_relative_token_maps(tokenizer)
    return {
        "context_length": FLAT_RELATIVE_CONTEXT_LENGTH,
        "minimum_distance": 1,
        "maximum_distance": FLAT_RELATIVE_CONTEXT_LENGTH - 1,
        "number_of_pointer_tokens": len(FLAT_RELATIVE_SPECIAL_TOKENS),
        "first_pointer_token_id": distance_to_id[1],
        "last_pointer_token_id": distance_to_id[FLAT_RELATIVE_CONTEXT_LENGTH - 1],
    }
