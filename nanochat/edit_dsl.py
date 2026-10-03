"""Plain-text edit DSL for autoregressive edit-script baselines."""

from __future__ import annotations

import json
import re
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
    _validate_edits,
    _validate_move_edits,
)


EDIT_SCRIPT_OPEN = "<edit-script>"
EDIT_SCRIPT_CLOSE = "</edit-script>"

AR_DSL_COORD_RE = re.compile(
    r"(?m)^(?:"
    r"D\s+(?P<D0>\d+)\s+(?P<D1>\d+)"
    r"|I\s+(?P<I0>\d+)\s+"
    r"|R\s+(?P<R0>\d+)\s+(?P<R1>\d+)\s+"
    r"|C\s+(?P<C0>\d+)\s+(?P<C1>\d+)\s+->\s+(?P<C2>\d+)"
    r"|M\s+(?P<M0>\d+)\s+(?P<M1>\d+)\s+->\s+(?P<M2>\d+)"
    r")"
)

_INT = r"(0|[1-9][0-9]*)"
_DELETE_RE = re.compile(rf"^D\s+{_INT}\s+{_INT}$")
_INSERT_RE = re.compile(rf"^I\s+{_INT}\s+(.+)$", re.DOTALL)
_REPLACE_RE = re.compile(rf"^R\s+{_INT}\s+{_INT}\s+(.+)$", re.DOTALL)
_COPY_RE = re.compile(rf"^C\s+{_INT}\s+{_INT}\s+->\s+{_INT}$")
_MOVE_RE = re.compile(rf"^M\s+{_INT}\s+{_INT}\s+->\s+{_INT}$")


def _decode_payload(tokeniser, tokens: np.ndarray) -> str:
    values = [int(t) for t in np.array(tokens).tolist()]
    return tokeniser.decode(values)


def _encode_payload(tokeniser, text: str) -> np.ndarray:
    enc = getattr(tokeniser, "enc", tokeniser)
    return np.array(enc.encode(text, allowed_special="all"), dtype=np.int64)


def sequence_edit_to_ar_dsl(seq_edit: SequenceEdit, tokeniser) -> str:
    """Render a token-coordinate ``SequenceEdit`` as a compact text program.

    Coordinates are over ``seq_edit.sequence`` only. They do not include DSL
    delimiter tokens or any text emitted after the source. If the caller includes
    BOS/document separator tokens in ``seq_edit.sequence``, those tokens are part
    of the coordinate space.
    Payloads are JSON string literals so arbitrary whitespace and quotes roundtrip.
    """

    lines = [EDIT_SCRIPT_OPEN]
    for edit in sorted(seq_edit.edits, key=_edit_key):
        if isinstance(edit, Delete):
            lines.append(f"D {int(edit.start)} {int(edit.end)}")
        elif isinstance(edit, Insert):
            payload = json.dumps(_decode_payload(tokeniser, edit.content), ensure_ascii=False)
            lines.append(f"I {int(edit.pos)} {payload}")
        elif isinstance(edit, Replace):
            payload = json.dumps(_decode_payload(tokeniser, edit.content), ensure_ascii=False)
            lines.append(f"R {int(edit.start)} {int(edit.end)} {payload}")
        elif isinstance(edit, Copy):
            lines.append(f"C {int(edit.start)} {int(edit.end)} -> {int(edit.dest)}")
        elif isinstance(edit, Move):
            lines.append(f"M {int(edit.start)} {int(edit.end)} -> {int(edit.dest)}")
        else:
            raise TypeError(f"Unsupported edit type: {type(edit)!r}")
    lines.append(EDIT_SCRIPT_CLOSE)
    return "\n".join(lines)


def sequence_edit_to_ar_dsl_tokens(seq_edit: SequenceEdit, tokenizer) -> list[int]:
    """Tokenize the AR DSL program for a token-coordinate edit script."""

    dsl = "\n" + sequence_edit_to_ar_dsl(seq_edit, tokenizer) + "\n"
    return tokenizer.encode(dsl)


def ar_dsl_coordinate_token_spans(tokenizer, dsl_text: str, source_token_count: int = 0) -> list[tuple[int, int]]:
    """Return token spans for integer coordinate fields in an AR-DSL script."""

    spans: list[tuple[int, int]] = []
    for match in AR_DSL_COORD_RE.finditer(dsl_text):
        for name, text in match.groupdict().items():
            if text is None:
                continue
            char_start, char_end = match.span(name)
            token_start = len(tokenizer.encode(dsl_text[:char_start]))
            token_end = len(tokenizer.encode(dsl_text[:char_end]))
            if token_end > token_start:
                spans.append((source_token_count + token_start, source_token_count + token_end))
    return spans


def ar_dsl_to_sequence_edit(source_tokens: list[int] | np.ndarray, dsl: str, tokeniser) -> SequenceEdit:
    """Parse an AR edit DSL program into a token-coordinate ``SequenceEdit``."""

    text = dsl.strip()
    if text.startswith(EDIT_SCRIPT_OPEN):
        text = text[len(EDIT_SCRIPT_OPEN):].lstrip()
    if text.endswith(EDIT_SCRIPT_CLOSE):
        text = text[: -len(EDIT_SCRIPT_CLOSE)].rstrip()

    edits: list[Any] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _DELETE_RE.match(line)
        if match:
            edits.append(Delete(int(match.group(1)), int(match.group(2))))
            continue
        match = _INSERT_RE.match(line)
        if match:
            edits.append(Insert(int(match.group(1)), _encode_payload(tokeniser, json.loads(match.group(2)))))
            continue
        match = _REPLACE_RE.match(line)
        if match:
            edits.append(
                Replace(
                    int(match.group(1)),
                    int(match.group(2)),
                    _encode_payload(tokeniser, json.loads(match.group(3))),
                )
            )
            continue
        match = _COPY_RE.match(line)
        if match:
            edits.append(Copy(int(match.group(1)), int(match.group(2)), int(match.group(3))))
            continue
        match = _MOVE_RE.match(line)
        if match:
            edits.append(Move(int(match.group(1)), int(match.group(2)), int(match.group(3))))
            continue
        raise ValueError(f"Invalid AR edit DSL line: {raw_line!r}")

    sequence = np.array(source_tokens, dtype=np.int64)
    edits.sort(key=_edit_key)
    _validate_move_edits(sequence, edits)
    _validate_edits(sequence, edits)
    return SequenceEdit(sequence=sequence, edits=edits)
