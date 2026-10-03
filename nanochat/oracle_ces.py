"""Oracle-location conditioning helpers for causal edit serialization (CES)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nanochat.causal_serialisation_extended import (
    Delete,
    Insert,
    Replace,
    SequenceEdit,
    TokensOperations,
)
from nanochat.text_edit_baselines import minimal_changed_envelope


class OraclePointerStructureMismatch(RuntimeError):
    """Raised when predicted operation structure cannot consume the next gold pointer."""


@dataclass(frozen=True)
class OracleSpanPrompt:
    """Target-derived operation skeleton whose target payload is not exposed."""

    prompt_tokens: tuple[int, ...]
    prompt_positions: tuple[int, ...]
    sequence_edit: SequenceEdit
    operation: str
    source_span: tuple[int, int]
    target_span: tuple[int, int]
    target_payload_tokens: tuple[int, ...]


class PointerOracleProvider:
    """Supply gold pointer values only when the predicted pointer-token type matches."""

    def __init__(
        self,
        gold_program_tokens: list[int],
        gold_program_positions: list[int],
        token_ops: TokensOperations,
    ) -> None:
        if len(gold_program_tokens) != len(gold_program_positions):
            raise ValueError("Gold program token and position streams must have equal length")
        pointer_tokens = {
            int(token_ops.deletion_start_token),
            int(token_ops.deletion_end_token),
            int(token_ops.insertion_loc_token),
            int(token_ops.replace_start_token),
            int(token_ops.replace_end_token),
            int(token_ops.copy_start_token),
            int(token_ops.copy_end_token),
            int(token_ops.copy_dest_token),
            int(token_ops.move_start_token),
            int(token_ops.move_end_token),
            int(token_ops.move_dest_token),
        }
        self.events = tuple(
            (int(token), int(position))
            for token, position in zip(gold_program_tokens, gold_program_positions)
            if int(token) in pointer_tokens
        )
        self.consumed = 0

    def __call__(self, row_index: int, token: int, _row, _write_index: int) -> int:
        if row_index != 0:
            raise ValueError("PointerOracleProvider supports one generated sample per evaluation row")
        if self.consumed >= len(self.events):
            raise OraclePointerStructureMismatch(
                f"Predicted unexpected pointer token {int(token)} after all gold pointers"
            )
        expected_token, position = self.events[self.consumed]
        if int(token) != expected_token:
            raise OraclePointerStructureMismatch(
                "Predicted pointer-token structure differs from gold at event "
                f"{self.consumed}: predicted={int(token)} expected={expected_token}"
            )
        self.consumed += 1
        return position

    @property
    def complete(self) -> bool:
        return self.consumed == len(self.events)


def build_oracle_span_prompt(
    source_tokens: list[int],
    target_tokens: list[int],
    token_ops: TokensOperations,
) -> OracleSpanPrompt | None:
    """Build the same target-derived changed envelope used by oracle-span FIM.

    Only the CES operation skeleton and source pointers are returned in the prompt.
    Target payload tokens occur strictly after the prompt boundary and therefore
    are never passed to the model.
    """

    source = [int(token) for token in source_tokens]
    target = [int(token) for token in target_tokens]
    source_start, source_end, target_start, target_end = minimal_changed_envelope(
        source,
        target,
    )
    if source_start == source_end and target_start == target_end:
        return None

    payload = np.asarray(target[target_start:target_end], dtype=np.int64)
    if source_start == source_end:
        operation = "insert"
        edit = Insert(source_start, payload)
        final_skeleton_token = int(token_ops.insertion_loc_token)
    elif target_start == target_end:
        operation = "delete"
        edit = Delete(source_start, source_end)
        final_skeleton_token = int(token_ops.deletion_end_token)
    else:
        operation = "replace"
        edit = Replace(source_start, source_end, payload)
        final_skeleton_token = int(token_ops.replace_end_token)

    sequence_edit = SequenceEdit(
        sequence=np.asarray(source, dtype=np.int64),
        edits=[edit],
    )
    serialised = token_ops.serialise(sequence_edit, indel_bos_pos=0)
    tokens = [int(token) for token in serialised.tokens]
    positions = [int(position) for position in serialised.positions]
    edit_start_indices = [
        index for index, token in enumerate(tokens)
        if token == int(token_ops.edit_start_token)
    ]
    if len(edit_start_indices) != 1:
        raise ValueError(f"Expected one edit-start token, found {len(edit_start_indices)}")
    edit_start = edit_start_indices[0]
    skeleton_indices = [
        index for index in range(edit_start + 1, len(tokens))
        if tokens[index] == final_skeleton_token
    ]
    if len(skeleton_indices) != 1:
        raise ValueError(
            f"Expected one final {operation} skeleton token, found {len(skeleton_indices)}"
        )
    prompt_end = skeleton_indices[0] + 1
    return OracleSpanPrompt(
        prompt_tokens=tuple(tokens[:prompt_end]),
        prompt_positions=tuple(positions[:prompt_end]),
        sequence_edit=sequence_edit,
        operation=operation,
        source_span=(source_start, source_end),
        target_span=(target_start, target_end),
        target_payload_tokens=tuple(int(token) for token in payload),
    )
