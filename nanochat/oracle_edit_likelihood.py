"""Canonical human-edit programs and conditional CES teacher-forced likelihoods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from nanochat.causal_serialisation_extended import (
    Delete,
    Insert,
    Replace,
    SequenceEdit,
    TokensOperations,
    TrainingData,
    TrainingTarget,
    resolve,
)


def canonical_sequence_edit_from_tokens(
    source_tokens: list[int],
    target_tokens: list[int],
) -> SequenceEdit:
    """Return one deterministic minimum-token-Levenshtein CES program.

    RapidFuzz returns a deterministic minimum-cost alignment for unit-cost
    insertions, deletions, and replacements. Contiguous primitive operations
    are coalesced into the corresponding CES operation.
    """

    edits: list[Any] = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(
        source_tokens, target_tokens
    ).as_list():
        if tag == "equal":
            continue
        if tag == "insert":
            edits.append(
                Insert(i1, np.asarray(target_tokens[j1:j2], dtype=np.int64))
            )
        elif tag == "delete":
            edits.append(Delete(i1, i2))
        elif tag == "replace":
            edits.append(
                Replace(
                    i1,
                    i2,
                    np.asarray(target_tokens[j1:j2], dtype=np.int64),
                )
            )
        else:  # pragma: no cover - guarded by RapidFuzz's opcode contract
            raise ValueError(f"Unsupported alignment opcode: {tag}")
    sequence_edit = SequenceEdit(
        sequence=np.asarray(source_tokens, dtype=np.int64),
        edits=edits,
    )
    resolved = [int(token) for token in resolve(sequence_edit).tolist()]
    if resolved != [int(token) for token in target_tokens]:
        raise ValueError("Canonical edit program does not resolve to target")
    return sequence_edit


def canonical_sequence_edit_from_text(tokenizer, source: str, target: str) -> SequenceEdit:
    bos = int(tokenizer.get_bos_token_id())
    source_tokens = [bos, *[int(token) for token in tokenizer.encode(source)]]
    target_tokens = [bos, *[int(token) for token in tokenizer.encode(target)]]
    return canonical_sequence_edit_from_tokens(source_tokens, target_tokens)


def _replace_target(training_data: TrainingData, **changes) -> TrainingData:
    target = training_data.target
    values = {
        "tokens": target.tokens,
        "gathered_pos_target": target.gathered_pos_target,
        "token_loss_weights": target.token_loss_weights,
        "gather_indices": target.gather_indices,
        "gather_valid": target.gather_valid,
        "overflow": target.overflow,
        "perplexity_masks": target.perplexity_masks,
    }
    values.update(changes)
    return TrainingData(
        input_tokens=training_data.input_tokens,
        input_positions=training_data.input_positions,
        target=TrainingTarget(**values),
    )


def conditional_program_training_data(
    token_ops: TokensOperations,
    sequence_edit: SequenceEdit,
    *,
    position_buffer_size: int,
) -> TrainingData:
    """Prepare a CES row whose token loss covers only the edit program suffix."""

    serialised = token_ops.serialise(sequence_edit, indel_bos_pos=0)
    training_data = token_ops.prepare_training_example(
        serialised,
        pos_buffer=position_buffer_size,
    )
    edit_start = np.flatnonzero(serialised.tokens == token_ops.edit_start_token)
    if len(edit_start) != 1:
        raise ValueError(f"Expected one edit-start token, found {len(edit_start)}")

    # target[k] predicts serialised token k + 1.
    target_serial_indices = np.arange(1, len(serialised.tokens))
    # The deployment prompt already ends with EDIT_START. Completion-only SFT
    # therefore begins with the first program action after it, not with the
    # EDIT_START token itself.
    suffix_mask = target_serial_indices > int(edit_start[0])
    weights = np.asarray(training_data.target.token_loss_weights, dtype=bool)
    weights = weights & suffix_mask
    return _replace_target(training_data, token_loss_weights=weights)


def pad_training_data(training_data: TrainingData, length: int) -> TrainingData:
    current = len(training_data.input_tokens)
    if current > length:
        raise ValueError(f"Training row length {current} exceeds padded length {length}")
    pad = length - current
    return TrainingData(
        input_tokens=np.pad(training_data.input_tokens, (0, pad), constant_values=0),
        input_positions=np.pad(
            training_data.input_positions, (0, pad), constant_values=0
        ),
        target=TrainingTarget(
            tokens=np.pad(
                training_data.target.tokens, (0, pad), constant_values=-1
            ),
            gathered_pos_target=training_data.target.gathered_pos_target,
            token_loss_weights=np.pad(
                training_data.target.token_loss_weights,
                (0, pad),
                constant_values=False,
            ),
            gather_indices=training_data.target.gather_indices,
            gather_valid=training_data.target.gather_valid,
            overflow=training_data.target.overflow,
            perplexity_masks=None,
        ),
    )


@dataclass(frozen=True)
class ProgramLikelihood:
    token_nll: float
    pointer_nll: float
    token_decisions: int
    pointer_decisions: int
    serialized_input_tokens: int

    @property
    def total_nll(self) -> float:
        return self.token_nll + self.pointer_nll

    @property
    def decisions(self) -> int:
        return self.token_decisions + self.pointer_decisions

    @property
    def bits_per_decision(self) -> float | None:
        if self.decisions == 0:
            return None
        return self.total_nll / self.decisions / np.log(2.0)


@torch.inference_mode()
def score_conditional_program(
    model,
    training_data: TrainingData,
    *,
    device: torch.device,
    padded_length: int,
    autocast_ctx,
) -> ProgramLikelihood:
    row = pad_training_data(training_data, padded_length)

    def tensor(value, dtype):
        return torch.as_tensor(value, dtype=dtype, device=device)[None, :]

    batch = TrainingData(
        input_tokens=tensor(row.input_tokens, torch.long),
        input_positions=tensor(row.input_positions, torch.long),
        target=TrainingTarget(
            tokens=tensor(row.target.tokens, torch.long),
            gathered_pos_target=tensor(
                row.target.gathered_pos_target, torch.long
            ),
            token_loss_weights=tensor(
                row.target.token_loss_weights, torch.bool
            ),
            gather_indices=tensor(row.target.gather_indices, torch.long),
            gather_valid=tensor(row.target.gather_valid, torch.bool),
            overflow=torch.as_tensor(
                [bool(row.target.overflow)], dtype=torch.bool, device=device
            ),
            perplexity_masks=None,
        ),
    )
    with autocast_ctx:
        _loss, aux = model(
            batch.input_tokens,
            tix=batch.input_positions,
            targets=batch.target,
            loss_reduction="sum",
        )
    return ProgramLikelihood(
        token_nll=float(aux["token_loss"].item()),
        pointer_nll=float(aux["position_loss"].item()),
        token_decisions=int(np.asarray(training_data.target.token_loss_weights).sum()),
        pointer_decisions=int(np.asarray(training_data.target.gather_valid).sum()),
        serialized_input_tokens=len(training_data.input_tokens),
    )
