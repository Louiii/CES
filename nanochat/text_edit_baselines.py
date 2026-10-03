"""Shared serialization for full-regeneration and oracle-span FIM baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nanochat.causal_serialisation_extended import TrainingData, TrainingTarget


REGEN_INSTRUCTION = "\nRewrite the corrupted text as clean text.\nCorrupted text:\n"
REGEN_OUTPUT_CUE = "\nClean text:\n"
FIM_PREFIX_CUE = "\n<fim-prefix>\n"
FIM_SUFFIX_CUE = "\n<fim-suffix>\n"
FIM_MIDDLE_CUE = "\n<fim-middle>\n"
FIM_LOCATE_START_CUE = "\n<fim-locate-start>\n"
FIM_LOCATE_END_CUE = "\n<fim-locate-end>\n"


@dataclass(frozen=True)
class BaselineExample:
    prompt_tokens: list[int]
    completion_tokens: list[int]
    source_prefix: list[int]
    source_suffix: list[int]
    gold_middle: list[int]
    position_query_indices: tuple[int, ...] = ()
    position_targets: tuple[int, ...] = ()
    source_span: tuple[int, int] | None = None


def build_predicted_fim_locator(
    source_tokens: list[int],
    tokenizer,
) -> tuple[list[int], tuple[int, int]]:
    """Return the source-only locator prefix and its two pointer-query indices."""

    source = [int(token) for token in source_tokens]
    if not source or source[0] != tokenizer.get_bos_token_id():
        raise ValueError("Predicted-span FIM source must begin with BOS")
    locator = [*source, *tokenizer.encode(FIM_LOCATE_START_CUE)]
    start_query = len(locator) - 1
    locator.extend(tokenizer.encode(FIM_LOCATE_END_CUE))
    end_query = len(locator) - 1
    return locator, (start_query, end_query)


def build_predicted_fim_infill_prompt(
    source_tokens: list[int],
    start: int,
    end: int,
    tokenizer,
) -> tuple[list[int], list[int], list[int]]:
    """Build the append-only FIM request for a predicted source-token span."""

    source = [int(token) for token in source_tokens]
    if not source or source[0] != tokenizer.get_bos_token_id():
        raise ValueError("Predicted-span FIM source must begin with BOS")
    if not 1 <= start <= end <= len(source):
        raise ValueError(
            f"Invalid predicted FIM span [{start}, {end}) for source length {len(source)}"
        )
    prefix = source[:start]
    suffix = source[end:]
    appended_prompt = [
        *tokenizer.encode(FIM_PREFIX_CUE),
        *prefix[1:],
        *tokenizer.encode(FIM_SUFFIX_CUE),
        *suffix,
        *tokenizer.encode(FIM_MIDDLE_CUE),
    ]
    return appended_prompt, prefix, suffix


def minimal_changed_envelope(
    source_tokens: list[int],
    target_tokens: list[int],
) -> tuple[int, int, int, int]:
    """Return the unmatched envelope after anchoring shared outer context.

    Anchoring the common prefix and suffix is important for packed examples, which
    may contain repeated BOS tokens. A global sequence alignment can otherwise
    match the leading BOS to a later occurrence and incorrectly propose a span
    beginning at position zero.
    """

    source_length = len(source_tokens)
    target_length = len(target_tokens)
    common_prefix = 0
    while (
        common_prefix < source_length
        and common_prefix < target_length
        and source_tokens[common_prefix] == target_tokens[common_prefix]
    ):
        common_prefix += 1

    if common_prefix == source_length and common_prefix == target_length:
        return source_length, source_length, target_length, target_length

    source_end = source_length
    target_end = target_length
    while (
        source_end > common_prefix
        and target_end > common_prefix
        and source_tokens[source_end - 1] == target_tokens[target_end - 1]
    ):
        source_end -= 1
        target_end -= 1

    return common_prefix, source_end, common_prefix, target_end


def build_baseline_example(
    mode: str,
    source_tokens: list[int],
    target_tokens: list[int],
    tokenizer,
) -> BaselineExample:
    """Build a source-only regeneration prompt or information-advantaged FIM prompt."""

    if mode not in {"regeneration", "fim", "fim_predicted"}:
        raise ValueError(f"Unknown text-edit baseline mode: {mode}")
    bos = tokenizer.get_bos_token_id()
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    source = [int(token) for token in source_tokens]
    target = [int(token) for token in target_tokens]
    if not source or source[0] != bos or not target or target[0] != bos:
        raise ValueError("Baseline source and target must begin with BOS")

    if mode == "regeneration":
        prompt = [
            bos,
            *tokenizer.encode(REGEN_INSTRUCTION),
            *source[1:],
            *tokenizer.encode(REGEN_OUTPUT_CUE),
        ]
        return BaselineExample(
            prompt_tokens=prompt,
            completion_tokens=[*target[1:], assistant_end],
            source_prefix=[bos],
            source_suffix=[],
            gold_middle=target[1:],
        )

    source_start, source_end, target_start, target_end = minimal_changed_envelope(source, target)
    source_prefix = source[:source_start]
    source_suffix = source[source_end:]
    gold_middle = target[target_start:target_end]
    if mode == "fim_predicted":
        locator, query_indices = build_predicted_fim_locator(source, tokenizer)
        infill_prompt, predicted_prefix, predicted_suffix = build_predicted_fim_infill_prompt(
            source,
            source_start,
            source_end,
            tokenizer,
        )
        # Boundary k is represented by a pointer to source token k-1. This
        # covers insertion (start == end), deletion, replacement, and the
        # identity/end-of-source case without requiring an out-of-range class.
        position_targets = (source_start - 1, source_end - 1)
        return BaselineExample(
            prompt_tokens=[*locator, *infill_prompt],
            completion_tokens=[*gold_middle, assistant_end],
            source_prefix=predicted_prefix,
            source_suffix=predicted_suffix,
            gold_middle=gold_middle,
            position_query_indices=query_indices,
            position_targets=position_targets,
            source_span=(source_start, source_end),
        )
    prompt = [
        bos,
        *tokenizer.encode(FIM_PREFIX_CUE),
        *source_prefix[1:],
        *tokenizer.encode(FIM_SUFFIX_CUE),
        *source_suffix,
        *tokenizer.encode(FIM_MIDDLE_CUE),
    ]
    return BaselineExample(
        prompt_tokens=prompt,
        completion_tokens=[*gold_middle, assistant_end],
        source_prefix=source_prefix,
        source_suffix=source_suffix,
        gold_middle=gold_middle,
        source_span=(source_start, source_end),
    )


def masked_baseline_training_data(
    example: BaselineExample,
    *,
    token_budget: int,
    pos_buffer_size: int,
    terminal_weight: float = 1.0,
) -> TrainingData | None:
    """Apply language-model loss only to completion and terminator tokens."""

    row = [*example.prompt_tokens, *example.completion_tokens]
    if len(row) > token_budget + 1 or len(example.prompt_tokens) < 1:
        return None
    inputs = np.zeros(token_budget, dtype=np.int64)
    targets = np.full(token_budget, -1, dtype=np.int64)
    weights = np.zeros(token_budget, dtype=np.float32)
    input_values = np.asarray(row[:-1], dtype=np.int64)
    target_values = np.asarray(row[1:], dtype=np.int64)
    inputs[:len(input_values)] = input_values
    targets[:len(target_values)] = target_values
    completion_start = len(example.prompt_tokens) - 1
    weights[completion_start:completion_start + len(example.completion_tokens)] = 1.0
    effective_terminal_weight = (
        max(len(example.completion_tokens) - 1, 1)
        if terminal_weight < 0
        else terminal_weight
    )
    if effective_terminal_weight <= 0:
        raise ValueError("terminal_weight must be positive or negative for automatic weighting")
    weights[completion_start + len(example.completion_tokens) - 1] = effective_terminal_weight
    gathered_pos_target = np.full(pos_buffer_size, -1, dtype=np.int64)
    gather_indices = np.zeros(pos_buffer_size, dtype=np.int64)
    gather_valid = np.zeros(pos_buffer_size, dtype=bool)
    if example.position_targets:
        if len(example.position_targets) != len(example.position_query_indices):
            raise ValueError("Position targets and query indices must have equal length")
        if len(example.position_targets) > pos_buffer_size:
            raise ValueError("Position buffer is too small for predicted-span FIM queries")
        count = len(example.position_targets)
        gathered_pos_target[:count] = np.asarray(example.position_targets, dtype=np.int64)
        gather_indices[:count] = np.asarray(example.position_query_indices, dtype=np.int64)
        gather_valid[:count] = True
    return TrainingData(
        input_tokens=inputs,
        input_positions=np.arange(token_budget, dtype=np.int64),
        target=TrainingTarget(
            tokens=targets,
            gathered_pos_target=gathered_pos_target,
            token_loss_weights=weights,
            gather_indices=gather_indices,
            gather_valid=gather_valid,
            overflow=False,
            perplexity_masks=None,
        ),
    )
