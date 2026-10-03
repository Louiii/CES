"""Core on-policy RL utilities for Causal Edit System (CES) programs.

This module intentionally contains no dataset or checkpoint policy.  It joins
the two CES action spaces (vocabulary tokens and source pointers), reconstructs
the exact constrained decoding distribution with gradients, and provides the
small RLOO/Dr.GRPO-style objective used by the training entry point.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import difflib
from typing import Sequence

import numpy as np
import torch
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from nanochat.causal_serialisation_extended import (
    Copy,
    Delete,
    Insert,
    Move,
    Replace,
    SerialisedData,
)
from nanochat.engine_extended import (
    EditController,
    PolicyRollout,
    selected_action_logprobs,
)


@dataclass(frozen=True)
class ReplayedPolicyLogprobs:
    """Differentiable probabilities for one sampled CES program."""

    token_logprobs: tuple[torch.Tensor, ...]
    pointer_logprobs: tuple[torch.Tensor, ...]

    @property
    def decision_count(self) -> int:
        return len(self.token_logprobs) + len(self.pointer_logprobs)

    @property
    def total(self) -> torch.Tensor:
        values = (*self.token_logprobs, *self.pointer_logprobs)
        if not values:
            raise ValueError("A policy rollout must contain at least one sampled decision")
        return torch.stack(values).sum()


@dataclass(frozen=True)
class RecoveryReward:
    """Terminal reward and diagnostics for target-aware corruption recovery."""

    reward: float
    valid: bool
    exact_target: bool
    source_distance: int
    candidate_distance: int | None
    edit_distance: int | None
    resolved_tokens: tuple[int, ...] | None
    likelihood_delta: float | None = None
    token_improvement: float | None = None
    character_improvement: float | None = None
    operation_local_improvement: float | None = None
    provenance_precision: float | None = None
    provenance_recall: float | None = None
    unmatched_program_ops: int | None = None
    verifier_pass_rate: float | None = None
    outside_edit_fraction: float | None = None
    error: str | None = None


def _model_device(model) -> torch.device:
    if hasattr(model, "get_device"):
        return torch.device(model.get_device())
    return next(model.parameters()).device


def replay_policy_logprobs(
    model,
    tokenizer,
    rollout: PolicyRollout,
) -> ReplayedPolicyLogprobs:
    """Teacher-force a rollout and reconstruct its constrained policy log-p.

    Token and pointer masks are rebuilt by replaying the same edit FSM.  The
    function remains differentiable with respect to both model heads; the
    rollout-time floating-point log-probabilities are used only for tests and
    diagnostics, never in the gradient path.
    """

    if rollout.temperature <= 0 or rollout.pointer_temperature <= 0:
        raise ValueError("RL replay requires positive token and pointer temperatures")
    if len(rollout.tokens) != len(rollout.pointers):
        raise ValueError("Rollout token and pointer streams must have equal length")
    if not 1 < rollout.prompt_length <= len(rollout.tokens):
        raise ValueError("Invalid rollout prompt length")

    expected_indices = list(range(rollout.prompt_length, len(rollout.tokens)))
    action_indices = [action.absolute_index for action in rollout.actions]
    if action_indices != expected_indices:
        raise ValueError(
            "Rollout actions must cover every generated token in stream order: "
            f"expected {expected_indices}, got {action_indices}"
        )

    device = _model_device(model)
    inputs = torch.tensor(
        [rollout.tokens[:-1]], dtype=torch.long, device=device
    )
    input_pointers = torch.tensor(
        [rollout.pointers[:-1]], dtype=torch.long, device=device
    )
    pointer_actions = [
        action for action in rollout.actions if action.pointer_sampled
    ]
    position_queries = None
    if pointer_actions:
        position_queries = torch.tensor(
            [[action.absolute_index - 1 for action in pointer_actions]],
            dtype=torch.long,
            device=device,
        )
        token_logits, pointer_logits = model(
            inputs,
            tix=input_pointers,
            return_pos_logits=True,
            position_queries=position_queries,
        )
    else:
        token_logits = model(inputs, tix=input_pointers)
        pointer_logits = None

    controller = EditController(tokenizer)
    controller.reset(
        1,
        max_payload_tokens=rollout.max_payload_tokens,
        max_edit_ops=rollout.max_edit_ops,
    )
    controller.prime_from_prompt(
        list(rollout.tokens[: rollout.prompt_length]),
        list(rollout.pointers[: rollout.prompt_length]),
        1,
    )

    token_logprobs: list[torch.Tensor] = []
    pointer_logprobs: list[torch.Tensor] = []
    pointer_cursor = 0
    for action in rollout.actions:
        query_index = action.absolute_index - 1
        action_token_logits = token_logits[0, query_index].clone()[None, :]
        forced_mask = torch.tensor(
            [not action.token_sampled], dtype=torch.bool, device=device
        )
        controller.apply_mask_(action_token_logits, forced_mask=forced_mask)
        controller.apply_closure_bias_(
            action_token_logits,
            rollout.closure_bias,
            min_payload_tokens=rollout.closure_bias_min_payload_tokens,
            forced_mask=forced_mask,
        )
        if rollout.no_edits:
            controller.ban_edit_tokens_(action_token_logits)

        if action.token_sampled:
            selected_token = torch.tensor(
                [[action.token_id]], dtype=torch.long, device=device
            )
            token_logprobs.append(selected_action_logprobs(
                action_token_logits,
                selected_token,
                rollout.temperature,
                rollout.top_k,
                rollout.top_p,
            )[0, 0])

        pointer_logits_row = None
        if action.pointer_sampled:
            assert pointer_logits is not None
            pointer_logits_row = pointer_logits[0, pointer_cursor]
            pointer_cursor += 1
            valid = controller.valid_ptr_range(
                controller.rows[0],
                action.token_id,
                action.absolute_index,
                device,
            )
            valid_tuple = tuple(int(value) for value in valid.tolist())
            if valid_tuple != action.valid_pointer_indices:
                raise ValueError(
                    f"Pointer support changed at action {action.absolute_index}: "
                    f"rollout={action.valid_pointer_indices}, replay={valid_tuple}"
                )
            if not valid_tuple:
                raise ValueError("Sampled pointer action has empty legal support")
            masked_pointer_logits = torch.full_like(
                pointer_logits_row, float("-inf")
            )
            masked_pointer_logits[list(valid_tuple)] = pointer_logits_row[list(valid_tuple)]
            selected_pointer = torch.tensor(
                [[action.pointer]], dtype=torch.long, device=device
            )
            pointer_logprobs.append(selected_action_logprobs(
                masked_pointer_logits[None, :],
                selected_pointer,
                rollout.pointer_temperature,
                rollout.top_k,
                rollout.top_p,
            )[0, 0])

        controller.step(
            i=0,
            tok=action.token_id,
            write_ix=action.absolute_index,
            position_logits_row=pointer_logits_row,
            rng=None,
            temperature=rollout.pointer_temperature,
            top_k=rollout.top_k,
            top_p=rollout.top_p,
            pointer_override=(
                action.pointer
                if action.token_id in controller.pointer_tokens
                else None
            ),
        )

    if pointer_cursor != len(pointer_actions):
        raise AssertionError("Not all pointer action logits were consumed")
    return ReplayedPolicyLogprobs(
        token_logprobs=tuple(token_logprobs),
        pointer_logprobs=tuple(pointer_logprobs),
    )


def _edit_signature(edit) -> tuple:
    """Canonical, hashable signature for one CES edit operation."""

    if isinstance(edit, dict):
        op = str(edit.get("op", ""))
        if op == "insert":
            return (op, int(edit["pos"]), tuple(map(int, edit.get("content", ()))))
        if op == "delete":
            return (op, int(edit["start"]), int(edit["end"]))
        if op == "replace":
            return (
                op,
                int(edit["start"]),
                int(edit["end"]),
                tuple(map(int, edit.get("content", ()))),
            )
        if op in {"copy", "move"}:
            return (
                op,
                int(edit["start"]),
                int(edit["end"]),
                int(edit["dest"]),
            )
        raise ValueError(f"Unsupported gold edit operation: {op!r}")
    if isinstance(edit, Insert):
        return ("insert", int(edit.pos), tuple(map(int, edit.content)))
    if isinstance(edit, Delete):
        return ("delete", int(edit.start), int(edit.end))
    if isinstance(edit, Replace):
        return (
            "replace", int(edit.start), int(edit.end), tuple(map(int, edit.content)),
        )
    if isinstance(edit, Copy):
        return ("copy", int(edit.start), int(edit.end), int(edit.dest))
    if isinstance(edit, Move):
        return ("move", int(edit.start), int(edit.end), int(edit.dest))
    raise TypeError(f"Unsupported sampled edit operation: {type(edit).__name__}")


def provenance_program_reward(
    rollout: PolicyRollout,
    token_ops,
    source_tokens: Sequence[int],
    target_tokens: Sequence[int],
    *,
    gold_edits: Sequence[dict],
    invalid_reward: float = -1.0,
    exact_bonus: float = 0.25,
    extra_op_penalty: float = 0.25,
    corrupt_noop_penalty: float = 0.0,
    negative_reward_scale: float = 1.0,
) -> RecoveryReward:
    """Verify a sampled repair against the known corruption program.

    This reward contains no sequence-distance term. Credit is the recall of
    exact gold repair operations, with a penalty for sampled operations that do
    not match provenance. An output-level exact bonus still recognizes an
    alternative program that happens to reconstruct the clean target.
    """

    if not gold_edits:
        raise ValueError("Provenance reward requires at least one gold edit")
    if extra_op_penalty < 0 or corrupt_noop_penalty < 0:
        raise ValueError("Provenance penalties must be non-negative")
    if negative_reward_scale < 1:
        raise ValueError("negative_reward_scale must be at least one")

    source = tuple(map(int, source_tokens))
    target = tuple(map(int, target_tokens))
    serialised = SerialisedData(
        tokens=np.asarray(rollout.tokens, dtype=np.int64),
        positions=np.asarray(rollout.pointers, dtype=np.int64),
    )
    try:
        parsed = token_ops.deserialise(serialised)
        resolved = token_ops.resolve(serialised)
        candidate = tuple(int(token) for token in np.asarray(resolved).tolist())
        predicted = Counter(_edit_signature(edit) for edit in parsed.edits)
        gold = Counter(_edit_signature(edit) for edit in gold_edits)
    except Exception as exc:
        return RecoveryReward(
            reward=float(invalid_reward),
            valid=False,
            exact_target=False,
            source_distance=0,
            candidate_distance=None,
            edit_distance=None,
            resolved_tokens=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    matched = sum((predicted & gold).values())
    predicted_count = sum(predicted.values())
    gold_count = sum(gold.values())
    unmatched = predicted_count - matched
    recall = matched / gold_count
    precision = matched / predicted_count if predicted_count else 0.0
    exact = candidate == target
    reward = recall + exact_bonus * float(exact)
    if predicted_count:
        reward -= extra_op_penalty * (unmatched / predicted_count)
    if source != target and candidate == source:
        reward -= corrupt_noop_penalty
    if reward < 0:
        reward *= negative_reward_scale
    return RecoveryReward(
        reward=float(reward),
        valid=True,
        exact_target=exact,
        source_distance=0,
        candidate_distance=None,
        edit_distance=None,
        resolved_tokens=candidate,
        provenance_precision=float(precision),
        provenance_recall=float(recall),
        unmatched_program_ops=int(unmatched),
    )


def _changed_opcodes(source: str, target: str) -> list[tuple[str, int, int, int, int]]:
    return [
        opcode
        for opcode in difflib.SequenceMatcher(
            a=source, b=target, autojunk=False
        ).get_opcodes()
        if opcode[0] != "equal"
    ]


def _span_overlaps_gold(
    start: int, end: int, gold: Sequence[tuple[str, int, int, int, int]]
) -> bool:
    for _tag, gold_start, gold_end, _j1, _j2 in gold:
        if start == end and gold_start == gold_end:
            if start == gold_start:
                return True
        elif start == end:
            if gold_start <= start <= gold_end:
                return True
        elif gold_start == gold_end:
            if start <= gold_start <= end:
                return True
        elif max(start, gold_start) < min(end, gold_end):
            return True
    return False


def local_repair_verifier_reward(
    rollout: PolicyRollout,
    token_ops,
    source_tokens: Sequence[int],
    target_tokens: Sequence[int],
    *,
    invalid_reward: float = -1.0,
    exact_bonus: float = 0.25,
    outside_edit_penalty: float = 0.25,
    anchor_chars: int = 12,
    gold_edits: Sequence[dict] = (),
    corrupt_noop_penalty: float = 0.0,
    negative_reward_scale: float = 1.0,
) -> RecoveryReward:
    """Distance-free verifier for exact repairs inside known local regions."""

    del gold_edits  # Regions are derived from source/target text, not edit provenance.
    if anchor_chars < 1:
        raise ValueError("anchor_chars must be positive")
    if outside_edit_penalty < 0 or corrupt_noop_penalty < 0:
        raise ValueError("verifier penalties must be non-negative")
    if negative_reward_scale < 1:
        raise ValueError("negative_reward_scale must be at least one")
    source = tuple(map(int, source_tokens))
    target = tuple(map(int, target_tokens))
    try:
        resolved = token_ops.resolve(SerialisedData(
            tokens=np.asarray(rollout.tokens, dtype=np.int64),
            positions=np.asarray(rollout.pointers, dtype=np.int64),
        ))
        candidate = tuple(int(token) for token in np.asarray(resolved).tolist())
        source_text = token_ops.tokeniser.decode(list(source))
        target_text = token_ops.tokeniser.decode(list(target))
        candidate_text = token_ops.tokeniser.decode(list(candidate))
    except Exception as exc:
        return RecoveryReward(
            reward=float(invalid_reward), valid=False, exact_target=False,
            source_distance=0, candidate_distance=None, edit_distance=None,
            resolved_tokens=None, error=f"{type(exc).__name__}: {exc}",
        )
    gold = _changed_opcodes(source_text, target_text)
    if not gold:
        passed_rate = float(candidate_text == source_text)
    else:
        passed = 0
        for _tag, i1, i2, j1, j2 in gold:
            left = source_text[max(0, i1 - anchor_chars):i1]
            right = source_text[i2:i2 + anchor_chars]
            repaired = target_text[j1:j2]
            passed += int(f"{left}{repaired}{right}" in candidate_text)
        passed_rate = passed / len(gold)
    predicted = _changed_opcodes(source_text, candidate_text)
    outside = sum(
        not _span_overlaps_gold(i1, i2, gold)
        for _tag, i1, i2, _j1, _j2 in predicted
    )
    outside_fraction = outside / max(len(predicted), 1)
    exact = candidate == target
    reward = passed_rate + exact_bonus * float(exact)
    reward -= outside_edit_penalty * outside_fraction
    if source != target and candidate == source:
        reward -= corrupt_noop_penalty
    if reward < 0:
        reward *= negative_reward_scale
    return RecoveryReward(
        reward=float(reward), valid=True, exact_target=exact,
        source_distance=0, candidate_distance=None, edit_distance=None,
        resolved_tokens=candidate, verifier_pass_rate=float(passed_rate),
        outside_edit_fraction=float(outside_fraction),
    )


def target_aware_recovery_reward(
    rollout: PolicyRollout,
    token_ops,
    source_tokens: Sequence[int],
    target_tokens: Sequence[int],
    *,
    invalid_reward: float = -1.0,
    exact_bonus: float = 0.25,
    edit_penalty: float = 0.0,
    character_reward_weight: float = 0.0,
    operation_local_reward_weight: float = 0.0,
    gold_edits: Sequence[dict] = (),
    local_context_tokens: int = 8,
    corrupt_noop_penalty: float = 0.0,
    negative_reward_scale: float = 1.0,
) -> RecoveryReward:
    """Score a complete applied CES program against a known clean target."""

    if character_reward_weight < 0 or operation_local_reward_weight < 0:
        raise ValueError("Reward component weights must be non-negative")
    if character_reward_weight + operation_local_reward_weight > 1:
        raise ValueError("Character and operation-local weights must sum to <= 1")
    if local_context_tokens < 1:
        raise ValueError("local_context_tokens must be positive")
    if corrupt_noop_penalty < 0:
        raise ValueError("corrupt_noop_penalty must be non-negative")
    if negative_reward_scale < 1:
        raise ValueError("negative_reward_scale must be at least one")

    source = tuple(int(token) for token in source_tokens)
    target = tuple(int(token) for token in target_tokens)
    source_distance = int(Levenshtein.distance(source, target))
    try:
        resolved = token_ops.resolve(SerialisedData(
            tokens=np.asarray(rollout.tokens, dtype=np.int64),
            positions=np.asarray(rollout.pointers, dtype=np.int64),
        ))
        candidate = tuple(int(token) for token in np.asarray(resolved).tolist())
    except Exception as exc:
        return RecoveryReward(
            reward=float(invalid_reward),
            valid=False,
            exact_target=False,
            source_distance=source_distance,
            candidate_distance=None,
            edit_distance=None,
            resolved_tokens=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    candidate_distance = int(Levenshtein.distance(candidate, target))
    edit_distance = int(Levenshtein.distance(source, candidate))
    denominator = max(source_distance, 1)
    directional_improvement = (source_distance - candidate_distance) / denominator
    directional_improvement = max(-1.0, min(1.0, directional_improvement))
    character_improvement = None
    if character_reward_weight > 0:
        source_text = token_ops.tokeniser.decode(list(source))
        target_text = token_ops.tokeniser.decode(list(target))
        candidate_text = token_ops.tokeniser.decode(list(candidate))
        source_char_distance = int(Levenshtein.distance(source_text, target_text))
        candidate_char_distance = int(Levenshtein.distance(candidate_text, target_text))
        character_improvement = (
            source_char_distance - candidate_char_distance
        ) / max(source_char_distance, 1)
        character_improvement = max(-1.0, min(1.0, character_improvement))

    operation_local_improvement = None
    if operation_local_reward_weight > 0:
        patterns = operation_local_patterns(
            source,
            target,
            gold_edits,
            context_tokens=local_context_tokens,
        )
        local_values = [
            _local_pattern_improvement(pattern, source, candidate)
            for pattern in patterns
            if pattern
        ]
        operation_local_improvement = (
            float(sum(local_values) / len(local_values))
            if local_values
            else directional_improvement
        )

    token_weight = 1.0 - character_reward_weight - operation_local_reward_weight
    blended_improvement = token_weight * directional_improvement
    if character_improvement is not None:
        blended_improvement += character_reward_weight * character_improvement
    if operation_local_improvement is not None:
        blended_improvement += (
            operation_local_reward_weight * operation_local_improvement
        )
    exact = candidate == target
    normalized_edit_mass = edit_distance / max(len(source), 1)
    reward = (
        blended_improvement
        + exact_bonus * float(exact)
        - edit_penalty * normalized_edit_mass
    )
    # On a known-corrupted prompt, immediate EDIT_END is valid but does not
    # attempt a repair. Give it a small explicit cost without applying the
    # same target-aware reward to genuinely clean examples (those belong in
    # the paired first-decision objective).
    if source != target and candidate == source:
        reward -= corrupt_noop_penalty
    if reward < 0:
        reward *= negative_reward_scale
    return RecoveryReward(
        reward=float(reward),
        valid=True,
        exact_target=exact,
        source_distance=source_distance,
        candidate_distance=candidate_distance,
        edit_distance=edit_distance,
        resolved_tokens=candidate,
        token_improvement=directional_improvement,
        character_improvement=character_improvement,
        operation_local_improvement=operation_local_improvement,
    )


def _find_subsequence(sequence: Sequence[int], content: Sequence[int]) -> list[int]:
    """Return every exact start index of ``content`` in ``sequence``."""

    content_tuple = tuple(int(token) for token in content)
    if not content_tuple:
        return []
    width = len(content_tuple)
    sequence_tuple = tuple(int(token) for token in sequence)
    return [
        index
        for index in range(max(len(sequence_tuple) - width + 1, 0))
        if sequence_tuple[index : index + width] == content_tuple
    ]


def operation_local_patterns(
    source_tokens: Sequence[int],
    target_tokens: Sequence[int],
    gold_edits: Sequence[dict],
    *,
    context_tokens: int = 8,
) -> tuple[tuple[int, ...], ...]:
    """Construct target-side local patterns for each known corruption operation.

    Patterns include both edited content and neighbouring context. Matching the
    content at an arbitrary location is therefore insufficient; the sampled
    repair must also restore its local boundary. Multiple CES scripts can still
    receive credit because only the applied sequence is inspected.
    """

    source = tuple(int(token) for token in source_tokens)
    target = tuple(int(token) for token in target_tokens)
    patterns: list[tuple[int, ...]] = []
    for edit in gold_edits:
        op = str(edit.get("op", ""))
        if op == "delete":
            start = int(edit["start"])
            end = int(edit["end"])
            expected = (
                source[max(0, start - context_tokens) : start]
                + source[end : min(len(source), end + context_tokens)]
            )
        elif op in {"insert", "replace"}:
            start = int(edit.get("pos", edit.get("start", 0)))
            end = int(edit.get("end", start))
            content = tuple(int(token) for token in edit.get("content", ()))
            expected = (
                source[max(0, start - context_tokens) : start]
                + content
                + source[end : min(len(source), end + context_tokens)]
            )
        elif op == "move":
            start = int(edit["start"])
            end = int(edit["end"])
            content = source[start:end]
            occurrences = _find_subsequence(target, content)
            if occurrences:
                dest_hint = int(edit.get("dest", occurrences[0]))
                location = min(occurrences, key=lambda value: abs(value - dest_hint))
                expected = target[
                    max(0, location - context_tokens) :
                    min(len(target), location + len(content) + context_tokens)
                ]
            else:
                expected = content
        else:
            continue
        if expected:
            patterns.append(tuple(expected))
    return tuple(patterns)


def _local_pattern_improvement(
    pattern: Sequence[int],
    source: Sequence[int],
    candidate: Sequence[int],
) -> float:
    """Directional improvement in best local sequence match, clipped to [-1, 1]."""

    baseline = float(fuzz.partial_ratio(tuple(pattern), tuple(source))) / 100.0
    achieved = float(fuzz.partial_ratio(tuple(pattern), tuple(candidate))) / 100.0
    if achieved >= baseline:
        value = (achieved - baseline) / max(1.0 - baseline, 1e-6)
    else:
        value = (achieved - baseline) / max(baseline, 1e-6)
    return max(-1.0, min(1.0, value))


def add_likelihood_auxiliary(
    target_reward: RecoveryReward,
    likelihood_reward: RecoveryReward,
    *,
    weight: float,
) -> RecoveryReward:
    """Add a weak frozen-LM preference to an otherwise target-aware reward."""

    if weight < 0:
        raise ValueError("Likelihood auxiliary weight must be non-negative")
    if not target_reward.valid:
        return target_reward
    if not likelihood_reward.valid:
        return replace(
            target_reward,
            likelihood_delta=likelihood_reward.likelihood_delta,
        )
    return replace(
        target_reward,
        reward=float(target_reward.reward + weight * likelihood_reward.reward),
        likelihood_delta=likelihood_reward.likelihood_delta,
    )


def target_blind_likelihood_reward(
    rollout: PolicyRollout,
    token_ops,
    source_tokens: Sequence[int],
    *,
    source_avg_logprob: float,
    candidate_avg_logprob: float | None,
    invalid_reward: float = -1.0,
    likelihood_scale: float = 0.25,
    edit_penalty: float = 0.0,
) -> RecoveryReward:
    """Reward a valid edit by frozen-LM likelihood gain, without a target.

    Sequence scores are length-normalized average token log-probabilities.
    The delta is scaled and clipped so rare likelihood outliers cannot dominate
    a rollout group. The scorer is supplied externally and must remain frozen.
    """

    if likelihood_scale <= 0:
        raise ValueError("likelihood_scale must be positive")
    source = tuple(int(token) for token in source_tokens)
    try:
        resolved = token_ops.resolve(SerialisedData(
            tokens=np.asarray(rollout.tokens, dtype=np.int64),
            positions=np.asarray(rollout.pointers, dtype=np.int64),
        ))
        candidate = tuple(int(token) for token in np.asarray(resolved).tolist())
    except Exception as exc:
        return RecoveryReward(
            reward=float(invalid_reward),
            valid=False,
            exact_target=False,
            source_distance=0,
            candidate_distance=None,
            edit_distance=None,
            resolved_tokens=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    if candidate_avg_logprob is None:
        return RecoveryReward(
            reward=float(invalid_reward),
            valid=False,
            exact_target=False,
            source_distance=0,
            candidate_distance=None,
            edit_distance=None,
            resolved_tokens=candidate,
            error="candidate sequence is not scoreable by the frozen LM",
        )
    likelihood_delta = float(candidate_avg_logprob - source_avg_logprob)
    scaled_delta = max(-1.0, min(1.0, likelihood_delta / likelihood_scale))
    edit_distance = int(Levenshtein.distance(source, candidate))
    normalized_edit_mass = edit_distance / max(len(source), 1)
    return RecoveryReward(
        reward=float(scaled_delta - edit_penalty * normalized_edit_mass),
        valid=True,
        exact_target=False,
        source_distance=0,
        candidate_distance=None,
        edit_distance=edit_distance,
        resolved_tokens=candidate,
        likelihood_delta=likelihood_delta,
    )


def group_advantages(
    rewards: torch.Tensor,
    *,
    method: str = "rloo",
) -> torch.Tensor:
    """Return unstandardized RLOO or centered Dr.GRPO-style advantages."""

    if rewards.ndim != 1:
        raise ValueError("Rewards must be a one-dimensional group")
    if method == "rloo":
        if rewards.numel() < 2:
            raise ValueError("RLOO requires at least two rollouts per prompt")
        return rewards - (rewards.sum() - rewards) / (rewards.numel() - 1)
    if method == "centered":
        return rewards - rewards.mean()
    raise ValueError(f"Unknown advantage method: {method!r}")


def reinforce_loss(
    replayed: Sequence[ReplayedPolicyLogprobs],
    advantages: torch.Tensor,
) -> torch.Tensor:
    """DAPO-style decision-normalized on-policy loss over token+pointer actions."""

    if len(replayed) != int(advantages.numel()):
        raise ValueError("Need one scalar advantage per rollout")
    decision_count = sum(item.decision_count for item in replayed)
    if decision_count == 0:
        raise ValueError("RL batch contains no sampled policy decisions")
    objective = sum(
        advantage.detach() * item.total
        for advantage, item in zip(advantages, replayed)
    ) / decision_count
    return -objective


def mix_supervised_replay_loss(
    rl_loss: torch.Tensor,
    supervised_loss: torch.Tensor,
    *,
    supervised_fraction: float = 0.25,
) -> torch.Tensor:
    """Mix an RL update with completion-only CES/clean-AR replay."""

    if not 0.0 <= supervised_fraction <= 1.0:
        raise ValueError("supervised_fraction must be in [0, 1]")
    return (
        (1.0 - supervised_fraction) * rl_loss
        + supervised_fraction * supervised_loss
    )
