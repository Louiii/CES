"""Live dataloading for source-anchored edit candidate datasets.

The generated edit parquet files are intentionally sparse: they store candidate
edit scripts keyed by source shard and row. This module joins those candidates
with the raw nanochat corpus at load time, samples a valid subset, converts
character-coordinate edits to token-coordinate edits, and serialises them with
the extended causal edit format.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch

from nanochat.causal_serialisation_extended import (
    Copy,
    Delete,
    Insert,
    Move,
    Replace,
    SequenceEdit,
    TokensOperations,
    TrainingData,
    TrainingTarget,
    _edit_key,
    _lower_moves,
    _validate_edits,
    complement,
    resolve,
)
from nanochat.causal_serialisation import (
    ChunkReservoir,
    InsertTypeProbs,
    gamma_shaped_partition,
    sample_insert,
)
from nanochat.common import get_dist_info
from nanochat.edit_dataset import sequence_edit_from_json
from nanochat.edit_dsl import sequence_edit_to_ar_dsl_tokens
from nanochat.flat_relative_dsl import sequence_edit_to_flat_relative_tokens
from nanochat.text_edit_baselines import build_baseline_example, masked_baseline_training_data


@dataclass(frozen=True)
class EditSamplingConfig:
    """Controls the live edit training distribution."""

    raw_data_dir: str | None = None
    generated_edit_paths: tuple[str, ...] = ()
    copy_scan_dir: str | None = None
    direction_probs: dict[str, float] = field(
        default_factory=lambda: {"inverse": 0.85, "forward": 0.15}
    )
    op_probs: dict[str, float] = field(
        default_factory=lambda: {
            "insert": 0.22,
            "delete": 0.22,
            "replace": 0.34,
            "move": 0.12,
            "copy": 0.10,
        }
    )
    max_edits_per_example: int = 4
    max_source_tokens: int | None = None
    max_serialised_tokens: int | None = None
    min_copy_chars: int = 20
    copy_length_alpha: float = 0.75
    copy_length_acceptance: dict[int, float] | None = None
    random_corruption_prob_when_no_candidates: float = 0.5
    random_plain_corruption_prob_when_no_candidates: float = 0.0
    noop_edit_prob_when_no_candidates: float = 0.0
    plain_corruption_only: bool = False
    random_no_del_mean: float = 128.0
    random_del_mean_len: float = 8.0
    random_no_ins_mean: float = 128.0
    random_ins_mean_len: float = 8.0
    random_replace_prob: float = 0.32
    random_delete_prob: float | None = 0.315
    random_insert_prob: float | None = 0.315
    random_move_prob: float = 0.05
    random_source_token_reserve: int = 128
    random_source_token_reserve_max: int | None = None
    random_source_length_jitter_prob: float = 0.0
    stream_clean_row_prob: float = 0.0
    stream_partition_row_prob: float = 0.0
    partition_min_size: int = 48
    partition_gamma_shape: float = 2.0
    partition_slice_mean_size: float = 360.0
    priority_candidate_lane: bool = True
    bos_aligned_row_packing: bool = True
    post_edit_clean_suffix: bool = True
    terminal_clean_suffix_min: int = 64
    terminal_edit_end_prob: float = 0.05
    random_insertion_type_probabilities: InsertTypeProbs = InsertTypeProbs(
        buffer=0.6,
        myself=0.1,
        uniform=0.2,
        marginal=0.1,
    )
    corrupted_indel_bos_random_prob: float = 0.7
    corrupted_indel_bos_start_prob: float = 0.3
    clean_indel_bos_random_prob: float = 0.5
    clean_indel_bos_absent_prob: float = 0.3
    clean_indel_bos_start_prob: float = 0.2
    seed: int = 1234


def deterministic_loader_mixture_with_state(
    primary_loader,
    replay_loader,
    replay_probability: float,
    resume_state_dict: dict[str, Any] | None = None,
):
    """Interleave two stateful loaders with a deterministic low-discrepancy schedule.

    The returned state preserves both child loader positions and the mixture
    accumulator, so checkpoint resumption does not silently change the replay
    schedule. At probability 0.25 the schedule is primary, primary, primary,
    replay, repeated; other probabilities are distributed by the same
    accumulator without sampling noise.
    """

    probability = float(replay_probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("replay_probability must be strictly between 0 and 1")

    resume = resume_state_dict or {}
    accumulator = float(resume.get("mixture_replay_accumulator", 0.0))
    mixture_index = int(resume.get("mixture_index", 0))
    primary_batches = int(resume.get("mixture_primary_batches", 0))
    replay_batches = int(resume.get("mixture_replay_batches", 0))
    primary_state = resume.get("primary_state")
    replay_state = resume.get("replay_state")

    while True:
        accumulator += probability
        use_replay = accumulator >= 1.0 - 1e-12
        if use_replay:
            accumulator -= 1.0
            batch, replay_state = next(replay_loader)
            selected_state = replay_state
            replay_batches += 1
        else:
            batch, primary_state = next(primary_loader)
            selected_state = primary_state
            primary_batches += 1
        mixture_index += 1

        state = dict(selected_state)
        state.update(
            {
                "mixture_index": mixture_index,
                "mixture_replay_accumulator": accumulator,
                "mixture_primary_batches": primary_batches,
                "mixture_replay_batches": replay_batches,
                "mixture_selected": "replay" if use_replay else "primary",
                "primary_state": primary_state,
                "replay_state": replay_state,
            }
        )
        yield batch, state


@dataclass(frozen=True)
class _EditCandidate:
    source_text: str
    forward_json: str
    inverse_json: str
    ops: tuple[str, ...]
    forward_token_json: str | None = None
    inverse_token_json: str | None = None


@dataclass(frozen=True)
class _CopyCandidate:
    repeated_text: str
    occurrences: tuple[int, ...]
    length_chars: int


@dataclass
class _PreparedDoc:
    shard_id: str
    row_index: int
    source_text: str
    tokens: list[int]
    token_edit: SequenceEdit | None


@dataclass
class _RawStreamDoc:
    shard_id: str
    row_index: int
    raw_text: str
    clean_tokens: list[int]
    offset: int = 0
    prepared: _PreparedDoc | None = None
    prepared_checked: bool = False
    priority_candidate_queued: bool = False


def _normalise_probs(probs: dict[str, float]) -> tuple[list[str], np.ndarray]:
    items = [(k, float(v)) for k, v in probs.items() if float(v) > 0]
    if not items:
        raise ValueError("At least one sampling probability must be positive.")
    keys = [k for k, _ in items]
    vals = np.array([v for _, v in items], dtype=np.float64)
    vals /= vals.sum()
    return keys, vals


def _iter_parquet_inputs(paths: Iterable[str]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            yield from sorted(path.glob("*.parquet"))
        else:
            yield path


def load_generated_edit_index(paths: Iterable[str]) -> dict[tuple[str, int], list[_EditCandidate]]:
    """Load generated LLM edit candidates keyed by ``(source_shard_id, row_index)``."""

    index: dict[tuple[str, int], list[_EditCandidate]] = {}
    columns = [
        "row_index",
        "source_shard_id",
        "source_text",
        "forward_edits_json",
        "inverse_edits_json",
        "has_valid_edit_script",
        "forward_edit_counts_json",
        "forward_token_edits_json",
        "inverse_token_edits_json",
    ]
    for path in _iter_parquet_inputs(paths):
        table = pq.read_table(path, columns=[c for c in columns if c in pq.read_schema(path).names])
        for row in table.to_pylist():
            if not row.get("has_valid_edit_script", True):
                continue
            shard_id = row.get("source_shard_id")
            if not shard_id:
                source_path = row.get("source_parquet_path")
                shard_id = Path(source_path).stem if source_path else None
            if not shard_id:
                continue
            ops = []
            counts_json = row.get("forward_edit_counts_json")
            if counts_json:
                counts = json.loads(counts_json)
                ops = [k for k, v in counts.items() for _ in range(int(v))]
            index.setdefault((str(shard_id), int(row["row_index"])), []).append(
                _EditCandidate(
                    source_text=row["source_text"],
                    forward_json=row["forward_edits_json"],
                    inverse_json=row["inverse_edits_json"],
                    ops=tuple(ops),
                    forward_token_json=row.get("forward_token_edits_json"),
                    inverse_token_json=row.get("inverse_token_edits_json"),
                )
            )
    return index


def load_copy_scan_index(copy_scan_dir: str | None) -> dict[tuple[str, int], list[_CopyCandidate]]:
    """Load repeated-span candidates from ``copy_scan`` JSONL files."""

    if copy_scan_dir is None:
        return {}
    index: dict[tuple[str, int], list[_CopyCandidate]] = {}
    for path in sorted(Path(copy_scan_dir).glob("shard_*_examples.jsonl")):
        shard_id = path.name.removesuffix("_examples.jsonl")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                occurrences = tuple(int(x) for x in row.get("occurrences", ()))
                if len(occurrences) < 2:
                    continue
                repeated_text = str(row.get("repeated_text", ""))
                if not repeated_text:
                    continue
                index.setdefault((shard_id, int(row["row_index"])), []).append(
                    _CopyCandidate(
                        repeated_text=repeated_text,
                        occurrences=occurrences,
                        length_chars=int(row.get("length_chars", len(repeated_text))),
                    )
                )
    return index


def _op_name(edit: Any) -> str:
    if isinstance(edit, Insert):
        return "insert"
    if isinstance(edit, Delete):
        return "delete"
    if isinstance(edit, Replace):
        return "replace"
    if isinstance(edit, Move):
        return "move"
    if isinstance(edit, Copy):
        return "copy"
    raise TypeError(f"Unsupported edit type: {type(edit)!r}")


def _edit_span_len(edit: Any) -> int:
    if isinstance(edit, Insert):
        return len(edit.content)
    return int(edit.end) - int(edit.start)


def script_edit_mass(edit: Any) -> int:
    """Token mass explicitly touched or written by an edit operation.

    This avoids sequence alignment and assumes the supplied script is the
    intended minimal description. Replace counts the removed span and written
    payload; Copy and Move count their source span once.
    """
    if isinstance(edit, Insert):
        return len(edit.content)
    if isinstance(edit, Delete):
        return int(edit.end) - int(edit.start)
    if isinstance(edit, Replace):
        return int(edit.end) - int(edit.start) + len(edit.content)
    if isinstance(edit, (Copy, Move)):
        return int(edit.end) - int(edit.start)
    raise TypeError(f"Unsupported edit type: {type(edit)!r}")


def sequence_edit_mass(seq_edit: SequenceEdit) -> int:
    """Return the raw script edit mass for a complete example."""
    return sum(script_edit_mass(edit) for edit in seq_edit.edits)


def _sample_valid_subset(
    seq_edit: SequenceEdit,
    *,
    op_probs: dict[str, float],
    max_edits: int,
    rng: np.random.Generator,
    required_op: str | None = None,
) -> SequenceEdit | None:
    edits = list(seq_edit.edits)
    if not edits:
        return None

    weights = []
    for edit in edits:
        op = _op_name(edit)
        op_weight = float(op_probs.get(op, 0.0))
        if required_op is not None and op == required_op:
            op_weight = max(op_weight, 1.0)
        # Tiny edits are plentiful in both generated and copy pools; this softly
        # shifts mass toward edits that exercise longer pointer spans.
        length_weight = max(_edit_span_len(edit), 1) ** 0.35
        weights.append(op_weight * length_weight)
    weights = np.array(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return None
    positive, = np.where(weights > 0)
    order = list(positive[rng.choice(len(positive), size=len(positive), replace=False, p=weights[positive] / weights[positive].sum())])
    zero, = np.where(weights <= 0)
    if len(zero):
        order.extend(list(zero[rng.permutation(len(zero))]))
    kept: list[Any] = []
    for ix in order:
        proposal = sorted(kept + [edits[int(ix)]], key=_edit_key)
        try:
            _validate_edits(seq_edit.sequence, proposal)
        except ValueError:
            continue
        kept = proposal
        if len(kept) >= max_edits:
            break
    if not kept:
        return None
    if required_op is not None and not any(_op_name(edit) == required_op for edit in kept):
        required = [edit for edit in edits if _op_name(edit) == required_op]
        for ix in rng.permutation(len(required)):
            edit = required[int(ix)]
            proposal = sorted([edit], key=_edit_key)
            try:
                _validate_edits(seq_edit.sequence, proposal)
            except ValueError:
                continue
            return SequenceEdit(sequence=np.array(seq_edit.sequence), edits=proposal)
        return None
    return SequenceEdit(sequence=np.array(seq_edit.sequence), edits=kept)


def _token_byte_offsets(enc, text: str) -> tuple[list[int], list[int], list[int]]:
    tokens = enc.encode(text, allowed_special="all")
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for token in tokens:
        starts.append(cursor)
        cursor += len(enc.decode_single_token_bytes(int(token)))
        ends.append(cursor)
    return tokens, starts, ends


def _char_to_byte_offsets(text: str) -> list[int]:
    offsets = [0]
    cursor = 0
    for ch in text:
        cursor += len(ch.encode("utf-8"))
        offsets.append(cursor)
    return offsets


def _boundary_token_index(byte_pos: int, starts: list[int], ends: list[int]) -> int | None:
    if byte_pos == 0:
        return 0
    if ends and byte_pos == ends[-1]:
        return len(ends)
    for ix, start in enumerate(starts):
        if byte_pos == start:
            return ix
    return None


def char_sequence_edit_to_token_sequence_edit(seq_edit: SequenceEdit, enc) -> SequenceEdit | None:
    """Convert a character-coordinate edit script to token coordinates.

    Boundaries that land inside a BPE token are expanded to the nearest token
    boundaries and represented as an equivalent replacement. This preserves
    the exact resolved bytes instead of discarding almost every naturally
    generated character-level edit. Move/copy retain their operation type and
    therefore still require exact token boundaries.
    """

    text = "".join(np.array(seq_edit.sequence).tolist())
    source_tokens, starts, ends = _token_byte_offsets(enc, text)
    char_bytes = _char_to_byte_offsets(text)
    byte_to_char = {byte_pos: char_pos for char_pos, byte_pos in enumerate(char_bytes)}
    text_bytes = text.encode("utf-8")

    def pos_to_token(pos: int) -> int | None:
        if pos < 0 or pos >= len(char_bytes):
            return None
        return _boundary_token_index(char_bytes[pos], starts, ends)

    def covering_token_span(start_pos: int, end_pos: int) -> tuple[int, int, int, int] | None:
        if not (0 <= start_pos <= end_pos <= len(text)):
            return None
        start_byte = char_bytes[start_pos]
        end_byte = char_bytes[end_pos]
        left = _boundary_token_index(start_byte, starts, ends)
        if left is None:
            left = next(
                (ix for ix, (token_start, token_end) in enumerate(zip(starts, ends))
                 if token_start < start_byte < token_end),
                None,
            )
        right = _boundary_token_index(end_byte, starts, ends)
        if right is None:
            containing = next(
                (ix for ix, (token_start, token_end) in enumerate(zip(starts, ends))
                 if token_start < end_byte < token_end),
                None,
            )
            right = None if containing is None else containing + 1
        if left is None or right is None:
            return None

        # A BPE boundary may itself split a multi-byte Unicode character. Grow
        # outward until the replacement span also has valid character bounds.
        while left > 0 and starts[left] not in byte_to_char:
            left -= 1
        while right < len(source_tokens) and ends[right - 1] not in byte_to_char:
            right += 1
        left_byte = starts[left] if left < len(starts) else len(text_bytes)
        right_byte = ends[right - 1] if right > 0 else 0
        if left_byte not in byte_to_char or right_byte not in byte_to_char:
            return None
        return left, right, byte_to_char[left_byte], byte_to_char[right_byte]

    def encode_text(value: str) -> np.ndarray:
        return np.asarray(enc.encode(value, allowed_special="all"), dtype=np.int64)

    token_edits = []
    for edit in seq_edit.edits:
        if isinstance(edit, Insert):
            pos = pos_to_token(int(edit.pos))
            content_text = "".join(np.array(edit.content).tolist())
            if pos is not None:
                token_edits.append(Insert(pos, encode_text(content_text)))
                continue
            span = covering_token_span(int(edit.pos), int(edit.pos))
            if span is None:
                return None
            left, right, char_left, char_right = span
            replacement = text[char_left:int(edit.pos)] + content_text + text[int(edit.pos):char_right]
            token_edits.append(Replace(left, right, encode_text(replacement)))
        elif isinstance(edit, Delete):
            start = pos_to_token(int(edit.start))
            end = pos_to_token(int(edit.end))
            if start is not None and end is not None and start < end:
                token_edits.append(Delete(start, end))
                continue
            span = covering_token_span(int(edit.start), int(edit.end))
            if span is None:
                return None
            left, right, char_left, char_right = span
            replacement = text[char_left:int(edit.start)] + text[int(edit.end):char_right]
            if replacement:
                token_edits.append(Replace(left, right, encode_text(replacement)))
            else:
                token_edits.append(Delete(left, right))
        elif isinstance(edit, Replace):
            start = pos_to_token(int(edit.start))
            end = pos_to_token(int(edit.end))
            content_text = "".join(np.array(edit.content).tolist())
            if start is not None and end is not None and start < end:
                token_edits.append(Replace(start, end, encode_text(content_text)))
                continue
            span = covering_token_span(int(edit.start), int(edit.end))
            if span is None:
                return None
            left, right, char_left, char_right = span
            replacement = (
                text[char_left:int(edit.start)]
                + content_text
                + text[int(edit.end):char_right]
            )
            token_edits.append(Replace(left, right, encode_text(replacement)))
        elif isinstance(edit, Move):
            start = pos_to_token(int(edit.start))
            end = pos_to_token(int(edit.end))
            dest = pos_to_token(int(edit.dest))
            if start is None or end is None or dest is None or start >= end:
                return None
            token_edits.append(Move(start, end, dest))
        elif isinstance(edit, Copy):
            start = pos_to_token(int(edit.start))
            end = pos_to_token(int(edit.end))
            dest = pos_to_token(int(edit.dest))
            if start is None or end is None or dest is None or start >= end:
                return None
            token_edits.append(Copy(start, end, dest))
        else:
            raise TypeError(f"Unsupported edit type: {type(edit)!r}")
    token_edits.sort(key=_edit_key)
    try:
        _validate_edits(np.array(source_tokens, dtype=np.int64), token_edits)
    except ValueError:
        return None
    token_sequence_edit = SequenceEdit(
        sequence=np.array(source_tokens, dtype=np.int64),
        edits=token_edits,
    )
    expected_text = "".join(np.asarray(resolve(seq_edit)).tolist()).encode("utf-8")
    resolved_bytes = b"".join(
        enc.decode_single_token_bytes(int(token))
        for token in np.asarray(resolve(token_sequence_edit)).tolist()
    )
    if resolved_bytes != expected_text:
        return None
    return token_sequence_edit


def _offset_edit(edit: Any, offset: int) -> Any:
    if isinstance(edit, Insert):
        return Insert(int(edit.pos) + offset, np.array(edit.content))
    if isinstance(edit, Delete):
        return Delete(int(edit.start) + offset, int(edit.end) + offset)
    if isinstance(edit, Replace):
        return Replace(int(edit.start) + offset, int(edit.end) + offset, np.array(edit.content))
    if isinstance(edit, Copy):
        return Copy(int(edit.start) + offset, int(edit.end) + offset, int(edit.dest) + offset)
    if isinstance(edit, Move):
        return Move(int(edit.start) + offset, int(edit.end) + offset, int(edit.dest) + offset)
    raise TypeError(f"Unsupported edit type: {type(edit)!r}")


def _plain_training_data(tokens: list[int] | np.ndarray, T: int, pos_buffer_size: int) -> TrainingData:
    row = np.array(tokens, dtype=np.int64)
    if len(row) < T + 1:
        row = np.pad(row, (0, T + 1 - len(row)), constant_values=0)
    row = row[:T + 1]
    return TrainingData(
        input_tokens=row[:-1],
        input_positions=np.arange(T, dtype=np.int64),
        target=TrainingTarget(
            tokens=row[1:],
            gathered_pos_target=np.full(pos_buffer_size, -1, dtype=np.int64),
            token_loss_weights=np.ones(T, dtype=bool),
            gather_indices=np.zeros(pos_buffer_size, dtype=np.int64),
            gather_valid=np.zeros(pos_buffer_size, dtype=bool),
            overflow=False,
            perplexity_masks=None,
        ),
    )


def _valid_random_span(
    n: int,
    *,
    start_hint: int | None = None,
    mean_gap: float,
    mean_len: float,
    protected: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    if n <= 1:
        return None
    for _ in range(64):
        start = int(start_hint) if start_hint is not None else int(rng.integers(0, n))
        if not (0 <= start < n):
            start_hint = None
            continue
        if protected[start]:
            start_hint = None
            continue
        length = max(1, int(rng.geometric(1.0 / max(mean_len, 1.0))))
        end = min(n, start + length)
        if end <= start or protected[start:end].any():
            start_hint = None
            continue
        return start, end
    return None


def _sample_random_boundary(
    n: int,
    *,
    mean_gap: float,
    protected: np.ndarray,
    rng: np.random.Generator,
    forbidden_span: tuple[int, int] | None = None,
) -> int | None:
    if n <= 0:
        return None
    for _ in range(64):
        pos = min(int(rng.geometric(1.0 / max(mean_gap, 1.0))), n)
        if forbidden_span is not None:
            start, end = forbidden_span
            if start < pos < end:
                continue
        if 0 < pos < n and protected[pos]:
            continue
        return pos
    return None


def _sample_random_payload(
    *,
    length: int,
    sequence: np.ndarray,
    token_ops: TokensOperations,
    reservoir: ChunkReservoir,
    cfg: EditSamplingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    probs = np.array(cfg.random_insertion_type_probabilities, dtype=np.float64)
    if length >= len(sequence):
        probs[1] = 0.0
    if len(reservoir) == 0 or length > reservoir.chunk_size:
        probs[0] = 0.0
    if probs.sum() <= 0:
        probs[2] = 1.0
    valid_tokens = getattr(
        token_ops, "_deterministic_non_special_tokens", None
    )
    if valid_tokens is None:
        special_tokens = set(token_ops.tokeniser._special_tokens.values())
        valid_tokens = np.array(
            [
                token
                for token in range(token_ops.tokeniser.max_token_value + 1)
                if token not in special_tokens
            ],
            dtype=np.int64,
        )
        token_ops._deterministic_non_special_tokens = valid_tokens
    deterministic_tokens = lambda size: rng.choice(
        valid_tokens, size=size, replace=True
    )
    return np.array(
        sample_insert(
            length=length,
            current_sequence=list(sequence),
            sample_uniform_random_tokens=deterministic_tokens,
            sample_marginal_tokens=deterministic_tokens,
            reservoir=reservoir,
            insertion_type_probabilities=probs,
            rng=rng,
        ),
        dtype=np.int64,
    )


def _sample_random_inverse_edit(
    sequence: np.ndarray,
    *,
    cfg: EditSamplingConfig,
    token_ops: TokensOperations,
    reservoir: ChunkReservoir,
    rng: np.random.Generator,
) -> SequenceEdit | None:
    n = len(sequence)
    if n <= 2:
        return None
    protected = sequence == token_ops.tokeniser._special_tokens.get("<|bos|>", -1)
    edits: list[Any] = []
    cursor = 0
    replace_prob = max(float(cfg.random_replace_prob), 0.0)
    move_prob = max(float(cfg.random_move_prob), 0.0)
    if cfg.random_delete_prob is None or cfg.random_insert_prob is None:
        residual = max(1.0 - replace_prob - move_prob, 0.0)
        delete_prob = residual / 2.0 if cfg.random_delete_prob is None else max(float(cfg.random_delete_prob), 0.0)
        insert_prob = residual / 2.0 if cfg.random_insert_prob is None else max(float(cfg.random_insert_prob), 0.0)
    else:
        delete_prob = max(float(cfg.random_delete_prob), 0.0)
        insert_prob = max(float(cfg.random_insert_prob), 0.0)
    op_keys = ("replace", "delete", "insert", "move")
    op_probs = np.array(
        [replace_prob, delete_prob, insert_prob, move_prob],
        dtype=np.float64,
    )
    if op_probs.sum() <= 0:
        op_keys = ("insert",)
        op_probs = np.array([1.0], dtype=np.float64)
    else:
        op_probs /= op_probs.sum()
    while cursor < n:
        gap = int(rng.geometric(1.0 / max(cfg.random_no_del_mean, cfg.random_no_ins_mean, 1.0)))
        cursor += gap
        if cursor >= n:
            break
        op = op_keys[int(rng.choice(len(op_keys), p=op_probs))]
        if op == "replace":
            span = _valid_random_span(
                n,
                start_hint=cursor,
                mean_gap=cfg.random_no_del_mean,
                mean_len=cfg.random_del_mean_len,
                protected=protected,
                rng=rng,
            )
            if span is None:
                continue
            start, end = span
            length = max(1, int(rng.geometric(1.0 / max(cfg.random_ins_mean_len, 1.0))))
            payload = _sample_random_payload(
                length=length,
                sequence=sequence,
                token_ops=token_ops,
                reservoir=reservoir,
                cfg=cfg,
                rng=rng,
            )
            edits.append(Replace(start, end, payload))
            cursor = end
        elif op == "delete":
            span = _valid_random_span(
                n,
                start_hint=cursor,
                mean_gap=cfg.random_no_del_mean,
                mean_len=cfg.random_del_mean_len,
                protected=protected,
                rng=rng,
            )
            if span is None:
                continue
            start, end = span
            edits.append(Delete(start, end))
            cursor = end
        elif op == "insert":
            pos = min(max(cursor, 0), n)
            length = max(1, int(rng.geometric(1.0 / max(cfg.random_ins_mean_len, 1.0))))
            payload = _sample_random_payload(
                length=length,
                sequence=sequence,
                token_ops=token_ops,
                reservoir=reservoir,
                cfg=cfg,
                rng=rng,
            )
            edits.append(Insert(pos, payload))
            cursor += 1
        elif op == "move":
            span = _valid_random_span(
                n,
                start_hint=cursor,
                mean_gap=cfg.random_no_del_mean,
                mean_len=cfg.random_del_mean_len,
                protected=protected,
                rng=rng,
            )
            if span is None:
                continue
            start, end = span
            dest = _sample_random_boundary(
                n,
                mean_gap=cfg.random_no_ins_mean,
                protected=protected,
                rng=rng,
                forbidden_span=(start, end),
            )
            if dest is None:
                continue
            edits.append(Move(start, end, dest))
            cursor = end
        if len(edits) >= cfg.max_edits_per_example:
            break
    if not edits:
        return None
    edits.sort(key=_edit_key)
    try:
        _validate_edits(sequence, edits)
        _validate_edits(sequence, _lower_moves(SequenceEdit(sequence=np.array(sequence), edits=edits)).edits)
    except ValueError:
        return None
    return complement(SequenceEdit(sequence=np.array(sequence), edits=edits))


def _copy_candidate_to_sequence_edit(
    raw_text: str,
    candidate: _CopyCandidate,
    *,
    rng: np.random.Generator,
) -> SequenceEdit | None:
    occurrences = sorted(candidate.occurrences)
    if len(occurrences) < 2 or candidate.length_chars <= 0:
        return None
    source_ix, remove_ix = occurrences[0], occurrences[int(rng.integers(1, len(occurrences)))]
    start = min(source_ix, remove_ix)
    end = max(source_ix, remove_ix)
    length = candidate.length_chars
    if raw_text[source_ix:source_ix + length] != candidate.repeated_text:
        return None
    if raw_text[remove_ix:remove_ix + length] != candidate.repeated_text:
        return None
    if start + length > end:
        return None

    base_text = raw_text[:remove_ix] + raw_text[remove_ix + length:]
    adjusted_source_ix = source_ix if source_ix < remove_ix else source_ix - length
    dest = remove_ix if remove_ix < source_ix else remove_ix
    return SequenceEdit(
        sequence=np.array(list(base_text)),
        edits=[Copy(adjusted_source_ix, adjusted_source_ix + length, dest)],
    )


def _choose_copy_candidate(
    candidates: list[_CopyCandidate],
    *,
    cfg: EditSamplingConfig,
    rng: np.random.Generator,
) -> _CopyCandidate | None:
    candidates = [c for c in candidates if c.length_chars >= cfg.min_copy_chars]
    if not candidates:
        return None
    if cfg.copy_length_acceptance is None:
        weights = np.array(
            [max(c.length_chars, 1) ** cfg.copy_length_alpha for c in candidates],
            dtype=np.float64,
        )
    else:
        weights = np.array(
            [float(cfg.copy_length_acceptance.get(int(c.length_chars), 0.0)) for c in candidates],
            dtype=np.float64,
        )
        if weights.sum() <= 0:
            return None
    weights /= weights.sum()
    return candidates[int(rng.choice(len(candidates), p=weights))]


def _pad_or_trim_training_data(training_data: TrainingData, T: int) -> TrainingData:
    n = len(training_data.input_tokens)
    if n == T:
        return training_data
    if n > T:
        raise ValueError(f"Serialised edit example has length {n}, exceeds T={T}")

    pad = T - n
    input_tokens = np.pad(training_data.input_tokens, (0, pad), constant_values=0)
    input_positions = np.pad(training_data.input_positions, (0, pad), constant_values=0)
    target_tokens = np.pad(training_data.target.tokens, (0, pad), constant_values=-1)
    token_loss_weights = np.pad(
        training_data.target.token_loss_weights, (0, pad), constant_values=False
    )
    return TrainingData(
        input_tokens=input_tokens,
        input_positions=input_positions,
        target=TrainingTarget(
            tokens=target_tokens,
            gathered_pos_target=training_data.target.gathered_pos_target,
            token_loss_weights=token_loss_weights,
            gather_indices=training_data.target.gather_indices,
            gather_valid=training_data.target.gather_valid,
            overflow=training_data.target.overflow,
            perplexity_masks=training_data.target.perplexity_masks,
        ),
    )


def _concat_training_data(parts: list[TrainingData]) -> TrainingData:
    """Concatenate ordered CES regions into one causal training row.

    The final token of one region becomes a normal input token whose target is
    the first source token of the next region. Pointer query/target indices and
    dual-RoPE address positions are shifted into the combined coordinate space.
    """

    if not parts:
        raise ValueError("At least one training-data part is required")
    if len(parts) == 1:
        return parts[0]

    full_lengths = np.array([len(part.input_tokens) + 1 for part in parts], dtype=np.int64)
    offsets = np.cumsum(np.concatenate([np.zeros(1, dtype=np.int64), full_lengths[:-1]]))
    pos_buffer = len(parts[0].target.gather_valid)

    gather_indices: list[np.ndarray] = []
    gathered_targets: list[np.ndarray] = []
    for part, offset in zip(parts, offsets):
        valid = np.asarray(part.target.gather_valid, dtype=bool)
        gather_indices.append(np.asarray(part.target.gather_indices)[valid] + offset)
        gathered_targets.append(np.asarray(part.target.gathered_pos_target)[valid] + offset)
    all_gather_indices = np.concatenate(gather_indices) if gather_indices else np.zeros(0, dtype=np.int64)
    all_gathered_targets = np.concatenate(gathered_targets) if gathered_targets else np.zeros(0, dtype=np.int64)
    used = min(len(all_gather_indices), pos_buffer)
    gather_indices_buf = np.zeros(pos_buffer, dtype=np.int64)
    gathered_targets_buf = np.full(pos_buffer, -1, dtype=np.int64)
    gather_valid_buf = np.zeros(pos_buffer, dtype=bool)
    gather_indices_buf[:used] = all_gather_indices[:used]
    gathered_targets_buf[:used] = all_gathered_targets[:used]
    gather_valid_buf[:used] = True

    full_tokens = []
    full_weights = []
    input_positions = []
    for part, offset in zip(parts, offsets):
        full_tokens.extend([part.input_tokens, part.target.tokens[-1:]])
        full_weights.extend(
            [part.target.token_loss_weights, np.ones(1, dtype=part.target.token_loss_weights.dtype)]
        )
        input_positions.extend(
            [
                np.array([offset - 1], dtype=part.input_positions.dtype),
                np.asarray(part.input_positions) + offset,
            ]
        )
    tokens = np.concatenate(full_tokens)
    weights = np.concatenate(full_weights)[:-1]
    positions = np.concatenate(input_positions[1:])

    return TrainingData(
        input_tokens=tokens[:-1],
        input_positions=positions,
        target=TrainingTarget(
            tokens=tokens[1:],
            gathered_pos_target=gathered_targets_buf,
            token_loss_weights=weights,
            gather_indices=gather_indices_buf,
            gather_valid=gather_valid_buf,
            overflow=bool(any(part.target.overflow for part in parts) or len(all_gather_indices) > pos_buffer),
            perplexity_masks=None,
        ),
    )
    return TrainingData(
        input_tokens=input_tokens,
        input_positions=input_positions,
        target=TrainingTarget(
            tokens=target_tokens,
            gathered_pos_target=training_data.target.gathered_pos_target,
            token_loss_weights=token_loss_weights,
            gather_indices=training_data.target.gather_indices,
            gather_valid=training_data.target.gather_valid,
            overflow=training_data.target.overflow,
            perplexity_masks=training_data.target.perplexity_masks,
        ),
    )


class EditCandidateStore:
    def __init__(self, cfg: EditSamplingConfig):
        self.cfg = cfg
        self.generated = load_generated_edit_index(cfg.generated_edit_paths)
        self.copy = load_copy_scan_index(cfg.copy_scan_dir)

    def sample_token_sequence_edit(
        self,
        *,
        shard_id: str,
        row_index: int,
        source_tokens: list[int],
        rng: np.random.Generator,
    ) -> SequenceEdit | None:
        candidates = [
            candidate
            for candidate in self.generated.get((shard_id, row_index), [])
            if candidate.forward_token_json
        ]
        if not candidates:
            return None
        candidate = candidates[int(rng.integers(0, len(candidates)))]
        direction_keys, direction_probs = _normalise_probs(self.cfg.direction_probs)
        direction = direction_keys[
            int(rng.choice(len(direction_keys), p=direction_probs))
        ]

        def parse(payload: str, sequence: list[int]) -> SequenceEdit:
            edits = []
            for item in json.loads(payload):
                kind = item["type"]
                if kind == "insert":
                    edits.append(
                        Insert(
                            int(item["pos"]),
                            np.asarray(item["content"], dtype=np.int64),
                        )
                    )
                elif kind == "delete":
                    edits.append(Delete(int(item["start"]), int(item["end"])))
                elif kind == "replace":
                    edits.append(
                        Replace(
                            int(item["start"]),
                            int(item["end"]),
                            np.asarray(item["content"], dtype=np.int64),
                        )
                    )
                else:
                    raise ValueError(f"Unsupported token edit type: {kind}")
            return SequenceEdit(
                sequence=np.asarray(sequence, dtype=np.int64),
                edits=edits,
            )

        if direction == "forward":
            sequence_edit = parse(candidate.forward_token_json, source_tokens)
        elif direction == "inverse":
            if not candidate.inverse_token_json:
                return None
            forward = parse(candidate.forward_token_json, source_tokens)
            target_tokens = [int(token) for token in resolve(forward).tolist()]
            sequence_edit = parse(candidate.inverse_token_json, target_tokens)
        else:
            raise ValueError(f"Unknown direction: {direction}")
        return _sample_valid_subset(
            sequence_edit,
            op_probs=self.cfg.op_probs,
            max_edits=self.cfg.max_edits_per_example,
            rng=rng,
            required_op=None,
        )

    def sample(
        self,
        *,
        shard_id: str,
        row_index: int,
        raw_text: str,
        enc,
        token_ops: TokensOperations,
        token_budget: int,
        pos_buffer_size: int,
        rng: np.random.Generator,
        target_op: str | None = None,
    ) -> TrainingData | None:
        op_keys, op_probs = _normalise_probs(self.cfg.op_probs)
        op = target_op if target_op is not None else op_keys[int(rng.choice(len(op_keys), p=op_probs))]
        sample = self.sample_sequence_edit(
            shard_id=shard_id,
            row_index=row_index,
            raw_text=raw_text,
            rng=rng,
            target_op=op,
        )
        if sample is None:
            return None
        token_seq_edit = char_sequence_edit_to_token_sequence_edit(sample, enc)
        if token_seq_edit is None or not token_seq_edit.edits:
            return None
        if self.cfg.max_source_tokens is not None and len(token_seq_edit.sequence) > self.cfg.max_source_tokens:
            return None

        indel_bos_pos = _sample_indel_bos_pos(
            token_seq_edit,
            cfg=self.cfg,
            rng=rng,
            edited=True,
        )
        assert indel_bos_pos is not None
        serialised = token_ops.serialise(token_seq_edit, indel_bos_pos=indel_bos_pos)
        if self.cfg.max_serialised_tokens is not None and len(serialised.tokens) > self.cfg.max_serialised_tokens:
            return None
        if len(serialised.tokens) > token_budget + 1:
            return None
        return _pad_or_trim_training_data(
            token_ops.prepare_training_example(serialised, pos_buffer_size),
            token_budget,
        )

    def sample_sequence_edit(
        self,
        *,
        shard_id: str,
        row_index: int,
        raw_text: str,
        rng: np.random.Generator,
        target_op: str | None = None,
    ) -> SequenceEdit | None:
        direction_keys, direction_probs = _normalise_probs(self.cfg.direction_probs)
        if target_op is None:
            generated = self.generated.get((shard_id, row_index), [])
            has_copy = bool(self.copy.get((shard_id, row_index)))
            pool_weights = []
            pools = []
            if generated:
                weight = sum(
                    float(self.cfg.op_probs.get(op_name, 0.0))
                    for cand in generated
                    for op_name in (cand.ops or ("replace",))
                )
                pools.append("generated")
                pool_weights.append(max(weight, 1e-12))
            if has_copy:
                pools.append("copy")
                pool_weights.append(max(float(self.cfg.op_probs.get("copy", 0.0)), 1e-12))
            if not pools:
                return None
            pool_probs = np.array(pool_weights, dtype=np.float64)
            pool_probs /= pool_probs.sum()
            pool = pools[int(rng.choice(len(pools), p=pool_probs))]
            if pool == "copy":
                copy_candidate = _choose_copy_candidate(
                    self.copy.get((shard_id, row_index), []), cfg=self.cfg, rng=rng
                )
                if copy_candidate is None:
                    return None
                return _copy_candidate_to_sequence_edit(raw_text, copy_candidate, rng=rng)

            cand = generated[int(rng.integers(0, len(generated)))]
            direction = direction_keys[int(rng.choice(len(direction_keys), p=direction_probs))]
            if direction == "forward":
                char_seq_edit = sequence_edit_from_json(cand.source_text, cand.forward_json)
            elif direction == "inverse":
                char_seq_edit = sequence_edit_from_json(cand.source_text, cand.forward_json)
                char_seq_edit = complement(char_seq_edit)
            else:
                raise ValueError(f"Unknown direction: {direction}")
            return _sample_valid_subset(
                char_seq_edit,
                op_probs=self.cfg.op_probs,
                max_edits=self.cfg.max_edits_per_example,
                rng=rng,
                required_op=None,
            )
        else:
            op = target_op

        char_seq_edit: SequenceEdit | None = None
        if op == "copy":
            copy_candidate = _choose_copy_candidate(
                self.copy.get((shard_id, row_index), []), cfg=self.cfg, rng=rng
            )
            if copy_candidate is not None:
                char_seq_edit = _copy_candidate_to_sequence_edit(raw_text, copy_candidate, rng=rng)
        else:
            candidates = [
                c for c in self.generated.get((shard_id, row_index), [])
                if (not c.ops or op in c.ops)
            ]
            if candidates:
                cand = candidates[int(rng.integers(0, len(candidates)))]
                direction = direction_keys[int(rng.choice(len(direction_keys), p=direction_probs))]
                if direction == "forward":
                    char_seq_edit = sequence_edit_from_json(cand.source_text, cand.forward_json)
                elif direction == "inverse":
                    char_seq_edit = sequence_edit_from_json(cand.source_text, cand.forward_json)
                    char_seq_edit = complement(char_seq_edit)
                else:
                    raise ValueError(f"Unknown direction: {direction}")
                char_seq_edit = _sample_valid_subset(
                    char_seq_edit,
                    op_probs=self.cfg.op_probs,
                    max_edits=self.cfg.max_edits_per_example,
                    rng=rng,
                    required_op=op,
                )

        if char_seq_edit is None:
            return None
        return char_seq_edit


def _document_rows(split: str, cfg: EditSamplingConfig, resume_state_dict, tokenizer_batch_size: int):
    from nanochat.dataset import list_parquet_files

    ddp, ddp_rank, _ddp_local_rank, ddp_world_size = get_dist_info()
    parquet_paths = list_parquet_files(data_dir=cfg.raw_data_dir, warn_on_legacy=ddp_rank == 0)
    assert parquet_paths, "No raw parquet files found."
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            shard_id = Path(filepath).stem
            pf = pq.ParquetFile(filepath)
            if first_pass and resume_rg_idx is not None and pq_idx == resume_pq_idx:
                base_idx = resume_rg_idx // ddp_world_size + 1
                rg_idx = base_idx * ddp_world_size + ddp_rank
                resume_rg_idx = None
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx, columns=["text"])
                texts = rg.column("text").to_pylist()
                row_start = sum(
                    pf.metadata.row_group(i).num_rows for i in range(rg_idx)
                )
                rows = [(shard_id, row_start + i, text) for i, text in enumerate(texts)]
                for i in range(0, len(rows), tokenizer_batch_size):
                    yield rows[i:i + tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def _prepare_doc(
    *,
    shard_id: str,
    row_index: int,
    raw_text: str,
    tokenizer,
    bos_token: int,
    store: EditCandidateStore,
    cfg: EditSamplingConfig,
    rng: np.random.Generator,
) -> _PreparedDoc:
    body_tokens = [int(token) for token in tokenizer.encode(raw_text)]
    sample_direct = getattr(store, "sample_token_sequence_edit", None)
    direct_token_edit = (
        None
        if sample_direct is None
        else sample_direct(
            shard_id=shard_id,
            row_index=row_index,
            source_tokens=body_tokens,
            rng=rng,
        )
    )
    if direct_token_edit is not None:
        tokens = [bos_token, *[int(token) for token in direct_token_edit.sequence]]
        shifted = SequenceEdit(
            sequence=np.asarray(tokens, dtype=np.int64),
            edits=[_offset_edit(edit, 1) for edit in direct_token_edit.edits],
        )
        return _PreparedDoc(
            shard_id=shard_id,
            row_index=row_index,
            source_text=raw_text,
            tokens=tokens,
            token_edit=shifted,
        )
    char_edit = store.sample_sequence_edit(
        shard_id=shard_id,
        row_index=row_index,
        raw_text=raw_text,
        rng=rng,
        target_op=None,
    )
    if char_edit is None:
        tokens = tokenizer.encode(raw_text, prepend=bos_token)
        return _PreparedDoc(shard_id, row_index, raw_text, tokens, None)

    token_edit = char_sequence_edit_to_token_sequence_edit(char_edit, tokenizer.enc)
    if token_edit is None:
        tokens = tokenizer.encode(raw_text, prepend=bos_token)
        return _PreparedDoc(shard_id, row_index, raw_text, tokens, None)

    tokens = [bos_token] + [int(t) for t in token_edit.sequence]
    shifted_edits = [_offset_edit(edit, 1) for edit in token_edit.edits]
    shifted = SequenceEdit(sequence=np.array(tokens, dtype=np.int64), edits=shifted_edits)
    return _PreparedDoc(
        shard_id=shard_id,
        row_index=row_index,
        source_text="".join(np.array(char_edit.sequence).tolist()),
        tokens=tokens,
        token_edit=shifted,
    )


class _OrderedDocumentStream:
    """Consume one ordered stream of document tokens without dropping suffixes."""

    def __init__(
        self,
        *,
        tokenizer,
        bos_token: int,
        store: EditCandidateStore,
        cfg: EditSamplingConfig,
        rng: np.random.Generator,
        refill_rows,
    ):
        self.tokenizer = tokenizer
        self.bos_token = int(bos_token)
        self.store = store
        self.cfg = cfg
        self.rng = rng
        self.refill_rows = refill_rows
        self.docs: list[_RawStreamDoc] = []
        self.priority_candidates: deque[_PreparedDoc] = deque()
        self.raw_documents_consumed = 0
        self.clean_tokens_consumed = 0
        self.candidate_docs_seen = 0
        self.candidate_docs_queued = 0

    def _refill(self) -> None:
        rows = self.refill_rows()
        for shard_id, row_index, raw_text in rows:
            clean_tokens = self.tokenizer.encode(raw_text, prepend=self.bos_token)
            self.docs.append(
                _RawStreamDoc(
                    shard_id=shard_id,
                    row_index=row_index,
                    raw_text=raw_text,
                    clean_tokens=[int(token) for token in clean_tokens],
                )
            )

    def _front(self) -> _RawStreamDoc:
        while not self.docs:
            self._refill()
        return self.docs[0]

    def _finish_front(self) -> None:
        self.docs.pop(0)
        self.raw_documents_consumed += 1

    def _prepared_doc(self, doc: _RawStreamDoc) -> _PreparedDoc | None:
        if doc.offset != 0:
            return None
        if not doc.prepared_checked:
            doc.prepared = _prepare_doc(
                shard_id=doc.shard_id,
                row_index=doc.row_index,
                raw_text=doc.raw_text,
                tokenizer=self.tokenizer,
                bos_token=self.bos_token,
                store=self.store,
                cfg=self.cfg,
                rng=self.rng,
            )
            doc.prepared_checked = True
            if doc.prepared is not None and doc.prepared.token_edit is not None:
                self.candidate_docs_seen += 1
        if doc.prepared is None or doc.prepared.token_edit is None:
            return None
        return doc.prepared

    def _queue_priority_candidate(self, doc: _RawStreamDoc) -> None:
        if not self.cfg.priority_candidate_lane or doc.priority_candidate_queued:
            return
        prepared = self._prepared_doc(doc)
        if prepared is not None:
            self.priority_candidates.append(prepared)
            doc.priority_candidate_queued = True
            self.candidate_docs_queued += 1

    def _take_clean_from_front(
        self, count: int, *, queue_priority_candidates: bool = True
    ) -> list[int]:
        doc = self._front()
        if queue_priority_candidates and doc.offset == 0:
            self._queue_priority_candidate(doc)
        available = len(doc.clean_tokens) - doc.offset
        take = min(max(int(count), 0), available)
        out = doc.clean_tokens[doc.offset : doc.offset + take]
        if take:
            doc.prepared_checked = True  # a split/clean-consumed document cannot later become one GPT unit
            doc.prepared = None
            doc.offset += take
            self.clean_tokens_consumed += take
        if doc.offset >= len(doc.clean_tokens):
            self._finish_front()
        return out

    def pop_clean_tokens(
        self, count: int, *, queue_priority_candidates: bool = True
    ) -> list[int]:
        out: list[int] = []
        while len(out) < count:
            out.extend(
                self._take_clean_from_front(
                    count - len(out),
                    queue_priority_candidates=queue_priority_candidates,
                )
            )
        return out

    def pop_row_end_tokens(
        self,
        count: int,
        *,
        max_context_tokens: int,
        queue_priority_candidates: bool = True,
    ) -> list[int]:
        """Best-fit complete documents into a physical row remainder.

        An active continuation is consumed first. Fresh documents are packed
        whole from bounded lookahead. A fresh document is split only when it
        is itself longer than a complete context.
        """

        count = max(int(count), 0)
        if count == 0:
            return []
        while len(self.docs) < 128:
            self._refill()

        out: list[int] = []
        if self.docs and self.docs[0].offset:
            out.extend(
                self._take_clean_from_front(
                    count,
                    queue_priority_candidates=queue_priority_candidates,
                )
            )
            if len(out) == count:
                return out

        while len(out) < count:
            remaining = count - len(out)
            fitting = [
                (len(doc.clean_tokens), ix)
                for ix, doc in enumerate(self.docs[:128])
                if doc.offset == 0 and len(doc.clean_tokens) <= remaining
            ]
            if fitting:
                _length, doc_ix = max(
                    fitting, key=lambda item: (item[0], -item[1])
                )
                doc = self.docs[doc_ix]
                if queue_priority_candidates:
                    self._queue_priority_candidate(doc)
                out.extend(doc.clean_tokens)
                self.clean_tokens_consumed += len(doc.clean_tokens)
                self.raw_documents_consumed += 1
                self.docs.pop(doc_ix)
                continue

            if not out:
                overlong_ix = next(
                    (
                        ix
                        for ix, doc in enumerate(self.docs[:128])
                        if doc.offset == 0
                        and len(doc.clean_tokens) > max_context_tokens
                    ),
                    None,
                )
                if overlong_ix is not None:
                    if overlong_ix:
                        self.docs.insert(0, self.docs.pop(overlong_ix))
                    out.extend(
                        self._take_clean_from_front(
                            remaining,
                            queue_priority_candidates=queue_priority_candidates,
                        )
                    )
            break
        return out

    def _prepared_front(self) -> _PreparedDoc | None:
        return self._prepared_doc(self._front())

    def pop_priority_candidate(self) -> _PreparedDoc | None:
        if not self.priority_candidates:
            return None
        return self.priority_candidates.popleft()

    def pop_candidate_source(
        self,
        *,
        full_token_budget: int,
        reserve: int,
        token_ops: TokensOperations,
        terminal_suffix_tokens: int = 0,
        defer_oversized_candidate: bool = False,
        max_deferred_candidate_tokens: int | None = None,
    ) -> tuple[SequenceEdit, int, bool, int]:
        """Pop an ordered source prefix and include every fitting whole-doc candidate.

        ``full_token_budget`` includes source, one indel BOS token, the edit
        program, and edit-end. Candidate documents are atomic. If a candidate
        cannot fit and ``defer_oversized_candidate`` is true, it remains at the
        stream head so the caller can transfer unused capacity to the next
        Gamma region. Otherwise its clean tokens are consumed; document
        suffixes are retained rather than discarded.
        """

        source: list[int] = []
        edits: list[Any] = []
        candidate_docs = 0
        deferred_candidate = False
        dropped_candidate_docs = 0
        overhead = 3  # indel BOS, edit-start, edit-end
        source_token_budget = max(
            2, full_token_budget - max(int(terminal_suffix_tokens), 0)
        )
        soft_source_target = max(2, source_token_budget - max(int(reserve), 3))

        while True:
            source_target = min(
                soft_source_target,
                max(2, source_token_budget - overhead),
            )
            if len(source) >= source_target:
                break
            prepared = self._prepared_front()
            if prepared is not None:
                candidate_overhead = (
                    len(token_ops.serialise(prepared.token_edit, indel_bos_pos=0).tokens)
                    - len(prepared.token_edit.sequence)
                )
                candidate_extra = candidate_overhead - 3
                would_fit = (
                    len(source)
                    + len(prepared.tokens)
                    + overhead
                    + candidate_extra
                    <= source_token_budget
                )
                if would_fit:
                    offset = len(source)
                    source.extend(int(token) for token in prepared.tokens)
                    edits.extend(_offset_edit(edit, offset) for edit in prepared.token_edit.edits)
                    overhead += candidate_extra
                    candidate_docs += 1
                    self.clean_tokens_consumed += len(self._front().clean_tokens)
                    self._finish_front()
                    continue
                can_defer = (
                    defer_oversized_candidate
                    and (
                        max_deferred_candidate_tokens is None
                        or len(prepared.tokens) + candidate_overhead
                        <= max_deferred_candidate_tokens
                    )
                )
                if can_defer:
                    deferred_candidate = True
                    break
                if not self.cfg.priority_candidate_lane:
                    dropped_candidate_docs += 1
            source.extend(self._take_clean_from_front(source_target - len(source)))

        return SequenceEdit(
            sequence=np.asarray(source, dtype=np.int64),
            edits=sorted(edits, key=_edit_key),
        ), candidate_docs, deferred_candidate, dropped_candidate_docs


def _partition_input_budgets(T: int, cfg: EditSamplingConfig, rng: np.random.Generator) -> list[int]:
    """Return per-region input lengths whose concatenation has input length ``T``."""

    sizes = gamma_shaped_partition(
        n=T,
        min_size=cfg.partition_min_size,
        shape=cfg.partition_gamma_shape,
        target_mean_size=cfg.partition_slice_mean_size,
        rng=rng,
    )
    if len(sizes) == 1:
        return sizes
    unshortened = int(rng.integers(0, len(sizes)))
    budgets = [size if ix == unshortened else size - 1 for ix, size in enumerate(sizes)]
    if any(size < 2 for size in budgets):
        raise ValueError(f"Partition produced an unusably small region: {budgets}")
    return budgets


def _append_source_tail(seq_edit: SequenceEdit, tail: list[int]) -> SequenceEdit:
    if not tail:
        return seq_edit
    return SequenceEdit(
        sequence=np.concatenate(
            [np.asarray(seq_edit.sequence, dtype=np.int64), np.asarray(tail, dtype=np.int64)]
        ),
        edits=list(seq_edit.edits),
    )


def _append_post_program_tail(
    training: TrainingData,
    tail: list[int],
    *,
    pos_buffer_size: int,
) -> TrainingData:
    """Append ordinary AR tokens after a complete serialized edit program.

    The concatenation helper preserves the causal transition from the
    serialized region's final edit-end token to the first continuation token
    and shifts the continuation's generation positions into the row.
    """

    if len(tail) < 2:
        raise ValueError("A post-program continuation needs at least two tokens")
    continuation = _plain_training_data(tail, len(tail) - 1, pos_buffer_size)
    return _concat_training_data([training, continuation])


def _edit_required_bounds(edit: Any) -> tuple[int, int]:
    """Return the source-boundary interval needed to retain an edit."""

    if isinstance(edit, Insert):
        pos = int(edit.pos)
        return pos, pos
    if isinstance(edit, (Delete, Replace)):
        return int(edit.start), int(edit.end)
    if isinstance(edit, (Copy, Move)):
        return min(int(edit.start), int(edit.dest)), max(int(edit.end), int(edit.dest))
    raise TypeError(f"Unsupported edit type: {type(edit)!r}")


def _window_candidate_sequence_edit(
    prepared: _PreparedDoc,
    *,
    full_token_budget: int,
    token_ops: TokensOperations,
    bos_token: int,
) -> SequenceEdit | None:
    """Crop a long candidate around the largest fitting subset of its edits.

    The source window always starts with a real BOS token. All address-bearing
    coordinates are remapped into the window and the result is accepted only
    after serializer and edit-script validation.
    """

    original = prepared.token_edit
    if original is None or not original.edits:
        return None
    sequence = np.asarray(original.sequence, dtype=np.int64)
    if not len(sequence) or int(sequence[0]) != int(bos_token):
        return None
    body = sequence[1:]
    if not len(body):
        return None

    edits = list(original.edits)
    subsets = [
        subset
        for subset_size in range(len(edits), 0, -1)
        for subset in combinations(edits, subset_size)
    ]
    best: SequenceEdit | None = None
    best_score = (-1, -1)

    for subset in subsets:
        body_edits = [_offset_edit(edit, -1) for edit in subset]
        bounds = [_edit_required_bounds(edit) for edit in body_edits]
        required_lo = min(lo for lo, _hi in bounds)
        required_hi = max(hi for _lo, hi in bounds)
        if not (0 <= required_lo <= required_hi <= len(body)):
            continue

        # Build the smallest non-empty valid probe to measure the edit-program
        # overhead, which is independent of the surrounding source length.
        probe_start = required_lo
        probe_end = required_hi
        if probe_start == probe_end:
            if probe_start == len(body):
                probe_start = max(0, probe_start - 1)
            else:
                probe_end = min(len(body), probe_end + 1)
        probe_edits = [
            _offset_edit(edit, 1 - probe_start) for edit in body_edits
        ]
        probe = SequenceEdit(
            sequence=np.concatenate(
                [np.asarray([bos_token], dtype=np.int64), body[probe_start:probe_end]]
            ),
            edits=sorted(probe_edits, key=_edit_key),
        )
        try:
            _validate_edits(np.asarray(probe.sequence), probe.edits)
            probe_serialised = token_ops.serialise(probe, indel_bos_pos=0)
        except (AssertionError, ValueError):
            continue
        overhead = len(probe_serialised.tokens) - len(probe.sequence)
        body_capacity = full_token_budget - overhead - 1  # one real document BOS
        required_width = required_hi - required_lo
        if body_capacity < max(required_width, 1):
            continue

        window_len = min(body_capacity, len(body))
        start_min = max(0, required_hi - window_len)
        start_max = min(required_lo, len(body) - window_len)
        if start_min > start_max:
            continue
        centred = (required_lo + required_hi - window_len) // 2
        window_start = min(max(centred, start_min), start_max)
        window_end = window_start + window_len
        window_edits = [
            _offset_edit(edit, 1 - window_start) for edit in body_edits
        ]
        candidate = SequenceEdit(
            sequence=np.concatenate(
                [np.asarray([bos_token], dtype=np.int64), body[window_start:window_end]]
            ),
            edits=sorted(window_edits, key=_edit_key),
        )
        try:
            _validate_edits(np.asarray(candidate.sequence), candidate.edits)
            serialised = token_ops.serialise(candidate, indel_bos_pos=0)
        except (AssertionError, ValueError):
            continue
        if len(serialised.tokens) > full_token_budget:
            continue
        score = (len(candidate.edits), len(candidate.sequence))
        if score > best_score:
            best = candidate
            best_score = score

    return best


def _window_autoregressive_sequence_edit(
    seq_edit: SequenceEdit,
    *,
    tokenizer,
    token_budget: int,
    pos_buffer_size: int,
    baseline_mode: str,
    bos_token: int,
) -> tuple[SequenceEdit, TrainingData] | None:
    """Crop around the largest edit subset whose textual program fits.

    CES pointer programs are more compact than AR-DSL text. A candidate that
    satisfies the shared CES source/program budget can therefore still overflow
    after baseline serialization. This recovery keeps a real BOS, preserves the
    largest fitting subset of edits, remaps every coordinate, and is used only
    after the full ordered example has failed to render.
    """

    sequence = np.asarray(seq_edit.sequence, dtype=np.int64)
    if not seq_edit.edits or len(sequence) < 2 or int(sequence[0]) != int(bos_token):
        return None
    body = sequence[1:]
    edits = list(seq_edit.edits)
    subsets = [
        subset
        for subset_size in range(len(edits), 0, -1)
        for subset in combinations(edits, subset_size)
    ]
    for subset in subsets:
        body_edits = [_offset_edit(edit, -1) for edit in subset]
        bounds = [_edit_required_bounds(edit) for edit in body_edits]
        required_lo = min(lo for lo, _hi in bounds)
        required_hi = max(hi for _lo, hi in bounds)
        if not (0 <= required_lo <= required_hi <= len(body)):
            continue
        minimum_width = max(required_hi - required_lo, 1)
        maximum_width = min(len(body), token_budget)
        for window_len in range(maximum_width, minimum_width - 1, -1):
            start_min = max(0, required_hi - window_len)
            start_max = min(required_lo, len(body) - window_len)
            if start_min > start_max:
                continue
            centred = (required_lo + required_hi - window_len) // 2
            window_start = min(max(centred, start_min), start_max)
            window_end = window_start + window_len
            window_edits = [
                _offset_edit(edit, 1 - window_start) for edit in body_edits
            ]
            candidate = SequenceEdit(
                sequence=np.concatenate(
                    [np.asarray([bos_token], dtype=np.int64), body[window_start:window_end]]
                ),
                edits=sorted(window_edits, key=_edit_key),
            )
            try:
                _validate_edits(np.asarray(candidate.sequence), candidate.edits)
            except (AssertionError, ValueError):
                continue
            training = _ordered_autoregressive_training(
                seq_edit=candidate,
                tokenizer=tokenizer,
                token_budget=token_budget,
                pos_buffer_size=pos_buffer_size,
                baseline_mode=baseline_mode,
            )
            if training is not None:
                return candidate, training
    return None


def _build_priority_candidate_row(
    prepared: _PreparedDoc,
    *,
    stream: _OrderedDocumentStream | None = None,
    input_budget: int,
    cfg: EditSamplingConfig,
    token_ops: TokensOperations,
    rng: np.random.Generator,
    pos_buffer_size: int,
    bos_token: int,
    baseline_mode: str | None = None,
    baseline_tokenizer=None,
) -> tuple[TrainingData | None, dict[str, int]]:
    """Build one dedicated full-context candidate row in the selected serialization."""

    if baseline_mode not in {None, "ar_dsl", "flat_relative"}:
        raise ValueError(f"Unsupported ordered baseline mode: {baseline_mode}")
    if baseline_mode is not None and baseline_tokenizer is None:
        raise ValueError("baseline_tokenizer is required for an ordered AR baseline")

    full_token_budget = input_budget + 1
    post_program_tail = bool(
        stream is not None
        and cfg.post_edit_clean_suffix
        and rng.random() >= min(max(float(cfg.terminal_edit_end_prob), 0.0), 1.0)
    )
    terminal_suffix_tokens = (
        min(
            max(2, int(cfg.terminal_clean_suffix_min)),
            max(2, input_budget // 4),
            max(full_token_budget - 8, 0),
        )
        if post_program_tail
        else 0
    )
    candidate_budget = full_token_budget - terminal_suffix_tokens
    seq_edit = prepared.token_edit
    if seq_edit is None or not seq_edit.edits:
        return None, {"windowed": 0, "padding_tokens": 0, "source_tokens": 0}

    indel_bos_pos = _sample_indel_bos_pos(seq_edit, cfg=cfg, rng=rng, edited=True)
    assert indel_bos_pos is not None
    if len(seq_edit.sequence) and int(seq_edit.sequence[0]) == int(bos_token):
        indel_bos_pos = max(1, indel_bos_pos)
    serialised = token_ops.serialise(seq_edit, indel_bos_pos=indel_bos_pos)
    windowed = False
    if len(serialised.tokens) > candidate_budget:
        seq_edit = _window_candidate_sequence_edit(
            prepared,
            full_token_budget=candidate_budget,
            token_ops=token_ops,
            bos_token=bos_token,
        )
        if seq_edit is None:
            return None, {"windowed": 0, "padding_tokens": 0, "source_tokens": 0}
        indel_bos_pos = _sample_indel_bos_pos(seq_edit, cfg=cfg, rng=rng, edited=True)
        assert indel_bos_pos is not None
        if len(seq_edit.sequence) and int(seq_edit.sequence[0]) == int(bos_token):
            indel_bos_pos = max(1, indel_bos_pos)
        serialised = token_ops.serialise(seq_edit, indel_bos_pos=indel_bos_pos)
        windowed = True

    if len(serialised.tokens) > candidate_budget:
        return None, {"windowed": int(windowed), "padding_tokens": 0, "source_tokens": 0}
    roundtripped = token_ops.deserialise(serialised)
    if not np.array_equal(resolve(roundtripped), resolve(seq_edit)):
        raise AssertionError("Priority candidate changed meaning during serialization")

    def render(edit: SequenceEdit) -> TrainingData | None:
        if baseline_mode is None:
            serial = token_ops.serialise(edit, indel_bos_pos=indel_bos_pos)
            return token_ops.prepare_training_example(serial, pos_buffer_size)
        return _ordered_autoregressive_training(
            seq_edit=edit,
            tokenizer=baseline_tokenizer,
            token_budget=input_budget,
            pos_buffer_size=pos_buffer_size,
            baseline_mode=baseline_mode,
        )

    training = render(seq_edit)
    if training is None and baseline_mode is not None:
        recovered = _window_autoregressive_sequence_edit(
            seq_edit,
            tokenizer=baseline_tokenizer,
            token_budget=input_budget,
            pos_buffer_size=pos_buffer_size,
            baseline_mode=baseline_mode,
            bos_token=bos_token,
        )
        if recovered is not None:
            seq_edit, training = recovered
            windowed = True
    if training is None:
        return None, {"windowed": int(windowed), "padding_tokens": 0, "source_tokens": 0}
    if len(training.input_tokens) + 1 > candidate_budget:
        # Textual coordinates can be slightly longer than CES pointer queries.
        # Preserve the candidate and full-row budget, but waive this row's
        # requested suffix instead of silently dropping or truncating the edit.
        post_program_tail = False
    padding_tokens = input_budget - len(training.input_tokens)
    terminal_suffix_len = 0
    if (
        stream is not None
        and padding_tokens > 0
        and (baseline_mode is None or post_program_tail)
    ):
        tail = stream.pop_row_end_tokens(
            padding_tokens,
            max_context_tokens=full_token_budget,
            queue_priority_candidates=False,
        )
        if post_program_tail and len(tail) >= 2:
            terminal_suffix_len = len(tail)
            training = _append_post_program_tail(
                training, tail, pos_buffer_size=pos_buffer_size
            )
        else:
            seq_edit = _append_source_tail(seq_edit, tail)
            training = render(seq_edit)
            if training is None:
                raise ValueError("Ordered priority baseline no longer fits after source tail")
        padding_tokens = input_budget - len(training.input_tokens)
    training = _pad_or_trim_training_data(training, input_budget)
    return training, {
        "windowed": int(windowed),
        "padding_tokens": int(padding_tokens),
        "source_tokens": len(seq_edit.sequence),
        "edit_mass": sequence_edit_mass(seq_edit),
        "terminal_suffix_tokens": terminal_suffix_len,
        "ended_at_edit_end": int(not post_program_tail),
    }


def _build_ordered_edit_piece(
    *,
    stream: _OrderedDocumentStream,
    input_budget: int,
    cfg: EditSamplingConfig,
    token_ops: TokensOperations,
    reservoir: ChunkReservoir,
    rng: np.random.Generator,
    pos_buffer_size: int,
    baseline_mode: str | None = None,
    baseline_tokenizer=None,
    defer_oversized_candidate: bool = False,
    max_deferred_candidate_tokens: int | None = None,
    row_terminal: bool = False,
    row_context_tokens: int | None = None,
) -> tuple[TrainingData | None, dict[str, int]]:
    """Build one repaired ordered edit region in CES or AR-baseline serialization."""

    if baseline_mode not in {None, "ar_dsl", "flat_relative"}:
        raise ValueError(f"Unsupported ordered baseline mode: {baseline_mode}")
    if baseline_mode is not None and baseline_tokenizer is None:
        raise ValueError("baseline_tokenizer is required for an ordered AR baseline")

    full_token_budget = input_budget + 1
    post_program_tail = bool(
        cfg.post_edit_clean_suffix
        and rng.random() >= min(max(float(cfg.terminal_edit_end_prob), 0.0), 1.0)
    )
    terminal_suffix_tokens = (
        max(2, int(cfg.terminal_clean_suffix_min)) if post_program_tail else 0
    )
    terminal_suffix_tokens = min(
        terminal_suffix_tokens,
        max(2, input_budget // 4),
        max(full_token_budget - 8, 0),
    )
    reserve = min(
        max(int(cfg.random_source_token_reserve), 3),
        max(8, (full_token_budget - terminal_suffix_tokens) // 3),
    )
    if baseline_mode is not None:
        # Textual operation names and coordinates need more room than CES
        # pointer queries. The unconsumed stream tokens still fill the same
        # repaired row after the complete AR program.
        available = max(full_token_budget - terminal_suffix_tokens, 3)
        reserve = max(reserve, min(max(24, 3 * input_budget // 4), available - 2))

    source_edit, candidate_docs, deferred_candidate, dropped_candidate_docs = stream.pop_candidate_source(
        full_token_budget=full_token_budget,
        reserve=reserve,
        token_ops=token_ops,
        terminal_suffix_tokens=terminal_suffix_tokens,
        defer_oversized_candidate=defer_oversized_candidate,
        max_deferred_candidate_tokens=max_deferred_candidate_tokens,
    )
    if deferred_candidate and len(source_edit.sequence) == 0:
        if row_terminal and post_program_tail:
            fallback, fallback_stats = _build_ordered_edit_piece(
                stream=stream,
                input_budget=input_budget,
                cfg=replace(cfg, post_edit_clean_suffix=False),
                token_ops=token_ops,
                reservoir=reservoir,
                rng=rng,
                pos_buffer_size=pos_buffer_size,
                defer_oversized_candidate=defer_oversized_candidate,
                baseline_mode=baseline_mode,
                baseline_tokenizer=baseline_tokenizer,
                max_deferred_candidate_tokens=max_deferred_candidate_tokens,
                row_terminal=row_terminal,
                row_context_tokens=row_context_tokens,
            )
            if fallback is not None:
                fallback_stats["terminal_suffix_fallback_rows"] = 1
                return fallback, fallback_stats
        return None, {
            "candidate_docs": 0,
            "candidate_docs_deferred": 1,
            "candidate_docs_dropped": 0,
            "random_edit_regions": 0,
            "noop_edit_regions": 0,
            "edit_regions": 0,
            "edit_mass": 0,
            "source_tokens": 0,
        }
    random_edit = False
    if not source_edit.edits and rng.random() < cfg.random_corruption_prob_when_no_candidates:
        scaled_cfg = replace(
            cfg,
            max_edits_per_example=max(
                1, min(cfg.max_edits_per_example, max(1, input_budget // 96))
            ),
        )
        for _ in range(16):
            sampled = _sample_random_inverse_edit(
                np.asarray(source_edit.sequence, dtype=np.int64),
                cfg=scaled_cfg,
                token_ops=token_ops,
                reservoir=reservoir,
                rng=rng,
            )
            if sampled is None:
                continue
            sampled_pos = _sample_indel_bos_pos(sampled, cfg=cfg, rng=rng, edited=True)
            assert sampled_pos is not None
            if len(sampled.sequence) and int(sampled.sequence[0]) == stream.bos_token:
                sampled_pos = max(1, sampled_pos)
            if (
                len(token_ops.serialise(sampled, indel_bos_pos=sampled_pos).tokens)
                <= full_token_budget - terminal_suffix_tokens
            ):
                source_edit = sampled
                random_edit = True
                break

    indel_bos_pos = _sample_indel_bos_pos(source_edit, cfg=cfg, rng=rng, edited=True)
    assert indel_bos_pos is not None
    if len(source_edit.sequence) and int(source_edit.sequence[0]) == stream.bos_token:
        indel_bos_pos = max(1, indel_bos_pos)
    serialised = token_ops.serialise(source_edit, indel_bos_pos=indel_bos_pos)
    if len(serialised.tokens) > full_token_budget - terminal_suffix_tokens:
        raise ValueError(
            "Ordered edit piece exceeded its source/program budget: "
            f"{len(serialised.tokens)} > {full_token_budget - terminal_suffix_tokens}; "
            f"source={len(source_edit.sequence)}, edits={len(source_edit.edits)}, "
            f"candidate_docs={candidate_docs}, random_edit={random_edit}, "
            f"deferred={deferred_candidate}, suffix={terminal_suffix_tokens}, "
            f"indel_bos_pos={indel_bos_pos}"
        )

    def render(edit: SequenceEdit) -> TrainingData | None:
        if baseline_mode is None:
            rendered = token_ops.serialise(edit, indel_bos_pos=indel_bos_pos)
            return token_ops.prepare_training_example(rendered, pos_buffer_size)
        return _ordered_autoregressive_training(
            seq_edit=edit,
            tokenizer=baseline_tokenizer,
            token_budget=input_budget,
            pos_buffer_size=pos_buffer_size,
            baseline_mode=baseline_mode,
        )

    training = render(source_edit)
    baseline_windowed = 0
    baseline_unrepresentable = 0
    if training is None and baseline_mode is not None:
        recovered = _window_autoregressive_sequence_edit(
            source_edit,
            tokenizer=baseline_tokenizer,
            token_budget=input_budget,
            pos_buffer_size=pos_buffer_size,
            baseline_mode=baseline_mode,
            bos_token=stream.bos_token,
        )
        if recovered is not None:
            source_edit, training = recovered
            baseline_windowed = 1
    if training is None and baseline_mode is not None:
        # A payload can itself consume the entire matched region. The baseline
        # cannot represent that example; train an explicit no-op and count it
        # instead of allowing one rare row to terminate a multi-hour run.
        sequence = np.asarray(source_edit.sequence, dtype=np.int64)
        for source_len in range(min(len(sequence), input_budget), 0, -1):
            fallback = SequenceEdit(sequence=sequence[:source_len], edits=[])
            training = render(fallback)
            if training is not None:
                source_edit = fallback
                baseline_unrepresentable = 1
                break
    if training is None:
        raise ValueError(
            f"Ordered {baseline_mode or 'ces'} row cannot fit within "
            f"{input_budget} input tokens, including an empty edit program"
        )

    # When a whole-document candidate is waiting at the stream head, do not
    # consume it merely to pad this region. The caller carries this unused
    # budget into the next Gamma piece, where the candidate can fit intact.
    tail_len = 0 if deferred_candidate else input_budget - len(training.input_tokens)
    if tail_len < 0:
        raise ValueError(
            f"Ordered {baseline_mode or 'ces'} edit program exceeded its input "
            f"budget: {len(training.input_tokens)} > {input_budget}"
        )
    if post_program_tail and tail_len < 2:
        post_program_tail = False
    terminal_suffix_len = 0
    if tail_len and (baseline_mode is None or post_program_tail):
        if row_terminal:
            tail = stream.pop_row_end_tokens(
                tail_len,
                max_context_tokens=row_context_tokens or full_token_budget,
            )
        else:
            tail = stream.pop_clean_tokens(tail_len)
        if post_program_tail and len(tail) >= 2:
            terminal_suffix_len = len(tail)
            training = _append_post_program_tail(
                training, tail, pos_buffer_size=pos_buffer_size
            )
        elif tail:
            source_edit = _append_source_tail(source_edit, tail)
            training = render(source_edit)
            if training is None:
                raise ValueError(
                    f"Ordered {baseline_mode or 'ces'} edit program no longer "
                    "fits after appending its clean source tail"
                )
    if (
        not deferred_candidate
        and not row_terminal
        and len(training.input_tokens) != input_budget
        and baseline_mode is None
    ):
        raise AssertionError(
            f"Ordered edit piece did not fill its budget: {len(training.input_tokens)} != {input_budget}"
        )

    return training, {
        "candidate_docs": candidate_docs,
        "candidate_docs_deferred": int(deferred_candidate),
        "candidate_docs_dropped": dropped_candidate_docs,
        "random_edit_regions": int(random_edit),
        "noop_edit_regions": int(not source_edit.edits),
        "edit_regions": 1,
        "source_tokens": len(source_edit.sequence),
        "edit_mass": sequence_edit_mass(source_edit) if source_edit.edits else 0,
        "terminal_suffix_tokens": terminal_suffix_len,
        "ended_at_edit_end": int(not post_program_tail and not deferred_candidate),
        "baseline_windowed_regions": baseline_windowed,
        "baseline_unrepresentable_regions": baseline_unrepresentable,
    }


def _edit_end_supervision_counts(training: TrainingData, edit_end_token: int) -> tuple[int, int]:
    targets = np.asarray(training.target.tokens)
    weights = np.asarray(training.target.token_loss_weights, dtype=bool)
    indices = np.where((targets == edit_end_token) & weights)[0]
    followed = sum(int(ix + 1 < len(targets) and weights[ix + 1]) for ix in indices)
    return int(len(indices)), int(followed)


def _pack_one_row_sequential(
    *,
    doc_buffer: list[_PreparedDoc],
    refill,
    row_capacity: int,
) -> tuple[list[int], list[tuple[int, _PreparedDoc]]]:
    row: list[int] = []
    full_docs: list[tuple[int, _PreparedDoc]] = []
    while len(row) < row_capacity:
        while not doc_buffer:
            refill()
        doc = doc_buffer.pop(0)
        remaining = row_capacity - len(row)
        if len(doc.tokens) <= remaining:
            offset = len(row)
            row.extend(doc.tokens)
            full_docs.append((offset, doc))
        else:
            row.extend(doc.tokens[:remaining])
    return row[:row_capacity], full_docs


def _sample_random_source_len(
    row_len: int,
    token_budget: int,
    cfg: EditSamplingConfig,
    rng: np.random.Generator,
) -> int:
    """Choose the source prefix length for a random edit example.

    Historically every random edit used the same reserved suffix, placing its
    edit program at almost the same absolute sequence position. Optional
    reserve jitter breaks that shortcut while leaving plain AR rows fully
    packed.
    """

    reserve_min = max(int(cfg.random_source_token_reserve), 0)
    reserve_max = cfg.random_source_token_reserve_max
    reserve_max = reserve_min if reserve_max is None else max(int(reserve_max), reserve_min)
    reserve_max = min(reserve_max, max(token_budget - 1, reserve_min))
    jitter_prob = min(max(float(cfg.random_source_length_jitter_prob), 0.0), 1.0)
    reserve = reserve_min
    if reserve_max > reserve_min and rng.random() < jitter_prob:
        reserve = int(rng.integers(reserve_min, reserve_max + 1))
    return max(2, min(int(row_len), token_budget + 1 - reserve))


def _edit_guard_boundary(edit: Any) -> int:
    """Earliest source boundary that must remain editable for this edit."""

    if isinstance(edit, Insert):
        return int(edit.pos)
    if isinstance(edit, (Copy, Move)):
        return min(int(edit.start), int(edit.dest))
    return int(edit.start)


def _sample_indel_bos_pos(
    seq_edit: SequenceEdit,
    *,
    cfg: EditSamplingConfig,
    rng: np.random.Generator,
    edited: bool,
) -> int | None:
    """Sample the guard token position for CES serialisation.

    ``None`` means leave the row as plain AR data for clean/noop examples.
    For edited examples, sample only from boundaries that keep all edit addresses
    after the guard, so training targets match inference-time pointer masking.
    """

    n = len(seq_edit.sequence)
    if edited:
        start_prob = max(float(cfg.corrupted_indel_bos_start_prob), 0.0)
        random_prob = max(float(cfg.corrupted_indel_bos_random_prob), 0.0)
        total = start_prob + random_prob
        roll = rng.random() * total if total > 0 else 0.0
        if total <= 0 or roll < start_prob:
            return 0
        if seq_edit.edits:
            max_pos = min(_edit_guard_boundary(edit) for edit in seq_edit.edits)
        else:
            max_pos = n
        max_pos = max(0, min(int(max_pos), n))
        return int(rng.integers(0, max_pos + 1))

    absent_prob = max(float(cfg.clean_indel_bos_absent_prob), 0.0)
    start_prob = max(float(cfg.clean_indel_bos_start_prob), 0.0)
    random_prob = max(float(cfg.clean_indel_bos_random_prob), 0.0)
    total = absent_prob + start_prob + random_prob
    roll = rng.random() * total if total > 0 else 0.0
    if total <= 0 or roll < absent_prob:
        return None
    if roll < absent_prob + start_prob:
        return 0
    return int(rng.integers(0, n + 1))


def _training_data_from_sequence_edit(
    *,
    seq_edit: SequenceEdit,
    token_ops: TokensOperations,
    token_budget: int,
    pos_buffer_size: int,
    cfg: EditSamplingConfig,
    rng: np.random.Generator,
    edited: bool,
) -> TrainingData | None:
    indel_bos_pos = _sample_indel_bos_pos(seq_edit, cfg=cfg, rng=rng, edited=edited)
    if indel_bos_pos is None:
        return _plain_training_data(seq_edit.sequence, token_budget, pos_buffer_size)
    try:
        serialised = token_ops.serialise(seq_edit, indel_bos_pos=indel_bos_pos)
    except ValueError:
        return None
    if len(serialised.tokens) > token_budget + 1:
        return None
    return _pad_or_trim_training_data(
        token_ops.prepare_training_example(serialised, pos_buffer_size),
        token_budget,
    )


def _training_data_from_ar_dsl_sequence_edit(
    *,
    seq_edit: SequenceEdit,
    tokenizer,
    token_budget: int,
    pos_buffer_size: int,
    pad_to_budget: bool = True,
    completion_only_loss: bool = False,
    terminal_weight: float = 1.0,
) -> TrainingData | None:
    """Build a plain AR row: source tokens followed by a textual edit script."""

    dsl_tokens = sequence_edit_to_ar_dsl_tokens(seq_edit, tokenizer)
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    row = [
        *[int(t) for t in np.array(seq_edit.sequence).tolist()],
        *dsl_tokens,
        assistant_end,
    ]
    if len(row) > token_budget + 1:
        return None
    training = _plain_training_data(row, len(row) - 1, pos_buffer_size)
    if completion_only_loss:
        weights = np.zeros(len(row) - 1, dtype=np.float32)
        completion_start = max(len(seq_edit.sequence) - 1, 0)
        weights[completion_start:] = 1.0
        effective_terminal_weight = (
            max(len(dsl_tokens), 1) if terminal_weight < 0 else terminal_weight
        )
        if effective_terminal_weight <= 0:
            raise ValueError(
                "terminal_weight must be positive or negative for automatic weighting"
            )
        weights[-1] = effective_terminal_weight
        training = TrainingData(
            input_tokens=training.input_tokens,
            input_positions=training.input_positions,
            target=training.target._replace(token_loss_weights=weights),
        )
    return (
        _pad_or_trim_training_data(training, token_budget)
        if pad_to_budget
        else training
    )


def _training_data_from_flat_relative_sequence_edit(
    *,
    seq_edit: SequenceEdit,
    tokenizer,
    token_budget: int,
    pos_buffer_size: int,
    pad_to_budget: bool = True,
    completion_only_loss: bool = False,
    terminal_weight: float = 1.0,
) -> TrainingData | None:
    """Build a plain AR row whose coordinate fields are BACK_k tokens."""

    try:
        dsl_tokens = sequence_edit_to_flat_relative_tokens(seq_edit, tokenizer)
    except ValueError:
        return None
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    row = [
        *[int(token) for token in np.asarray(seq_edit.sequence).tolist()],
        *dsl_tokens,
        assistant_end,
    ]
    if len(row) > token_budget + 1:
        return None
    training = _plain_training_data(row, len(row) - 1, pos_buffer_size)
    if completion_only_loss:
        weights = np.zeros(len(row) - 1, dtype=np.float32)
        completion_start = max(len(seq_edit.sequence) - 1, 0)
        weights[completion_start:] = 1.0
        effective_terminal_weight = (
            max(len(dsl_tokens), 1) if terminal_weight < 0 else terminal_weight
        )
        if effective_terminal_weight <= 0:
            raise ValueError(
                "terminal_weight must be positive or negative for automatic weighting"
            )
        weights[-1] = effective_terminal_weight
        training = TrainingData(
            input_tokens=training.input_tokens,
            input_positions=training.input_positions,
            target=training.target._replace(token_loss_weights=weights),
        )
    return (
        _pad_or_trim_training_data(training, token_budget)
        if pad_to_budget
        else training
    )


def _ordered_autoregressive_training(
    *,
    seq_edit: SequenceEdit,
    tokenizer,
    token_budget: int,
    pos_buffer_size: int,
    baseline_mode: str,
) -> TrainingData | None:
    """Render one repaired ordered-loader edit region without early padding."""

    builder = (
        _training_data_from_ar_dsl_sequence_edit
        if baseline_mode == "ar_dsl"
        else _training_data_from_flat_relative_sequence_edit
    )
    return builder(
        seq_edit=seq_edit,
        tokenizer=tokenizer,
        token_budget=token_budget,
        pos_buffer_size=pos_buffer_size,
        pad_to_budget=False,
    )


def tokenizing_distributed_partitioned_edit_loader_with_state(
    tokenizer,
    B: int,
    T: int,
    split: str,
    cfg: EditSamplingConfig,
    token_ops: TokensOperations,
    tokenizer_batch_size: int = 128,
    device: str = "cuda",
    resume_state_dict=None,
    pos_buffer_size: int = 64,
    reservoir_max_chunks: int = 8192,
    reservoir_chunk_size: int = 2048,
    baseline_mode: str | None = None,
):
    """Ordered repaired mixture with clean, partitioned, and full rows.

    Gamma regions consume consecutive tokens from one shared document stream.
    Each edited region incorporates fitting whole-document GPT/copy candidates,
    otherwise samples a random inverse edit. Concatenated regions remain in one
    causal context. ``baseline_mode`` changes only the edit-program serialization.
    """

    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    if baseline_mode not in {None, "ar_dsl", "flat_relative"}:
        raise ValueError(f"Unsupported ordered baseline mode: {baseline_mode}")

    clean_prob = float(cfg.stream_clean_row_prob)
    partition_prob = float(cfg.stream_partition_row_prob)
    if clean_prob < 0 or partition_prob < 0 or clean_prob + partition_prob > 1:
        raise ValueError(
            "stream_clean_row_prob and stream_partition_row_prob must be non-negative and sum to <= 1"
        )

    _ddp, ddp_rank, _ddp_local_rank, _ddp_world_size = get_dist_info()
    rng = np.random.default_rng(cfg.seed + ddp_rank)
    store = EditCandidateStore(cfg)
    rows = _document_rows(split, cfg, resume_state_dict, tokenizer_batch_size)
    pq_idx, rg_idx, epoch = 0, 0, 1
    bos_token = tokenizer.get_bos_token_id()
    use_cuda = device == "cuda"
    reservoir = ChunkReservoir(
        max_chunks=reservoir_max_chunks,
        chunk_size=reservoir_chunk_size,
        rng=rng,
    )

    def refill_rows():
        nonlocal pq_idx, rg_idx, epoch
        row_batch, (pq_idx, rg_idx, epoch) = next(rows)
        return row_batch

    stream = _OrderedDocumentStream(
        tokenizer=tokenizer,
        bos_token=bos_token,
        store=store,
        cfg=cfg,
        rng=rng,
        refill_rows=refill_rows,
    )
    force_full_edit_row = False

    def ten(values):
        array = np.stack(values)
        dtype = torch.long if np.issubdtype(array.dtype, np.integer) else torch.bool
        tensor = torch.as_tensor(array, dtype=dtype)
        if use_cuda:
            tensor = tensor.pin_memory()
        return tensor.to(device=device, non_blocking=use_cuda)

    while True:
        batch: list[TrainingData] = []
        stats = {
            "clean_rows": 0,
            "partition_rows": 0,
            "full_rows": 0,
            "candidate_docs": 0,
            "candidate_docs_deferred": 0,
            "candidate_docs_dropped": 0,
            "priority_candidate_rows": 0,
            "priority_candidate_windowed_rows": 0,
            "baseline_windowed_regions": 0,
            "baseline_unrepresentable_regions": 0,
            "priority_candidate_unusable": 0,
            "priority_candidate_padding_tokens": 0,
            "forced_candidate_rows": 0,
            "candidate_boundary_padding_tokens": 0,
            "random_edit_regions": 0,
            "noop_edit_regions": 0,
            "edit_regions": 0,
            "edited_source_tokens": 0,
            "edit_mass": 0,
            "edit_end_targets": 0,
            "edit_end_followed_by_supervised": 0,
            "edited_rows": 0,
            "terminal_suffix_fallback_rows": 0,
            "edited_rows_ending_at_edit_end": 0,
            "independent_rows": 0,
            "continuation_rows": 0,
            "independent_rows_bos_at_zero": 0,
        }
        docs_before = stream.raw_documents_consumed
        tokens_before = stream.clean_tokens_consumed
        candidates_seen_before = stream.candidate_docs_seen
        candidates_queued_before = stream.candidate_docs_queued

        for _ in range(B):
            forced_this_row = force_full_edit_row
            force_full_edit_row = False
            force_after_row = False
            priority_training = None
            row_is_continuation = bool(stream._front().offset)
            if not forced_this_row and cfg.priority_candidate_lane:
                while stream.priority_candidates:
                    prepared = stream.pop_priority_candidate()
                    assert prepared is not None
                    candidate_training, candidate_stats = _build_priority_candidate_row(
                        prepared,
                        stream=stream if cfg.bos_aligned_row_packing else None,
                        input_budget=T,
                        cfg=cfg,
                        token_ops=token_ops,
                        rng=rng,
                        pos_buffer_size=pos_buffer_size,
                        bos_token=bos_token,
                        baseline_mode=baseline_mode,
                        baseline_tokenizer=tokenizer,
                    )
                    if candidate_training is None:
                        stats["priority_candidate_unusable"] += 1
                        continue
                    priority_training = candidate_training
                    stats["priority_candidate_rows"] += 1
                    stats["priority_candidate_windowed_rows"] += candidate_stats["windowed"]
                    stats["priority_candidate_padding_tokens"] += candidate_stats[
                        "padding_tokens"
                    ]
                    stats["edited_source_tokens"] += candidate_stats["source_tokens"]
                    stats["edit_mass"] += candidate_stats["edit_mass"]
                    stats["candidate_docs"] += 1
                    stats["edit_regions"] += 1
                    break
            roll = 1.0 if forced_this_row else float(rng.random())
            if priority_training is not None:
                row_is_continuation = False
                training = priority_training
            elif not forced_this_row and roll < clean_prob:
                if cfg.bos_aligned_row_packing:
                    clean_tokens = stream.pop_row_end_tokens(
                        T + 1, max_context_tokens=T + 1
                    )
                    if len(clean_tokens) < 2:
                        raise AssertionError("BOS packer made no progress")
                    training = _plain_training_data(
                        clean_tokens, len(clean_tokens) - 1, pos_buffer_size
                    )
                    training = _pad_or_trim_training_data(training, T)
                else:
                    clean_tokens = stream.pop_clean_tokens(T + 1)
                    training = _plain_training_data(clean_tokens, T, pos_buffer_size)
                stats["clean_rows"] += 1
                if len(clean_tokens) >= reservoir_chunk_size:
                    reservoir.offer(clean_tokens[:reservoir_chunk_size])
            elif roll < clean_prob + partition_prob:
                parts = []
                budgets = _partition_input_budgets(T, cfg, rng)
                if baseline_mode is not None and len(budgets) > 1:
                    # The repaired physical row is unchanged, but textual DSL
                    # headers cannot fit in arbitrarily tiny Gamma regions.
                    minimum = min(T, 96)
                    coalesced = []
                    pending = 0
                    for budget in budgets:
                        pending = budget if pending == 0 else pending + budget + 1
                        if pending >= minimum:
                            coalesced.append(pending)
                            pending = 0
                    if pending:
                        if coalesced:
                            coalesced[-1] += pending + 1
                        else:
                            coalesced.append(pending)
                    budgets = coalesced
                budget_ix = 0
                while budget_ix < len(budgets):
                    budget = budgets[budget_ix]
                    has_next_budget = budget_ix + 1 < len(budgets)
                    part, part_stats = _build_ordered_edit_piece(
                        stream=stream,
                        input_budget=budget,
                        cfg=cfg,
                        token_ops=token_ops,
                        reservoir=reservoir,
                        rng=rng,
                        pos_buffer_size=pos_buffer_size,
                        defer_oversized_candidate=True,
                        max_deferred_candidate_tokens=T + 1,
                        row_terminal=(
                            cfg.bos_aligned_row_packing and not has_next_budget
                        ),
                        row_context_tokens=T + 1,
                        baseline_mode=baseline_mode,
                        baseline_tokenizer=tokenizer,
                    )
                    if part is None:
                        stats["candidate_docs_deferred"] += part_stats[
                            "candidate_docs_deferred"
                        ]
                        if not has_next_budget:
                            force_after_row = True
                            break
                        # No stream token was consumed: remove this boundary and
                        # give its complete physical capacity to the next region.
                        budgets[budget_ix + 1] += budget + 1
                        budgets.pop(budget_ix)
                        continue
                    parts.append(part)
                    for key in (
                        "candidate_docs",
                        "candidate_docs_deferred",
                        "candidate_docs_dropped",
                        "random_edit_regions",
                        "noop_edit_regions",
                        "baseline_windowed_regions",
                        "baseline_unrepresentable_regions",
                        "edit_mass",
                        "edit_regions",
                    ):
                        stats[key] += part_stats[key]
                    stats["edited_source_tokens"] += part_stats["source_tokens"]
                    stats["terminal_suffix_fallback_rows"] += part_stats.get(
                        "terminal_suffix_fallback_rows", 0
                    )
                    unused_budget = budget - len(part.input_tokens)
                    if unused_budget:
                        if has_next_budget:
                            budgets[budget_ix + 1] += unused_budget
                        else:
                            break
                    budget_ix += 1
                training = _concat_training_data(parts)
                stats["partition_rows"] += 1
            else:
                training, part_stats = _build_ordered_edit_piece(
                    stream=stream,
                    input_budget=T,
                    cfg=cfg,
                    token_ops=token_ops,
                    reservoir=reservoir,
                    rng=rng,
                    pos_buffer_size=pos_buffer_size,
                    defer_oversized_candidate=True,
                    max_deferred_candidate_tokens=T + 1,
                    row_terminal=cfg.bos_aligned_row_packing,
                    row_context_tokens=T + 1,
                    baseline_mode=baseline_mode,
                    baseline_tokenizer=tokenizer,
                )
                assert training is not None
                for key in (
                    "candidate_docs",
                    "candidate_docs_deferred",
                    "candidate_docs_dropped",
                    "random_edit_regions",
                    "noop_edit_regions",
                    "edit_mass",
                    "baseline_windowed_regions",
                    "baseline_unrepresentable_regions",
                    "edit_regions",
                ):
                    stats[key] += part_stats[key]
                stats["edited_source_tokens"] += part_stats["source_tokens"]
                stats["terminal_suffix_fallback_rows"] += part_stats.get(
                    "terminal_suffix_fallback_rows", 0
                )
                stats["full_rows"] += 1
                stats["forced_candidate_rows"] += int(forced_this_row)
                force_after_row = bool(part_stats["candidate_docs_deferred"])

            if len(training.input_tokens) < T:
                padding = T - len(training.input_tokens)
                training = _pad_or_trim_training_data(training, T)
                stats["candidate_boundary_padding_tokens"] += padding
                force_full_edit_row = force_after_row

            if baseline_mode is None:
                ends, followed = _edit_end_supervision_counts(training, token_ops.edit_end_token)
            else:
                ends, followed = 0, 0
            stats["edit_end_targets"] += ends
            stats["edit_end_followed_by_supervised"] += followed
            stats["edited_rows"] += int(ends > 0)
            stats["edited_rows_ending_at_edit_end"] += int(
                ends > 0
                and int(training.target.tokens[-1]) == int(token_ops.edit_end_token)
                and bool(training.target.token_loss_weights[-1])
            )
            stats["continuation_rows"] += int(row_is_continuation)
            stats["independent_rows"] += int(not row_is_continuation)
            stats["independent_rows_bos_at_zero"] += int(
                not row_is_continuation and int(training.input_tokens[0]) == int(bos_token)
            )
            if len(training.input_tokens) != T or len(training.target.tokens) != T:
                raise AssertionError("Partitioned loader produced a non-T training row")
            batch.append(training)

        target_kwargs = {
            field_name: ten([getattr(item.target, field_name) for item in batch])
            for field_name in [
                "tokens",
                "gathered_pos_target",
                "token_loss_weights",
                "gather_indices",
                "gather_valid",
                "overflow",
            ]
        }
        training_batch = TrainingData(
            input_tokens=ten([item.input_tokens for item in batch]),
            input_positions=ten([item.input_positions for item in batch]),
            target=TrainingTarget(**target_kwargs),
        )
        stats.update(
            {
                "pq_idx": pq_idx,
                "rg_idx": rg_idx,
                "epoch": epoch,
                "raw_examples_consumption": stream.raw_documents_consumed - docs_before,
                "raw_token_consumption": stream.clean_tokens_consumed - tokens_before,
                "candidate_docs_seen": stream.candidate_docs_seen - candidates_seen_before,
                "candidate_docs_queued": stream.candidate_docs_queued - candidates_queued_before,
                "priority_candidate_queue_remaining": len(stream.priority_candidates),
                "edit_mass_rho": stats["edit_mass"] / max(stream.clean_tokens_consumed - tokens_before, 1),
                "valid_pointer_targets": int(
                    sum(np.asarray(item.target.gather_valid, dtype=bool).sum() for item in batch)
                ),
            }
        )
        yield training_batch, stats


def tokenizing_distributed_sequential_edit_loader_with_state(
    tokenizer,
    B: int,
    T: int,
    split: str,
    cfg: EditSamplingConfig,
    token_ops: TokensOperations,
    tokenizer_batch_size: int = 128,
    device: str = "cuda",
    resume_state_dict=None,
    pos_buffer_size: int = 64,
    reservoir_max_chunks: int = 8192,
    reservoir_chunk_size: int = 2048,
):
    """Sequential BOS-packed edit loader.

    This follows the corpus stream order. Documents with generated/copy candidates
    are converted to their sampled edit-source sequence before tokenization. Packed
    rows with no candidate edits can optionally receive random inverse corruption.
    """

    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    _ddp, ddp_rank, _ddp_local_rank, _ddp_world_size = get_dist_info()
    rng = np.random.default_rng(cfg.seed + ddp_rank)
    store = EditCandidateStore(cfg)
    rows = _document_rows(split, cfg, resume_state_dict, tokenizer_batch_size)
    doc_buffer: list[_PreparedDoc] = []
    pq_idx, rg_idx, epoch = 0, 0, 1
    bos_token = tokenizer.get_bos_token_id()
    row_capacity = max(2, T + 1 - max(int(cfg.random_source_token_reserve), 0))
    use_cuda = device == "cuda"
    reservoir = ChunkReservoir(
        max_chunks=reservoir_max_chunks,
        chunk_size=reservoir_chunk_size,
        rng=rng,
    )

    def refill():
        nonlocal pq_idx, rg_idx, epoch
        row_batch, (pq_idx, rg_idx, epoch) = next(rows)
        for shard_id, row_index, raw_text in row_batch:
            doc_buffer.append(
                _prepare_doc(
                    shard_id=shard_id,
                    row_index=row_index,
                    raw_text=raw_text,
                    tokenizer=tokenizer,
                    bos_token=bos_token,
                    store=store,
                    cfg=cfg,
                    rng=rng,
                )
            )

    def ten(x_list):
        x = np.stack(x_list)
        dtype = torch.long if np.issubdtype(x.dtype, np.integer) else torch.bool
        tensor = torch.as_tensor(x, dtype=dtype)
        if use_cuda:
            tensor = tensor.pin_memory()
        return tensor.to(device=device, non_blocking=use_cuda)

    while True:
        batch: list[TrainingData] = []
        candidate_rows = 0
        random_rows = 0
        random_plain_rows = 0
        noop_rows = 0
        plain_rows = 0
        raw_examples_consumption = 0
        raw_token_consumption = 0
        edit_mass = 0
        edited_source_tokens = 0
        valid_pointer_targets = 0
        for _ in range(B):
            row, docs = _pack_one_row_sequential(
                doc_buffer=doc_buffer,
                refill=refill,
                row_capacity=row_capacity,
            )
            raw_examples_consumption += len(docs)
            raw_token_consumption += len(row)
            if len(row) >= reservoir_chunk_size:
                reservoir.offer(row[:reservoir_chunk_size])

            edits: list[Any] = []
            for offset, doc in docs:
                if doc.token_edit is None:
                    continue
                edits.extend(_offset_edit(edit, offset) for edit in doc.token_edit.edits)

            training_data = None
            accepted_edit = None
            if edits:
                seq_edit = SequenceEdit(sequence=np.array(row, dtype=np.int64), edits=sorted(edits, key=_edit_key))
                if cfg.plain_corruption_only:
                    training_data = _plain_training_data(seq_edit.sequence, T, pos_buffer_size)
                else:
                    training_data = _training_data_from_sequence_edit(
                        seq_edit=seq_edit,
                        token_ops=token_ops,
                        token_budget=T,
                        pos_buffer_size=pos_buffer_size,
                        cfg=cfg,
                        rng=rng,
                        edited=True,
                    )
                if training_data is not None:
                    candidate_rows += 1
                    accepted_edit = seq_edit

            fallback_roll = rng.random() if training_data is None and not edits else None
            random_plain_threshold = cfg.random_plain_corruption_prob_when_no_candidates

            if (
                training_data is None
                and not edits
                and fallback_roll is not None
                and fallback_roll < random_plain_threshold
            ):
                random_source_len = _sample_random_source_len(len(row), T, cfg, rng)
                random_edit = _sample_random_inverse_edit(
                    np.array(row[:random_source_len], dtype=np.int64),
                    cfg=cfg,
                    token_ops=token_ops,
                    reservoir=reservoir,
                    rng=rng,
                )
                if random_edit is not None:
                    corrupted = [
                        *[int(x) for x in random_edit.sequence],
                        *[int(x) for x in row[random_source_len:]],
                    ]
                    training_data = _plain_training_data(corrupted, T, pos_buffer_size)
                    random_plain_rows += 1

            random_edit_threshold = (
                cfg.random_plain_corruption_prob_when_no_candidates
                + cfg.random_corruption_prob_when_no_candidates
            )
            if (
                training_data is None
                and not edits
                and fallback_roll is not None
                and random_plain_threshold <= fallback_roll < random_edit_threshold
            ):
                random_source_len = _sample_random_source_len(len(row), T, cfg, rng)
                random_edit = _sample_random_inverse_edit(
                    np.array(row[:random_source_len], dtype=np.int64),
                    cfg=cfg,
                    token_ops=token_ops,
                    reservoir=reservoir,
                    rng=rng,
                )
                if random_edit is not None:
                    training_data = _training_data_from_sequence_edit(
                        seq_edit=random_edit,
                        token_ops=token_ops,
                        token_budget=T,
                        pos_buffer_size=pos_buffer_size,
                        cfg=cfg,
                        rng=rng,
                        edited=True,
                    )
                    if training_data is not None:
                        random_rows += 1
                        accepted_edit = random_edit

            noop_threshold = random_edit_threshold + cfg.noop_edit_prob_when_no_candidates
            if (
                training_data is None
                and not edits
                and fallback_roll is not None
                and random_edit_threshold <= fallback_roll < noop_threshold
            ):
                noop_source_len = _sample_random_source_len(len(row), T, cfg, rng)
                noop_edit = SequenceEdit(sequence=np.array(row[:noop_source_len], dtype=np.int64), edits=[])
                training_data = _training_data_from_sequence_edit(
                    seq_edit=noop_edit,
                    token_ops=token_ops,
                    token_budget=T,
                    pos_buffer_size=pos_buffer_size,
                    cfg=cfg,
                    rng=rng,
                    edited=False,
                )
                if training_data is not None:
                    noop_rows += 1

            if training_data is None:
                training_data = _plain_training_data(row, T, pos_buffer_size)
                plain_rows += 1
            if accepted_edit is not None:
                edit_mass += sequence_edit_mass(accepted_edit)
                edited_source_tokens += len(accepted_edit.sequence)
            valid_pointer_targets += int(np.asarray(training_data.target.gather_valid).sum())
            batch.append(training_data)

        kwgs = {
            field_name: ten([getattr(x.target, field_name) for x in batch])
            for field_name in [
                "tokens",
                "gathered_pos_target",
                "token_loss_weights",
                "gather_indices",
                "gather_valid",
                "overflow",
            ]
        }
        training_batch = TrainingData(
            input_tokens=ten([x.input_tokens for x in batch]),
            input_positions=ten([x.input_positions for x in batch]),
            target=TrainingTarget(**kwgs),
        )
        yield training_batch, {
            "pq_idx": pq_idx,
            "rg_idx": rg_idx,
            "epoch": epoch,
            "candidate_rows": candidate_rows,
            "random_rows": random_rows,
            "random_plain_rows": random_plain_rows,
            "noop_rows": noop_rows,
            "plain_rows": plain_rows,
            "raw_examples_consumption": raw_examples_consumption,
            "raw_token_consumption": raw_token_consumption,
            "edit_mass": edit_mass,
            "edit_mass_rho": edit_mass / max(raw_token_consumption, 1),
            "edited_source_tokens": edited_source_tokens,
            "valid_pointer_targets": valid_pointer_targets,
        }


def tokenizing_distributed_sequential_ar_dsl_loader_with_state(
    tokenizer,
    B: int,
    T: int,
    split: str,
    cfg: EditSamplingConfig,
    token_ops: TokensOperations,
    tokenizer_batch_size: int = 128,
    device: str = "cuda",
    resume_state_dict=None,
    pos_buffer_size: int = 64,
    reservoir_max_chunks: int = 8192,
    reservoir_chunk_size: int = 2048,
    baseline_mode: str = "ar_dsl",
    completion_only_loss: bool = False,
    terminal_weight: float = 1.0,
    single_document_pairs: bool = False,
):
    """Sequential loader for autoregressive edit-coordinate baselines."""

    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    if baseline_mode not in {"ar_dsl", "flat_relative"}:
        raise ValueError(f"Unsupported autoregressive edit baseline: {baseline_mode}")
    training_data_builder = (
        _training_data_from_ar_dsl_sequence_edit
        if baseline_mode == "ar_dsl"
        else _training_data_from_flat_relative_sequence_edit
    )
    _ddp, ddp_rank, _ddp_local_rank, _ddp_world_size = get_dist_info()
    rng = np.random.default_rng(cfg.seed + ddp_rank)
    store = EditCandidateStore(cfg)
    rows = _document_rows(split, cfg, resume_state_dict, tokenizer_batch_size)
    doc_buffer: list[_PreparedDoc] = []
    pq_idx, rg_idx, epoch = 0, 0, 1
    bos_token = tokenizer.get_bos_token_id()
    row_capacity = max(2, T + 1 - max(int(cfg.random_source_token_reserve), 0))
    use_cuda = device == "cuda"
    reservoir = ChunkReservoir(
        max_chunks=reservoir_max_chunks,
        chunk_size=reservoir_chunk_size,
        rng=rng,
    )

    def refill():
        nonlocal pq_idx, rg_idx, epoch
        row_batch, (pq_idx, rg_idx, epoch) = next(rows)
        for shard_id, row_index, raw_text in row_batch:
            doc_buffer.append(
                _prepare_doc(
                    shard_id=shard_id,
                    row_index=row_index,
                    raw_text=raw_text,
                    tokenizer=tokenizer,
                    bos_token=bos_token,
                    store=store,
                    cfg=cfg,
                    rng=rng,
                )
            )

    def ten(x_list):
        x = np.stack(x_list)
        dtype = torch.long if np.issubdtype(x.dtype, np.integer) else torch.bool
        tensor = torch.as_tensor(x, dtype=dtype)
        if use_cuda:
            tensor = tensor.pin_memory()
        return tensor.to(device=device, non_blocking=use_cuda)

    while True:
        batch: list[TrainingData] = []
        candidate_rows = 0
        random_rows = 0
        plain_rows = 0
        raw_examples_consumption = 0
        raw_token_consumption = 0
        for _ in range(B):
            if single_document_pairs:
                while not doc_buffer:
                    refill()
                document = doc_buffer.pop(0)
                row = [int(token) for token in document.tokens[:row_capacity]]
                docs = [(0, document)] if len(document.tokens) <= row_capacity else []
            else:
                row, docs = _pack_one_row_sequential(
                    doc_buffer=doc_buffer,
                    refill=refill,
                    row_capacity=row_capacity,
                )
            raw_examples_consumption += len(docs)
            raw_token_consumption += len(row)
            if len(row) >= reservoir_chunk_size:
                reservoir.offer(row[:reservoir_chunk_size])

            edits: list[Any] = []
            for offset, doc in docs:
                if doc.token_edit is None:
                    continue
                edits.extend(_offset_edit(edit, offset) for edit in doc.token_edit.edits)

            clean_probability = float(cfg.stream_clean_row_prob)
            force_clean = clean_probability > 0 and rng.random() < clean_probability
            training_data = (
                _plain_training_data(row, T, pos_buffer_size) if force_clean else None
            )
            plain_rows += int(force_clean)
            if edits and not force_clean:
                seq_edit = SequenceEdit(
                    sequence=np.array(row, dtype=np.int64),
                    edits=sorted(edits, key=_edit_key),
                )
                training_data = training_data_builder(
                    seq_edit=seq_edit,
                    tokenizer=tokenizer,
                    token_budget=T,
                    pos_buffer_size=pos_buffer_size,
                    completion_only_loss=completion_only_loss,
                    terminal_weight=terminal_weight,
                )
                if training_data is not None:
                    candidate_rows += 1

            fallback_roll = (
                rng.random()
                if training_data is None and not edits and not force_clean
                else None
            )
            if (
                training_data is None
                and not edits
                and fallback_roll is not None
                and fallback_roll < cfg.random_corruption_prob_when_no_candidates
            ):
                random_source_len = _sample_random_source_len(len(row), T, cfg, rng)
                random_edit = _sample_random_inverse_edit(
                    np.array(row[:random_source_len], dtype=np.int64),
                    cfg=cfg,
                    token_ops=token_ops,
                    reservoir=reservoir,
                    rng=rng,
                )
                if random_edit is not None:
                    suffix = [int(x) for x in row[random_source_len:]]
                    if suffix:
                        random_edit = SequenceEdit(
                            sequence=np.array([*random_edit.sequence.tolist(), *suffix], dtype=np.int64),
                            edits=random_edit.edits,
                        )
                    training_data = training_data_builder(
                        seq_edit=random_edit,
                        tokenizer=tokenizer,
                        token_budget=T,
                        pos_buffer_size=pos_buffer_size,
                        completion_only_loss=completion_only_loss,
                        terminal_weight=terminal_weight,
                    )
                    if training_data is not None:
                        random_rows += 1

            if training_data is None:
                training_data = _plain_training_data(row, T, pos_buffer_size)
                plain_rows += 1
            batch.append(training_data)

        kwgs = {
            field_name: ten([getattr(x.target, field_name) for x in batch])
            for field_name in [
                "tokens",
                "gathered_pos_target",
                "token_loss_weights",
                "gather_indices",
                "gather_valid",
                "overflow",
            ]
        }
        training_batch = TrainingData(
            input_tokens=ten([x.input_tokens for x in batch]),
            input_positions=ten([x.input_positions for x in batch]),
            target=TrainingTarget(**kwgs),
        )
        yield training_batch, {
            "pq_idx": pq_idx,
            "rg_idx": rg_idx,
            "epoch": epoch,
            "candidate_rows": candidate_rows,
            "random_rows": random_rows,
            "plain_rows": plain_rows,
            "raw_examples_consumption": raw_examples_consumption,
            "raw_token_consumption": raw_token_consumption,
        }


def tokenizing_distributed_text_edit_baseline_loader_with_state(
    tokenizer,
    B: int,
    T: int,
    split: str,
    cfg: EditSamplingConfig,
    token_ops: TokensOperations,
    tokenizer_batch_size: int = 128,
    device: str = "cuda",
    resume_state_dict=None,
    pos_buffer_size: int = 64,
    baseline_mode: str = "regeneration",
    baseline_terminal_weight: float = 1.0,
    reservoir_max_chunks: int = 8192,
    reservoir_chunk_size: int = 2048,
    single_document_pairs: bool = False,
):
    """Matched source-to-target loader for regeneration and FIM variants."""

    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    if baseline_mode not in {"regeneration", "fim", "fim_predicted"}:
        raise ValueError(f"Unsupported baseline mode: {baseline_mode}")
    _ddp, ddp_rank, _ddp_local_rank, _ddp_world_size = get_dist_info()
    rng = np.random.default_rng(cfg.seed + ddp_rank)
    store = EditCandidateStore(cfg)
    rows = _document_rows(split, cfg, resume_state_dict, tokenizer_batch_size)
    doc_buffer: list[_PreparedDoc] = []
    pq_idx, rg_idx, epoch = 0, 0, 1
    bos_token = tokenizer.get_bos_token_id()
    pair_reserve = max(int(cfg.random_source_token_reserve), 32)
    row_capacity = max(32, (T + 1 - pair_reserve) // 2)
    use_cuda = device == "cuda"
    reservoir = ChunkReservoir(
        max_chunks=reservoir_max_chunks,
        chunk_size=reservoir_chunk_size,
        rng=rng,
    )

    def refill():
        nonlocal pq_idx, rg_idx, epoch
        row_batch, (pq_idx, rg_idx, epoch) = next(rows)
        for shard_id, row_index, raw_text in row_batch:
            doc_buffer.append(
                _prepare_doc(
                    shard_id=shard_id,
                    row_index=row_index,
                    raw_text=raw_text,
                    tokenizer=tokenizer,
                    bos_token=bos_token,
                    store=store,
                    cfg=cfg,
                    rng=rng,
                )
            )

    def ten(values):
        array = np.stack(values)
        if np.issubdtype(array.dtype, np.integer):
            dtype = torch.long
        elif np.issubdtype(array.dtype, np.floating):
            dtype = torch.float32
        else:
            dtype = torch.bool
        tensor = torch.as_tensor(array, dtype=dtype)
        if use_cuda:
            tensor = tensor.pin_memory()
        return tensor.to(device=device, non_blocking=use_cuda)

    while True:
        batch: list[TrainingData] = []
        candidate_rows = 0
        random_rows = 0
        identity_rows = 0
        raw_examples_consumption = 0
        raw_token_consumption = 0
        edit_mass = 0
        for _ in range(B):
            if single_document_pairs:
                while not doc_buffer:
                    refill()
                document = doc_buffer.pop(0)
                row = [int(token) for token in document.tokens[:row_capacity]]
                docs = [(0, document)] if len(document.tokens) <= row_capacity else []
            else:
                row, docs = _pack_one_row_sequential(
                    doc_buffer=doc_buffer,
                    refill=refill,
                    row_capacity=row_capacity,
                )
            raw_examples_consumption += len(docs)
            raw_token_consumption += len(row)
            if len(row) >= reservoir_chunk_size:
                reservoir.offer(row[:reservoir_chunk_size])

            edits: list[Any] = []
            for offset, doc in docs:
                if doc.token_edit is not None:
                    edits.extend(_offset_edit(edit, offset) for edit in doc.token_edit.edits)

            seq_edit = None
            row_kind = "identity"
            if edits:
                seq_edit = SequenceEdit(
                    sequence=np.asarray(row, dtype=np.int64),
                    edits=sorted(edits, key=_edit_key),
                )
                row_kind = "candidate"
            elif rng.random() < cfg.random_corruption_prob_when_no_candidates:
                seq_edit = _sample_random_inverse_edit(
                    np.asarray(row, dtype=np.int64),
                    cfg=cfg,
                    token_ops=token_ops,
                    reservoir=reservoir,
                    rng=rng,
                )
                if seq_edit is not None:
                    row_kind = "random"
            if seq_edit is None:
                seq_edit = SequenceEdit(sequence=np.asarray(row, dtype=np.int64), edits=[])

            source_tokens = [int(token) for token in np.asarray(seq_edit.sequence).tolist()]
            target_tokens = [int(token) for token in np.asarray(resolve(seq_edit)).tolist()]
            example = build_baseline_example(
                baseline_mode,
                source_tokens,
                target_tokens,
                tokenizer,
            )
            training_data = masked_baseline_training_data(
                example,
                token_budget=T,
                pos_buffer_size=pos_buffer_size,
                terminal_weight=baseline_terminal_weight,
            )
            if training_data is None:
                # The half-context capacity should make this exceptional. Use a
                # short identity row rather than silently training a malformed pair.
                source_tokens = source_tokens[:max(2, row_capacity // 2)]
                target_tokens = source_tokens
                example = build_baseline_example(
                    baseline_mode,
                    source_tokens,
                    target_tokens,
                    tokenizer,
                )
                training_data = masked_baseline_training_data(
                    example,
                    token_budget=T,
                    pos_buffer_size=pos_buffer_size,
                    terminal_weight=baseline_terminal_weight,
                )
                if training_data is None:
                    raise RuntimeError("Could not fit fallback text-edit baseline row")
                row_kind = "identity"

            candidate_rows += int(row_kind == "candidate")
            random_rows += int(row_kind == "random")
            identity_rows += int(row_kind == "identity")
            edit_mass += sequence_edit_mass(seq_edit) if row_kind != "identity" else 0
            batch.append(training_data)

        target_fields = {
            field_name: ten([getattr(item.target, field_name) for item in batch])
            for field_name in (
                "tokens",
                "gathered_pos_target",
                "token_loss_weights",
                "gather_indices",
                "gather_valid",
                "overflow",
            )
        }
        training_batch = TrainingData(
            input_tokens=ten([item.input_tokens for item in batch]),
            input_positions=ten([item.input_positions for item in batch]),
            target=TrainingTarget(**target_fields),
        )
        yield training_batch, {
            "pq_idx": pq_idx,
            "rg_idx": rg_idx,
            "epoch": epoch,
            "candidate_rows": candidate_rows,
            "random_rows": random_rows,
            "identity_rows": identity_rows,
            "raw_examples_consumption": raw_examples_consumption,
            "raw_token_consumption": raw_token_consumption,
            "edit_mass": edit_mass,
            "valid_pointer_targets": int(sum(
                np.asarray(item.target.gather_valid).sum() for item in batch
            )),
        }


def tokenizing_distributed_extended_edit_loader_with_state(
    tokenizer,
    B: int,
    T: int,
    split: str,
    cfg: EditSamplingConfig,
    token_ops: TokensOperations,
    tokenizer_batch_size: int = 128,
    device: str = "cuda",
    resume_state_dict=None,
    pos_buffer_size: int = 64,
):
    """Yield batches from raw corpus rows joined with sparse edit candidates."""

    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    ddp, ddp_rank, _ddp_local_rank, _ddp_world_size = get_dist_info()
    rng = np.random.default_rng(cfg.seed + ddp_rank)
    store = EditCandidateStore(cfg)
    op_keys, op_probs = _normalise_probs(cfg.op_probs)
    rows = _document_rows(split, cfg, resume_state_dict, tokenizer_batch_size)
    row_buffer: list[tuple[str, int, str]] = []
    pq_idx, rg_idx, epoch = 0, 0, 1
    use_cuda = device == "cuda"

    def ten(x_list):
        x = np.stack(x_list)
        dtype = torch.long if np.issubdtype(x.dtype, np.integer) else torch.bool
        tensor = torch.as_tensor(x, dtype=dtype)
        if use_cuda:
            tensor = tensor.pin_memory()
        return tensor.to(device=device, non_blocking=use_cuda)

    while True:
        batch: list[TrainingData] = []
        while len(batch) < B:
            target_op = op_keys[int(rng.choice(len(op_keys), p=op_probs))]
            attempts = 0
            item = None
            while item is None:
                attempts += 1
                if attempts > max(10_000, B * 2000):
                    raise RuntimeError(
                        f"Could not construct a {target_op!r} edit example; "
                        "check candidate coverage and length filters."
                    )
                if not row_buffer:
                    row_buffer, (pq_idx, rg_idx, epoch) = next(rows)
                    rng.shuffle(row_buffer)
                shard_id, row_index, raw_text = row_buffer.pop()
                item = store.sample(
                    shard_id=shard_id,
                    row_index=row_index,
                    raw_text=raw_text,
                    enc=tokenizer.enc,
                    token_ops=token_ops,
                    token_budget=T,
                    pos_buffer_size=pos_buffer_size,
                    rng=rng,
                    target_op=target_op,
                )
            batch.append(item)

        kwgs = {
            field_name: ten([getattr(x.target, field_name) for x in batch])
            for field_name in [
                "tokens",
                "gathered_pos_target",
                "token_loss_weights",
                "gather_indices",
                "gather_valid",
                "overflow",
            ]
        }
        training_batch = TrainingData(
            input_tokens=ten([x.input_tokens for x in batch]),
            input_positions=ten([x.input_positions for x in batch]),
            target=TrainingTarget(**kwgs),
        )
        yield training_batch, {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
