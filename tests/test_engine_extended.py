import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nanochat.causal_serialisation_extended import (
    EDIT_START,
    INDL_BOS,
    Copy,
    Delete,
    Insert,
    Move,
    Replace,
    SequenceEdit,
    SerialisedData,
    TokensOperations,
    _edit_key,
    _lower_moves,
    _validate_edits,
    get_specials,
)
from nanochat.ces_rl import (
    group_advantages,
    reinforce_loss,
    replay_policy_logprobs,
    target_aware_recovery_reward,
)
from nanochat.engine_extended import EditController, Engine, Mode, sample_next_token
from nanochat.oracle_ces import (
    PointerOracleProvider,
    build_oracle_span_prompt,
)
from nanochat.oracle_edit_likelihood import conditional_program_training_data

from ces.workflows.oracle_ces_eval import _generate_record

class ExtendedByteTokenizer:
    def __init__(self):
        specials = get_specials({}, 255)
        next_id = max(specials.values()) + 1
        extras = {
            "<|python_start|>": next_id,
            "<|python_end|>": next_id + 1,
            "<|output_start|>": next_id + 2,
            "<|output_end|>": next_id + 3,
            "<|assistant_end|>": next_id + 4,
            "<|bos|>": next_id + 5,
        }
        self._special_tokens = {**specials, **extras}
        self._id_to_special = {v: k for k, v in self._special_tokens.items()}
        self._bos = self._special_tokens["<|bos|>"]
        self.max_token_value = max(self._special_tokens.values())

    def encode_special(self, s):
        return self._special_tokens[s]

    def get_bos_token_id(self):
        return self._bos

    def get_vocab_size(self):
        return self.max_token_value + 1

    def encode(self, text, prepend=None):
        tokens = list(text.encode("utf-8"))
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        parts = []
        for token in tokens:
            token = int(token)
            if token in self._id_to_special:
                parts.append(self._id_to_special[token])
            elif 0 <= token < 256:
                parts.append(bytes([token]).decode("utf-8", errors="replace"))
        return "".join(parts)


@dataclass
class DummyConfig:
    vocab_size: int
    sequence_len: int = 1024
    n_layer: int = 1
    n_head: int = 1
    n_kv_head: int = 1
    n_embd: int = 64


class ScriptedExtendedModel:
    def __init__(self, tokenizer, expected_tokens, expected_positions):
        self.tokenizer = tokenizer
        self.config = DummyConfig(vocab_size=tokenizer.get_vocab_size(), sequence_len=len(expected_tokens) + 8)
        self._device = torch.device("cpu")
        self.expected_tokens = [int(tok) for tok in expected_tokens]
        self.expected_positions = [int(pos) for pos in expected_positions]
        self.assistant_end_id = tokenizer.encode_special("<|assistant_end|>")

    def get_device(self):
        return self._device

    def _target_token(self, next_abs_ix: int) -> int:
        if next_abs_ix < len(self.expected_tokens):
            return self.expected_tokens[next_abs_ix]
        return self.assistant_end_id

    def _target_ptr(self, next_abs_ix: int, tctx: int) -> int:
        if next_abs_ix < len(self.expected_positions):
            ptr = self.expected_positions[next_abs_ix]
            if 0 <= ptr < tctx:
                return ptr
        return max(0, tctx - 1)

    def forward(self, idx, tix=None, kv_cache=None, pos_k_cache=None, **kwargs):
        idx = idx.to(self._device)
        batch_size, seq_len = idx.shape
        vocab_size = self.config.vocab_size
        consumed = kv_cache.get_pos() if kv_cache is not None else 0

        logits = torch.full(
            (batch_size, seq_len, vocab_size),
            fill_value=-1e9,
            dtype=torch.float32,
            device=self._device,
        )
        for local_ix in range(seq_len):
            next_abs_ix = consumed + local_ix + 1
            logits[:, local_ix, self._target_token(next_abs_ix)] = 0.0

        if kv_cache is not None:
            kv_cache.advance(seq_len)
            tctx = kv_cache.get_pos()
        else:
            tctx = consumed + seq_len

        if pos_k_cache is not None:
            pos_k_cache.advance(seq_len)

        if tix is None:
            return logits

        position_logits = torch.full(
            (batch_size, seq_len, tctx),
            fill_value=-1e9,
            dtype=torch.float32,
            device=self._device,
        )
        for local_ix in range(seq_len):
            next_abs_ix = consumed + local_ix + 1
            position_logits[:, local_ix, self._target_ptr(next_abs_ix, tctx)] = 0.0
        return logits, position_logits


class DifferentiableScriptedModel(nn.Module):
    """Finite scripted logits with identical cached and full-sequence behavior."""

    def __init__(self, tokenizer, expected_tokens, expected_positions):
        super().__init__()
        self.config = DummyConfig(
            vocab_size=tokenizer.get_vocab_size(),
            sequence_len=len(expected_tokens) + 16,
        )
        self.expected_tokens = [int(token) for token in expected_tokens]
        self.expected_positions = [int(pos) for pos in expected_positions]
        self.assistant_end_id = tokenizer.encode_special("<|assistant_end|>")
        self.token_bias = nn.Parameter(torch.zeros(self.config.vocab_size))
        self.pointer_bias = nn.Parameter(torch.zeros(self.config.sequence_len))

    def get_device(self):
        return self.token_bias.device

    def _next_token(self, absolute_index):
        if absolute_index < len(self.expected_tokens):
            return self.expected_tokens[absolute_index]
        return self.assistant_end_id

    def _next_pointer(self, absolute_index, context_length):
        if absolute_index < len(self.expected_positions):
            pointer = self.expected_positions[absolute_index]
            if 0 <= pointer < context_length:
                return pointer
        return max(0, context_length - 1)

    def forward(
        self,
        idx,
        tix=None,
        kv_cache=None,
        pos_k_cache=None,
        return_pos_logits=False,
        position_queries=None,
        **_kwargs,
    ):
        batch_size, seq_len = idx.shape
        consumed = kv_cache.get_pos() if kv_cache is not None else 0
        logits = self.token_bias[None, None, :].expand(
            batch_size, seq_len, -1
        ).clone()
        for local_index in range(seq_len):
            next_index = consumed + local_index + 1
            logits[:, local_index, self._next_token(next_index)] += 12.0

        if kv_cache is not None:
            kv_cache.advance(seq_len)
            context_length = kv_cache.get_pos()
        else:
            context_length = seq_len
        if pos_k_cache is not None:
            pos_k_cache.advance(seq_len)

        if not return_pos_logits:
            return logits
        if position_queries is None:
            query_indices = torch.full(
                (batch_size, 1),
                consumed + seq_len - 1,
                dtype=torch.long,
                device=idx.device,
            )
        else:
            query_indices = position_queries
        pointer_logits = self.pointer_bias[:context_length][None, None, :].expand(
            batch_size, query_indices.size(1), -1
        ).clone()
        for row in range(batch_size):
            for query_slot, query_index in enumerate(query_indices[row].tolist()):
                next_index = int(query_index) + 1
                target = self._next_pointer(next_index, context_length)
                pointer_logits[row, query_slot, target] += 12.0
        return logits, pointer_logits


class FlatExtendedModel:
    """Uniform finite logits for asynchronous batched termination tests."""

    def __init__(self, tokenizer):
        self.config = DummyConfig(
            vocab_size=tokenizer.get_vocab_size(), sequence_len=128
        )
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def forward(self, idx, tix=None, kv_cache=None, pos_k_cache=None, **_kwargs):
        batch_size, seq_len = idx.shape
        logits = torch.zeros(
            (batch_size, seq_len, self.config.vocab_size), dtype=torch.float32
        )
        if kv_cache is not None:
            kv_cache.advance(seq_len)
            context_length = kv_cache.get_pos()
        else:
            context_length = seq_len
        if pos_k_cache is not None:
            pos_k_cache.advance(seq_len)
        position_logits = torch.zeros(
            (batch_size, seq_len, context_length), dtype=torch.float32
        )
        return logits, position_logits


def test_top_p_keeps_minimal_probability_prefix():
    logits = torch.tensor([[10.0, 9.0, 8.0]])
    rng = torch.Generator().manual_seed(0)
    draws = [int(sample_next_token(logits, rng, temperature=1.0, top_p=0.5).item()) for _ in range(20)]
    assert draws == [0] * 20


def test_top_p_validation():
    logits = torch.zeros((1, 3))
    rng = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="top_p"):
        sample_next_token(logits, rng, top_p=0.0)


def test_closure_bias_targets_legal_closure_token():
    tokenizer = ExtendedByteTokenizer()
    controller = EditController(tokenizer)
    controller.reset(1)
    controller.rows[0].mode = Mode.IN_REPLACE
    controller.rows[0].payload_len = 4
    logits = torch.zeros((1, tokenizer.get_vocab_size()))
    controller.apply_mask_(logits)
    controller.apply_closure_bias_(logits, 2.5, min_payload_tokens=2)
    close = controller.tok["REPLACE_EOS"]
    assert float(logits[0, close]) == pytest.approx(2.5)


def test_hierarchical_first_decision_matches_marginal_edit_class():
    tokenizer = ExtendedByteTokenizer()
    controller = EditController(tokenizer)
    controller.reset(2)
    for row in controller.rows:
        row.mode = Mode.AFTER_EDIT
    logits = torch.full((2, tokenizer.get_vocab_size()), -10.0)
    edit_tokens = [
        controller.tok["DEL_OPEN"],
        controller.tok["INS_OPEN"],
        controller.tok["REPLACE_OPEN"],
    ]
    noop = controller.tok["EDIT_END"]
    logits[0, edit_tokens] = 1.0
    logits[0, noop] = 2.0  # Each opener loses, but their combined class wins.
    logits[1, edit_tokens] = 0.0
    logits[1, noop] = 2.0
    controller.apply_first_decision_gate_(logits, threshold=0.0)
    assert torch.isfinite(logits[0, edit_tokens]).all()
    assert torch.isneginf(logits[0, noop])
    assert torch.isfinite(logits[1, noop])
    assert torch.isneginf(logits[1, edit_tokens]).all()
    assert controller.rows[0].hierarchical_edit_selected is True
    assert controller.rows[1].hierarchical_edit_selected is False


def _edit_equal(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Insert):
        return left.pos == right.pos and np.array_equal(np.array(left.content), np.array(right.content))
    if isinstance(left, Delete):
        return left.start == right.start and left.end == right.end
    if isinstance(left, (Copy, Move)):
        return left.start == right.start and left.end == right.end and left.dest == right.dest
    return (
        left.start == right.start
        and left.end == right.end
        and np.array_equal(np.array(left.content), np.array(right.content))
    )


def _seq_edit_equal(left: SequenceEdit, right: SequenceEdit) -> bool:
    if not np.array_equal(np.array(left.sequence), np.array(right.sequence)):
        return False
    if len(left.edits) != len(right.edits):
        return False
    return all(_edit_equal(le, re) for le, re in zip(left.edits, right.edits))


def _random_ascii_tokens(rng: np.random.Generator, *, min_len: int = 1, max_len: int = 4) -> np.ndarray:
    length = int(rng.integers(min_len, max_len + 1))
    alphabet = np.array(list(b"uvwxyzUVWXYZ!?+-<>"), dtype=np.uint8)
    return alphabet[rng.integers(0, len(alphabet), size=length)].astype(np.int64)


def _random_source_tokens(rng: np.random.Generator, *, min_len: int = 3, max_len: int = 12) -> np.ndarray:
    length = int(rng.integers(min_len, max_len + 1))
    alphabet = np.array(list(b"abcdefghiJKLMNOPQR"), dtype=np.uint8)
    return alphabet[rng.integers(0, len(alphabet), size=length)].astype(np.int64)


def _random_source_span(rng: np.random.Generator, n: int) -> tuple[int, int]:
    start = int(rng.integers(0, n))
    end = int(rng.integers(start + 1, n + 1))
    return start, end


def _random_valid_sequence_edit(rng: np.random.Generator, *, require_extended: bool = True) -> SequenceEdit:
    while True:
        sequence = _random_source_tokens(rng)
        n = len(sequence)
        edits = []
        boundary = 0
        has_extended = False

        while boundary <= n:
            extra_ops = int(rng.integers(0, 3))
            for _ in range(extra_ops):
                if n > 0 and rng.random() < 0.45:
                    start, end = _random_source_span(rng, n)
                    edits.append(Copy(start, end, boundary))
                    has_extended = True
                else:
                    edits.append(Insert(boundary, _random_ascii_tokens(rng)))

            if boundary == n:
                break

            if rng.random() < 0.55:
                end = int(rng.integers(boundary + 1, n + 1))
                roll = rng.random()
                if roll < 0.34:
                    edits.append(Delete(boundary, end))
                elif roll < 0.67:
                    edits.append(Replace(boundary, end, _random_ascii_tokens(rng)))
                    has_extended = True
                else:
                    dest = int(rng.integers(end, n + 1))
                    edits.append(Move(boundary, end, dest))
                    has_extended = True
                boundary = end
            else:
                boundary += 1

        edits.sort(key=_edit_key)
        if require_extended and not has_extended:
            continue
        try:
            _validate_edits(sequence, edits)
            # Move is lowered to Copy+Delete by the serializer. Some scripts
            # are valid before lowering but overlap after lowering, so only
            # generate cases the serializer can actually accept.
            _validate_edits(sequence, _lower_moves(SequenceEdit(sequence, edits)).edits)
        except ValueError:
            continue
        return SequenceEdit(sequence=sequence, edits=edits)


def _drive_engine_on_serialised(
    serialised: SerialisedData, pointer_token_mode: str = "role_specific"
):
    tokenizer = ExtendedByteTokenizer()
    model = ScriptedExtendedModel(tokenizer, serialised.tokens.tolist(), serialised.positions.tolist())
    engine = Engine(model, tokenizer, pointer_token_mode=pointer_token_mode)

    edit_start_token = tokenizer.encode_special(EDIT_START)
    edit_start_ix = serialised.tokens.tolist().index(edit_start_token)
    prompt_tokens = serialised.tokens[: edit_start_ix + 1].tolist()
    prompt_tix = serialised.positions[: edit_start_ix + 1].tolist()

    results, _, pointers = engine.generate_batch(
        prompt_tokens,
        tix=prompt_tix,
        num_samples=1,
        max_tokens=len(serialised.tokens) - len(prompt_tokens) + 2,
        temperature=0.0,
    )
    return np.array(results[0], dtype=np.int64), np.array(pointers[0], dtype=np.int64)


def test_random_generator_produces_valid_extended_scripts():
    rng = np.random.default_rng(0)

    for _ in range(100):
        seq_edit = _random_valid_sequence_edit(rng)
        assert any(isinstance(edit, (Replace, Copy, Move)) for edit in seq_edit.edits)
        _validate_edits(seq_edit.sequence, seq_edit.edits)


def test_engine_extended_replays_random_valid_extended_serialisations():
    rng = np.random.default_rng(0)
    tokenizer = ExtendedByteTokenizer()
    token_ops = TokensOperations(tokenizer)

    for _ in range(50):
        seq_edit = _random_valid_sequence_edit(rng)
        serialised = token_ops.serialise(
            seq_edit,
            indel_bos_pos=int(rng.integers(0, len(seq_edit.sequence) + 1)),
        )

        generated_tokens, generated_positions = _drive_engine_on_serialised(serialised)

        assert generated_tokens.tolist() == serialised.tokens.tolist()
        assert generated_positions.tolist() == serialised.positions.tolist()

        generated = SerialisedData(tokens=generated_tokens, positions=generated_positions)
        assert _seq_edit_equal(token_ops.deserialise(generated), seq_edit)


def test_engine_extended_replays_shared_pointer_serialisations():
    rng = np.random.default_rng(7)
    tokenizer = ExtendedByteTokenizer()
    token_ops = TokensOperations(tokenizer, pointer_token_mode="shared")
    controller = EditController(tokenizer, pointer_token_mode="shared")
    assert len(controller.pointer_tokens) == 1

    for _ in range(50):
        seq_edit = _random_valid_sequence_edit(rng)
        serialised = token_ops.serialise(
            seq_edit,
            indel_bos_pos=int(rng.integers(0, len(seq_edit.sequence) + 1)),
        )
        generated_tokens, generated_positions = _drive_engine_on_serialised(
            serialised, pointer_token_mode="shared"
        )
        assert generated_tokens.tolist() == serialised.tokens.tolist()
        assert generated_positions.tolist() == serialised.positions.tolist()
        generated = SerialisedData(
            tokens=generated_tokens,
            positions=generated_positions,
        )
        assert _seq_edit_equal(token_ops.deserialise(generated), seq_edit)


def test_engine_extended_can_replay_a_minimal_replace_script():
    tokenizer = ExtendedByteTokenizer()
    token_ops = TokensOperations(tokenizer)
    seq_edit = SequenceEdit(
        sequence=np.array(tokenizer.encode("abcd"), dtype=np.int64),
        edits=[Replace(1, 3, np.array(tokenizer.encode("XY"), dtype=np.int64))],
    )

    serialised = token_ops.serialise(seq_edit, indel_bos_pos=0)
    generated_tokens, generated_positions = _drive_engine_on_serialised(serialised)

    assert generated_tokens.tolist() == serialised.tokens.tolist()
    assert generated_positions.tolist() == serialised.positions.tolist()
    assert tokenizer.decode(token_ops.resolve(SerialisedData(generated_tokens, generated_positions))) == "aXYd"
    assert int(serialised.tokens[0]) == tokenizer.encode_special(INDL_BOS)


def test_terminal_edit_constraints_close_a_runaway_insert():
    tokenizer = ExtendedByteTokenizer()
    token_ops = TokensOperations(tokenizer)
    source = np.array(tokenizer.encode("abcd"), dtype=np.int64)
    serialised = token_ops.serialise(
        SequenceEdit(source, [Insert(2, np.array(tokenizer.encode("x"), dtype=np.int64))]),
        indel_bos_pos=0,
    )
    edit_start_ix = serialised.tokens.tolist().index(tokenizer.encode_special(EDIT_START))
    prompt_tokens = serialised.tokens[: edit_start_ix + 1].tolist()
    prompt_positions = serialised.positions[: edit_start_ix + 1].tolist()

    insert_open = token_ops.insertion_open_token
    insert_loc = token_ops.insertion_loc_token
    edit_end = token_ops.edit_end_token
    payload = tokenizer.encode("z") * 32
    expected_tokens = prompt_tokens + [insert_open, insert_loc] + payload
    insertion_anchor = 2
    expected_positions = (
        prompt_positions
        + [len(prompt_tokens), insertion_anchor]
        + [insertion_anchor] * len(payload)
    )
    model = ScriptedExtendedModel(tokenizer, expected_tokens, expected_positions)
    engine = Engine(model, tokenizer)

    results, _, pointers = engine.generate_batch(
        prompt_tokens,
        tix=prompt_positions,
        num_samples=1,
        max_tokens=32,
        temperature=0.0,
        max_payload_tokens=3,
        max_edit_ops=1,
        stop_after_edit_end=True,
    )

    assert results[0][-1] == edit_end
    assert len(results[0]) == len(prompt_tokens) + 2 + 3 + 1
    diagnostics = engine.last_generation_diagnostics[0]
    assert diagnostics == {
        "final_mode": "NORMAL",
        "payload_len": 0,
        "max_payload_len_seen": 3,
            "completed_ops": 1,
            "forced_payload_closures": 1,
            "forced_program_closures": 0,
            "first_decision_edit_margin": None,
            "hierarchical_edit_selected": None,
        }
    resolved = token_ops.resolve(
        SerialisedData(
            tokens=np.array(results[0], dtype=np.int64),
            positions=np.array(pointers[0], dtype=np.int64),
        )
    )
    assert tokenizer.decode(resolved) == "abzzzcd"


def test_batched_stop_after_edit_end_does_not_append_to_completed_rows():
    tokenizer = ExtendedByteTokenizer()
    token_ops = TokensOperations(tokenizer)
    prompt_tokens = [
        tokenizer.encode_special(INDL_BOS),
        *tokenizer.encode("ab"),
        token_ops.edit_start_token,
    ]
    prompt_positions = list(range(len(prompt_tokens)))
    rollouts, _masks = Engine(FlatExtendedModel(tokenizer), tokenizer).generate_rollout_batch(
        prompt_tokens,
        tix=prompt_positions,
        num_samples=16,
        max_tokens=12,
        temperature=1.0,
        pointer_temperature=1.0,
        max_payload_tokens=1,
        max_edit_ops=1,
        stop_after_edit_end=True,
        seed=19,
    )

    generated_lengths = set()
    for rollout in rollouts:
        generated_lengths.add(len(rollout.actions))
        assert [action.absolute_index for action in rollout.actions] == list(
            range(rollout.prompt_length, len(rollout.tokens))
        )
        generated = rollout.tokens[rollout.prompt_length:]
        if token_ops.edit_end_token in generated:
            assert generated[-1] == token_ops.edit_end_token
    assert len(generated_lengths) > 1


def test_pointer_oracle_provider_overrides_model_pointer_logits():
    tokenizer = ExtendedByteTokenizer()
    token_ops = TokensOperations(tokenizer)
    sequence_edit = SequenceEdit(
        sequence=np.array(tokenizer.encode("abcd"), dtype=np.int64),
        edits=[Replace(1, 3, np.array(tokenizer.encode("XY"), dtype=np.int64))],
    )
    serialised = token_ops.serialise(sequence_edit, indel_bos_pos=0)
    tokens = serialised.tokens.tolist()
    positions = serialised.positions.tolist()
    edit_start = tokens.index(token_ops.edit_start_token)
    prompt_tokens = tokens[:edit_start + 1]
    prompt_positions = positions[:edit_start + 1]
    provider = PointerOracleProvider(
        tokens[edit_start + 1:],
        positions[edit_start + 1:],
        token_ops,
    )
    wrong_model_positions = [0] * len(positions)
    model = ScriptedExtendedModel(tokenizer, tokens, wrong_model_positions)
    engine = Engine(model, tokenizer)

    results, _, generated_positions = engine.generate_batch(
        prompt_tokens,
        tix=prompt_positions,
        num_samples=1,
        max_tokens=len(tokens) - len(prompt_tokens) + 2,
        temperature=0.0,
        pointer_provider=provider,
    )

    assert results[0] == tokens
    assert generated_positions[0] == positions
    assert provider.complete


def test_ces_policy_rollout_replay_matches_token_and_pointer_logprobs():
    tokenizer = ExtendedByteTokenizer()
    token_ops = TokensOperations(tokenizer)
    source = np.array(tokenizer.encode("abcd"), dtype=np.int64)
    target = tokenizer.encode("aXYd")
    serialised = token_ops.serialise(
        SequenceEdit(
            source,
            [Replace(1, 3, np.array(tokenizer.encode("XY"), dtype=np.int64))],
        ),
        indel_bos_pos=0,
    )
    expected_tokens = serialised.tokens.tolist()
    expected_positions = serialised.positions.tolist()
    edit_start = expected_tokens.index(token_ops.edit_start_token)
    prompt_tokens = expected_tokens[: edit_start + 1]
    prompt_positions = expected_positions[: edit_start + 1]
    model = DifferentiableScriptedModel(
        tokenizer, expected_tokens, expected_positions
    )
    engine = Engine(model, tokenizer)

    rollouts, masks = engine.generate_rollout_batch(
        prompt_tokens,
        tix=prompt_positions,
        num_samples=1,
        max_tokens=len(expected_tokens) - len(prompt_tokens) + 2,
        temperature=0.7,
        pointer_temperature=0.9,
        seed=7,
        stop_after_edit_end=True,
    )
    rollout = rollouts[0]
    assert list(rollout.tokens) == expected_tokens
    assert list(rollout.pointers) == expected_positions
    assert masks[0][: len(prompt_tokens)] == [0] * len(prompt_tokens)

    replayed = replay_policy_logprobs(model, tokenizer, rollout)
    rollout_token_logprobs = [
        action.token_logprob
        for action in rollout.actions
        if action.token_sampled
    ]
    rollout_pointer_logprobs = [
        action.pointer_logprob
        for action in rollout.actions
        if action.pointer_sampled
    ]
    assert [float(value.detach()) for value in replayed.token_logprobs] == pytest.approx(
        rollout_token_logprobs, abs=1e-6
    )
    assert [float(value.detach()) for value in replayed.pointer_logprobs] == pytest.approx(
        rollout_pointer_logprobs, abs=1e-6
    )

    replayed.total.backward()
    assert model.token_bias.grad is not None
    assert model.pointer_bias.grad is not None
    assert float(model.token_bias.grad.abs().sum()) > 0
    assert float(model.pointer_bias.grad.abs().sum()) > 0

    reward = target_aware_recovery_reward(
        rollout,
        token_ops,
        source_tokens=source.tolist(),
        target_tokens=target,
    )
    assert reward.valid
    assert reward.exact_target
    assert reward.candidate_distance == 0
    assert reward.reward == pytest.approx(1.25)


def test_completion_only_ces_sft_starts_after_prompt_edit_start():
    tokenizer = ExtendedByteTokenizer()
    token_ops = TokensOperations(tokenizer)
    sequence_edit = SequenceEdit(
        np.array(tokenizer.encode("abcd"), dtype=np.int64),
        [Replace(1, 3, np.array(tokenizer.encode("XY"), dtype=np.int64))],
    )
    data = conditional_program_training_data(
        token_ops, sequence_edit, position_buffer_size=16
    )
    serialised = token_ops.serialise(sequence_edit, indel_bos_pos=0)
    edit_start = serialised.tokens.tolist().index(token_ops.edit_start_token)
    supervised_serial_indices = (
        np.flatnonzero(data.target.token_loss_weights) + 1
    )

    assert len(supervised_serial_indices) > 0
    assert int(supervised_serial_indices.min()) > edit_start
    assert int(data.target.gather_valid.sum()) > 0


def test_rloo_joint_policy_loss_is_decision_normalized():
    rewards = torch.tensor([-1.0, 0.0, 1.0])
    advantages = group_advantages(rewards, method="rloo")
    assert advantages.tolist() == pytest.approx([-1.5, 0.0, 1.5])

    tokenizer = ExtendedByteTokenizer()
    token_ops = TokensOperations(tokenizer)
    serialised = token_ops.serialise(
        SequenceEdit(
            np.array(tokenizer.encode("ab"), dtype=np.int64),
            [Insert(1, np.array(tokenizer.encode("X"), dtype=np.int64))],
        ),
        indel_bos_pos=0,
    )
    tokens = serialised.tokens.tolist()
    pointers = serialised.positions.tolist()
    edit_start = tokens.index(token_ops.edit_start_token)
    model = DifferentiableScriptedModel(tokenizer, tokens, pointers)
    rollout, _ = Engine(model, tokenizer).generate_rollout_batch(
        tokens[: edit_start + 1],
        tix=pointers[: edit_start + 1],
        num_samples=1,
        max_tokens=len(tokens) - edit_start + 1,
        temperature=0.7,
        pointer_temperature=0.9,
        seed=11,
        stop_after_edit_end=True,
    )
    replayed = replay_policy_logprobs(model, tokenizer, rollout[0])
    loss = reinforce_loss(
        [replayed, replayed, replayed], advantages
    )
    # Equal log-probability sums and zero-sum group advantages cancel exactly.
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-7)


def test_oracle_span_prompt_matches_fim_envelope_without_exposing_payload():
    tokenizer = ExtendedByteTokenizer()
    token_ops = TokensOperations(tokenizer)
    bos = tokenizer.get_bos_token_id()
    source = [bos, *tokenizer.encode("abcd")]
    target = [bos, *tokenizer.encode("aXYd")]

    oracle = build_oracle_span_prompt(source, target, token_ops)

    assert oracle is not None
    assert oracle.operation == "replace"
    assert oracle.source_span == (2, 4)
    assert oracle.target_span == (2, 4)
    assert oracle.target_payload_tokens == tuple(tokenizer.encode("XY"))
    assert oracle.prompt_tokens[-1] == token_ops.replace_end_token
    assert all(token not in oracle.prompt_tokens for token in tokenizer.encode("XY"))
    assert tokenizer.decode(oracle.sequence_edit.edits[0].content) == "XY"
    alternate = build_oracle_span_prompt(
        source, [bos, *tokenizer.encode("aZZd")], token_ops
    )
    assert alternate is not None


def test_oracle_ces_evaluator_resolves_scripted_payload_in_both_modes():
    tokenizer = ExtendedByteTokenizer()
    token_ops = TokensOperations(tokenizer)
    bos = tokenizer.get_bos_token_id()
    source = [bos, *tokenizer.encode("abcd")]
    target = [bos, *tokenizer.encode("aXYd")]
    sequence_edit = SequenceEdit(
        sequence=np.asarray(source, dtype=np.int64),
        edits=[Replace(2, 4, np.asarray(tokenizer.encode("XY"), dtype=np.int64))],
    )
    serialised = token_ops.serialise(sequence_edit, indel_bos_pos=0)
    tokens = serialised.tokens.tolist()
    positions = serialised.positions.tolist()
    edit_start = tokens.index(token_ops.edit_start_token)
    row = {
        "benchmark_id": "scripted",
        "source_text": "abcd",
        "target_text": "aXYd",
        "source_tokens": source,
        "target_tokens": target,
        "gold_edits": [{"op": "replace", "start": 2, "end": 4, "content": tokenizer.encode("XY")}],
        "gold_op_counts": {"replace": 1},
        "prompt_tokens": tokens[:edit_start + 1],
        "prompt_positions": positions[:edit_start + 1],
        "gold_program_tokens": tokens[edit_start + 1:],
        "gold_program_positions": positions[edit_start + 1:],
        "stratum": "replace_k1",
        "repair_op_mode": "replace",
        "repair_num_ops": 1,
        "density_regime": "training_span_distribution",
        "source_shard_id": "test",
        "source_row_index": 0,
        "source_token_offset": 0,
        "construction_seed": 0,
    }

    for mode in ("pointer_oracle", "oracle_span"):
        model = ScriptedExtendedModel(tokenizer, tokens, positions)
        record = _generate_record(
            mode=mode,
            row=row,
            row_index=0,
            model=model,
            tokenizer=tokenizer,
            token_ops=token_ops,
            engine=Engine(model, tokenizer),
            device=torch.device("cpu"),
            max_edit_tokens=32,
            max_payload_tokens=16,
            max_edit_ops=64,
            temperature=0.0,
            generation_seed=7,
        )

        assert record["valid"]
        assert record["exact_target"]
        assert record["prediction_text"] == "aXYd"
        assert record["target_payload_exposed"] is False


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_cpu_generation_cache_matches_compute_dtype(monkeypatch, compute_dtype):
    import nanochat.engine_extended as engine_module

    monkeypatch.setattr(engine_module, "COMPUTE_DTYPE", compute_dtype)
    original_init = engine_module.KVCache.__init__
    cache_dtypes = []

    def record_cache(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        cache_dtypes.append(self.k_cache.dtype)

    monkeypatch.setattr(engine_module.KVCache, "__init__", record_cache)
    tokenizer = ExtendedByteTokenizer()
    token_ops = TokensOperations(tokenizer)
    source = SequenceEdit(
        sequence=np.array([65, 66, 67]),
        edits=[Delete(1, 2)],
    )
    serialised = token_ops.serialise(source, indel_bos_pos=0)
    tokens, positions = _drive_engine_on_serialised(serialised)

    assert tokens.tolist() == serialised.tokens.tolist()
    assert positions.tolist() == serialised.positions.tolist()
    assert cache_dtypes
    assert all(dtype == compute_dtype for dtype in cache_dtypes)
