import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nanochat.causal_serialisation import ChunkReservoir
from nanochat.causal_serialisation_extended import Copy, Delete, Insert, Move, Replace, SequenceEdit, TokensOperations, get_specials, resolve
from nanochat.edit_dataloader import (
    EditSamplingConfig,
    _CopyCandidate,
    _copy_candidate_to_sequence_edit,
    _sample_indel_bos_pos,
    _sample_random_inverse_edit,
    _sample_valid_subset,
    _training_data_from_ar_dsl_sequence_edit,
    _training_data_from_flat_relative_sequence_edit,
    char_sequence_edit_to_token_sequence_edit,
    script_edit_mass,
    sequence_edit_mass,
)
from nanochat.tokenizer import FLAT_RELATIVE_CONTEXT_LENGTH, flat_relative_token
from nanochat.edit_dsl import (
    EDIT_SCRIPT_CLOSE,
    EDIT_SCRIPT_OPEN,
    ar_dsl_coordinate_token_spans,
    ar_dsl_to_sequence_edit,
    sequence_edit_to_ar_dsl,
    sequence_edit_to_ar_dsl_tokens,
)


def test_script_edit_mass_uses_no_alignment_heuristic():
    edits = [
        Insert(0, np.array([1, 2])),
        Delete(2, 5),
        Replace(5, 7, np.array([3, 4, 5])),
        Copy(7, 11, 12),
        Move(12, 17, 20),
    ]
    assert [script_edit_mass(edit) for edit in edits] == [2, 3, 5, 4, 5]
    assert sequence_edit_mass(
        SequenceEdit(sequence=np.arange(24), edits=edits)
    ) == 19


class ByteEnc:
    def __init__(self):
        self._special_tokens = get_specials({}, 255)
        self._special_tokens["<|assistant_end|>"] = max(self._special_tokens.values()) + 1
        self.max_token_value = max(self._special_tokens.values())

    def encode(self, text, allowed_special=None):
        return list(text.encode("utf-8"))

    def decode_single_token_bytes(self, token):
        return bytes([int(token)])

    def decode(self, tokens):
        return bytes([int(t) for t in tokens]).decode("utf-8")

    def encode_special(self, text):
        return self._special_tokens[text]


def test_ar_dsl_row_supervises_terminal_and_masks_padding():
    enc = ByteEnc()
    sequence_edit = SequenceEdit(
        sequence=np.array(list(b"abc")),
        edits=[Delete(1, 2)],
    )

    row = _training_data_from_ar_dsl_sequence_edit(
        seq_edit=sequence_edit,
        tokenizer=enc,
        token_budget=64,
        pos_buffer_size=8,
        pad_to_budget=True,
    )

    assert row is not None
    terminal = enc.encode_special("<|assistant_end|>")
    supervised = np.flatnonzero(row.target.token_loss_weights)
    assert row.target.tokens[supervised[-1]] == terminal
    assert np.all(row.target.tokens[supervised[-1] + 1 :] == -1)
    assert not np.any(row.target.token_loss_weights[supervised[-1] + 1 :])


class FlatByteEnc(ByteEnc):
    def __init__(self):
        super().__init__()
        self.enc = self
        first_id = self.max_token_value + 1
        self._special_tokens.update(
            {
                flat_relative_token(distance): first_id + distance - 1
                for distance in range(1, FLAT_RELATIVE_CONTEXT_LENGTH)
            }
        )
        self.max_token_value = max(self._special_tokens.values())


def test_flat_relative_row_supervises_terminal_and_masks_padding():
    enc = FlatByteEnc()
    sequence_edit = SequenceEdit(
        sequence=np.array(list(b"abc")),
        edits=[Delete(1, 2)],
    )

    row = _training_data_from_flat_relative_sequence_edit(
        seq_edit=sequence_edit,
        tokenizer=enc,
        token_budget=64,
        pos_buffer_size=8,
        pad_to_budget=True,
    )

    assert row is not None
    terminal = enc.encode_special("<|assistant_end|>")
    supervised = np.flatnonzero(row.target.token_loss_weights)
    assert row.target.tokens[supervised[-1]] == terminal
    assert np.all(row.target.tokens[supervised[-1] + 1 :] == -1)
    assert not np.any(row.target.token_loss_weights[supervised[-1] + 1 :])


def test_ar_dsl_completion_only_loss_masks_the_corrupted_source():
    enc = ByteEnc()
    sequence_edit = SequenceEdit(
        sequence=np.array(list(b"abc")),
        edits=[Delete(1, 2)],
    )

    row = _training_data_from_ar_dsl_sequence_edit(
        seq_edit=sequence_edit,
        tokenizer=enc,
        token_budget=64,
        pos_buffer_size=8,
        completion_only_loss=True,
        terminal_weight=-1,
    )

    assert row is not None
    weights = np.asarray(row.target.token_loss_weights)
    completion_start = len(sequence_edit.sequence) - 1
    supervised = np.flatnonzero(weights)
    assert not np.any(weights[:completion_start])
    assert supervised[0] == completion_start
    assert row.target.tokens[supervised[-1]] == enc.encode_special("<|assistant_end|>")
    assert weights[supervised[-1]] > 1


def test_flat_relative_completion_only_loss_masks_the_corrupted_source():
    enc = FlatByteEnc()
    sequence_edit = SequenceEdit(
        sequence=np.array(list(b"abc")),
        edits=[Delete(1, 2)],
    )

    row = _training_data_from_flat_relative_sequence_edit(
        seq_edit=sequence_edit,
        tokenizer=enc,
        token_budget=64,
        pos_buffer_size=8,
        completion_only_loss=True,
        terminal_weight=-1,
    )

    assert row is not None
    weights = np.asarray(row.target.token_loss_weights)
    completion_start = len(sequence_edit.sequence) - 1
    supervised = np.flatnonzero(weights)
    assert not np.any(weights[:completion_start])
    assert supervised[0] == completion_start
    assert row.target.tokens[supervised[-1]] == enc.encode_special("<|assistant_end|>")
    assert weights[supervised[-1]] > 1


class MergingEnc(ByteEnc):
    """Tiny BPE-like encoder where common triples are single tokens."""

    chunks = {300: b"abc", 301: b"def"}

    def encode(self, text, allowed_special=None):
        data = text.encode("utf-8")
        out = []
        cursor = 0
        while cursor < len(data):
            merged = next(
                ((token, chunk) for token, chunk in self.chunks.items() if data.startswith(chunk, cursor)),
                None,
            )
            if merged is None:
                out.append(data[cursor])
                cursor += 1
            else:
                token, chunk = merged
                out.append(token)
                cursor += len(chunk)
        return out

    def decode_single_token_bytes(self, token):
        token = int(token)
        return self.chunks[token] if token in self.chunks else bytes([token])


def test_char_sequence_edit_to_token_sequence_edit_preserves_boundary_edits():
    seq_edit = SequenceEdit(
        sequence=np.array(list("abcdef")),
        edits=[
            Insert(1, np.array(list("X"))),
            Delete(2, 4),
            Replace(5, 6, np.array(list("Y"))),
        ],
    )

    out = char_sequence_edit_to_token_sequence_edit(seq_edit, ByteEnc())

    assert out is not None
    assert out.sequence.tolist() == list(b"abcdef")
    assert isinstance(out.edits[0], Insert)
    assert out.edits[0].pos == 1
    assert np.array_equal(out.edits[0].content, np.array([ord("X")]))
    assert out.edits[1] == Delete(2, 4)
    assert isinstance(out.edits[2], Replace)
    assert out.edits[2].start == 5
    assert out.edits[2].end == 6
    assert np.array_equal(out.edits[2].content, np.array([ord("Y")]))


def test_char_sequence_edit_expands_inside_bpe_boundary_exactly():
    enc = MergingEnc()
    char_edit = SequenceEdit(
        sequence=np.array(list("abcdef")),
        edits=[Insert(1, np.array(list("XYZ")))],
    )

    token_edit = char_sequence_edit_to_token_sequence_edit(char_edit, enc)

    assert token_edit is not None
    assert token_edit.sequence.tolist() == [300, 301]
    assert token_edit.edits[0].start == 0
    assert token_edit.edits[0].end == 1
    expected = "".join(resolve(char_edit).tolist()).encode()
    actual = b"".join(enc.decode_single_token_bytes(t) for t in resolve(token_edit))
    assert actual == expected == b"aXYZbcdef"


def test_copy_scan_candidate_removes_one_occurrence_and_copies_it_back():
    raw = "abc XX def XX ghi"
    candidate = _CopyCandidate(repeated_text="XX", occurrences=(4, 11), length_chars=2)

    out = _copy_candidate_to_sequence_edit(raw, candidate, rng=np.random.default_rng(0))

    assert out is not None
    assert "".join(out.sequence.tolist()) == "abc XX def  ghi"
    assert out.edits == [Copy(4, 6, 11)]


def test_sample_valid_subset_drops_overlapping_edits():
    seq_edit = SequenceEdit(
        sequence=np.array(list("abcdef")),
        edits=[Delete(1, 4), Replace(2, 5, np.array(list("X")))],
    )

    out = _sample_valid_subset(
        seq_edit,
        op_probs={"delete": 1.0, "replace": 1.0},
        max_edits=2,
        rng=np.random.default_rng(2),
    )

    assert out is not None
    assert len(out.edits) == 1


def test_sample_valid_subset_can_require_an_op():
    seq_edit = SequenceEdit(
        sequence=np.array(list("abcdef")),
        edits=[
            Delete(1, 2),
            Replace(3, 4, np.array(list("X"))),
        ],
    )

    out = _sample_valid_subset(
        seq_edit,
        op_probs={"delete": 1.0, "replace": 0.0},
        max_edits=1,
        rng=np.random.default_rng(3),
        required_op="replace",
    )

    assert out is not None
    assert len(out.edits) == 1
    assert isinstance(out.edits[0], Replace)


def test_random_inverse_edit_serialises_with_extended_ops():
    rng = np.random.default_rng(0)
    tokeniser = ByteEnc()
    ops = TokensOperations(tokeniser)
    cfg = EditSamplingConfig(
        random_no_del_mean=4,
        random_del_mean_len=2,
        random_no_ins_mean=4,
        random_ins_mean_len=2,
        max_edits_per_example=2,
    )
    reservoir = ChunkReservoir(max_chunks=2, chunk_size=8, rng=rng)
    sequence = np.array(list(b"abcdefghijk"), dtype=np.int64)

    seq_edit = _sample_random_inverse_edit(
        sequence,
        cfg=cfg,
        token_ops=ops,
        reservoir=reservoir,
        rng=rng,
    )

    assert seq_edit is not None
    serialised = ops.serialise(seq_edit)
    training = ops.prepare_training_example(serialised, pos_buffer=8)
    assert len(training.input_tokens) == len(serialised.tokens) - 1


def test_random_inverse_edit_can_sample_move_ops():
    rng = np.random.default_rng(4)
    tokeniser = ByteEnc()
    ops = TokensOperations(tokeniser)
    cfg = EditSamplingConfig(
        random_no_del_mean=1,
        random_del_mean_len=2,
        random_no_ins_mean=1,
        random_ins_mean_len=2,
        random_replace_prob=0.0,
        random_delete_prob=0.0,
        random_insert_prob=0.0,
        random_move_prob=1.0,
        max_edits_per_example=1,
    )
    reservoir = ChunkReservoir(max_chunks=2, chunk_size=8, rng=rng)
    sequence = np.array(list(b"abcdefghijk"), dtype=np.int64)

    seq_edit = _sample_random_inverse_edit(
        sequence,
        cfg=cfg,
        token_ops=ops,
        reservoir=reservoir,
        rng=rng,
    )

    assert seq_edit is not None
    assert len(seq_edit.edits) == 1
    assert isinstance(seq_edit.edits[0], Move)
    ops.prepare_training_example(ops.serialise(seq_edit), pos_buffer=8)


def test_ar_dsl_roundtrips_token_coordinate_edits():
    tokeniser = ByteEnc()
    seq_edit = SequenceEdit(
        sequence=np.array(list(b"abcdefghi"), dtype=np.int64),
        edits=[
            Insert(1, np.array(list(b"X Y"), dtype=np.int64)),
            Delete(2, 4),
            Replace(5, 7, np.array(list(b"Q\"Z"), dtype=np.int64)),
            Move(7, 8, 9),
        ],
    )

    dsl = sequence_edit_to_ar_dsl(seq_edit, tokeniser)
    parsed = ar_dsl_to_sequence_edit(seq_edit.sequence, dsl, tokeniser)

    assert dsl.startswith(EDIT_SCRIPT_OPEN)
    assert dsl.endswith(EDIT_SCRIPT_CLOSE)
    assert parsed.sequence.tolist() == seq_edit.sequence.tolist()
    assert len(parsed.edits) == len(seq_edit.edits)
    for left, right in zip(parsed.edits, seq_edit.edits):
        assert type(left) is type(right)
        if isinstance(left, Insert):
            assert left.pos == right.pos
            assert np.array_equal(left.content, right.content)
        elif isinstance(left, Replace):
            assert left.start == right.start
            assert left.end == right.end
            assert np.array_equal(left.content, right.content)
        else:
            assert left == right


def test_ar_dsl_tokens_are_plain_autoregressive_text():
    class Tok:
        enc = ByteEnc()

        def encode(self, text):
            return self.enc.encode(text)

        def decode(self, tokens):
            return self.enc.decode(tokens)

    tok = Tok()
    seq_edit = SequenceEdit(
        sequence=np.array(list(b"abcdef"), dtype=np.int64),
        edits=[Replace(1, 3, np.array(list(b"XY"), dtype=np.int64))],
    )

    tokens = sequence_edit_to_ar_dsl_tokens(seq_edit, tok)
    text = tok.decode(tokens)

    assert "<edit-script>" in text
    assert 'R 1 3 "XY"' in text


def test_ar_dsl_coordinate_token_spans_cover_only_position_fields():
    class Tok:
        enc = ByteEnc()

        def encode(self, text):
            return self.enc.encode(text)

    tok = Tok()
    dsl = "\n<edit-script>\nI 12 \"34\"\nD 5 6\nR 7 8 \"9\"\nM 1 2 -> 3\n</edit-script>\n"
    source_len = 10

    spans = ar_dsl_coordinate_token_spans(tok, dsl, source_len)
    fields = [dsl[start - source_len:end - source_len] for start, end in spans]

    assert fields == ["12", "5", "6", "7", "8", "1", "2", "3"]


def test_sample_indel_bos_for_edited_examples_respects_earliest_guard_boundary():
    seq_edit = SequenceEdit(
        sequence=np.arange(20, dtype=np.int64),
        edits=[
            Replace(5, 8, np.array([100, 101], dtype=np.int64)),
            Copy(12, 15, 3),
        ],
    )
    cfg = EditSamplingConfig(
        corrupted_indel_bos_random_prob=1.0,
        corrupted_indel_bos_start_prob=0.0,
    )
    rng = np.random.default_rng(0)

    positions = [
        _sample_indel_bos_pos(seq_edit, cfg=cfg, rng=rng, edited=True)
        for _ in range(100)
    ]

    assert all(pos is not None and 0 <= pos <= 3 for pos in positions)
    assert any(pos > 0 for pos in positions)


def test_sample_indel_bos_can_omit_clean_noop_examples():
    seq_edit = SequenceEdit(sequence=np.arange(10, dtype=np.int64), edits=[])
    cfg = EditSamplingConfig(
        clean_indel_bos_absent_prob=1.0,
        clean_indel_bos_random_prob=0.0,
        clean_indel_bos_start_prob=0.0,
    )

    assert _sample_indel_bos_pos(
        seq_edit,
        cfg=cfg,
        rng=np.random.default_rng(0),
        edited=False,
    ) is None
