import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nanochat.causal_serialisation_extended import (
    COPY_DEST,
    COPY_END,
    COPY_OPEN,
    COPY_START,
    DEL_END,
    DEL_START,
    INS_LOC,
    INDL_BOS,
    MOVE_DEST,
    MOVE_END,
    MOVE_OPEN,
    MOVE_START,
    REPLACE_END,
    REPLACE_EOS,
    REPLACE_OPEN,
    REPLACE_START,
    Copy,
    Delete,
    Insert,
    Move,
    Replace,
    SequenceEdit,
    SerialisedData,
    TokensOperations,
    _edit_key,
    _validate_edits,
    complement,
    get_specials,
    print_training_examples,
    resolve,
)


class ByteTokenizer:
    def __init__(self):
        self._special_tokens = get_specials({}, 255)
        self.max_token_value = max(self._special_tokens.values())
        self._id_to_special = {v: k for k, v in self._special_tokens.items()}

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        parts = []
        for token in tokens:
            token = int(token)
            if token in self._id_to_special:
                parts.append(self._id_to_special[token])
            elif 0 <= token < 256:
                parts.append(bytes([token]).decode("utf-8", errors="replace"))
        return "".join(parts)

    def decode_batch(self, batch):
        return [self.decode(tokens) for tokens in batch]


def _edit_equal(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Insert):
        return left.pos == right.pos and np.array_equal(np.array(left.content), np.array(right.content))
    if isinstance(left, Delete):
        return left.start == right.start and left.end == right.end
    if isinstance(left, Copy):
        return left.start == right.start and left.end == right.end and left.dest == right.dest
    if isinstance(left, Move):
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


def _random_source_tokens(rng: np.random.Generator, *, min_len: int = 1, max_len: int = 12) -> np.ndarray:
    length = int(rng.integers(min_len, max_len + 1))
    alphabet = np.array(list(b"abcdefghiJKLMNOPQR"), dtype=np.uint8)
    return alphabet[rng.integers(0, len(alphabet), size=length)].astype(np.int64)


def _random_source_span(rng: np.random.Generator, n: int) -> tuple[int, int]:
    start = int(rng.integers(0, n))
    end = int(rng.integers(start + 1, n + 1))
    return start, end


def _random_source_span_from(rng: np.random.Generator, n: int, min_start: int) -> tuple[int, int]:
    start = int(rng.integers(min_start, n))
    end = int(rng.integers(start + 1, n + 1))
    return start, end


def _random_valid_sequence_edit(rng: np.random.Generator) -> SequenceEdit:
    while True:
        sequence = _random_source_tokens(rng)
        n = len(sequence)
        edits = []
        boundary = 0

        while boundary <= n:
            extra_ops = int(rng.integers(0, 3))
            for _ in range(extra_ops):
                op = "insert"
                if n > 0 and boundary < n:
                    roll = rng.random()
                    if roll < 0.5:
                        op = "copy"
                if op == "insert":
                    edits.append(Insert(boundary, _random_ascii_tokens(rng)))
                elif op == "copy":
                    start, end = _random_source_span(rng, n)
                    edits.append(Copy(start, end, boundary))

            if boundary == n:
                break

            if rng.random() < 0.45:
                end = int(rng.integers(boundary + 1, n + 1))
                if rng.random() < 0.5:
                    edits.append(Delete(boundary, end))
                else:
                    edits.append(Replace(boundary, end, _random_ascii_tokens(rng)))
                boundary = end
            else:
                boundary += 1

        edits.sort(key=_edit_key)
        try:
            _validate_edits(sequence, edits)
        except ValueError:
            continue
        return SequenceEdit(sequence=sequence, edits=edits)


def test_replace_roundtrip_serialise_deserialise_and_resolve():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[Replace(2, 4, np.array(tokeniser.encode("XY")))],
    )

    serialised = ops.serialise(seq_edit)
    deserialised = ops.deserialise(serialised)

    assert _seq_edit_equal(seq_edit, deserialised)
    assert tokeniser.decode(ops.resolve(serialised)) == "abXYef"


def test_copy_roundtrip_serialise_deserialise_and_resolve():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[Copy(1, 3, 6)],
    )

    serialised = ops.serialise(seq_edit)
    deserialised = ops.deserialise(serialised)

    assert _seq_edit_equal(seq_edit, deserialised)
    assert tokeniser.decode(ops.resolve(serialised)) == "abcdefbc"


def test_move_roundtrip_serialise_deserialise_and_resolve():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[Move(1, 3, 6)],
    )

    serialised = ops.serialise(seq_edit)
    deserialised = ops.deserialise(serialised)

    assert _seq_edit_equal(seq_edit, deserialised)
    assert tokeniser.decode(ops.resolve(serialised)) == "adefbc"


def test_move_can_shift_span_to_earlier_boundary():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[Move(2, 4, 0)],
    )

    serialised = ops.serialise(seq_edit)
    assert tokeniser.decode(ops.resolve(serialised)) == "cdabef"


def test_shared_pointer_token_mode_preserves_edit_semantics_and_supervision():
    tokeniser = ByteTokenizer()
    original_ops = TokensOperations(tokeniser)
    shared_ops = TokensOperations(tokeniser, pointer_token_mode="shared")
    shared_id = tokeniser._special_tokens[INS_LOC]
    shared_roles = {
        shared_ops.deletion_start_token,
        shared_ops.deletion_end_token,
        shared_ops.insertion_loc_token,
        shared_ops.replace_start_token,
        shared_ops.replace_end_token,
        shared_ops.copy_start_token,
        shared_ops.copy_end_token,
        shared_ops.copy_dest_token,
        shared_ops.move_start_token,
        shared_ops.move_end_token,
        shared_ops.move_dest_token,
    }
    assert shared_roles == {shared_id}
    assert original_ops.deletion_start_token != original_ops.deletion_end_token

    examples = [
        SequenceEdit(
            sequence=np.array(tokeniser.encode("abcdef")),
            edits=[Insert(2, np.array(tokeniser.encode("XY")))],
        ),
        SequenceEdit(
            sequence=np.array(tokeniser.encode("abcdef")),
            edits=[Delete(2, 4)],
        ),
        SequenceEdit(
            sequence=np.array(tokeniser.encode("abcdef")),
            edits=[Replace(2, 4, np.array(tokeniser.encode("XY")))],
        ),
        SequenceEdit(
            sequence=np.array(tokeniser.encode("abcdef")),
            edits=[Copy(1, 3, 5)],
        ),
        SequenceEdit(
            sequence=np.array(tokeniser.encode("abcdef")),
            edits=[Move(1, 3, 5)],
        ),
    ]
    for seq_edit in examples:
        original = original_ops.serialise(seq_edit, indel_bos_pos=1)
        shared = shared_ops.serialise(seq_edit, indel_bos_pos=1)
        assert _seq_edit_equal(shared_ops.deserialise(shared), seq_edit)
        assert np.array_equal(shared_ops.resolve(shared), original_ops.resolve(original))

        original_training = original_ops.prepare_training_example(original, pos_buffer=8)
        shared_training = shared_ops.prepare_training_example(shared, pos_buffer=8)
        assert np.array_equal(
            original_training.target.gathered_pos_target,
            shared_training.target.gathered_pos_target,
        )
        assert np.array_equal(
            original_training.target.token_loss_weights,
            shared_training.target.token_loss_weights,
        )


def test_shared_pointer_role_scanner_handles_multiple_packed_programs():
    tokeniser = ByteTokenizer()
    original_ops = TokensOperations(tokeniser)
    shared_ops = TokensOperations(tokeniser, pointer_token_mode="shared")
    examples = [
        SequenceEdit(
            sequence=np.array(tokeniser.encode("abcdef")),
            edits=[Delete(1, 3), Insert(5, np.array(tokeniser.encode("X")))],
        ),
        SequenceEdit(
            sequence=np.array(tokeniser.encode("uvwxyz")),
            edits=[Replace(2, 4, np.array(tokeniser.encode("YY")))],
        ),
    ]

    def pack(ops: TokensOperations) -> SerialisedData:
        rows = [ops.serialise(example, indel_bos_pos=0) for example in examples]
        token_parts = []
        position_parts = []
        offset = 0
        for row in rows:
            token_parts.append(row.tokens)
            position_parts.append(row.positions + offset)
            offset += len(row.tokens)
        return SerialisedData(
            tokens=np.concatenate(token_parts),
            positions=np.concatenate(position_parts),
        )

    original = original_ops.prepare_training_example(pack(original_ops), pos_buffer=16)
    shared = shared_ops.prepare_training_example(pack(shared_ops), pos_buffer=16)
    assert original.target.gather_valid.sum() == shared.target.gather_valid.sum() == 5
    assert np.array_equal(
        original.target.gathered_pos_target,
        shared.target.gathered_pos_target,
    )
    assert np.array_equal(
        original.target.token_loss_weights,
        shared.target.token_loss_weights,
    )


def test_complement_of_copy_is_delete_over_copied_output_span():
    tokeniser = ByteTokenizer()
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[Copy(1, 3, 6)],
    )

    inverse = complement(seq_edit)

    assert tokeniser.decode(inverse.sequence) == "abcdefbc"
    assert len(inverse.edits) == 1
    edit = inverse.edits[0]
    assert isinstance(edit, Delete)
    assert edit.start == 6
    assert edit.end == 8


def test_complement_of_replacement_is_inverse_replacement():
    tokeniser = ByteTokenizer()
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[Replace(2, 4, np.array(tokeniser.encode("XY")))],
    )

    inverse = complement(seq_edit)

    assert tokeniser.decode(inverse.sequence) == "abXYef"
    assert len(inverse.edits) == 1
    edit = inverse.edits[0]
    assert isinstance(edit, Replace)
    assert edit.start == 2
    assert edit.end == 4
    assert tokeniser.decode(edit.content) == "cd"


def test_complement_of_move_to_later_boundary_is_inverse_move():
    tokeniser = ByteTokenizer()
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[Move(1, 3, 6)],
    )

    inverse = complement(seq_edit)

    assert tokeniser.decode(inverse.sequence) == "adefbc"
    assert len(inverse.edits) == 1
    edit = inverse.edits[0]
    assert isinstance(edit, Move)
    assert edit == Move(4, 6, 1)
    assert tokeniser.decode(resolve(inverse)) == "abcdef"


def test_complement_of_move_to_earlier_boundary_is_inverse_move():
    tokeniser = ByteTokenizer()
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[Move(3, 5, 1)],
    )

    inverse = complement(seq_edit)

    assert tokeniser.decode(inverse.sequence) == "adebcf"
    assert len(inverse.edits) == 1
    edit = inverse.edits[0]
    assert isinstance(edit, Move)
    assert edit == Move(1, 3, 5)
    assert tokeniser.decode(resolve(inverse)) == "abcdef"


def test_complement_of_move_delete_same_boundary_preserves_inverse_order():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcXXXYYY")),
        edits=[Move(6, 9, 3), Delete(3, 6)],
    )

    inverse = complement(seq_edit)

    assert tokeniser.decode(inverse.sequence) == "abcYYY"
    assert [type(edit) for edit in inverse.edits] == [Insert, Move]
    assert tokeniser.decode(resolve(inverse)) == "abcXXXYYY"

    roundtripped = ops.deserialise(ops.serialise(inverse))
    assert tokeniser.decode(resolve(roundtripped)) == "abcXXXYYY"


def test_complement_of_move_delete_same_boundary_preserves_reverse_inverse_order():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcYYYXXXqq")),
        edits=[Move(3, 6, 11), Delete(6, 9)],
    )

    inverse = complement(seq_edit)

    assert tokeniser.decode(inverse.sequence) == "abcqqYYY"
    assert [type(edit) for edit in inverse.edits] == [Move, Insert]
    assert tokeniser.decode(resolve(inverse)) == "abcYYYXXXqq"

    roundtripped = ops.deserialise(ops.serialise(inverse))
    assert tokeniser.decode(resolve(roundtripped)) == "abcYYYXXXqq"


def test_replace_can_mix_with_insertions_and_deletions():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[
            Delete(0, 1),
            Replace(2, 4, np.array(tokeniser.encode("W"))),
            Insert(6, np.array(tokeniser.encode("XY"))),
        ],
    )

    serialised = ops.serialise(seq_edit)
    deserialised = ops.deserialise(serialised)

    assert _seq_edit_equal(seq_edit, deserialised)
    assert tokeniser.decode(ops.resolve(serialised)) == "bWefXY"


def test_copy_can_mix_with_replacements_and_insertions():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[
            Insert(0, np.array(tokeniser.encode("^"))),
            Copy(1, 3, 4),
            Replace(4, 6, np.array(tokeniser.encode("!"))),
        ],
    )

    serialised = ops.serialise(seq_edit)
    deserialised = ops.deserialise(serialised)

    assert _seq_edit_equal(seq_edit, deserialised)
    assert tokeniser.decode(ops.resolve(serialised)) == "^abcdbc!"


def test_move_serialises_all_three_pointer_tokens():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcd")),
        edits=[Move(1, 3, 4)],
    )

    serialised = ops.serialise(seq_edit)
    tokens = serialised.tokens.tolist()
    positions = serialised.positions.tolist()

    move_open = tokeniser._special_tokens[MOVE_OPEN]
    move_start = tokeniser._special_tokens[MOVE_START]
    move_end = tokeniser._special_tokens[MOVE_END]
    move_dest = tokeniser._special_tokens[MOVE_DEST]

    move_ix = tokens.index(move_open)
    assert tokens[move_ix:move_ix + 4] == [
        move_open,
        move_start,
        move_end,
        move_dest,
    ]
    assert positions[move_ix + 1] == 2
    assert positions[move_ix + 2] == 3
    assert positions[move_ix + 3] == 4


def test_insertions_can_land_immediately_before_and_after_a_replacement():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[
            Insert(2, np.array(tokeniser.encode("<"))),
            Replace(2, 4, np.array(tokeniser.encode("XY"))),
            Insert(4, np.array(tokeniser.encode(">"))),
        ],
    )

    serialised = ops.serialise(seq_edit)
    deserialised = ops.deserialise(serialised)

    assert _seq_edit_equal(seq_edit, deserialised)
    assert tokeniser.decode(ops.resolve(serialised)) == "ab<XY>ef"


def test_replace_allows_payload_length_to_differ_from_removed_span_length():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[Replace(2, 4, np.array(tokeniser.encode("XYZ")))],
    )

    serialised = ops.serialise(seq_edit)
    deserialised = ops.deserialise(serialised)

    assert _seq_edit_equal(seq_edit, deserialised)
    assert tokeniser.decode(ops.resolve(serialised)) == "abXYZef"


def test_complement_roundtrip_restores_original_sequence():
    tokeniser = ByteTokenizer()
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[
            Insert(0, np.array(tokeniser.encode("^"))),
            Delete(1, 2),
            Replace(3, 5, np.array(tokeniser.encode("XYZ"))),
            Insert(6, np.array(tokeniser.encode("$"))),
        ],
    )

    inverse = complement(seq_edit)
    restored = complement(inverse)

    assert tokeniser.decode(inverse.sequence) == "^acXYZf$"
    assert tokeniser.decode(restored.sequence) == "abcdef"
    assert _seq_edit_equal(restored, seq_edit)


def test_invalid_insertion_inside_replaced_span_is_rejected():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[
            Replace(2, 4, np.array(tokeniser.encode("XY"))),
            Insert(3, np.array(tokeniser.encode("!"))),
        ],
    )

    with pytest.raises(ValueError, match="inside a removed span"):
        ops.serialise(seq_edit)


def test_invalid_move_destination_inside_moved_span_is_rejected():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[Move(2, 5, 3)],
    )

    with pytest.raises(ValueError, match="strictly inside the moved span"):
        ops.serialise(seq_edit)


def test_overlapping_span_edits_are_rejected():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[
            Delete(1, 4),
            Replace(3, 5, np.array(tokeniser.encode("XY"))),
        ],
    )

    with pytest.raises(ValueError, match="Overlapping span edits"):
        ops.serialise(seq_edit)


def test_copy_serialises_all_three_pointer_tokens():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcd")),
        edits=[Copy(1, 3, 4)],
    )

    serialised = ops.serialise(seq_edit)
    tokens = serialised.tokens.tolist()
    positions = serialised.positions.tolist()

    copy_open = tokeniser._special_tokens[COPY_OPEN]
    copy_start = tokeniser._special_tokens[COPY_START]
    copy_end = tokeniser._special_tokens[COPY_END]
    copy_dest = tokeniser._special_tokens[COPY_DEST]

    copy_ix = tokens.index(copy_open)
    assert tokens[copy_ix:copy_ix + 4] == [
        copy_open,
        copy_start,
        copy_end,
        copy_dest,
    ]
    assert positions[copy_ix + 1] == 2
    assert positions[copy_ix + 2] == 3
    assert positions[copy_ix + 3] == 4


def test_debug_print_helpers_handle_replacements(capsys):
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcdef")),
        edits=[
            Insert(2, np.array(tokeniser.encode("<"))),
            Replace(2, 4, np.array(tokeniser.encode("XYZ"))),
            Delete(5, 6),
            Insert(6, np.array(tokeniser.encode(">"))),
        ],
    )

    serialised = ops.serialise(seq_edit)
    training = ops.prepare_training_example(serialised, pos_buffer=8)

    serialised.print(tokeniser)
    print_training_examples([training], tokeniser, max_rows=16)

    captured = capsys.readouterr()
    assert "replace-open" in captured.out


def test_position_control_tokens_do_not_receive_token_loss():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    pointer_token_ids = {
        tokeniser._special_tokens[name]
        for name in (
            INS_LOC,
            DEL_START,
            DEL_END,
            REPLACE_START,
            REPLACE_END,
            COPY_START,
            COPY_END,
            COPY_DEST,
            MOVE_START,
            MOVE_END,
            MOVE_DEST,
        )
    }
    examples = [
        SequenceEdit(
            sequence=np.array(tokeniser.encode("abcdef")),
            edits=[Insert(2, np.array(tokeniser.encode("XY")))],
        ),
        SequenceEdit(sequence=np.array(tokeniser.encode("abcdef")), edits=[Delete(2, 4)]),
        SequenceEdit(
            sequence=np.array(tokeniser.encode("abcdef")),
            edits=[Replace(2, 4, np.array(tokeniser.encode("XY")))],
        ),
        SequenceEdit(sequence=np.array(tokeniser.encode("abcdef")), edits=[Copy(1, 3, 5)]),
        SequenceEdit(sequence=np.array(tokeniser.encode("abcdef")), edits=[Move(1, 3, 5)]),
    ]

    for seq_edit in examples:
        serialised = ops.serialise(seq_edit)
        training = ops.prepare_training_example(serialised, pos_buffer=8)
        for absolute_ix, token in enumerate(serialised.tokens):
            if int(token) in pointer_token_ids and absolute_ix > 0:
                assert training.target.token_loss_weights[absolute_ix - 1] == 0


def test_random_mixed_edit_scripts_roundtrip_and_invert():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    rng = np.random.default_rng(0)

    for _ in range(250):
        seq_edit = _random_valid_sequence_edit(rng)

        serialised = ops.serialise(seq_edit)
        deserialised = ops.deserialise(serialised)
        assert _seq_edit_equal(seq_edit, deserialised)

        inverse = complement(seq_edit)
        restored = complement(inverse)
        assert np.array_equal(inverse.sequence, ops.resolve(serialised))
        assert np.array_equal(resolve(inverse), seq_edit.sequence)
        assert np.array_equal(resolve(restored), resolve(seq_edit))
        if not any(isinstance(edit, (Copy, Move)) for edit in seq_edit.edits):
            assert _seq_edit_equal(restored, seq_edit)


def test_replace_segment_and_training_targets_are_explicit():
    tokeniser = ByteTokenizer()
    ops = TokensOperations(tokeniser)
    seq_edit = SequenceEdit(
        sequence=np.array(tokeniser.encode("abcd")),
        edits=[Replace(1, 3, np.array(tokeniser.encode("XY")))],
    )

    serialised = ops.serialise(seq_edit)
    tokens = serialised.tokens.tolist()
    positions = serialised.positions.tolist()

    replace_open = tokeniser._special_tokens[REPLACE_OPEN]
    replace_start = tokeniser._special_tokens[REPLACE_START]
    replace_end = tokeniser._special_tokens[REPLACE_END]
    replace_eos = tokeniser._special_tokens[REPLACE_EOS]
    indel_bos = tokeniser._special_tokens[INDL_BOS]

    bos_ix = tokens.index(indel_bos)
    replace_ix = tokens.index(replace_open)

    assert bos_ix == 0
    assert tokens[replace_ix:replace_ix + 6] == [
        replace_open,
        replace_start,
        replace_end,
        ord("X"),
        ord("Y"),
        replace_eos,
    ]
    assert positions[replace_ix + 1] == 2
    assert positions[replace_ix + 2] == 3
    assert positions[replace_ix + 3] == 2
    assert positions[replace_ix + 4] == 2

    training = ops.prepare_training_example(serialised, pos_buffer=8)
    valid_targets = training.target.gathered_pos_target[training.target.gather_valid]

    assert valid_targets.tolist() == [2, 3]
    assert training.target.token_loss_weights[1] == 0
    assert training.target.token_loss_weights[2] == 0
    assert training.target.token_loss_weights[3] == 0
