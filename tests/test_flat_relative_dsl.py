import numpy as np
import pytest

from nanochat.causal_serialisation_extended import (
    Copy,
    Delete,
    Insert,
    Move,
    Replace,
    SequenceEdit,
    resolve,
)
from nanochat.flat_relative_dsl import (
    flat_relative_token_maps,
    flat_relative_tokens_to_sequence_edit,
    sequence_edit_to_flat_relative_tokens,
)
from nanochat.tokenizer import (
    FLAT_RELATIVE_CONTEXT_LENGTH,
    flat_relative_token,
    get_extended_indel_tokeniser,
    get_flat_relative_2048_tokeniser,
)


def test_flat_tokenizer_preserves_canonical_ids_and_adds_2047_offsets():
    canonical = get_extended_indel_tokeniser()
    flat = get_flat_relative_2048_tokeniser()

    assert canonical.get_vocab_size() == 32768
    assert flat.get_vocab_size() == 32768 + FLAT_RELATIVE_CONTEXT_LENGTH - 1
    for name, token_id in canonical.enc._special_tokens.items():
        assert flat.enc._special_tokens[name] == token_id
    assert flat.encode_special(flat_relative_token(1)) == 32768
    assert flat.encode_special(flat_relative_token(2047)) == 34814


@pytest.mark.parametrize(
    "edits",
    [
        [Delete(2, 4)],
        [Insert(3, np.asarray([65], dtype=np.int64))],
        [Replace(2, 4, np.asarray([67], dtype=np.int64))],
        [Copy(2, 4, 6)],
        [Move(2, 4, 6)],
    ],
)
def test_flat_relative_program_roundtrips_every_operation(edits):
    tokenizer = get_flat_relative_2048_tokeniser()
    source = np.asarray(
        [tokenizer.get_bos_token_id(), *tokenizer.encode("one two three four five")],
        dtype=np.int64,
    )
    sequence_edit = SequenceEdit(source, edits)

    program = sequence_edit_to_flat_relative_tokens(sequence_edit, tokenizer)
    parsed = flat_relative_tokens_to_sequence_edit(source, program, tokenizer)

    assert np.array_equal(resolve(parsed), resolve(sequence_edit))
    _, id_to_distance = flat_relative_token_maps(tokenizer)
    expected_coordinates = sum(
        1 if isinstance(edit, Insert) else 2 if isinstance(edit, (Delete, Replace)) else 3
        for edit in edits
    )
    assert sum(token in id_to_distance for token in program) == expected_coordinates


def test_flat_relative_coordinate_is_counted_from_its_serialized_index():
    tokenizer = get_flat_relative_2048_tokeniser()
    source = np.asarray(
        [tokenizer.get_bos_token_id(), *tokenizer.encode("abcdef")],
        dtype=np.int64,
    )
    sequence_edit = SequenceEdit(source, [Delete(1, 2)])
    program = sequence_edit_to_flat_relative_tokens(sequence_edit, tokenizer)
    _, id_to_distance = flat_relative_token_maps(tokenizer)

    coordinate_indices = [index for index, token in enumerate(program) if token in id_to_distance]
    recovered = [
        len(source) + index - id_to_distance[program[index]]
        for index in coordinate_indices
    ]
    assert recovered == [1, 2]


def test_flat_relative_rejects_a_back_token_before_the_source():
    tokenizer = get_flat_relative_2048_tokeniser()
    source = [tokenizer.get_bos_token_id(), *tokenizer.encode("short")]
    program = [
        *tokenizer.encode("\n<edit-script>\nD "),
        tokenizer.encode_special(flat_relative_token(2047)),
        *tokenizer.encode(" 1\n</edit-script>"),
    ]
    with pytest.raises(ValueError, match="out-of-range coordinate"):
        flat_relative_tokens_to_sequence_edit(source, program, tokenizer)
