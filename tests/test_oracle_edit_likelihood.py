import numpy as np
import pytest

from nanochat.causal_serialisation_extended import (
    Delete,
    Insert,
    Replace,
    TokensOperations,
    resolve,
)
from nanochat.oracle_edit_likelihood import (
    canonical_sequence_edit_from_tokens,
    conditional_program_training_data,
)
from nanochat.tokenizer import get_extended_indel_tokeniser


def test_canonical_minimum_token_edit_resolves_target():
    source = [99, 1, 2, 3, 4, 5]
    target = [99, 1, 8, 3, 5, 6]
    edit = canonical_sequence_edit_from_tokens(source, target)

    assert resolve(edit).tolist() == target
    assert any(isinstance(op, Replace) for op in edit.edits)


def test_canonical_identical_pair_is_noop():
    edit = canonical_sequence_edit_from_tokens([99, 1, 2], [99, 1, 2])
    assert edit.edits == []
    assert np.array_equal(edit.sequence, np.array([99, 1, 2]))


def test_conditional_mask_excludes_source_and_keeps_edit_suffix():
    canonical_tokenizer = get_extended_indel_tokeniser()
    token_ops = TokensOperations(canonical_tokenizer.enc)
    bos = canonical_tokenizer.get_bos_token_id()
    source = [bos, *canonical_tokenizer.encode("A bad sentence.")]
    target = [bos, *canonical_tokenizer.encode("A good sentence.")]
    edit = canonical_sequence_edit_from_tokens(source, target)
    data = conditional_program_training_data(
        token_ops,
        edit,
        position_buffer_size=64,
    )
    serialised = token_ops.serialise(edit, indel_bos_pos=0)
    edit_start = int(np.flatnonzero(serialised.tokens == token_ops.edit_start_token)[0])
    weighted_target_indices = np.flatnonzero(data.target.token_loss_weights) + 1

    assert len(weighted_target_indices) > 0
    assert int(weighted_target_indices.min()) > edit_start
    assert int(data.target.gather_valid.sum()) > 0
