from nanochat.causal_serialisation_extended import TokensOperations
from nanochat.tokenizer import get_extended_indel_char_tokeniser
from ces.workflows.benchmark_tokenization import prepare_benchmark_for_tokenizer


def _incompatible_row():
    return {
        "benchmark_id": "example",
        "source_text": "abc",
        "target_text": "axc",
        "source_tokens": [999],
        "target_tokens": [999],
        "prompt_tokens": [999],
        "prompt_positions": [0],
        "gold_edits": [{"op": "replace", "start": 2, "end": 3, "content": [120]}],
        "gold_op_counts": {"replace": 1},
    }


def test_retokenizes_frozen_text_when_manifest_ids_are_incompatible():
    tokenizer = get_extended_indel_char_tokeniser()
    token_ops = TokensOperations(tokenizer.enc)

    rows, metadata = prepare_benchmark_for_tokenizer(
        [_incompatible_row()],
        tokenizer=tokenizer,
        token_ops=token_ops,
    )

    row = rows[0]
    assert metadata["mode"] == "retokenized_text"
    assert metadata["incompatibility_counts"]
    assert row["source_tokens"] == [tokenizer.get_bos_token_id(), 97, 98, 99]
    assert row["target_tokens"] == [tokenizer.get_bos_token_id(), 97, 120, 99]
    assert row["prompt_tokens"][-1] == token_ops.edit_start_token
    assert row["gold_program_tokens"][-1] == token_ops.edit_end_token
    assert row["gold_edits"] == [
        {"op": "replace", "start": 2, "end": 3, "content": [120]}
    ]
    assert row["construction_gold_edits"] == _incompatible_row()["gold_edits"]


def test_preserves_ids_after_retokenized_row_is_compatible():
    tokenizer = get_extended_indel_char_tokeniser()
    token_ops = TokensOperations(tokenizer.enc)
    rebuilt, _metadata = prepare_benchmark_for_tokenizer(
        [_incompatible_row()],
        tokenizer=tokenizer,
        token_ops=token_ops,
    )

    rows, metadata = prepare_benchmark_for_tokenizer(
        rebuilt,
        tokenizer=tokenizer,
        token_ops=token_ops,
    )

    assert metadata["mode"] == "manifest_tokens"
    assert rows == rebuilt
