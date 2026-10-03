from ces.workflows.ces_balanced_sft import (
    balanced_objective_weights,
    cyclic_examples,
    distributed_cyclic_examples,
    row_fits_context,
)


def test_balanced_objective_weights_require_unit_mass():
    assert balanced_objective_weights(human=0.4, noop=0.4, synthetic=0.1, clean=0.1)["noop"] == 0.4
    try:
        balanced_objective_weights(human=0.5, noop=0.5, synthetic=0.1, clean=0.0)
    except ValueError as exc:
        assert "sum to one" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected invalid mixture to be rejected")


def test_cyclic_examples_cover_sequential_rows_then_wrap():
    rows = [{"id": index} for index in range(5)]
    assert [row["id"] for row in cyclic_examples(rows, step=0, count=3)] == [0, 1, 2]
    assert [row["id"] for row in cyclic_examples(rows, step=1, count=3)] == [3, 4, 0]


def test_distributed_cyclic_examples_partition_global_batch():
    rows = [{"id": index} for index in range(12)]
    rank0 = distributed_cyclic_examples(
        rows, step=1, global_count=4, rank=0, world_size=2
    )
    rank1 = distributed_cyclic_examples(
        rows, step=1, global_count=4, rank=1, world_size=2
    )
    assert [row["id"] for row in rank0] == [4, 5]
    assert [row["id"] for row in rank1] == [6, 7]


def test_row_fits_context_checks_program_and_clean_lengths():
    row = {
        "prompt_tokens": [1, 2, 3],
        "gold_program_tokens": [4, 5],
        "target_tokens": [1, 2, 3, 4],
    }
    assert row_fits_context(row, 4)
    assert not row_fits_context(row, 3)
