import math
from types import SimpleNamespace

import pytest
import torch

from ces.workflows.controlled_edit_eval import (
    _gold_signature,
    _tokens_hash,
    score_unique_sequences,
    select_benchmark_rows,
    summarise_records,
)


class _MinimumWidthModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(sequence_len=8)
        self.input_widths = []

    def forward(self, inputs, tix):
        assert inputs.shape[1] >= 2
        assert inputs.shape == tix.shape
        self.input_widths.append(inputs.shape[1])
        return torch.zeros((*inputs.shape, 4), device=inputs.device)


def test_score_unique_sequences_pads_two_token_sequence_for_model_forward():
    model = _MinimumWidthModel()
    scores = score_unique_sequences(
        model,
        [[1], [1, 2]],
        device=torch.device("cpu"),
        batch_size=8,
    )

    assert model.input_widths == [2]
    assert scores[_tokens_hash([1])] is None
    assert scores[_tokens_hash([1, 2])] == pytest.approx(-math.log(4))


def _record(**overrides):
    record = {
        "stratum": "replace_k1",
        "repair_num_ops": 1,
        "repair_op_mode": "replace",
        "density_regime": "training_span_distribution",
        "source_length_bin": "le128",
        "generation_seed": 7,
        "status": "valid",
        "valid": True,
        "exact_target": False,
        "exact_program": False,
        "char_similarity_delta": 0.1,
        "token_similarity_delta": 0.2,
        "likelihood_delta": 0.3,
        "likelihood_improved": True,
        "target_similarity_improved": True,
        "generation_seconds": 0.4,
        "apply_seconds": 0.01,
    }
    record.update(overrides)
    return record


def test_gold_signature_supports_every_edit_operation():
    edits = [
        {"op": "insert", "pos": 2, "content": [10, 11]},
        {"op": "delete", "start": 4, "end": 6},
        {"op": "replace", "start": 7, "end": 8, "content": [12]},
        {"op": "copy", "start": 9, "end": 10, "dest": 3},
        {"op": "move", "start": 11, "end": 13, "dest": 5},
    ]
    assert _gold_signature(edits) == (
        ("insert", 2, (10, 11)),
        ("delete", 4, 6),
        ("replace", 7, 8, (12,)),
        ("copy", 9, 10, 3),
        ("move", 11, 13, 5),
    )


def test_summary_keeps_invalid_programs_in_unconditional_denominator():
    records = [
        _record(),
        _record(
            status="invalid_edit_end_count_0",
            valid=False,
            char_similarity_delta=0.0,
            token_similarity_delta=0.0,
            likelihood_delta=0.0,
            likelihood_improved=False,
            target_similarity_improved=False,
            generation_seconds=0.2,
            apply_seconds=0.0,
        ),
    ]
    summary = summarise_records(records, bootstrap_samples=0, bootstrap_seed=123)["overall"]
    assert summary["examples"] == 2
    assert summary["valid_program_rate"] == 0.5
    assert summary["mean_char_similarity_delta"] == 0.05
    assert summary["mean_likelihood_delta"] == 0.15
    assert summary["mean_likelihood_improved"] == 0.5
    assert summary["valid_target_similarity_improvement_rate"] == 1.0
    assert summary["status_counts"] == {"invalid_edit_end_count_0": 1, "valid": 1}


def test_bootstrap_summary_is_deterministic():
    records = [_record(char_similarity_delta=value) for value in (-0.2, 0.0, 0.1, 0.5)]
    first = summarise_records(records, bootstrap_samples=100, bootstrap_seed=99)
    second = summarise_records(records, bootstrap_samples=100, bootstrap_seed=99)
    assert first == second


def test_stratified_selection_is_exact_balanced_and_deterministic():
    rows = [
        {"stratum": f"s{stratum}", "row": row}
        for stratum in range(20)
        for row in range(50)
    ]
    first = select_benchmark_rows(
        rows,
        limit=250,
        selection="stratified",
        seed=123,
    )
    second = select_benchmark_rows(
        rows,
        limit=250,
        selection="stratified",
        seed=123,
    )

    counts = {}
    for row in first:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
    assert first == second
    assert len(first) == 250
    assert len(counts) == 20
    assert set(counts.values()) == {12, 13}


def test_head_selection_preserves_legacy_prefix_behavior():
    rows = [{"stratum": "s", "row": row} for row in range(5)]
    assert select_benchmark_rows(
        rows,
        limit=2,
        selection="head",
        seed=1,
    ) == rows[:2]
