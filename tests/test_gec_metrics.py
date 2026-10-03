from __future__ import annotations

import pytest

from nanochat.gec_metrics import (
    ErrantCounts,
    load_errant_annotator,
    risk_coverage_curve,
    score_errant_example,
    summarise_scored_examples,
)


@pytest.fixture(scope="module")
def annotator():
    return load_errant_annotator()


def test_errant_known_partial_correction(annotator) -> None:
    scored = score_errant_example(
        annotator=annotator,
        source_text="This are gramamtical sentence.",
        target_text="This is a grammatical sentence.",
        prediction_text="This is grammatical sentence.",
    )
    assert scored["counts"] == ErrantCounts(tp=2, fp=0, fn=1)
    assert scored["target_distance_direction"] == "improves"


def test_invalid_or_missing_prediction_is_source_noop(annotator) -> None:
    scored = score_errant_example(
        annotator=annotator,
        source_text="This are wrong.",
        target_text="This is wrong.",
        prediction_text=None,
    )
    assert scored["counts"] == ErrantCounts(tp=0, fp=0, fn=1)
    assert scored["changed"] is False
    assert scored["target_distance_direction"] == "same"


def test_corpus_f05_and_risk_coverage(annotator) -> None:
    examples = [
        score_errant_example(
            annotator=annotator,
            source_text="This are wrong.",
            target_text="This is wrong.",
            prediction_text="This is wrong.",
        ),
        score_errant_example(
            annotator=annotator,
            source_text="They is ready.",
            target_text="They are ready.",
            prediction_text="They was ready.",
        ),
    ]
    rows = [
        {**row, "_errant_counts": row["counts"], "confidence": confidence}
        for row, confidence in zip(examples, (0.9, 0.1))
    ]
    summary = summarise_scored_examples(examples, beta=0.5)
    assert summary["tp"] == 1
    assert summary["fp"] == 1
    assert summary["fn"] == 1
    assert summary["precision"] == pytest.approx(0.5)
    assert summary["recall"] == pytest.approx(0.5)
    assert summary["f0.5"] == pytest.approx(0.5)

    curve = risk_coverage_curve(rows, confidence_field="confidence", beta=0.5)
    assert curve[0]["coverage"] == 0
    assert curve[1]["coverage"] == pytest.approx(0.5)
    # The rejected second example is an unchanged-source false negative.
    assert curve[1]["f0.5"] == pytest.approx(5 / 6)
    assert curve[-1]["coverage"] == 1.0
    assert curve[-1]["f0.5"] == pytest.approx(0.5)
