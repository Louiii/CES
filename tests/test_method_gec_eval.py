from ces.workflows.method_gec_eval import _summary


def test_summary_reports_validity_change_and_target_blind_confidence():
    records = [
        {
            "valid": True,
            "source_text": "A sentence.",
            "prediction_text": "A better sentence.",
            "exact_target": True,
            "edit_confidence": -0.2,
        },
        {
            "valid": False,
            "source_text": "No change.",
            "prediction_text": "No change.",
            "exact_target": False,
            "edit_confidence": None,
        },
    ]

    summary = _summary(records, {"method": "test"})

    assert summary["examples"] == 2
    assert summary["validity_rate"] == 0.5
    assert summary["change_rate"] == 0.5
    assert summary["exact_target_rate"] == 0.5
    assert summary["records_with_confidence"] == 1
    assert summary["confidence_field"] == "edit_confidence"
    assert summary["confidence_is_target_blind"] is True


def test_oracle_span_fim_summary_discloses_target_derived_boundary():
    records = [
        {
            "valid": True,
            "source_text": "A sentence.",
            "prediction_text": "A better sentence.",
            "exact_target": True,
            "edit_confidence": -0.2,
        }
    ]

    summary = _summary(records, {"method": "fim", "oracle_span": True})

    assert summary["confidence_is_target_blind"] is False
