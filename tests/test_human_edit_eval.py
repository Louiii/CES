import json
import torch

from ces.workflows.human_edit_eval import (
    _confidence_summary,
    _load_manifest_examples,
    _resolved_change_features,
    _resolved_change_signature,
    _token_edit_from_pair,
)


def test_confidence_summary_combines_token_and_pointer_decisions():
    result = _confidence_summary(
        (torch.tensor(-1.0), torch.tensor(-3.0)),
        (torch.tensor(-2.0),),
    )
    assert result["program_mean_logprob"] == -2.0
    assert result["program_min_logprob"] == -3.0
    assert result["program_token_mean_logprob"] == -2.0
    assert result["program_pointer_mean_logprob"] == -2.0
    assert result["program_decision_count"] == 3


def test_resolved_change_signature_identifies_single_primitive_types():
    assert _resolved_change_signature("abc", "abXc") == ("insert",)
    assert _resolved_change_signature("abc", "ac") == ("delete",)
    assert _resolved_change_signature("abc", "aXc") == ("replace",)
    assert _resolved_change_features("abcdef", "abf") == (("delete",), 3)
    assert _resolved_change_features("abc", "aXc") == (("replace",), 2)


def test_resolved_change_signature_rejects_multi_change_shape():
    assert _resolved_change_signature("abc def", "aXc deY") == ("replace", "replace")

class _TinyTokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, text):
        return [ord(char) for char in text]


def test_token_edit_from_pair_remains_callable():
    edit = _token_edit_from_pair(_TinyTokenizer(), "This are.", "This is.")
    assert edit is not None
    assert edit.edits


def test_token_edit_from_identity_pair_builds_noop_program_source():
    edit = _token_edit_from_pair(_TinyTokenizer(), "Already clean.", "Already clean.")
    assert edit is not None
    assert edit.edits == []


def test_load_frozen_manifest(tmp_path):
    path = tmp_path / "development.jsonl"
    path.write_text(
        json.dumps(
            {
                "benchmark_id": "coedit_gec_v1:development:abc",
                "role": "development",
                "dataset": "coedit",
                "task": "gec",
                "source_text": "This are wrong.",
                "target_text": "This is wrong.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = _load_manifest_examples(path, limit=None)
    assert len(examples) == 1
    assert examples[0].split == "development"
    assert examples[0].example_id == "coedit_gec_v1:development:abc"
