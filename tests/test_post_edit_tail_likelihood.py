import random

import numpy as np

from nanochat.causal_serialisation_extended import TokensOperations, resolve
from ces.workflows.post_edit_tail_likelihood import (
    CES_CONDITIONS,
    build_condition_prefixes,
    make_corruption,
    summarise_records,
)


class TinyEncoding:
    def __init__(self):
        self._special_tokens = {
            "<|bos|>": 100,
            "<|assistant_end|>": 101,
            "<|del-open|>": 102,
            "<|del-start|>": 103,
            "<|del-end|>": 104,
            "<|ins-open|>": 105,
            "<|ins-loc|>": 106,
            "<|replace-open|>": 107,
            "<|replace-start|>": 108,
            "<|replace-end|>": 109,
            "<|replace-eos|>": 110,
            "<|copy-open|>": 111,
            "<|copy-start|>": 112,
            "<|copy-end|>": 113,
            "<|copy-dest|>": 114,
            "<|move-open|>": 115,
            "<|move-start|>": 116,
            "<|move-end|>": 117,
            "<|move-dest|>": 118,
            "<|edit-start|>": 119,
            "<|edit-end|>": 120,
            "<|indel-bos|>": 121,
        }
        self.max_token_value = 121


def _clean_prefix():
    return [100, *list(range(1, 41))]


def test_all_corruption_families_have_gold_and_noncorrecting_sham_programs():
    clean = _clean_prefix()
    for operation in ("replace", "delete", "insert"):
        made = make_corruption(
            clean,
            operation=operation,
            span_length=4,
            edit_distance=8,
            rng=random.Random(13),
        )
        assert made["gold_target"] == clean
        assert made["sham_target"] != clean
        assert made["sham_target"] != made["corrupt_prefix"]


def test_condition_programs_roundtrip_to_expected_prefixes():
    clean = _clean_prefix()
    made = make_corruption(
        clean,
        operation="replace",
        span_length=4,
        edit_distance=8,
        rng=random.Random(7),
    )
    case = {
        "clean_prefix": clean,
        "corrupt_prefix": made["corrupt_prefix"],
        "gold_edit": made["gold_edit"],
        "sham_edit": made["sham_edit"],
    }
    token_ops = TokensOperations(TinyEncoding())
    prefixes = build_condition_prefixes(case, token_ops, role="ces")
    assert tuple(prefixes) == CES_CONDITIONS
    for name in ("clean_noop_program", "corrupt_noop_program", "corrupt_sham_program", "corrupt_gold_program"):
        tokens, positions = prefixes[name]
        assert len(tokens) == len(positions)
        parsed = token_ops.deserialise(
            type("Serial", (), {
                "tokens": np.asarray(tokens),
                "positions": np.asarray(positions),
            })()
        )
        resolved = [int(token) for token in resolve(parsed).tolist()]
        if name == "clean_noop_program":
            assert resolved == clean
        elif name == "corrupt_noop_program":
            assert resolved == made["corrupt_prefix"]
        elif name == "corrupt_gold_program":
            assert resolved == clean
        else:
            assert resolved == made["sham_target"]


def _condition(nll):
    return {
        "tail_nll": float(nll),
        "tail_tokens": 2,
        "tail_bytes": 2,
        "tail_token_nll": [float(nll) / 2, float(nll) / 2],
    }


def test_summary_uses_dataset_level_recovery_ratio():
    records = []
    for clean, corrupt, sham, repair in ((2, 6, 7, 3), (2, 10, 9, 5)):
        records.append({
            "operation": "replace",
            "span_length": 1,
            "edit_distance": 16,
            "conditions": {
                "clean_plain": _condition(clean),
                "corrupt_plain": _condition(corrupt),
                "clean_noop_program": _condition(clean),
                "corrupt_noop_program": _condition(corrupt),
                "corrupt_sham_program": _condition(sham),
                "corrupt_gold_program": _condition(repair),
            },
        })
    overall = summarise_records(records, role="ces")["overall"]
    # Aggregate mean NLLs are clean=1, corrupt=4, repair=2 tokens^-1.
    assert overall["corruption_excess_mean_nll"] == 3.0
    assert overall["repair_mean_nll_gain_vs_corrupt_noop"] == 2.0
    assert overall["aggregate_recovery_fraction"] == 2.0 / 3.0
