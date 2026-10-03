import numpy as np

from nanochat.text_edit_baselines import (
    BaselineExample,
    build_baseline_example,
    build_predicted_fim_infill_prompt,
    build_predicted_fim_locator,
    masked_baseline_training_data,
    minimal_changed_envelope,
)


class FakeTokenizer:
    def get_bos_token_id(self):
        return 1

    def encode_special(self, text):
        assert text == "<|assistant_end|>"
        return 2

    def encode(self, text):
        return [100 + (ord(char) % 23) for char in text]


def test_minimal_changed_envelope_contains_multiple_changes():
    source = [1, 10, 11, 12, 13, 14]
    target = [1, 10, 21, 12, 13, 24]
    assert minimal_changed_envelope(source, target) == (2, 6, 2, 6)


def test_minimal_changed_envelope_anchors_repeated_leading_bos():
    # A global SequenceMatcher alignment pairs the first target BOS with the
    # second source token here, incorrectly producing a source span at zero.
    source = [1, 0, 0]
    target = [1, 1, 0]
    assert minimal_changed_envelope(source, target) == (1, 2, 1, 2)


def test_minimal_changed_envelope_handles_insertion_after_shared_prefix():
    source = [1, 10, 20]
    target = [1, 10, 99, 20]
    assert minimal_changed_envelope(source, target) == (2, 2, 2, 3)


def test_oracle_fim_reconstructs_target_from_gold_middle():
    tokenizer = FakeTokenizer()
    source = [1, 10, 11, 12, 13]
    target = [1, 10, 21, 22, 13]
    example = build_baseline_example("fim", source, target, tokenizer)
    reconstructed = [*example.source_prefix, *example.gold_middle, *example.source_suffix]
    assert reconstructed == target
    assert example.completion_tokens == [21, 22, 2]


def test_predicted_fim_combines_target_blind_locator_and_gold_training_infill():
    tokenizer = FakeTokenizer()
    source = [1, 10, 11, 12, 13]
    target = [1, 10, 21, 22, 13]

    locator, query_indices = build_predicted_fim_locator(source, tokenizer)
    example = build_baseline_example("fim_predicted", source, target, tokenizer)

    assert locator == example.prompt_tokens[:len(locator)]
    assert example.position_query_indices == query_indices
    assert example.position_targets == (1, 3)
    assert example.source_span == (2, 4)
    assert [*example.source_prefix, *example.gold_middle, *example.source_suffix] == target


def test_predicted_fim_infill_prompt_uses_predicted_slice_not_target_tokens():
    tokenizer = FakeTokenizer()
    source = [1, 10, 11, 12, 13]
    appended, prefix, suffix = build_predicted_fim_infill_prompt(
        source, 2, 4, tokenizer
    )

    assert prefix == [1, 10]
    assert suffix == [13]
    assert 11 not in prefix + suffix
    assert 12 not in prefix + suffix
    assert appended


def test_predicted_fim_training_scores_two_pointer_queries():
    tokenizer = FakeTokenizer()
    example = build_baseline_example(
        "fim_predicted", [1, 10, 11, 12, 13], [1, 10, 21, 22, 13], tokenizer
    )
    training = masked_baseline_training_data(
        example, token_budget=128, pos_buffer_size=2
    )

    assert training is not None
    assert training.target.gathered_pos_target.tolist() == [1, 3]
    assert training.target.gather_indices.tolist() == list(example.position_query_indices)
    assert training.target.gather_valid.tolist() == [True, True]



def test_regeneration_completion_is_complete_target_without_second_bos():
    tokenizer = FakeTokenizer()
    example = build_baseline_example("regeneration", [1, 8, 9], [1, 8, 7], tokenizer)
    assert example.completion_tokens == [8, 7, 2]
    assert example.gold_middle == [8, 7]


def test_masked_training_data_scores_only_completion():
    example = BaselineExample(
        prompt_tokens=[1, 4, 5],
        completion_tokens=[6, 2],
        source_prefix=[1],
        source_suffix=[],
        gold_middle=[6],
    )
    training = masked_baseline_training_data(example, token_budget=8, pos_buffer_size=4)
    assert training is not None
    assert training.input_tokens.tolist()[:4] == [1, 4, 5, 6]
    assert training.target.tokens.tolist()[:4] == [4, 5, 6, 2]
    assert np.flatnonzero(training.target.token_loss_weights).tolist() == [2, 3]
    assert training.target.tokens.tolist()[4:] == [-1, -1, -1, -1]


def test_automatic_terminal_weight_matches_content_length():
    example = BaselineExample(
        prompt_tokens=[1, 4],
        completion_tokens=[6, 7, 8, 2],
        source_prefix=[1],
        source_suffix=[],
        gold_middle=[6, 7, 8],
    )
    training = masked_baseline_training_data(
        example,
        token_budget=8,
        pos_buffer_size=4,
        terminal_weight=-1,
    )
    assert training is not None
    assert training.target.token_loss_weights.tolist()[:5] == [0.0, 1.0, 1.0, 1.0, 3.0]
