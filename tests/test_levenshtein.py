import torch

from nanochat.levenshtein import (
    LevenshteinConfig,
    LevenshteinTransformer,
    build_oracle_round,
    build_round_example,
    collate_rounds,
)


def tiny_config(**overrides):
    values = dict(
        sequence_len=128,
        vocab_size=64,
        pad_id=60,
        mask_id=61,
        bos_id=62,
        eos_id=63,
        n_layer=2,
        n_head=2,
        n_embd=32,
        max_insertions=32,
        max_rounds=4,
    )
    values.update(overrides)
    return LevenshteinConfig(**values)


def test_replace_is_rewritten_as_delete_plus_insert():
    config = tiny_config()
    oracle = build_round_example([1, 2, 3], [1, 4, 3], round_index=0, config=config)

    assert oracle.current == [config.bos_id, 1, 2, 3, config.eos_id]
    assert oracle.delete_labels == [0, 0, 1, 0, 0]
    assert oracle.kept == [config.bos_id, 1, 3, config.eos_id]
    assert oracle.gap_labels == [0, 1, 0]
    assert oracle.completed == [config.bos_id, 1, 4, 3, config.eos_id]
    assert oracle.masked == [config.bos_id, 1, config.mask_id, 3, config.eos_id]


def test_large_gap_is_completed_over_multiple_rounds():
    config = tiny_config()
    source = [1, 2]
    target = [1, *range(10, 80), 2]

    first = build_round_example(source, target, round_index=0, config=config)
    second = build_round_example(source, target, round_index=1, config=config)
    third = build_round_example(source, target, round_index=2, config=config)

    assert max(first.gap_labels) == 32
    assert max(second.gap_labels) == 32
    assert max(third.gap_labels) == 6
    assert third.completed == [config.bos_id, *target, config.eos_id]


def test_identity_batch_has_finite_three_pass_loss_and_gradients():
    config = tiny_config()
    oracle = build_round_example([1, 2, 3], [1, 2, 3], round_index=0, config=config)
    batch = collate_rounds([oracle], pad_id=config.pad_id, device="cpu")
    model = LevenshteinTransformer(config)
    model.init_weights()

    loss, metrics = model.compute_loss(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["encoder_passes"].item() == 3
    assert metrics["token_count"].item() == 0
    assert model.delete_head.weight.grad is not None
    assert model.delete_head.bias.grad is not None
    assert model.gap_head.bias.grad is not None
    assert model.gap_head.weight.grad is not None


def test_gap_head_pads_optimizer_rows_without_exposing_an_extra_class():
    config = tiny_config()
    model = LevenshteinTransformer(config)
    model.init_weights()
    hidden = torch.randn(2, 5, config.n_embd)

    logits = model.gap_logits(hidden)
    logits.sum().backward()

    assert logits.shape == (2, 4, 33)
    assert model.gap_head.weight.shape[0] == 34
    assert model.gap_head.bias.shape[0] == 34
    assert model.gap_head.weight.grad.shape[0] % 2 == 0
    assert torch.count_nonzero(model.gap_head.weight.grad[-1]) == 0
    assert model.gap_head.bias.grad[-1] == 0



def test_collate_pads_each_of_the_three_passes_independently():
    config = tiny_config()
    rounds = [
        build_round_example([1, 2, 3], [1, 4, 3], round_index=0, config=config),
        build_round_example([5], [5, 6, 7], round_index=0, config=config),
    ]
    batch = collate_rounds(rounds, pad_id=config.pad_id, device="cpu")

    assert batch.current_ids.shape == (2, 5)
    assert batch.kept_ids.shape[0] == 2
    assert batch.gap_labels.shape[1] == batch.kept_ids.shape[1] - 1
    assert batch.masked_ids.shape == batch.token_labels.shape
    assert not batch.current_padding_mask[1, -1]


def test_oracle_round_handles_pure_insert_and_delete():
    config = tiny_config()
    source = [config.bos_id, 1, 2, 3, config.eos_id]
    target = [config.bos_id, 1, 8, 9, 3, config.eos_id]
    oracle = build_oracle_round(
        source,
        target,
        mask_id=config.mask_id,
        max_insertions=config.max_insertions,
    )
    assert oracle.completed == target
    assert sum(oracle.delete_labels) == 1
    assert sum(oracle.gap_labels) == 2


def test_iterative_decode_reports_target_blind_greedy_confidence():
    config = tiny_config(sequence_len=32, max_rounds=1)
    model = LevenshteinTransformer(config)
    model.init_weights()

    prediction, stats = model.iterative_decode([1, 2, 3])

    assert isinstance(prediction, list)
    assert stats["decision_count"] > 0
    assert stats["mean_decision_logprob"] is not None
    assert torch.isfinite(torch.tensor(stats["mean_decision_logprob"]))
    assert stats["mean_decision_logprob"] <= 0
