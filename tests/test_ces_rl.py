import pathlib
import sys

import pytest
import torch
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nanochat.ces_rl import (
    add_likelihood_auxiliary,
    group_advantages,
    operation_local_patterns,
    local_repair_verifier_reward,
    provenance_program_reward,
    target_aware_recovery_reward,
    target_blind_likelihood_reward,
)
from nanochat.engine_extended import selected_action_logprobs
from nanochat.causal_serialisation_extended import (
    Insert,
    SequenceEdit,
    TokensOperations,
)
from nanochat.rl_pair_dataloader import StreamingEditPair
from nanochat.rl_pair_dataloader import StreamingEditPairSampler
from nanochat.tokenizer import get_extended_indel_tokeniser
from ces.workflows.ces_rl import (
    clean_token_kl_loss,
    clean_ar_completion_targets,
    cyclic_component_rows,
    cyclic_prompt_rows,
    distributed_cyclic_component_rows,
    first_decision_loss,
    first_decision_pair_weights,
    human_identity_rows,
    joint_objective_weights,
    local_rollout_group_size,
    materialize_missing_gradients,
    oversample_controlled_rows,
    reject_heldout_rows,
    stream_pair_to_row,
)
from ces.workflows.ces_sft import matched_sft_loss


def test_training_split_guard_rejects_validation_rows():
    with pytest.raises(ValueError, match="held-out"):
        reject_heldout_rows([
            {
                "benchmark_id": "controlled:1",
                "split": "validation",
            }
        ])


def test_training_split_guard_accepts_finetune_train_rows():
    reject_heldout_rows([
        {
            "benchmark_id": "coedit:1",
            "role": "finetune_train",
            "source_split": "train",
        }
    ])


def test_centered_advantages_are_not_variance_standardized():
    rewards = torch.tensor([-2.0, 0.0, 4.0])
    advantages = group_advantages(rewards, method="centered")
    assert advantages.tolist() == pytest.approx([-8 / 3, -2 / 3, 10 / 3])


def test_clean_ar_replay_masks_prefix_and_predicts_suffix():
    inputs, positions, labels = clean_ar_completion_targets(
        [10, 11, 12, 13, 14, 15], prefix_fraction=0.5
    )
    assert inputs == [10, 11, 12, 13, 14]
    assert positions == [0, 1, 2, 3, 4]
    assert labels == [-1, -1, 13, 14, 15]


def test_clean_ar_zero_prefix_trains_every_next_token():
    _inputs, _positions, labels = clean_ar_completion_targets(
        [10, 11, 12, 13], prefix_fraction=0.0
    )
    assert labels == [11, 12, 13]


def test_cyclic_component_rows_averaging_schedule_wraps():
    rows = [{"id": index} for index in range(5)]
    assert [row["id"] for row in cyclic_component_rows(rows, step=0, count=4)] == [0, 1, 2, 3]
    assert [row["id"] for row in cyclic_component_rows(rows, step=1, count=4)] == [4, 0, 1, 2]


def test_cyclic_prompt_rows_increases_independent_prompt_coverage():
    rows = [{"id": index} for index in range(10)]
    assert [row["id"] for row in cyclic_prompt_rows(
        rows, step=0, prompts_per_step=4
    )] == [0, 1, 2, 3]
    assert [row["id"] for row in cyclic_prompt_rows(
        rows, step=2, prompts_per_step=4
    )] == [8, 9, 0, 1]


def test_stream_pair_adapter_builds_existing_rl_row_contract():
    tokenizer = get_extended_indel_tokeniser()
    token_ops = TokensOperations(tokenizer.enc)
    source = [tokenizer.get_bos_token_id(), *tokenizer.encode("alpha omega")]
    inserted = tokenizer.encode(" beta")
    sequence_edit = SequenceEdit(
        sequence=np.asarray(source, dtype=np.int64),
        edits=[Insert(2, np.asarray(inserted, dtype=np.int64))],
    )
    row = stream_pair_to_row(
        StreamingEditPair(
            example_id="stream:test",
            kind="gpt",
            source_shard_id="shard_00000",
            source_row_index=3,
            epoch=1,
            sequence_edit=sequence_edit,
        ),
        token_ops,
    )

    assert row["benchmark_id"] == "stream:test"
    assert row["stream_kind"] == "gpt"
    assert row["prompt_tokens"][-1] == token_ops.edit_start_token
    assert row["gold_program_tokens"][-1] == token_ops.edit_end_token
    assert row["gold_op_counts"] == {"insert": 1}
    assert row["source_tokens"] != row["target_tokens"]


def test_stream_pair_batch_uses_exact_largest_remainder_mixture():
    class FakeSampler:
        probabilities = np.asarray([0.5, 0.3, 0.2])
        rng = np.random.default_rng(42)

        def next(self, *, kind=None):
            return kind

    sampler = FakeSampler()
    batch = StreamingEditPairSampler.next_batch(sampler, 32)

    assert len(batch) == 32
    assert batch.count("gpt") == 16
    assert batch.count("random") == 10
    assert batch.count("noop") == 6


def test_distributed_component_rows_are_disjoint_global_slice():
    rows = [{"id": index} for index in range(10)]
    rank0 = distributed_cyclic_component_rows(
        rows, step=1, global_count=4, rank=0, world_size=2
    )
    rank1 = distributed_cyclic_component_rows(
        rows, step=1, global_count=4, rank=1, world_size=2
    )
    assert [row["id"] for row in rank0] == [4, 5]
    assert [row["id"] for row in rank1] == [6, 7]


def test_illegal_pointer_logit_has_zero_policy_gradient():
    logits = torch.tensor([1.0, 0.0, 9.0], requires_grad=True)
    masked = torch.full_like(logits, float("-inf"))
    masked[[0, 1]] = logits[[0, 1]]
    logprob = selected_action_logprobs(
        masked[None, :], torch.tensor([[0]]), temperature=0.5
    )[0, 0]
    logprob.backward()

    assert float(logits.grad[0]) > 0
    assert float(logits.grad[1]) < 0
    assert float(logits.grad[2]) == 0


def test_missing_policy_gradients_are_materialized_as_zeros():
    model = torch.nn.Linear(2, 2)
    model.weight.sum().backward()
    assert model.bias.grad is None

    assert materialize_missing_gradients(model) == 1
    assert torch.equal(model.bias.grad, torch.zeros_like(model.bias))


def test_global_rollout_group_splits_evenly_across_workers():
    assert local_rollout_group_size(8, 2) == 4
    assert local_rollout_group_size(8, 1) == 8
    with pytest.raises(ValueError, match="must be divisible"):
        local_rollout_group_size(7, 2)


def test_target_blind_likelihood_reward_scales_clips_and_penalizes_edits(monkeypatch):
    class TokenOps:
        @staticmethod
        def resolve(_serialised):
            return [1, 2, 4]

    rollout = type("Rollout", (), {"tokens": [1], "pointers": [0]})()
    reward = target_blind_likelihood_reward(
        rollout,
        TokenOps(),
        [1, 2, 3],
        source_avg_logprob=-2.0,
        candidate_avg_logprob=-1.5,
        likelihood_scale=0.25,
        edit_penalty=0.3,
    )
    assert reward.valid
    assert reward.likelihood_delta == pytest.approx(0.5)
    assert reward.reward == pytest.approx(0.9)


def test_hybrid_recovery_reward_blends_token_character_and_local_credit():
    class Tokenizer:
        @staticmethod
        def decode(tokens):
            return "".join(chr(96 + token) for token in tokens)

    class TokenOps:
        tokeniser = Tokenizer()

        @staticmethod
        def resolve(_serialised):
            return [1, 2, 9, 3, 4]

    rollout = type("Rollout", (), {"tokens": [1], "pointers": [0]})()
    reward = target_aware_recovery_reward(
        rollout,
        TokenOps(),
        [1, 2, 3, 4],
        [1, 2, 9, 3, 4],
        exact_bonus=0.0,
        character_reward_weight=0.25,
        operation_local_reward_weight=0.25,
        gold_edits=[{"op": "insert", "pos": 2, "content": [9]}],
        local_context_tokens=2,
    )
    assert reward.reward == pytest.approx(1.0)
    assert reward.token_improvement == pytest.approx(1.0)
    assert reward.character_improvement == pytest.approx(1.0)
    assert reward.operation_local_improvement == pytest.approx(1.0)


def test_revised_reward_penalizes_corrupt_noop_and_scales_valid_harm():
    rollout = type("Rollout", (), {"tokens": [1], "pointers": [0]})()
    noop_ops = type(
        "NoopOps", (), {"resolve": staticmethod(lambda _data: [1, 2, 3])}
    )()
    noop = target_aware_recovery_reward(
        rollout,
        noop_ops,
        [1, 2, 3],
        [1, 2, 4],
        exact_bonus=0.0,
        corrupt_noop_penalty=0.1,
        negative_reward_scale=2.0,
    )
    assert noop.reward == pytest.approx(-0.2)

    harmful_ops = type(
        "HarmfulOps", (), {"resolve": staticmethod(lambda _data: [9, 9, 9])}
    )()
    harmful = target_aware_recovery_reward(
        rollout,
        harmful_ops,
        [1, 2, 3],
        [1, 2, 4],
        exact_bonus=0.0,
        negative_reward_scale=1.5,
    )
    assert harmful.reward == pytest.approx(-1.5)


def test_provenance_reward_matches_exact_operations_without_distance_shaping():
    rollout = type("Rollout", (), {"tokens": [1], "pointers": [0]})()
    parsed = type("Parsed", (), {"edits": [Insert(2, [9])]})()
    token_ops = type(
        "TokenOps",
        (),
        {
            "deserialise": staticmethod(lambda _data: parsed),
            "resolve": staticmethod(lambda _data: [1, 2, 9, 3]),
        },
    )()

    reward = provenance_program_reward(
        rollout,
        token_ops,
        [1, 2, 3],
        [1, 2, 9, 3],
        gold_edits=[{"op": "insert", "pos": 2, "content": [9]}],
        exact_bonus=0.25,
    )

    assert reward.reward == pytest.approx(1.25)
    assert reward.provenance_precision == pytest.approx(1.0)
    assert reward.provenance_recall == pytest.approx(1.0)
    assert reward.source_distance == 0
    assert reward.candidate_distance is None
    assert reward.edit_distance is None


def test_provenance_reward_penalizes_non_provenance_operation():
    rollout = type("Rollout", (), {"tokens": [1], "pointers": [0]})()
    parsed = type("Parsed", (), {"edits": [Insert(1, [8])]})()
    token_ops = type(
        "TokenOps",
        (),
        {
            "deserialise": staticmethod(lambda _data: parsed),
            "resolve": staticmethod(lambda _data: [1, 8, 2, 3]),
        },
    )()

    reward = provenance_program_reward(
        rollout,
        token_ops,
        [1, 2, 3],
        [1, 2, 9, 3],
        gold_edits=[{"op": "insert", "pos": 2, "content": [9]}],
        exact_bonus=0.25,
        extra_op_penalty=0.25,
        negative_reward_scale=2.0,
    )

    assert reward.reward == pytest.approx(-0.5)
    assert reward.provenance_precision == 0.0
    assert reward.provenance_recall == 0.0
    assert reward.unmatched_program_ops == 1


def test_local_verifier_rewards_exact_local_phrase_without_distance():
    class Tokenizer:
        @staticmethod
        def decode(tokens):
            return "".join(chr(token) for token in tokens)

    class TokenOps:
        tokeniser = Tokenizer()

        @staticmethod
        def resolve(_data):
            return [ord(c) for c in "She goes home."]

    rollout = type("Rollout", (), {"tokens": [1], "pointers": [0]})()
    reward = local_repair_verifier_reward(
        rollout,
        TokenOps(),
        [ord(c) for c in "She go home."],
        [ord(c) for c in "She goes home."],
        exact_bonus=0.25,
        anchor_chars=3,
    )
    assert reward.reward == pytest.approx(1.25)
    assert reward.verifier_pass_rate == pytest.approx(1.0)
    assert reward.outside_edit_fraction == pytest.approx(0.0)


def test_first_decision_loss_marginalizes_gec_edit_openers():
    class TokenOps:
        edit_start_token = 1
        edit_end_token = 2
        deletion_open_token = 3
        insertion_open_token = 4
        replace_open_token = 5

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.next_logits = torch.nn.Parameter(torch.zeros(8))

        def forward(self, inputs, *, tix):
            del tix
            return self.next_logits[None, None, :].expand(1, inputs.shape[1], -1)

    model = ToyModel()
    with torch.no_grad():
        model.next_logits[2] = 4.0
    noop_loss = first_decision_loss(
        model,
        TokenOps(),
        [0, 1],
        [0, 1],
        should_edit=False,
        device=torch.device("cpu"),
    )
    edit_loss = first_decision_loss(
        model,
        TokenOps(),
        [0, 1],
        [0, 1],
        should_edit=True,
        device=torch.device("cpu"),
    )
    assert noop_loss < edit_loss


def test_first_decision_pair_weights_preserve_pair_mean():
    assert first_decision_pair_weights(0.5) == pytest.approx((1.0, 1.0))
    assert first_decision_pair_weights(0.35) == pytest.approx((0.7, 1.3))
    with pytest.raises(ValueError, match="edit_prior"):
        first_decision_pair_weights(-0.1)


def test_operation_local_patterns_include_changed_content_and_boundaries():
    patterns = operation_local_patterns(
        [1, 2, 3, 4, 5],
        [1, 2, 9, 3, 5],
        [
            {"op": "insert", "pos": 2, "content": [9]},
            {"op": "delete", "start": 3, "end": 4},
        ],
        context_tokens=2,
    )
    assert patterns == ((1, 2, 9, 3, 4), (2, 3, 5))


def test_likelihood_auxiliary_preserves_target_diagnostics():
    target = target_blind_likelihood_reward(
        type("Rollout", (), {"tokens": [1], "pointers": [0]})(),
        type("TokenOps", (), {"resolve": staticmethod(lambda _data: [1, 2, 4])})(),
        [1, 2, 3],
        source_avg_logprob=-2.0,
        candidate_avg_logprob=-1.75,
    )
    target = target.__class__(**{**target.__dict__, "reward": 0.4, "exact_target": True})
    likelihood = target_blind_likelihood_reward(
        type("Rollout", (), {"tokens": [1], "pointers": [0]})(),
        type("TokenOps", (), {"resolve": staticmethod(lambda _data: [1, 2, 4])})(),
        [1, 2, 3],
        source_avg_logprob=-2.0,
        candidate_avg_logprob=-1.75,
        likelihood_scale=0.25,
    )
    combined = add_likelihood_auxiliary(target, likelihood, weight=0.05)
    assert combined.reward == pytest.approx(0.45)
    assert combined.exact_target
    assert combined.likelihood_delta == pytest.approx(0.25)


def test_clean_token_kl_is_zero_for_identical_models():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 8)

        def forward(self, inputs, *, tix):
            del tix
            return self.embedding(inputs)

    model = ToyModel()
    reference = ToyModel()
    reference.load_state_dict(model.state_dict())
    loss = clean_token_kl_loss(
        model,
        reference,
        [1, 2, 3, 4],
        device=torch.device("cpu"),
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_controlled_operation_oversampling_repeats_only_selected_modes():
    rows = [
        {"benchmark_id": "a", "repair_op_mode": "insert"},
        {"benchmark_id": "b", "repair_op_mode": "delete"},
        {"benchmark_id": "c", "repair_op_mode": "move"},
    ]
    sampled = oversample_controlled_rows(
        rows, modes=["insert", "move"], factor=3
    )
    assert [row["benchmark_id"] for row in sampled] == [
        "a", "a", "a", "b", "c", "c", "c"
    ]


def test_joint_objective_weights_match_named_five_way_mixture():
    weights = joint_objective_weights(
        supervised_replay_fraction=0.2,
        ces_replay_share=0.5,
        human_supervised_fraction=0.2,
        noop_supervised_fraction=0.1,
        retention_mode="paired_mixed",
        first_decision_fraction=0.05,
    )
    assert weights == pytest.approx({
        "rl": 0.45,
        "synthetic_ces": 0.1,
        "retention": 0.1,
        "human_ces": 0.2,
        "noop_ces": 0.1,
        "first_decision": 0.05,
    })


def test_joint_objective_rejects_more_than_unit_mass():
    with pytest.raises(ValueError, match="exceed one"):
        joint_objective_weights(
            supervised_replay_fraction=0.5,
            ces_replay_share=0.5,
            human_supervised_fraction=0.4,
            noop_supervised_fraction=0.2,
            retention_mode="paired_mixed",
            first_decision_fraction=0.1,
        )


def test_human_identity_rows_train_immediate_noop_on_the_clean_target():
    rows = human_identity_rows([
        {
            "benchmark_id": "human:1",
            "source_text": "bad sentence",
            "target_text": "Good sentence.",
        }
    ])
    assert rows[0]["source_text"] == "Good sentence."
    assert rows[0]["target_text"] == "Good sentence."
    assert rows[0]["role"] == "joint_noop_training"


def test_matched_sft_loss_preserves_clean_replay_weight():
    assert matched_sft_loss(
        torch.tensor(2.0),
        torch.tensor(6.0),
        clean_replay_fraction=0.125,
    ).item() == pytest.approx(2.5)


def test_matched_sft_loss_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="clean_replay_fraction"):
        matched_sft_loss(
            torch.tensor(1.0),
            torch.tensor(1.0),
            clean_replay_fraction=1.1,
        )
