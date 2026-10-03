"""Target-aware on-policy RL post-training for Causal Edit System models.

The trainer consumes paired source/target rows. Each prompt gets a global group
of stochastic CES programs; in a distributed run the group is split evenly
across workers and gathered before computing RLOO advantages. Complete programs
are applied and rewarded by normalized recovery toward the known clean
sequence. Gradients use exact token+pointer policy replay, an unstandardized
RLOO baseline, and optional completion-only gold CES replay.

Example smoke run:

    python -m ces.workflows.ces_rl --model-tag d12 --model-step 74800 --steps 2 \
        --input-format text_pairs \
        --manifest-jsonl data/evals/coedit_gec_transfer_v1/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanochat.causal_serialisation_extended import (
    Copy,
    Delete,
    Insert,
    Move,
    Replace,
    SequenceEdit,
    SerialisedData,
    TokensOperations,
    TrainingData,
    TrainingTarget,
    resolve,
)
from nanochat.ces_rl import (
    add_likelihood_auxiliary,
    group_advantages,
    local_repair_verifier_reward,
    provenance_program_reward,
    replay_policy_logprobs,
    target_blind_likelihood_reward,
    target_aware_recovery_reward,
)
from nanochat.checkpoint_manager import build_model, load_model, save_checkpoint
from nanochat.common import (
    COMPUTE_DTYPE,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
)
from nanochat.dataset import list_parquet_files
from nanochat.engine_extended import Engine, PolicyRollout
from nanochat.oracle_edit_likelihood import conditional_program_training_data
from nanochat.rl_pair_dataloader import StreamingEditPairSampler
from nanochat.edit_dataloader import EditSamplingConfig
from ces.workflows.benchmark_tokenization import prepare_benchmark_for_tokenizer
from ces.workflows.controlled_edit_eval import _tokens_hash, score_unique_sequences
from ces.workflows.oracle_ces_eval import _text_pair_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, default=None)
    parser.add_argument(
        "--input-format",
        choices=("controlled", "text_pairs", "edit_stream"),
        required=True,
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help="ClimbMix source shards for edit_stream; final shard stays held out.",
    )
    parser.add_argument(
        "--generated-edit-paths",
        nargs="*",
        type=Path,
        default=[],
        help="GPT edit-candidate Parquet files/directories for edit_stream.",
    )
    parser.add_argument("--stream-gpt-probability", type=float, default=0.5)
    parser.add_argument("--stream-random-probability", type=float, default=0.3)
    parser.add_argument("--stream-noop-probability", type=float, default=0.2)
    parser.add_argument("--stream-max-source-tokens", type=int, default=512)
    parser.add_argument("--stream-max-serialised-tokens", type=int, default=768)
    parser.add_argument(
        "--allow-eval-split",
        action="store_true",
        help="Permit held-out rows for an explicit non-reportable smoke test",
    )
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--model-step", type=int, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "Optional direct cold-start checkpoint directory. This keeps "
            "post-training checkpoints outside the curated base namespace."
        ),
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Exact CES-RL checkpoint directory to resume",
    )
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device-type", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--prompts-per-step",
        type=int,
        default=1,
        help=(
            "Independent prompt groups accumulated before each optimizer step. "
            "Each prompt receives --group-size rollouts and its own RLOO baseline."
        ),
    )
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument(
        "--max-resample-attempts",
        type=int,
        default=0,
        help="Regenerate a rollout group up to this many times when all rewards are equal",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--pointer-temperature", type=float, default=None)
    parser.add_argument("--max-edit-tokens", type=int, default=128)
    parser.add_argument("--max-payload-tokens", type=int, default=64)
    parser.add_argument("--max-edit-ops", type=int, default=8)
    parser.add_argument("--closure-bias", type=float, default=0.0)
    parser.add_argument("--closure-bias-min-payload-tokens", type=int, default=0)
    parser.add_argument("--invalid-reward", type=float, default=-1.0)
    parser.add_argument(
        "--reward-mode",
        choices=("target_aware", "provenance", "local_verifier", "likelihood"),
        default="target_aware",
    )
    parser.add_argument(
        "--likelihood-scorer-dir",
        type=Path,
        default=None,
        help="Frozen clean-LM checkpoint directory for target-blind likelihood reward",
    )
    parser.add_argument("--likelihood-scorer-step", type=int, default=None)
    parser.add_argument(
        "--likelihood-reward-scale",
        type=float,
        default=0.25,
        help="Average-logprob delta mapped to unit target-blind reward before clipping",
    )
    parser.add_argument("--exact-bonus", type=float, default=0.25)
    parser.add_argument(
        "--provenance-extra-op-penalty",
        type=float,
        default=0.25,
        help="Penalty weight for sampled CES operations absent from gold provenance.",
    )
    parser.add_argument(
        "--verifier-outside-edit-penalty", type=float, default=0.25,
        help="Penalty for candidate edit regions outside known corruptions.",
    )
    parser.add_argument(
        "--verifier-anchor-chars", type=int, default=12,
        help="Unchanged characters bracketing each local repair check.",
    )
    parser.add_argument("--edit-penalty", type=float, default=0.0)
    parser.add_argument(
        "--corrupt-noop-penalty",
        type=float,
        default=0.0,
        help="Extra terminal penalty when a known-corrupted prompt is left unchanged.",
    )
    parser.add_argument(
        "--negative-reward-scale",
        type=float,
        default=1.0,
        help="Multiplier >=1 for valid candidates whose shaped reward is negative.",
    )
    parser.add_argument(
        "--character-reward-weight",
        type=float,
        default=0.0,
        help="Weight on decoded-character recovery; remaining mass stays on token recovery",
    )
    parser.add_argument(
        "--operation-local-reward-weight",
        type=float,
        default=0.0,
        help="Weight on recovery of target-side local patterns around known corruptions",
    )
    parser.add_argument("--local-context-tokens", type=int, default=8)
    parser.add_argument(
        "--likelihood-aux-weight",
        type=float,
        default=0.0,
        help="Weak frozen clean-LM reward added to target-aware recovery",
    )
    parser.add_argument(
        "--operation-oversample",
        nargs="*",
        choices=("insert", "delete", "replace", "move", "mixed"),
        default=[],
        help="Duplicate rows of these controlled corruption modes in the shuffled schedule",
    )
    parser.add_argument(
        "--operation-oversample-factor",
        type=int,
        default=2,
        help="Total copies for modes selected by --operation-oversample",
    )
    parser.add_argument("--advantage", choices=("rloo", "centered"), default="rloo")
    parser.add_argument("--supervised-replay-fraction", type=float, default=0.25)
    parser.add_argument(
        "--ces-replay-share",
        type=float,
        default=0.5,
        help="Fraction of supervised replay assigned to gold CES; the rest is clean AR",
    )
    parser.add_argument(
        "--human-manifest-jsonl",
        type=Path,
        default=None,
        help="Optional train-only human source/target pairs for joint CES completion SFT",
    )
    parser.add_argument(
        "--human-supervised-fraction",
        type=float,
        default=0.0,
        help="Objective weight for completion-only CES loss on human edit pairs",
    )
    parser.add_argument(
        "--noop-supervised-fraction",
        type=float,
        default=0.0,
        help="Objective weight for explicit immediate-EDIT_END programs on clean human targets",
    )
    parser.add_argument(
        "--first-decision-fraction",
        type=float,
        default=0.0,
        help=(
            "Objective weight for a paired edit-vs-EDIT_END loss. Each corrupted "
            "prompt is paired with a clean prompt made from its known target."
        ),
    )
    parser.add_argument(
        "--first-decision-edit-prior",
        type=float,
        default=0.5,
        help=(
            "Relative weight assigned to the corrupted/edit member of each "
            "paired first-decision example. Values below 0.5 favor clean "
            "abstention while preserving both members of every pair."
        ),
    )
    parser.add_argument(
        "--human-examples-per-step",
        type=int,
        default=1,
        help="Human CES examples averaged into each on-policy optimizer update",
    )
    parser.add_argument(
        "--noop-examples-per-step",
        type=int,
        default=1,
        help="Clean no-op CES examples averaged into each on-policy optimizer update",
    )
    parser.add_argument(
        "--human-replace-oversample-factor",
        type=int,
        default=1,
        help="Total copies of human rows containing a replacement operation",
    )
    parser.add_argument("--clean-ar-prefix-fraction", type=float, default=0.5)
    parser.add_argument(
        "--retention-mode",
        choices=("paired_mixed", "independent_mixed", "alternating_ar", "clean_kl"),
        default="paired_mixed",
        help="How clean-language retention is combined with the on-policy update",
    )
    parser.add_argument(
        "--clean-data-dir",
        type=Path,
        default=None,
        help="ClimbMix parquet directory for an independent, train-only clean stream",
    )
    parser.add_argument("--clean-seq-len", type=int, default=256)
    parser.add_argument(
        "--clean-update-every",
        type=int,
        default=4,
        help="One standalone clean AR step after this many RL steps in alternating_ar mode",
    )
    parser.add_argument(
        "--clean-step-lr-scale",
        type=float,
        default=1.0,
        help="Temporary LR multiplier for standalone alternating clean-AR steps",
    )
    parser.add_argument(
        "--kl-reference-dir",
        type=Path,
        default=None,
        help="Optional direct checkpoint directory for clean-token KL; defaults to the initial CES checkpoint",
    )
    parser.add_argument("--kl-reference-step", type=int, default=None)
    parser.add_argument("--position-buffer-size", type=int, default=128)
    parser.add_argument("--lr-scale", type=float, default=0.05)
    parser.add_argument("--embedding-lr", type=float, default=0.2)
    parser.add_argument("--unembedding-lr", type=float, default=0.004)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--low-memory-distributed-optimizer",
        action="store_true",
        help=(
            "Process distributed optimizer groups sequentially to reduce peak "
            "VRAM. Recommended for full-parameter d24 tuning on 16 GiB GPUs."
        ),
    )
    parser.add_argument(
        "--distributed-muon-group-size",
        type=int,
        default=0,
        help="Maximum same-shaped matrices per distributed Muon group; 0 keeps the default.",
    )
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--save-steps",
        nargs="*",
        type=int,
        default=[],
        help="Additional exact optimizer steps to checkpoint.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run updates and logging without writing checkpoint tensors",
    )
    parser.add_argument(
        "--save-model-only",
        action="store_true",
        help=(
            "Save model and metadata but omit optimizer state. This is useful "
            "for terminal sweep checkpoints that will only be evaluated; such "
            "checkpoints cannot be used with --resume-dir."
        ),
    )
    parser.add_argument("--log-jsonl", type=Path, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def materialize_missing_gradients(model: torch.nn.Module) -> int:
    """Give optimizers explicit zero gradients for parameters unused by a batch.

    Pure policy-gradient batches can legitimately omit a policy head or route.
    Nanochat's fused Muon update stacks every parameter gradient, so represent
    those mathematically zero gradients explicitly instead of leaving them as
    ``None``.
    """
    missing = 0
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
            missing += 1
    return missing


def local_rollout_group_size(global_group_size: int, world_size: int) -> int:
    """Split one RLOO group evenly while keeping CLI semantics device-count invariant."""

    if global_group_size < 1:
        raise ValueError("global rollout group size must be positive")
    if world_size < 1:
        raise ValueError("world size must be positive")
    if global_group_size % world_size:
        raise ValueError(
            f"--group-size ({global_group_size}) must be divisible by distributed "
            f"world size ({world_size})"
        )
    return global_group_size // world_size


def gather_rollout_rewards(local_rewards: torch.Tensor, world_size: int) -> torch.Tensor:
    """Gather rank-local rewards in rank order to form one global RLOO group."""

    if world_size == 1:
        return local_rewards
    gathered = [torch.empty_like(local_rewards) for _ in range(world_size)]
    dist.all_gather(gathered, local_rewards)
    return torch.cat(gathered)


def reduce_scalar(
    value: float | int,
    *,
    device: torch.device,
    world_size: int,
    reduction: str = "mean",
) -> float:
    """Reduce a logging/control scalar without constructing model-sized buffers."""

    result = torch.tensor(float(value), dtype=torch.float64, device=device)
    if world_size > 1:
        operations = {
            "sum": dist.ReduceOp.SUM,
            "mean": dist.ReduceOp.SUM,
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }
        if reduction not in operations:
            raise ValueError(f"Unknown scalar reduction: {reduction}")
        dist.all_reduce(result, op=operations[reduction])
        if reduction == "mean":
            result /= world_size
    return float(result.item())


def distributed_clip_grad_norm_(
    model: torch.nn.Module,
    max_norm: float,
    *,
    device: torch.device,
    world_size: int,
) -> torch.Tensor:
    """Apply one conservative clipping coefficient on every data-parallel rank.

    The optimizer averages gradients internally. Using the maximum local norm
    gives every rank the same coefficient, so the averaged update remains the
    gradient of a consistently scaled distributed objective.
    """

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not parameters:
        return torch.zeros((), dtype=torch.float32, device=device)
    norms = torch.stack([
        torch.linalg.vector_norm(parameter.grad.detach().float())
        for parameter in parameters
    ])
    local_norm = torch.linalg.vector_norm(norms)
    shared_norm = local_norm.clone()
    if world_size > 1:
        dist.all_reduce(shared_norm, op=dist.ReduceOp.MAX)
    coefficient = (float(max_norm) / (shared_norm + 1e-6)).clamp(max=1.0)
    for parameter in parameters:
        parameter.grad.mul_(coefficient.to(parameter.grad.dtype))
    return shared_norm


def optional_detail_mean(
    details: list[Any],
    attribute: str,
    *,
    device: torch.device,
    world_size: int,
) -> float | None:
    """Global mean of an optional numeric reward diagnostic."""

    values = [
        float(value)
        for detail in details
        if (value := getattr(detail, attribute)) is not None
    ]
    total = reduce_scalar(
        sum(values), device=device, world_size=world_size, reduction="sum"
    )
    count = reduce_scalar(
        len(values), device=device, world_size=world_size, reduction="sum"
    )
    return total / count if count else None


def reject_heldout_rows(rows: list[dict[str, Any]]) -> None:
    """Prevent accidental post-training on validation/test benchmarks."""

    heldout_labels = {
        "validation", "development", "confirmation", "sweep", "test", "eval"
    }
    offenders = []
    for row in rows:
        labels = {
            str(row.get(field, "")).strip().lower()
            for field in ("split", "role", "source_split")
        }
        if labels & heldout_labels:
            offenders.append(str(row.get("benchmark_id", "<unknown>")))
    if offenders:
        preview = ", ".join(offenders[:3])
        raise ValueError(
            "Refusing to train on held-out rows "
            f"({preview}{' ...' if len(offenders) > 3 else ''}). "
            "Use a disjoint training manifest. --allow-eval-split is only for "
            "a non-reportable plumbing smoke test."
        )


def oversample_controlled_rows(
    rows: list[dict[str, Any]],
    *,
    modes: list[str],
    factor: int,
) -> list[dict[str, Any]]:
    """Repeat difficult controlled modes without changing any row contents."""
    if factor < 1:
        raise ValueError("operation oversample factor must be >= 1")
    selected = set(modes)
    return [
        row
        for row in rows
        for _ in range(factor if str(row.get("repair_op_mode")) in selected else 1)
    ]


def cyclic_component_rows(
    rows: list[dict[str, Any]], *, step: int, count: int
) -> list[dict[str, Any]]:
    """Take deterministic sequential component examples and wrap at epoch end."""
    if count < 1:
        raise ValueError("Component examples per step must be positive")
    if not rows:
        raise ValueError("Cannot sample an empty component")
    start = step * count
    return [rows[(start + offset) % len(rows)] for offset in range(count)]


def cyclic_prompt_rows(
    rows: list[dict[str, Any]], *, step: int, prompts_per_step: int
) -> list[dict[str, Any]]:
    """Select the independent prompts assigned to one optimizer update."""

    return cyclic_component_rows(rows, step=step, count=prompts_per_step)


def _stream_edit_record(edit: Any) -> dict[str, Any]:
    if isinstance(edit, Insert):
        return {
            "op": "insert",
            "pos": int(edit.pos),
            "content": [int(token) for token in edit.content],
        }
    if isinstance(edit, Delete):
        return {"op": "delete", "start": int(edit.start), "end": int(edit.end)}
    if isinstance(edit, Replace):
        return {
            "op": "replace",
            "start": int(edit.start),
            "end": int(edit.end),
            "content": [int(token) for token in edit.content],
        }
    if isinstance(edit, (Copy, Move)):
        return {
            "op": "copy" if isinstance(edit, Copy) else "move",
            "start": int(edit.start),
            "end": int(edit.end),
            "dest": int(edit.dest),
        }
    raise TypeError(f"Unsupported stream edit: {type(edit).__name__}")


def stream_pair_to_row(pair, token_ops: TokensOperations) -> dict[str, Any]:
    """Render one pretraining-stream pair into the existing CES-RL row API."""

    sequence_edit: SequenceEdit = pair.sequence_edit
    serialised = token_ops.serialise(sequence_edit, indel_bos_pos=0)
    tokens = [int(token) for token in serialised.tokens]
    positions = [int(position) for position in serialised.positions]
    starts = [
        index
        for index, token in enumerate(tokens)
        if token == int(token_ops.edit_start_token)
    ]
    if len(starts) != 1:
        raise ValueError(
            f"Expected one edit-start token for {pair.example_id}, found {len(starts)}"
        )
    edit_start = starts[0]
    source_tokens = [
        int(token) for token in np.asarray(sequence_edit.sequence).tolist()
    ]
    target_tokens = [
        int(token) for token in np.asarray(resolve(sequence_edit)).tolist()
    ]
    gold_edits = [_stream_edit_record(edit) for edit in sequence_edit.edits]
    return {
        "benchmark_id": pair.example_id,
        "stream_kind": pair.kind,
        "stream_epoch": pair.epoch,
        "source_shard_id": pair.source_shard_id,
        "source_row_index": pair.source_row_index,
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "gold_edits": gold_edits,
        "gold_op_counts": dict(
            sorted(Counter(edit["op"] for edit in gold_edits).items())
        ),
        "prompt_tokens": tokens[: edit_start + 1],
        "prompt_positions": positions[: edit_start + 1],
        "gold_program_tokens": tokens[edit_start + 1 :],
        "gold_program_positions": positions[edit_start + 1 :],
    }


def distributed_cyclic_component_rows(
    rows: list[dict[str, Any]],
    *,
    step: int,
    global_count: int,
    rank: int,
    world_size: int,
) -> list[dict[str, Any]]:
    """Partition a deterministic global auxiliary batch across ranks."""

    if global_count % world_size != 0:
        raise ValueError(
            f"Global component count ({global_count}) must be divisible by "
            f"world size ({world_size})"
        )
    local_count = global_count // world_size
    if local_count < 1:
        raise ValueError("Each rank must receive at least one component example")
    if not rows:
        raise ValueError("Cannot sample an empty component")
    start = step * global_count + rank * local_count
    return [rows[(start + offset) % len(rows)] for offset in range(local_count)]


def joint_objective_weights(
    *,
    supervised_replay_fraction: float,
    ces_replay_share: float,
    human_supervised_fraction: float,
    noop_supervised_fraction: float,
    retention_mode: str,
    first_decision_fraction: float = 0.0,
) -> dict[str, float]:
    """Resolve user-facing mixture fractions into additive loss weights."""

    values = {
        "supervised_replay_fraction": supervised_replay_fraction,
        "ces_replay_share": ces_replay_share,
        "human_supervised_fraction": human_supervised_fraction,
        "noop_supervised_fraction": noop_supervised_fraction,
        "first_decision_fraction": first_decision_fraction,
    }
    if any(not 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError(f"Joint objective fractions must be in [0, 1]: {values}")
    synthetic = supervised_replay_fraction * ces_replay_share
    retention = supervised_replay_fraction * (1.0 - ces_replay_share)
    if retention_mode == "alternating_ar":
        # Alternating clean AR is a separate optimizer update and therefore is
        # not subtracted from the mixed update's unit mass.
        rl = (
            1.0
            - synthetic
            - human_supervised_fraction
            - noop_supervised_fraction
            - first_decision_fraction
        )
    else:
        rl = (
            1.0
            - synthetic
            - retention
            - human_supervised_fraction
            - noop_supervised_fraction
            - first_decision_fraction
        )
    if rl < -1e-9:
        raise ValueError(
            "Joint objective weights exceed one: "
            f"synthetic={synthetic}, retention={retention}, "
            f"human={human_supervised_fraction}, noop={noop_supervised_fraction}, "
            f"first_decision={first_decision_fraction}"
        )
    return {
        "rl": max(0.0, rl),
        "synthetic_ces": synthetic,
        "retention": retention,
        "human_ces": human_supervised_fraction,
        "noop_ces": noop_supervised_fraction,
        "first_decision": first_decision_fraction,
    }


def human_identity_rows(
    raw_human_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn human targets into clean source=target abstention examples."""

    return [
        {
            "benchmark_id": f"noop:{row.get('benchmark_id', index)}",
            "source_text": str(row["target_text"]),
            "target_text": str(row["target_text"]),
            "split": "training",
            "role": "joint_noop_training",
            "source_split": "train",
        }
        for index, row in enumerate(raw_human_rows)
    ]


def autocast_context(device_type: str):
    if device_type == "cuda" and COMPUTE_DTYPE in (torch.bfloat16, torch.float16):
        return torch.amp.autocast(device_type="cuda", dtype=COMPUTE_DTYPE)
    return nullcontext()


def tensor_training_data(data: TrainingData, device: torch.device) -> TrainingData:
    def tensor(value, dtype):
        return torch.as_tensor(value, dtype=dtype, device=device)[None, :]

    return TrainingData(
        input_tokens=tensor(data.input_tokens, torch.long),
        input_positions=tensor(data.input_positions, torch.long),
        target=TrainingTarget(
            tokens=tensor(data.target.tokens, torch.long),
            gathered_pos_target=tensor(
                data.target.gathered_pos_target, torch.long
            ),
            token_loss_weights=tensor(
                data.target.token_loss_weights, torch.bool
            ),
            gather_indices=tensor(data.target.gather_indices, torch.long),
            gather_valid=tensor(data.target.gather_valid, torch.bool),
            overflow=torch.as_tensor(
                [bool(data.target.overflow)], dtype=torch.bool, device=device
            ),
            perplexity_masks=None,
        ),
    )


def clean_ar_completion_targets(
    tokens: list[int],
    *,
    prefix_fraction: float,
) -> tuple[list[int], list[int], list[int]]:
    """Build AR inputs/positions and labels masked before a clean suffix."""

    if len(tokens) < 2:
        raise ValueError("Clean AR replay needs at least two tokens")
    if not 0.0 <= prefix_fraction < 1.0:
        raise ValueError("prefix_fraction must be in [0, 1)")
    split = min(max(int(round(len(tokens) * prefix_fraction)), 1), len(tokens) - 1)
    inputs = [int(token) for token in tokens[:-1]]
    positions = list(range(len(inputs)))
    labels = [int(token) for token in tokens[1:]]
    # label[j] predicts token[j + 1]. The prefix includes tokens[:split], so
    # the first supervised label predicts tokens[split].
    labels[: max(split - 1, 0)] = [-1] * max(split - 1, 0)
    return inputs, positions, labels


class IndependentCleanTextStream:
    """Deterministic clean-document stream disjoint from corruption shards."""

    def __init__(
        self,
        data_dir: Path,
        tokenizer,
        *,
        max_tokens: int,
        seed: int,
        excluded_shards: set[str],
    ) -> None:
        if max_tokens < 2:
            raise ValueError("Clean sequence length must be at least two")
        paths = [Path(path) for path in list_parquet_files(str(data_dir))]
        if len(paths) < 2:
            raise ValueError("Independent clean replay needs train and validation shards")
        paths = [
            path for path in paths[:-1]
            if path.stem not in excluded_shards
        ]
        if not paths:
            raise ValueError("No clean training shards remain after exclusions")
        rng = random.Random(seed)
        rng.shuffle(paths)
        self.paths = paths
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.rng = rng
        self.path_index = 0
        self.row_group_index = 0
        self.text_buffer: list[str] = []
        self.documents_seen = 0

    def _refill(self) -> None:
        while not self.text_buffer:
            path = self.paths[self.path_index]
            parquet = pq.ParquetFile(path)
            if self.row_group_index >= parquet.num_row_groups:
                self.path_index = (self.path_index + 1) % len(self.paths)
                self.row_group_index = 0
                continue
            table = parquet.read_row_group(self.row_group_index, columns=["text"])
            self.row_group_index += 1
            texts = [str(text) for text in table.column("text").to_pylist() if text]
            self.rng.shuffle(texts)
            self.text_buffer.extend(texts)

    def next_tokens(self) -> list[int]:
        bos = int(self.tokenizer.get_bos_token_id())
        while True:
            self._refill()
            text = self.text_buffer.pop()
            self.documents_seen += 1
            body = [int(token) for token in self.tokenizer.encode(text)]
            if not body:
                continue
            if len(body) >= self.max_tokens:
                # Preserve the native BOS-aligned pretraining convention. A
                # random mid-document crop immediately after BOS is a different
                # and potentially harmful conditional distribution.
                body = body[: self.max_tokens - 1]
            return [bos, *body]


def clean_ar_sequence_loss(
    model,
    tokens: list[int],
    *,
    device: torch.device,
    prefix_fraction: float,
) -> torch.Tensor:
    """Completion-only AR loss for one already-tokenized clean document."""

    inputs, positions, labels = clean_ar_completion_targets(
        tokens,
        prefix_fraction=prefix_fraction,
    )
    input_tensor = torch.tensor([inputs], dtype=torch.long, device=device)
    position_tensor = torch.tensor([positions], dtype=torch.long, device=device)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    targets = TrainingTarget(
        tokens=label_tensor,
        gathered_pos_target=None,
        token_loss_weights=torch.ones_like(label_tensor, dtype=torch.bool),
        gather_indices=None,
        gather_valid=None,
        overflow=False,
        perplexity_masks=None,
    )
    loss = model(
        input_tensor,
        tix=position_tensor,
        targets=targets,
        loss_reduction="sum",
    )
    decisions = (label_tensor != -1).sum().clamp(min=1)
    return loss / decisions


def clean_ar_completion_loss(
    model,
    row: dict[str, Any],
    *,
    device: torch.device,
    prefix_fraction: float,
) -> torch.Tensor:
    return clean_ar_sequence_loss(
        model,
        [int(token) for token in row["target_tokens"]],
        device=device,
        prefix_fraction=prefix_fraction,
    )


def clean_token_kl_loss(
    model,
    reference_model,
    tokens: list[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Forward KL from a frozen starting policy on independent clean text."""

    if len(tokens) < 2:
        raise ValueError("Clean KL needs at least two tokens")
    inputs = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    positions = torch.arange(
        inputs.shape[1], dtype=torch.long, device=device
    )[None, :]
    with torch.no_grad():
        reference_logits = reference_model(inputs, tix=positions).float()
        reference_probs = F.softmax(reference_logits, dim=-1)
    policy_logprobs = F.log_softmax(model(inputs, tix=positions).float(), dim=-1)
    per_token = F.kl_div(
        policy_logprobs,
        reference_probs,
        reduction="none",
    ).sum(dim=-1)
    return per_token.mean()


def completion_only_supervised_loss(
    model,
    token_ops: TokensOperations,
    row: dict[str, Any],
    *,
    device: torch.device,
    position_buffer_size: int,
) -> torch.Tensor:
    tokens = [*row["prompt_tokens"], *row["gold_program_tokens"]]
    positions = [*row["prompt_positions"], *row["gold_program_positions"]]
    sequence_edit = token_ops.deserialise(SerialisedData(
        tokens=np.asarray(tokens, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.int64),
    ))
    data = conditional_program_training_data(
        token_ops,
        sequence_edit,
        position_buffer_size=position_buffer_size,
    )
    if data.target.overflow:
        raise ValueError(
            f"Gold program needs more than {position_buffer_size} pointer slots"
        )
    batch = tensor_training_data(data, device)
    _loss, auxiliary = model(
        batch.input_tokens,
        tix=batch.input_positions,
        targets=batch.target,
        loss_reduction="sum",
    )
    decisions = (
        batch.target.token_loss_weights.sum()
        + batch.target.gather_valid.sum()
    ).clamp(min=1)
    return (auxiliary["token_loss"] + auxiliary["position_loss"]) / decisions


def identity_prompt_from_target(
    token_ops: TokensOperations,
    target_tokens: list[int],
) -> tuple[list[int], list[int]]:
    """Serialise a clean target through EDIT_START, ready for a no-op decision."""

    serialised = token_ops.serialise(
        SequenceEdit(
            sequence=np.asarray(target_tokens, dtype=np.int64),
            edits=[],
        ),
        indel_bos_pos=0,
    )
    tokens = [int(token) for token in serialised.tokens]
    positions = [int(position) for position in serialised.positions]
    starts = [
        index
        for index, token in enumerate(tokens)
        if token == int(token_ops.edit_start_token)
    ]
    if len(starts) != 1:
        raise ValueError(f"Expected exactly one EDIT_START, found {len(starts)}")
    end = starts[0] + 1
    return tokens[:end], positions[:end]


def first_decision_loss(
    model,
    token_ops: TokensOperations,
    prompt_tokens: list[int],
    prompt_positions: list[int],
    *,
    should_edit: bool,
    device: torch.device,
) -> torch.Tensor:
    """Binary edit/abstain loss at the token immediately after EDIT_START.

    The edit class marginalises over the three operation openers used by the
    GEC post-training distribution. This teaches whether to act without
    imposing an arbitrary operation label or pointer on a one-to-many repair.
    """

    if not prompt_tokens or prompt_tokens[-1] != int(token_ops.edit_start_token):
        raise ValueError("First-decision prompts must end at EDIT_START")
    inputs = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    positions = torch.tensor([prompt_positions], dtype=torch.long, device=device)
    logits = model(inputs, tix=positions)[0, -1].float()
    edit_tokens = torch.tensor(
        [
            int(token_ops.deletion_open_token),
            int(token_ops.insertion_open_token),
            int(token_ops.replace_open_token),
        ],
        dtype=torch.long,
        device=device,
    )
    edit_logit = torch.logsumexp(logits.index_select(0, edit_tokens), dim=0)
    noop_logit = logits[int(token_ops.edit_end_token)]
    class_logits = torch.stack((noop_logit, edit_logit))[None, :]
    label = torch.tensor([int(should_edit)], dtype=torch.long, device=device)
    return F.cross_entropy(class_logits, label)


def first_decision_pair_weights(edit_prior: float) -> tuple[float, float]:
    """Return mean-preserving weights for corrupt/edit and clean/no-op pairs."""

    if not 0.0 <= edit_prior <= 1.0:
        raise ValueError("edit_prior must be in [0, 1]")
    return 2.0 * edit_prior, 2.0 * (1.0 - edit_prior)


def rollout_decision_count(rollout: PolicyRollout) -> int:
    return sum(
        int(action.token_sampled) + int(action.pointer_sampled)
        for action in rollout.actions
    )


def checkpoint_metadata(
    original: dict[str, Any],
    args: argparse.Namespace,
    *,
    posttraining_step: int,
    tokenization: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(original)
    metadata.update({
        "step": posttraining_step,
        "phase": "ces_rl_posttraining",
        "initial_model_tag": args.model_tag,
        "initial_model_step": args.model_step,
        "user_config": vars(args) | {
            "manifest_jsonl": (
                str(args.manifest_jsonl) if args.manifest_jsonl else None
            ),
            "raw_data_dir": (
                str(args.raw_data_dir) if args.raw_data_dir else None
            ),
            "generated_edit_paths": [
                str(path) for path in args.generated_edit_paths
            ],
            "output_dir": str(args.output_dir) if args.output_dir else None,
            "model_dir": str(args.model_dir) if args.model_dir else None,
            "log_jsonl": str(args.log_jsonl) if args.log_jsonl else None,
            "resume_dir": str(args.resume_dir) if args.resume_dir else None,
            "likelihood_scorer_dir": (
                str(args.likelihood_scorer_dir)
                if args.likelihood_scorer_dir
                else None
            ),
            "clean_data_dir": str(args.clean_data_dir) if args.clean_data_dir else None,
            "human_manifest_jsonl": (
                str(args.human_manifest_jsonl)
                if args.human_manifest_jsonl
                else None
            ),
            "kl_reference_dir": (
                str(args.kl_reference_dir) if args.kl_reference_dir else None
            ),
        },
        "benchmark_tokenization": tokenization,
    })
    return metadata


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.prompts_per_step < 1:
        raise ValueError("--prompts-per-step must be positive")
    if args.input_format == "edit_stream":
        if args.raw_data_dir is None or not args.generated_edit_paths:
            raise ValueError(
                "edit_stream requires --raw-data-dir and --generated-edit-paths"
            )
        stream_probabilities = (
            args.stream_gpt_probability,
            args.stream_random_probability,
            args.stream_noop_probability,
        )
        if any(value < 0 for value in stream_probabilities) or not np.isclose(
            sum(stream_probabilities), 1.0
        ):
            raise ValueError(
                "Stream GPT/random/no-op probabilities must be non-negative "
                "and sum to one"
            )
    elif args.manifest_jsonl is None:
        raise ValueError(
            "controlled and text_pairs inputs require --manifest-jsonl"
        )
    if any(step < 1 or step > args.steps for step in args.save_steps):
        raise ValueError("--save-steps values must lie in [1, --steps]")
    if args.group_size < 2 and args.advantage == "rloo":
        raise ValueError("RLOO requires --group-size >= 2")
    if args.temperature <= 0:
        raise ValueError("RL rollouts require --temperature > 0")
    if args.max_resample_attempts < 0:
        raise ValueError("--max-resample-attempts must be non-negative")
    if args.operation_oversample_factor < 1:
        raise ValueError("--operation-oversample-factor must be >= 1")
    if args.likelihood_reward_scale <= 0:
        raise ValueError("--likelihood-reward-scale must be positive")
    if (args.reward_mode == "likelihood" or args.likelihood_aux_weight > 0) and (
        args.likelihood_scorer_dir is None or args.likelihood_scorer_step is None
    ):
        raise ValueError(
            "Likelihood reward/auxiliary requires --likelihood-scorer-dir and --likelihood-scorer-step"
        )
    if args.character_reward_weight < 0 or args.operation_local_reward_weight < 0:
        raise ValueError("Reward component weights must be non-negative")
    if args.character_reward_weight + args.operation_local_reward_weight > 1:
        raise ValueError("Character and operation-local reward weights must sum to <= 1")
    if args.likelihood_aux_weight < 0:
        raise ValueError("--likelihood-aux-weight must be non-negative")
    if args.corrupt_noop_penalty < 0:
        raise ValueError("--corrupt-noop-penalty must be non-negative")
    if args.negative_reward_scale < 1:
        raise ValueError("--negative-reward-scale must be at least one")
    if args.provenance_extra_op_penalty < 0:
        raise ValueError("--provenance-extra-op-penalty must be non-negative")
    if args.verifier_outside_edit_penalty < 0 or args.verifier_anchor_chars < 1:
        raise ValueError("Local verifier penalty/anchor settings are invalid")
    if args.local_context_tokens < 1:
        raise ValueError("--local-context-tokens must be positive")
    if args.pointer_temperature is not None and args.pointer_temperature <= 0:
        raise ValueError("--pointer-temperature must be positive")
    if not 0.0 <= args.supervised_replay_fraction <= 1.0:
        raise ValueError("--supervised-replay-fraction must be in [0, 1]")
    if not 0.0 <= args.ces_replay_share <= 1.0:
        raise ValueError("--ces-replay-share must be in [0, 1]")
    if (
        args.human_supervised_fraction < 0
        or args.noop_supervised_fraction < 0
        or args.first_decision_fraction < 0
    ):
        raise ValueError(
            "Human, no-op, and first-decision fractions must be non-negative"
        )
    if not 0.0 <= args.first_decision_edit_prior <= 1.0:
        raise ValueError("--first-decision-edit-prior must be in [0, 1]")
    if args.human_replace_oversample_factor < 1:
        raise ValueError("--human-replace-oversample-factor must be >= 1")
    if args.human_examples_per_step < 1 or args.noop_examples_per_step < 1:
        raise ValueError("Human and no-op examples per step must be positive")
    if (
        args.human_supervised_fraction > 0
        or args.noop_supervised_fraction > 0
    ) and args.human_manifest_jsonl is None:
        raise ValueError(
            "Human/no-op supervision requires --human-manifest-jsonl"
        )
    if not 0.0 <= args.clean_ar_prefix_fraction < 1.0:
        raise ValueError("--clean-ar-prefix-fraction must be in [0, 1)")
    if args.clean_seq_len < 2:
        raise ValueError("--clean-seq-len must be at least two")
    if args.clean_update_every < 1:
        raise ValueError("--clean-update-every must be positive")
    if args.clean_step_lr_scale <= 0:
        raise ValueError("--clean-step-lr-scale must be positive")
    if args.retention_mode != "paired_mixed" and args.clean_data_dir is None:
        raise ValueError(
            "Independent/alternating/KL retention requires --clean-data-dir"
        )
    if (args.kl_reference_dir is None) != (args.kl_reference_step is None):
        raise ValueError("Pass --kl-reference-dir and --kl-reference-step together")
    if (args.resume_dir is None) != (args.resume_step is None):
        raise ValueError("Pass --resume-dir and --resume-step together")
    if args.model_dir is not None and args.resume_dir is not None:
        raise ValueError("--model-dir and --resume-dir are mutually exclusive")
    if args.resume_step is not None and not 0 < args.resume_step < args.steps:
        raise ValueError("Require 0 < --resume-step < --steps (the total horizon)")
    objective_weights = joint_objective_weights(
        supervised_replay_fraction=args.supervised_replay_fraction,
        ces_replay_share=args.ces_replay_share,
        human_supervised_fraction=args.human_supervised_fraction,
        noop_supervised_fraction=args.noop_supervised_fraction,
        retention_mode=args.retention_mode,
        first_decision_fraction=args.first_decision_fraction,
    )

    device_type = args.device_type or autodetect_device_type()
    ddp, rank, _local_rank, world_size, device = compute_init(
        device_type, seed=args.seed
    )
    local_group_size = local_rollout_group_size(args.group_size, world_size)
    for name, count in (
        ("human", args.human_examples_per_step),
        ("noop", args.noop_examples_per_step),
    ):
        if count % world_size != 0:
            raise ValueError(
                f"--{name}-examples-per-step={count} must be divisible by "
                f"world size {world_size}"
            )
    local_human_examples = args.human_examples_per_step // world_size
    local_noop_examples = args.noop_examples_per_step // world_size
    if args.advantage == "rloo" and args.group_size < 2:
        raise ValueError("RLOO requires a global --group-size >= 2")

    start_step = int(args.resume_step or 0)
    if args.resume_dir is None:
        if args.model_dir is None:
            model, tokenizer, original_metadata = load_model(
                "base",
                device,
                phase="train",
                model_tag=args.model_tag,
                step=args.model_step,
            )
        else:
            model, tokenizer, original_metadata = build_model(
                str(args.model_dir.resolve()),
                args.model_step,
                device,
                phase="train",
            )
    else:
        resume_dir = args.resume_dir.resolve()
        model, tokenizer, original_metadata = build_model(
            str(resume_dir), start_step, device, phase="train"
        )
    token_ops = TokensOperations(getattr(tokenizer, "enc", tokenizer))
    scorer = None
    if args.reward_mode == "likelihood" or args.likelihood_aux_weight > 0:
        scorer, scorer_tokenizer, _scorer_metadata = build_model(
            str(args.likelihood_scorer_dir.resolve()),
            int(args.likelihood_scorer_step),
            device,
            phase="eval",
        )
        if scorer_tokenizer.get_vocab_size() != tokenizer.get_vocab_size():
            raise ValueError("Policy and frozen likelihood scorer vocabularies differ")
        scorer.eval()
        for parameter in scorer.parameters():
            parameter.requires_grad_(False)
    raw_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    stream_sampler: StreamingEditPairSampler | None = None
    if args.input_format == "edit_stream":
        stream_sampler = StreamingEditPairSampler(
            tokenizer=tokenizer,
            token_ops=token_ops,
            cfg=EditSamplingConfig(
                raw_data_dir=str(args.raw_data_dir.resolve()),
                generated_edit_paths=tuple(
                    str(path.resolve()) for path in args.generated_edit_paths
                ),
                direction_probs={"inverse": 1.0},
                seed=args.seed,
            ),
            gpt_probability=args.stream_gpt_probability,
            random_probability=args.stream_random_probability,
            noop_probability=args.stream_noop_probability,
            max_source_tokens=args.stream_max_source_tokens,
            max_serialised_tokens=args.stream_max_serialised_tokens,
            seed=args.seed,
        )
        # Reconstruct the deterministic prompt-stream position on resume.
        for _ in range(start_step):
            stream_sampler.next_batch(args.prompts_per_step)
        tokenization = {
            "mode": "pretraining_edit_stream",
            "examples_per_epoch": stream_sampler.examples_per_epoch,
            "gpt_probability": args.stream_gpt_probability,
            "random_probability": args.stream_random_probability,
            "noop_probability": args.stream_noop_probability,
            "max_source_tokens": args.stream_max_source_tokens,
            "max_serialised_tokens": args.stream_max_serialised_tokens,
        }
    else:
        assert args.manifest_jsonl is not None
        raw_rows = read_jsonl(args.manifest_jsonl)
        if not args.allow_eval_split:
            reject_heldout_rows(raw_rows)
        if args.input_format == "controlled":
            rows, tokenization = prepare_benchmark_for_tokenizer(
                raw_rows,
                tokenizer=tokenizer,
                token_ops=token_ops,
            )
        else:
            rows, tokenization = _text_pair_rows(
                raw_rows,
                tokenizer=tokenizer,
                token_ops=token_ops,
            )
        if not rows:
            raise ValueError("The training manifest is empty")
        rows = oversample_controlled_rows(
            rows,
            modes=args.operation_oversample,
            factor=args.operation_oversample_factor,
        )
        random.Random(args.seed).shuffle(rows)

    human_rows: list[dict[str, Any]] = []
    noop_rows: list[dict[str, Any]] = []
    human_tokenization: dict[str, Any] | None = None
    if args.human_manifest_jsonl is not None:
        raw_human_rows = read_jsonl(args.human_manifest_jsonl)
        reject_heldout_rows(raw_human_rows)
        human_rows, human_tokenization = _text_pair_rows(
            raw_human_rows,
            tokenizer=tokenizer,
            token_ops=token_ops,
        )
        if args.human_replace_oversample_factor > 1:
            human_rows = [
                row
                for row in human_rows
                for _ in range(
                    args.human_replace_oversample_factor
                    if int(row.get("gold_op_counts", {}).get("replace", 0)) > 0
                    else 1
                )
            ]
        noop_rows, _noop_tokenization = _text_pair_rows(
            human_identity_rows(raw_human_rows),
            tokenizer=tokenizer,
            token_ops=token_ops,
        )
        if not human_rows or not noop_rows:
            raise ValueError("The human/no-op training streams are empty")
        random.Random(args.seed + 10_000).shuffle(human_rows)
        random.Random(args.seed + 11_000).shuffle(noop_rows)

    clean_stream = None
    if args.retention_mode != "paired_mixed":
        excluded_shards = {
            str(row.get("source_shard_id"))
            for row in raw_rows
            if row.get("source_shard_id")
        }
        clean_stream = IndependentCleanTextStream(
            args.clean_data_dir.resolve(),
            tokenizer,
            max_tokens=args.clean_seq_len,
            seed=args.seed + 20_000 + rank,
            excluded_shards=excluded_shards,
        )

    kl_reference = None
    if args.retention_mode == "clean_kl":
        if args.kl_reference_dir is None:
            kl_reference, kl_tokenizer, _kl_metadata = load_model(
                "base",
                device,
                phase="eval",
                model_tag=args.model_tag,
                step=args.model_step,
            )
        else:
            kl_reference, kl_tokenizer, _kl_metadata = build_model(
                str(args.kl_reference_dir.resolve()),
                int(args.kl_reference_step),
                device,
                phase="eval",
            )
        if kl_tokenizer.get_vocab_size() != tokenizer.get_vocab_size():
            raise ValueError("Policy and KL reference vocabularies differ")
        kl_reference.eval()
        for parameter in kl_reference.parameters():
            parameter.requires_grad_(False)

    engine = Engine(model, tokenizer)
    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        scalar_lr=args.scalar_lr,
        weight_decay=args.weight_decay,
        distributed_optimizer_overlap=(
            not args.low_memory_distributed_optimizer
        ),
        distributed_muon_group_size=(args.distributed_muon_group_size or None),
    )
    for group in optimizer.param_groups:
        group["lr"] *= args.lr_scale
        group["initial_lr"] = group["lr"]

    if args.resume_dir is not None:
        optimizer_path = (
            args.resume_dir.resolve()
            / f"optim_{start_step:06d}_rank{rank}.pt"
        )
        if not optimizer_path.is_file():
            raise FileNotFoundError(f"Missing resume optimizer: {optimizer_path}")
        optimizer.load_state_dict(torch.load(
            optimizer_path, map_location=device, weights_only=False
        ))

    output_tag = args.output_tag or f"{args.model_tag}-ces-rl-seed{args.seed}"
    output_dir = args.output_dir or args.resume_dir or (
        Path(get_base_dir()) / "cesrl_checkpoints" / output_tag
    )
    if (
        args.resume_dir is not None
        and output_dir.resolve() != args.resume_dir.resolve()
    ):
        raise ValueError("A resumed run must write back to its --resume-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()
    log_path = args.log_jsonl or output_dir / "train.jsonl"
    if rank == 0:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()

    pointer_temperature = (
        args.temperature
        if args.pointer_temperature is None
        else args.pointer_temperature
    )
    if rank == 0:
        print(json.dumps({
            "device": str(device),
            "world_size": world_size,
            "model_tag": args.model_tag,
            "model_step": args.model_step,
            "rows": (
                stream_sampler.examples_per_epoch
                if stream_sampler is not None
                else len(rows)
            ),
            "input_format": args.input_format,
            "prompts_per_step": args.prompts_per_step,
            "rollouts_per_step": args.prompts_per_step * args.group_size,
            "planned_prompt_exposures": args.prompts_per_step * args.steps,
            "group_size": args.group_size,
            "local_group_size": local_group_size,
            "start_step": start_step,
            "target_step": args.steps,
            "output_dir": str(output_dir),
            "retention_mode": args.retention_mode,
            "objective_weights": objective_weights,
            "human_rows": len(human_rows),
            "noop_rows": len(noop_rows),
            "human_examples_per_step": args.human_examples_per_step,
            "noop_examples_per_step": args.noop_examples_per_step,
            "local_human_examples_per_step": local_human_examples,
            "local_noop_examples_per_step": local_noop_examples,
            "human_replace_oversample_factor": (
                args.human_replace_oversample_factor
            ),
            "human_tokenization": human_tokenization,
            "low_memory_distributed_optimizer": (
                args.low_memory_distributed_optimizer
            ),
            "clean_stream_shard_count": (
                len(clean_stream.paths) if clean_stream is not None else 0
            ),
            "clean_stream_first_shards": (
                [path.stem for path in clean_stream.paths[:8]]
                if clean_stream is not None
                else []
            ),
            "tokenization": tokenization,
        }, indent=2), flush=True)

    try:
        for step in range(start_step, args.steps):
            started = time.perf_counter()
            if device_type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            ces_weight = objective_weights["synthetic_ces"]
            clean_retention_weight = objective_weights["retention"]
            human_weight = objective_weights["human_ces"]
            noop_weight = objective_weights["noop_ces"]
            first_decision_weight = objective_weights["first_decision"]
            rl_weight = objective_weights["rl"]
            if stream_sampler is None:
                prompt_rows = cyclic_prompt_rows(
                    rows, step=step, prompts_per_step=args.prompts_per_step
                )
            else:
                prompt_rows = [
                    stream_pair_to_row(pair, token_ops)
                    for pair in stream_sampler.next_batch(
                        args.prompts_per_step
                    )
                ]
            all_reward_details: list[Any] = []
            all_rollouts: list[PolicyRollout] = []
            reward_groups: list[torch.Tensor] = []
            resample_attempt_counts: list[int] = []
            total_decisions = 0
            rl_loss_value = 0.0
            nonzero_advantage_prompts = 0

            # Each prompt has its own RLOO group. Replay and backward are
            # sequential, so activations are released while all prompts still
            # see the identical pre-update policy parameters.
            for prompt_index, row in enumerate(prompt_rows):
                prompt_ordinal = step * args.prompts_per_step + prompt_index
                resample_attempts = 0
                while True:
                    model.eval()
                    with autocast_context(device_type):
                        rollouts, _masks = engine.generate_rollout_batch(
                            [int(token) for token in row["prompt_tokens"]],
                            tix=[int(pointer) for pointer in row["prompt_positions"]],
                            num_samples=local_group_size,
                            max_tokens=args.max_edit_tokens,
                            temperature=args.temperature,
                            pointer_temperature=pointer_temperature,
                            closure_bias=args.closure_bias,
                            closure_bias_min_payload_tokens=(
                                args.closure_bias_min_payload_tokens
                            ),
                            max_payload_tokens=args.max_payload_tokens,
                            max_edit_ops=args.max_edit_ops,
                            stop_after_edit_end=True,
                            seed=(
                                hash((
                                    args.seed, prompt_ordinal,
                                    resample_attempts, rank,
                                ))
                                & 0x7FFFFFFF
                            ),
                        )
                    if args.reward_mode in {"target_aware", "provenance", "local_verifier"}:
                        reward_function = (
                            target_aware_recovery_reward
                            if args.reward_mode == "target_aware"
                            else provenance_program_reward
                            if args.reward_mode == "provenance"
                            else local_repair_verifier_reward
                        )
                        reward_details = [
                            reward_function(
                                rollout,
                                token_ops,
                                row["source_tokens"],
                                row["target_tokens"],
                                invalid_reward=args.invalid_reward,
                                exact_bonus=args.exact_bonus,
                                gold_edits=row.get("gold_edits", ()),
                                corrupt_noop_penalty=args.corrupt_noop_penalty,
                                negative_reward_scale=args.negative_reward_scale,
                                **(
                                    {
                                        "edit_penalty": args.edit_penalty,
                                        "character_reward_weight": args.character_reward_weight,
                                        "operation_local_reward_weight": args.operation_local_reward_weight,
                                        "local_context_tokens": args.local_context_tokens,
                                    }
                                    if args.reward_mode == "target_aware"
                                    else {
                                        "extra_op_penalty": args.provenance_extra_op_penalty,
                                    }
                                    if args.reward_mode == "provenance"
                                    else {
                                        "outside_edit_penalty": args.verifier_outside_edit_penalty,
                                        "anchor_chars": args.verifier_anchor_chars,
                                    }
                                ),
                            )
                            for rollout in rollouts
                        ]
                        if args.likelihood_aux_weight > 0:
                            assert scorer is not None
                            sequences = [list(map(int, row["source_tokens"]))]
                            sequences.extend(
                                list(detail.resolved_tokens)
                                for detail in reward_details
                                if detail.valid
                                and detail.resolved_tokens is not None
                            )
                            scores = score_unique_sequences(
                                scorer,
                                sequences,
                                device=device,
                                batch_size=local_group_size + 1,
                            )
                            source_score = scores[
                                _tokens_hash(row["source_tokens"])
                            ]
                            if source_score is None:
                                raise ValueError(
                                    "Training source is not scoreable by frozen LM"
                                )
                            reward_details = [
                                add_likelihood_auxiliary(
                                    detail,
                                    target_blind_likelihood_reward(
                                        rollout,
                                        token_ops,
                                        row["source_tokens"],
                                        source_avg_logprob=source_score,
                                        candidate_avg_logprob=(
                                            scores.get(_tokens_hash(
                                                detail.resolved_tokens
                                            ))
                                            if detail.resolved_tokens is not None
                                            else None
                                        ),
                                        invalid_reward=args.invalid_reward,
                                        likelihood_scale=(
                                            args.likelihood_reward_scale
                                        ),
                                    ),
                                    weight=args.likelihood_aux_weight,
                                )
                                for rollout, detail in zip(
                                    rollouts, reward_details
                                )
                            ]
                    else:
                        assert scorer is not None
                        provisional = [
                            target_blind_likelihood_reward(
                                rollout,
                                token_ops,
                                row["source_tokens"],
                                source_avg_logprob=0.0,
                                candidate_avg_logprob=0.0,
                                invalid_reward=args.invalid_reward,
                                likelihood_scale=args.likelihood_reward_scale,
                                edit_penalty=args.edit_penalty,
                            )
                            for rollout in rollouts
                        ]
                        sequences = [list(map(int, row["source_tokens"]))]
                        sequences.extend(
                            list(detail.resolved_tokens)
                            for detail in provisional
                            if detail.valid
                            and detail.resolved_tokens is not None
                        )
                        scores = score_unique_sequences(
                            scorer,
                            sequences,
                            device=device,
                            batch_size=local_group_size + 1,
                        )
                        source_score = scores[_tokens_hash(row["source_tokens"])]
                        if source_score is None:
                            raise ValueError(
                                "Training source is not scoreable by frozen LM"
                            )
                        reward_details = [
                            (
                                target_blind_likelihood_reward(
                                    rollout,
                                    token_ops,
                                    row["source_tokens"],
                                    source_avg_logprob=source_score,
                                    candidate_avg_logprob=(
                                        scores.get(_tokens_hash(
                                            detail.resolved_tokens
                                        ))
                                        if detail.resolved_tokens is not None
                                        else None
                                    ),
                                    invalid_reward=args.invalid_reward,
                                    likelihood_scale=(
                                        args.likelihood_reward_scale
                                    ),
                                    edit_penalty=args.edit_penalty,
                                )
                                if detail.valid
                                else detail
                            )
                            for rollout, detail in zip(rollouts, provisional)
                        ]
                    local_rewards = torch.tensor(
                        [detail.reward for detail in reward_details],
                        dtype=torch.float32,
                        device=device,
                    )
                    rewards = gather_rollout_rewards(local_rewards, world_size)
                    if (
                        float(rewards.max()) != float(rewards.min())
                        or resample_attempts >= args.max_resample_attempts
                    ):
                        break
                    resample_attempts += 1

                global_advantages = group_advantages(
                    rewards, method=args.advantage
                )
                local_offset = rank * local_group_size
                advantages = global_advantages[
                    local_offset : local_offset + local_group_size
                ]
                local_decisions = sum(
                    rollout_decision_count(item) for item in rollouts
                )
                prompt_decisions = int(reduce_scalar(
                    local_decisions,
                    device=device,
                    world_size=world_size,
                    reduction="sum",
                ))
                if prompt_decisions == 0:
                    raise RuntimeError(
                        "Rollout prompt group contained no sampled decisions"
                    )
                if float(rewards.max()) != float(rewards.min()):
                    nonzero_advantage_prompts += 1

                model.eval()
                for rollout, advantage in zip(rollouts, advantages):
                    with autocast_context(device_type):
                        replayed = replay_policy_logprobs(
                            model, tokenizer, rollout
                        )
                        micro_loss = (
                            -rl_weight * advantage.detach() * replayed.total
                            / prompt_decisions
                            * world_size
                            / args.prompts_per_step
                        )
                    micro_loss.backward()
                    rl_loss_value += float(micro_loss.detach())

                all_reward_details.extend(reward_details)
                all_rollouts.extend(rollouts)
                reward_groups.append(rewards)
                resample_attempt_counts.append(resample_attempts)
                total_decisions += prompt_decisions

            rewards = torch.cat(reward_groups)

            reward_by_prompt_kind: dict[str, float] = {}
            valid_by_prompt_kind: dict[str, float] = {}
            exact_by_prompt_kind: dict[str, float] = {}
            noop_by_prompt_kind: dict[str, float] = {}
            prompt_kinds = sorted({
                str(row.get("stream_kind", "manifest"))
                for row in prompt_rows
            })
            for kind in prompt_kinds:
                prompt_indices = [
                    index
                    for index, row in enumerate(prompt_rows)
                    if str(row.get("stream_kind", "manifest")) == kind
                ]
                kind_rewards = torch.cat([
                    reward_groups[index] for index in prompt_indices
                ])
                reward_by_prompt_kind[kind] = float(kind_rewards.mean())
                local_details = [
                    detail
                    for index in prompt_indices
                    for detail in all_reward_details[
                        index * local_group_size : (index + 1) * local_group_size
                    ]
                ]
                global_kind_rollouts = len(prompt_indices) * args.group_size
                valid_by_prompt_kind[kind] = reduce_scalar(
                    sum(detail.valid for detail in local_details),
                    device=device,
                    world_size=world_size,
                    reduction="sum",
                ) / global_kind_rollouts
                exact_by_prompt_kind[kind] = reduce_scalar(
                    sum(detail.exact_target for detail in local_details),
                    device=device,
                    world_size=world_size,
                    reduction="sum",
                ) / global_kind_rollouts
                local_noops = 0
                for index in prompt_indices:
                    source_signature = tuple(map(int, prompt_rows[index]["source_tokens"]))
                    for detail in all_reward_details[
                        index * local_group_size : (index + 1) * local_group_size
                    ]:
                        local_noops += int(
                            detail.valid
                            and detail.resolved_tokens is not None
                            and tuple(detail.resolved_tokens) == source_signature
                        )
                noop_by_prompt_kind[kind] = reduce_scalar(
                    local_noops,
                    device=device,
                    world_size=world_size,
                    reduction="sum",
                ) / global_kind_rollouts

            ces_supervised_loss_value = 0.0
            human_supervised_loss_value = 0.0
            noop_supervised_loss_value = 0.0
            first_decision_loss_value = 0.0
            clean_ar_loss_value = 0.0
            clean_kl_loss_value = 0.0
            if (
                args.supervised_replay_fraction > 0
                or human_weight > 0
                or noop_weight > 0
                or first_decision_weight > 0
            ):
                model.train()
                if ces_weight > 0:
                    ces_losses = []
                    for row in prompt_rows:
                        with autocast_context(device_type):
                            ces_supervised_loss = completion_only_supervised_loss(
                                model,
                                token_ops,
                                row,
                                device=device,
                                position_buffer_size=args.position_buffer_size,
                            )
                            weighted_ces_loss = (
                                ces_weight
                                * ces_supervised_loss
                                / args.prompts_per_step
                            )
                        weighted_ces_loss.backward()
                        ces_losses.append(float(ces_supervised_loss.detach()))
                    ces_supervised_loss_value = sum(ces_losses) / len(ces_losses)
                if human_weight > 0:
                    human_losses = []
                    for human_row in distributed_cyclic_component_rows(
                        human_rows,
                        step=step,
                        global_count=args.human_examples_per_step,
                        rank=rank,
                        world_size=world_size,
                    ):
                        with autocast_context(device_type):
                            human_supervised_loss = completion_only_supervised_loss(
                                model,
                                token_ops,
                                human_row,
                                device=device,
                                position_buffer_size=args.position_buffer_size,
                            )
                            weighted_human_loss = (
                                human_weight
                                * human_supervised_loss
                                / local_human_examples
                            )
                        weighted_human_loss.backward()
                        human_losses.append(float(human_supervised_loss.detach()))
                    human_supervised_loss_value = sum(human_losses) / len(human_losses)
                if noop_weight > 0:
                    noop_losses = []
                    for noop_row in distributed_cyclic_component_rows(
                        noop_rows,
                        step=step,
                        global_count=args.noop_examples_per_step,
                        rank=rank,
                        world_size=world_size,
                    ):
                        with autocast_context(device_type):
                            noop_supervised_loss = completion_only_supervised_loss(
                                model,
                                token_ops,
                                noop_row,
                                device=device,
                                position_buffer_size=args.position_buffer_size,
                            )
                            weighted_noop_loss = (
                                noop_weight
                                * noop_supervised_loss
                                / local_noop_examples
                            )
                        weighted_noop_loss.backward()
                        noop_losses.append(float(noop_supervised_loss.detach()))
                    noop_supervised_loss_value = sum(noop_losses) / len(noop_losses)
                if first_decision_weight > 0:
                    decision_losses = []
                    edit_pair_weight, clean_pair_weight = (
                        first_decision_pair_weights(
                            args.first_decision_edit_prior
                        )
                    )
                    for row in prompt_rows:
                        clean_prompt_tokens, clean_prompt_positions = (
                            identity_prompt_from_target(
                                token_ops,
                                [int(token) for token in row["target_tokens"]],
                            )
                        )
                        paired_prompts = (
                            (
                                [int(token) for token in row["prompt_tokens"]],
                                [int(position) for position in row["prompt_positions"]],
                                row["source_tokens"] != row["target_tokens"],
                                edit_pair_weight,
                            ),
                            (
                                clean_prompt_tokens,
                                clean_prompt_positions,
                                False,
                                clean_pair_weight,
                            ),
                        )
                        for (
                            gate_tokens,
                            gate_positions,
                            should_edit,
                            pair_weight,
                        ) in paired_prompts:
                            with autocast_context(device_type):
                                gate_loss = first_decision_loss(
                                    model,
                                    token_ops,
                                    gate_tokens,
                                    gate_positions,
                                    should_edit=bool(should_edit),
                                    device=device,
                                )
                                weighted_gate_loss = (
                                    first_decision_weight
                                    * pair_weight
                                    * gate_loss
                                    / (2 * args.prompts_per_step)
                                )
                            weighted_gate_loss.backward()
                            decision_losses.append(float(gate_loss.detach()))
                    first_decision_loss_value = (
                        sum(decision_losses) / len(decision_losses)
                    )
                if (
                    clean_retention_weight > 0
                    and args.retention_mode in {"paired_mixed", "independent_mixed"}
                ):
                    clean_losses = []
                    for row in prompt_rows:
                        with autocast_context(device_type):
                            if args.retention_mode == "paired_mixed":
                                clean_ar_loss = clean_ar_completion_loss(
                                    model,
                                    row,
                                    device=device,
                                    prefix_fraction=(
                                        args.clean_ar_prefix_fraction
                                    ),
                                )
                            else:
                                assert clean_stream is not None
                                clean_ar_loss = clean_ar_sequence_loss(
                                    model,
                                    clean_stream.next_tokens(),
                                    device=device,
                                    prefix_fraction=(
                                        args.clean_ar_prefix_fraction
                                    ),
                                )
                            weighted_clean_ar_loss = (
                                clean_retention_weight
                                * clean_ar_loss
                                / args.prompts_per_step
                            )
                        weighted_clean_ar_loss.backward()
                        clean_losses.append(float(clean_ar_loss.detach()))
                    clean_ar_loss_value = sum(clean_losses) / len(clean_losses)
                if clean_retention_weight > 0 and args.retention_mode == "clean_kl":
                    assert clean_stream is not None and kl_reference is not None
                    kl_losses = []
                    for _row in prompt_rows:
                        with autocast_context(device_type):
                            kl_loss = clean_token_kl_loss(
                                model,
                                kl_reference,
                                clean_stream.next_tokens(),
                                device=device,
                            )
                            weighted_kl_loss = (
                                clean_retention_weight
                                * kl_loss
                                / args.prompts_per_step
                            )
                        weighted_kl_loss.backward()
                        kl_losses.append(float(kl_loss.detach()))
                    clean_kl_loss_value = sum(kl_losses) / len(kl_losses)

            missing_gradient_count = materialize_missing_gradients(model)
            grad_norm = distributed_clip_grad_norm_(
                model,
                args.grad_clip,
                device=device,
                world_size=world_size,
            )
            if (
                device_type == "cuda"
                and world_size > 1
                and args.low_memory_distributed_optimizer
            ):
                torch.cuda.empty_cache()
            optimizer.step()

            clean_grad_norm_value = 0.0
            optimizer_updates = 1
            if (
                args.retention_mode == "alternating_ar"
                and (step + 1) % args.clean_update_every == 0
            ):
                assert clean_stream is not None
                optimizer.zero_grad(set_to_none=True)
                model.train()
                with autocast_context(device_type):
                    alternating_clean_loss = clean_ar_sequence_loss(
                        model,
                        clean_stream.next_tokens(),
                        device=device,
                        prefix_fraction=args.clean_ar_prefix_fraction,
                    )
                alternating_clean_loss.backward()
                materialize_missing_gradients(model)
                clean_grad_norm = distributed_clip_grad_norm_(
                    model,
                    args.grad_clip,
                    device=device,
                    world_size=world_size,
                )
                if (
                    device_type == "cuda"
                    and world_size > 1
                    and args.low_memory_distributed_optimizer
                ):
                    torch.cuda.empty_cache()
                original_lrs = [group["lr"] for group in optimizer.param_groups]
                for group in optimizer.param_groups:
                    group["lr"] *= args.clean_step_lr_scale
                optimizer.step()
                for group, original_lr in zip(optimizer.param_groups, original_lrs):
                    group["lr"] = original_lr
                clean_ar_loss_value = float(alternating_clean_loss.detach())
                clean_grad_norm_value = float(clean_grad_norm)
                optimizer_updates += 1

            global_valid_count = reduce_scalar(
                sum(detail.valid for detail in all_reward_details),
                device=device,
                world_size=world_size,
                reduction="sum",
            )
            global_exact_count = reduce_scalar(
                sum(detail.exact_target for detail in all_reward_details),
                device=device,
                world_size=world_size,
                reduction="sum",
            )
            local_noop_count = 0
            for prompt_index, row in enumerate(prompt_rows):
                source_signature = tuple(map(int, row["source_tokens"]))
                for detail in all_reward_details[
                    prompt_index * local_group_size :
                    (prompt_index + 1) * local_group_size
                ]:
                    local_noop_count += int(
                        detail.valid
                        and detail.resolved_tokens is not None
                        and tuple(detail.resolved_tokens) == source_signature
                    )
            global_noop_count = reduce_scalar(
                local_noop_count,
                device=device,
                world_size=world_size,
                reduction="sum",
            )
            total_rollouts = args.prompts_per_step * args.group_size
            record = {
                "step": step + 1,
                "benchmark_id": (
                    prompt_rows[0].get("benchmark_id")
                    if args.prompts_per_step == 1
                    else None
                ),
                "benchmark_ids": [
                    row.get("benchmark_id") for row in prompt_rows
                ],
                "prompt_kind_counts": dict(sorted(Counter(
                    str(row.get("stream_kind", "manifest"))
                    for row in prompt_rows
                ).items())),
                "stream_state": (
                    stream_sampler.state_dict()
                    if stream_sampler is not None
                    else None
                ),
                "prompts_per_step": args.prompts_per_step,
                "prompt_exposures": (step + 1) * args.prompts_per_step,
                "rollouts_per_step": total_rollouts,
                "reward_mean": float(rewards.mean()),
                "reward_min": float(rewards.min()),
                "reward_max": float(rewards.max()),
                "reward_std": float(rewards.std(unbiased=False)),
                "reward_by_prompt_kind": reward_by_prompt_kind,
                "valid_by_prompt_kind": valid_by_prompt_kind,
                "exact_by_prompt_kind": exact_by_prompt_kind,
                "noop_by_prompt_kind": noop_by_prompt_kind,
                "resample_attempts": int(reduce_scalar(
                    max(resample_attempt_counts, default=0),
                    device=device,
                    world_size=world_size,
                    reduction="max",
                )),
                "resample_attempts_total": int(reduce_scalar(
                    sum(resample_attempt_counts),
                    device=device,
                    world_size=world_size,
                    reduction="max",
                )),
                "nonzero_advantage_prompt_fraction": (
                    nonzero_advantage_prompts / args.prompts_per_step
                ),
                "valid_fraction": global_valid_count / total_rollouts,
                "exact_fraction": global_exact_count / total_rollouts,
                "noop_fraction": global_noop_count / total_rollouts,
                "mean_likelihood_delta": optional_detail_mean(
                    all_reward_details,
                    "likelihood_delta",
                    device=device,
                    world_size=world_size,
                ),
                "mean_token_improvement": optional_detail_mean(
                    all_reward_details,
                    "token_improvement",
                    device=device,
                    world_size=world_size,
                ),
                "mean_character_improvement": optional_detail_mean(
                    all_reward_details,
                    "character_improvement",
                    device=device,
                    world_size=world_size,
                ),
                "mean_operation_local_improvement": optional_detail_mean(
                    all_reward_details,
                    "operation_local_improvement",
                    device=device,
                    world_size=world_size,
                ),
                "mean_provenance_precision": optional_detail_mean(
                    all_reward_details,
                    "provenance_precision",
                    device=device,
                    world_size=world_size,
                ),
                "mean_provenance_recall": optional_detail_mean(
                    all_reward_details,
                    "provenance_recall",
                    device=device,
                    world_size=world_size,
                ),
                "mean_verifier_pass_rate": optional_detail_mean(
                    all_reward_details,
                    "verifier_pass_rate",
                    device=device,
                    world_size=world_size,
                ),
                "mean_outside_edit_fraction": optional_detail_mean(
                    all_reward_details,
                    "outside_edit_fraction",
                    device=device,
                    world_size=world_size,
                ),
                "mean_generated_tokens": reduce_scalar(
                    sum(len(item.actions) for item in all_rollouts),
                    device=device,
                    world_size=world_size,
                    reduction="sum",
                ) / total_rollouts,
                "policy_decisions": total_decisions,
                "rl_loss": reduce_scalar(
                    rl_loss_value,
                    device=device,
                    world_size=world_size,
                    reduction="mean",
                ),
                "ces_supervised_loss": reduce_scalar(
                    ces_supervised_loss_value,
                    device=device,
                    world_size=world_size,
                    reduction="mean",
                ),
                "human_supervised_loss": reduce_scalar(
                    human_supervised_loss_value,
                    device=device,
                    world_size=world_size,
                    reduction="mean",
                ),
                "noop_supervised_loss": reduce_scalar(
                    noop_supervised_loss_value,
                    device=device,
                    world_size=world_size,
                    reduction="mean",
                ),
                "first_decision_loss": reduce_scalar(
                    first_decision_loss_value,
                    device=device,
                    world_size=world_size,
                    reduction="mean",
                ),
                "clean_ar_loss": reduce_scalar(
                    clean_ar_loss_value,
                    device=device,
                    world_size=world_size,
                    reduction="mean",
                ),
                "clean_kl_loss": reduce_scalar(
                    clean_kl_loss_value,
                    device=device,
                    world_size=world_size,
                    reduction="mean",
                ),
                "grad_norm": float(grad_norm),
                "clean_grad_norm": reduce_scalar(
                    clean_grad_norm_value,
                    device=device,
                    world_size=world_size,
                    reduction="max",
                ),
                "optimizer_updates": optimizer_updates,
                "clean_documents_seen": int(reduce_scalar(
                    clean_stream.documents_seen if clean_stream is not None else 0,
                    device=device,
                    world_size=world_size,
                    reduction="sum",
                )),
                "missing_gradient_count": int(reduce_scalar(
                    missing_gradient_count,
                    device=device,
                    world_size=world_size,
                    reduction="max",
                )),
                "peak_memory_allocated_gib": reduce_scalar(
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device_type == "cuda" else 0.0,
                    device=device,
                    world_size=world_size,
                    reduction="max",
                ),
                "peak_memory_reserved_gib": reduce_scalar(
                    torch.cuda.max_memory_reserved(device) / 1024**3
                    if device_type == "cuda" else 0.0,
                    device=device,
                    world_size=world_size,
                    reduction="max",
                ),
                "elapsed_seconds": reduce_scalar(
                    time.perf_counter() - started,
                    device=device,
                    world_size=world_size,
                    reduction="max",
                ),
            }
            if rank == 0:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)

            should_save = (
                not args.no_save
                and (
                    (args.save_every > 0 and (step + 1) % args.save_every == 0)
                    or (step + 1) in set(args.save_steps)
                    or step + 1 == args.steps
                )
            )
            if should_save:
                save_checkpoint(
                    str(output_dir),
                    step + 1,
                    model.state_dict(),
                    None if args.save_model_only else optimizer.state_dict(),
                    checkpoint_metadata(
                        original_metadata,
                        args,
                        posttraining_step=step + 1,
                        tokenization=tokenization,
                    ),
                    rank=rank,
                )
                if ddp:
                    dist.barrier()
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
