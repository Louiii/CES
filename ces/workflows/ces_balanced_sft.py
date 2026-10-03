"""Balanced human/no-op CES SFT cold start for subsequent joint RL.

Each optimizer update averages several human edit programs and matched clean
no-op programs before applying component weights. Synthetic CES replay and
clean AR completion keep controlled editing and ordinary language behavior in
the mixture. The trainer is deliberately separate from ``ces.workflows.ces_sft`` so
the earlier matched-compute baseline remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from nanochat.causal_serialisation_extended import TokensOperations
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir
from ces.workflows.benchmark_tokenization import prepare_benchmark_for_tokenizer
from ces.workflows.ces_rl import (
    autocast_context,
    clean_ar_completion_loss,
    completion_only_supervised_loss,
    human_identity_rows,
    materialize_missing_gradients,
    read_jsonl,
    reject_heldout_rows,
    distributed_clip_grad_norm_,
    reduce_scalar,
)
from ces.workflows.oracle_ces_eval import _text_pair_rows


COMPONENTS = ("human", "noop", "synthetic", "clean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--human-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--model-step", type=int, required=True)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device-type", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--human-weight", type=float, default=0.4)
    parser.add_argument("--noop-weight", type=float, default=0.4)
    parser.add_argument("--synthetic-weight", type=float, default=0.1)
    parser.add_argument("--clean-weight", type=float, default=0.1)
    parser.add_argument("--human-examples-per-step", type=int, default=4)
    parser.add_argument("--noop-examples-per-step", type=int, default=4)
    parser.add_argument("--synthetic-examples-per-step", type=int, default=1)
    parser.add_argument("--clean-examples-per-step", type=int, default=1)
    parser.add_argument("--clean-ar-prefix-fraction", type=float, default=0.5)
    parser.add_argument("--position-buffer-size", type=int, default=128)
    parser.add_argument("--lr-scale", type=float, default=0.005)
    parser.add_argument("--embedding-lr", type=float, default=0.2)
    parser.add_argument("--unembedding-lr", type=float, default=0.004)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument(
        "--low-memory-distributed-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable communication overlap buffers in the distributed optimizer.",
    )
    parser.add_argument(
        "--distributed-muon-group-size",
        type=int,
        default=0,
        help="Maximum same-shaped matrices per distributed Muon group; 0 keeps the default.",
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--log-jsonl", type=Path, default=None)
    return parser.parse_args()


def balanced_objective_weights(**values: float) -> dict[str, float]:
    unknown = set(values) - set(COMPONENTS)
    if unknown or set(values) != set(COMPONENTS):
        raise ValueError(f"Expected weights for {COMPONENTS}, received {sorted(values)}")
    if any(value < 0 for value in values.values()):
        raise ValueError(f"Mixture weights must be non-negative: {values}")
    total = sum(values.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Mixture weights must sum to one, received {total}: {values}")
    return dict(values)


def cyclic_examples(rows: list[dict[str, Any]], *, step: int, count: int) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot sample from an empty component")
    if count < 0:
        raise ValueError("Examples per step must be non-negative")
    start = step * count
    return [rows[(start + offset) % len(rows)] for offset in range(count)]


def distributed_cyclic_examples(
    rows: list[dict[str, Any]],
    *,
    step: int,
    global_count: int,
    rank: int,
    world_size: int,
) -> list[dict[str, Any]]:
    """Return a disjoint rank-local slice of one deterministic global batch."""

    if global_count % world_size != 0:
        raise ValueError(
            f"Global examples per step ({global_count}) must be divisible by "
            f"world size ({world_size})"
        )
    local_count = global_count // world_size
    if not rows:
        raise ValueError("Cannot sample from an empty component")
    start = step * global_count + rank * local_count
    return [rows[(start + offset) % len(rows)] for offset in range(local_count)]


def row_fits_context(row: dict[str, Any], max_seq_len: int) -> bool:
    serialised = len(row["prompt_tokens"]) + len(row["gold_program_tokens"])
    clean = len(row["target_tokens"])
    return serialised - 1 <= max_seq_len and clean - 1 <= max_seq_len


def component_mean_backward(
    model,
    token_ops: TokensOperations,
    rows: list[dict[str, Any]],
    *,
    weight: float,
    device: torch.device,
    device_type: str,
    position_buffer_size: int,
    clean_ar: bool = False,
    clean_ar_prefix_fraction: float = 0.5,
) -> float:
    if not rows or weight == 0:
        return 0.0
    values = []
    scale = weight / len(rows)
    for row in rows:
        with autocast_context(device_type):
            if clean_ar:
                loss = clean_ar_completion_loss(
                    model,
                    row,
                    device=device,
                    prefix_fraction=clean_ar_prefix_fraction,
                )
            else:
                loss = completion_only_supervised_loss(
                    model,
                    token_ops,
                    row,
                    device=device,
                    position_buffer_size=position_buffer_size,
                )
            weighted = scale * loss
        weighted.backward()
        values.append(float(loss.detach()))
    return sum(values) / len(values)


def checkpoint_metadata(
    original: dict[str, Any],
    args: argparse.Namespace,
    *,
    step: int,
    data_summary: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(original)
    metadata.update({
        "step": step,
        "phase": "ces_balanced_sft_posttraining",
        "initial_model_tag": args.model_tag,
        "initial_model_step": args.model_step,
        "user_config": vars(args) | {
            "controlled_manifest_jsonl": str(args.controlled_manifest_jsonl),
            "human_manifest_jsonl": str(args.human_manifest_jsonl),
            "output_dir": str(args.output_dir) if args.output_dir else None,
            "log_jsonl": str(args.log_jsonl) if args.log_jsonl else None,
        },
        "balanced_sft_data": data_summary,
    })
    return metadata


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not 0.0 <= args.clean_ar_prefix_fraction < 1.0:
        raise ValueError("--clean-ar-prefix-fraction must be in [0, 1)")
    weights = balanced_objective_weights(
        human=args.human_weight,
        noop=args.noop_weight,
        synthetic=args.synthetic_weight,
        clean=args.clean_weight,
    )
    counts = {
        "human": args.human_examples_per_step,
        "noop": args.noop_examples_per_step,
        "synthetic": args.synthetic_examples_per_step,
        "clean": args.clean_examples_per_step,
    }
    for component in COMPONENTS:
        if counts[component] < 0:
            raise ValueError(f"{component} examples per step must be non-negative")
        if weights[component] > 0 and counts[component] == 0:
            raise ValueError(f"Positive {component} weight requires examples per step")

    device_type = args.device_type or autodetect_device_type()
    ddp, rank, _local_rank, world_size, device = compute_init(device_type, seed=args.seed)
    for component in COMPONENTS:
        if counts[component] % world_size != 0:
            raise ValueError(
                f"--{component}-examples-per-step={counts[component]} must be "
                f"divisible by world size {world_size}"
            )
    local_counts = {name: count // world_size for name, count in counts.items()}
    model, tokenizer, original_metadata = load_model(
        "base", device, phase="train", model_tag=args.model_tag, step=args.model_step
    )
    token_ops = TokensOperations(getattr(tokenizer, "enc", tokenizer))
    max_seq_len = int(original_metadata.get("model_config", {}).get("sequence_len", 2048))

    raw_controlled = read_jsonl(args.controlled_manifest_jsonl)
    raw_human = read_jsonl(args.human_manifest_jsonl)
    reject_heldout_rows(raw_controlled)
    reject_heldout_rows(raw_human)
    synthetic_rows, synthetic_tokenization = prepare_benchmark_for_tokenizer(
        raw_controlled, tokenizer=tokenizer, token_ops=token_ops
    )
    human_rows, human_tokenization = _text_pair_rows(
        raw_human, tokenizer=tokenizer, token_ops=token_ops
    )
    noop_rows, noop_tokenization = _text_pair_rows(
        human_identity_rows(raw_human), tokenizer=tokenizer, token_ops=token_ops
    )
    raw_sizes = {
        "human": len(human_rows),
        "noop": len(noop_rows),
        "synthetic": len(synthetic_rows),
    }
    human_rows = [row for row in human_rows if row_fits_context(row, max_seq_len)]
    noop_rows = [row for row in noop_rows if row_fits_context(row, max_seq_len)]
    synthetic_rows = [row for row in synthetic_rows if row_fits_context(row, max_seq_len)]
    pools = {
        "human": human_rows,
        "noop": noop_rows,
        "synthetic": synthetic_rows,
        # Keep an independent ordering: clean AR and CES no-op consume the
        # same source texts but should not be locked to identical examples.
        "clean": list(noop_rows),
    }
    for offset, component in enumerate(COMPONENTS):
        random.Random(args.seed + 10_000 * (offset + 1)).shuffle(pools[component])
    data_summary = {
        "max_seq_len": max_seq_len,
        "raw_rows": raw_sizes,
        "usable_rows": {name: len(pools[name]) for name in ("human", "noop", "synthetic")},
        "weights": weights,
        "examples_per_step": counts,
        "local_examples_per_step": local_counts,
        "world_size": world_size,
        "planned_exposures": {name: args.steps * count for name, count in counts.items()},
        "human_tokenization": human_tokenization,
        "noop_tokenization": noop_tokenization,
        "synthetic_tokenization": synthetic_tokenization,
    }
    if any(weights[name] > 0 and not pools[name] for name in COMPONENTS):
        raise ValueError(f"A weighted component has no usable rows: {data_summary}")

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
    output_dir = args.output_dir or Path(get_base_dir()) / "base_checkpoints" / args.output_tag
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()
    log_path = args.log_jsonl or output_dir / "train.jsonl"
    if rank == 0:
        print(json.dumps({
            "device": str(device),
            "model_tag": args.model_tag,
            "model_step": args.model_step,
            "output_dir": str(output_dir),
            "steps": args.steps,
            "data": data_summary,
        }, indent=2), flush=True)

    try:
        for step in range(args.steps):
            started = time.perf_counter()
            if device_type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            selected = {
                name: distributed_cyclic_examples(
                    pools[name], step=step, global_count=counts[name],
                    rank=rank, world_size=world_size,
                )
                for name in COMPONENTS
            }
            losses = {
                "human": component_mean_backward(
                    model, token_ops, selected["human"], weight=weights["human"],
                    device=device, device_type=device_type,
                    position_buffer_size=args.position_buffer_size,
                ),
                "noop": component_mean_backward(
                    model, token_ops, selected["noop"], weight=weights["noop"],
                    device=device, device_type=device_type,
                    position_buffer_size=args.position_buffer_size,
                ),
                "synthetic": component_mean_backward(
                    model, token_ops, selected["synthetic"], weight=weights["synthetic"],
                    device=device, device_type=device_type,
                    position_buffer_size=args.position_buffer_size,
                ),
                "clean": component_mean_backward(
                    model, token_ops, selected["clean"], weight=weights["clean"],
                    device=device, device_type=device_type,
                    position_buffer_size=args.position_buffer_size,
                    clean_ar=True,
                    clean_ar_prefix_fraction=args.clean_ar_prefix_fraction,
                ),
            }
            missing_gradient_count = materialize_missing_gradients(model)
            grad_norm = distributed_clip_grad_norm_(
                model, args.grad_clip, device=device, world_size=world_size
            )
            if (
                device_type == "cuda"
                and world_size > 1
                and args.low_memory_distributed_optimizer
            ):
                # Variable-length microbatches leave cached activation blocks
                # resident while the sharded Muon step needs a large temporary.
                # Releasing only unused cache avoids allocator-fragmentation OOMs.
                torch.cuda.empty_cache()
            optimizer.step()
            losses = {
                name: reduce_scalar(value, device=device, world_size=world_size)
                for name, value in losses.items()
            }
            weighted_loss = sum(weights[name] * losses[name] for name in COMPONENTS)
            record = {
                "step": step + 1,
                "loss": weighted_loss,
                **{f"{name}_loss": losses[name] for name in COMPONENTS},
                "grad_norm": float(grad_norm),
                "missing_gradient_count": int(reduce_scalar(
                    missing_gradient_count, device=device, world_size=world_size,
                    reduction="max",
                )),
                "peak_memory_allocated_gib": reduce_scalar(
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device_type == "cuda" else 0.0,
                    device=device, world_size=world_size, reduction="max",
                ),
                "peak_memory_reserved_gib": reduce_scalar(
                    torch.cuda.max_memory_reserved(device) / 1024**3
                    if device_type == "cuda" else 0.0,
                    device=device, world_size=world_size, reduction="max",
                ),
                "elapsed_seconds": reduce_scalar(
                    time.perf_counter() - started, device=device,
                    world_size=world_size, reduction="max",
                ),
            }
            if rank == 0:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                print(json.dumps(record, separators=(",", ":")), flush=True)
            should_save = not args.no_save and (
                (args.save_every > 0 and (step + 1) % args.save_every == 0)
                or step + 1 == args.steps
            )
            if should_save:
                save_checkpoint(
                    str(output_dir),
                    step + 1,
                    model.state_dict(),
                    optimizer.state_dict() if args.save_optimizer else None,
                    checkpoint_metadata(
                        original_metadata, args, step=step + 1, data_summary=data_summary
                    ),
                    rank=rank,
                )
                if ddp:
                    dist.barrier()
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
