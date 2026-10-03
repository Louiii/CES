"""Matched completion-only CES continued-SFT baseline.

The trainer uses the same controlled source/target rows and optimizer family as
``ces.workflows.ces_rl``. Gold CES program actions receive completion-only loss; a
configurable clean-AR suffix loss preserves ordinary language modeling. This
is the update-matched behavior-cloning control for CES RL post-training.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from nanochat.causal_serialisation_extended import TokensOperations
from nanochat.checkpoint_manager import build_model, load_model, save_checkpoint
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir
from ces.workflows.benchmark_tokenization import prepare_benchmark_for_tokenizer
from ces.workflows.ces_rl import (
    autocast_context,
    clean_ar_completion_loss,
    completion_only_supervised_loss,
    materialize_missing_gradients,
    read_jsonl,
    reject_heldout_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--model-step", type=int, required=True)
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device-type", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--clean-replay-fraction", type=float, default=0.125)
    parser.add_argument("--clean-ar-prefix-fraction", type=float, default=0.5)
    parser.add_argument("--position-buffer-size", type=int, default=128)
    parser.add_argument("--lr-scale", type=float, default=0.01)
    parser.add_argument("--embedding-lr", type=float, default=0.2)
    parser.add_argument("--unembedding-lr", type=float, default=0.004)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--log-jsonl", type=Path, default=None)
    return parser.parse_args()


def matched_sft_loss(
    ces_loss: torch.Tensor,
    clean_ar_loss: torch.Tensor,
    *,
    clean_replay_fraction: float,
) -> torch.Tensor:
    if not 0.0 <= clean_replay_fraction <= 1.0:
        raise ValueError("clean_replay_fraction must be in [0, 1]")
    return (
        (1.0 - clean_replay_fraction) * ces_loss
        + clean_replay_fraction * clean_ar_loss
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
        "phase": "ces_sft_posttraining",
        "initial_model_tag": args.model_tag,
        "initial_model_step": args.model_step,
        "user_config": vars(args) | {
            "manifest_jsonl": str(args.manifest_jsonl),
            "resume_dir": str(args.resume_dir) if args.resume_dir else None,
            "output_dir": str(args.output_dir) if args.output_dir else None,
            "log_jsonl": str(args.log_jsonl) if args.log_jsonl else None,
        },
        "benchmark_tokenization": tokenization,
    })
    return metadata


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not 0.0 <= args.clean_replay_fraction <= 1.0:
        raise ValueError("--clean-replay-fraction must be in [0, 1]")
    if not 0.0 < args.clean_ar_prefix_fraction < 1.0:
        raise ValueError("--clean-ar-prefix-fraction must be in (0, 1)")
    if (args.resume_dir is None) != (args.resume_step is None):
        raise ValueError("Pass --resume-dir and --resume-step together")
    if args.resume_step is not None and not 0 < args.resume_step < args.steps:
        raise ValueError("Require 0 < --resume-step < --steps")

    device_type = args.device_type or autodetect_device_type()
    ddp, rank, _local_rank, world_size, device = compute_init(
        device_type, seed=args.seed
    )
    if ddp or rank != 0 or world_size != 1:
        raise RuntimeError("The matched CES-SFT trainer is single-device")

    start_step = int(args.resume_step or 0)
    if args.resume_dir is None:
        model, tokenizer, original_metadata = load_model(
            "base",
            device,
            phase="train",
            model_tag=args.model_tag,
            step=args.model_step,
        )
    else:
        model, tokenizer, original_metadata = build_model(
            str(args.resume_dir.resolve()), start_step, device, phase="train"
        )
    token_ops = TokensOperations(getattr(tokenizer, "enc", tokenizer))
    raw_rows = read_jsonl(args.manifest_jsonl)
    reject_heldout_rows(raw_rows)
    rows, tokenization = prepare_benchmark_for_tokenizer(
        raw_rows,
        tokenizer=tokenizer,
        token_ops=token_ops,
    )
    if not rows:
        raise ValueError("The training manifest is empty")
    random.Random(args.seed).shuffle(rows)

    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        scalar_lr=args.scalar_lr,
        weight_decay=args.weight_decay,
    )
    for group in optimizer.param_groups:
        group["lr"] *= args.lr_scale
        group["initial_lr"] = group["lr"]
    if args.resume_dir is not None:
        optimizer_path = args.resume_dir.resolve() / f"optim_{start_step:06d}_rank0.pt"
        if not optimizer_path.is_file():
            raise FileNotFoundError(f"Missing resume optimizer: {optimizer_path}")
        optimizer.load_state_dict(torch.load(
            optimizer_path,
            map_location=device,
            weights_only=False,
        ))

    output_dir = args.output_dir or args.resume_dir or (
        Path(get_base_dir()) / "base_checkpoints" / args.output_tag
    )
    if args.resume_dir is not None and output_dir.resolve() != args.resume_dir.resolve():
        raise ValueError("A resumed run must write back to its --resume-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_jsonl or output_dir / "train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "device": str(device),
        "model_tag": args.model_tag,
        "model_step": args.model_step,
        "rows": len(rows),
        "start_step": start_step,
        "target_step": args.steps,
        "clean_replay_fraction": args.clean_replay_fraction,
        "output_dir": str(output_dir),
        "tokenization": tokenization,
    }, indent=2), flush=True)

    try:
        for step in range(start_step, args.steps):
            started = time.perf_counter()
            if device_type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            row = rows[step % len(rows)]
            optimizer.zero_grad(set_to_none=True)
            model.train()
            with autocast_context(device_type):
                ces_loss = completion_only_supervised_loss(
                    model,
                    token_ops,
                    row,
                    device=device,
                    position_buffer_size=args.position_buffer_size,
                )
                clean_loss = clean_ar_completion_loss(
                    model,
                    row,
                    device=device,
                    prefix_fraction=args.clean_ar_prefix_fraction,
                )
                loss = matched_sft_loss(
                    ces_loss,
                    clean_loss,
                    clean_replay_fraction=args.clean_replay_fraction,
                )
            loss.backward()
            missing_gradient_count = materialize_missing_gradients(model)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            optimizer.step()

            record = {
                "step": step + 1,
                "benchmark_id": row.get("benchmark_id"),
                "loss": float(loss.detach()),
                "ces_supervised_loss": float(ces_loss.detach()),
                "clean_ar_loss": float(clean_loss.detach()),
                "grad_norm": float(grad_norm),
                "missing_gradient_count": missing_gradient_count,
                "peak_memory_allocated_gib": (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device_type == "cuda" else 0.0
                ),
                "peak_memory_reserved_gib": (
                    torch.cuda.max_memory_reserved(device) / 1024**3
                    if device_type == "cuda" else 0.0
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            print(json.dumps(record, separators=(",", ":")), flush=True)

            should_save = (
                not args.no_save
                and (
                    (args.save_every > 0 and (step + 1) % args.save_every == 0)
                    or step + 1 == args.steps
                )
            )
            if should_save:
                save_checkpoint(
                    str(output_dir),
                    step + 1,
                    model.state_dict(),
                    optimizer.state_dict(),
                    checkpoint_metadata(
                        original_metadata,
                        args,
                        posttraining_step=step + 1,
                        tokenization=tokenization,
                    ),
                    rank=0,
                )
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
