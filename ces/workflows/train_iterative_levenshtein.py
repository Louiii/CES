#!/usr/bin/env python
"""Train the independent iterative Levenshtein-style baseline from scratch."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist

from nanochat.causal_serialisation_extended import TokensOperations
from nanochat.checkpoint_manager import delete_checkpoint, load_checkpoint, save_checkpoint
from nanochat.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
    print_banner,
)
from nanochat.edit_dataloader import EditSamplingConfig
from nanochat.levenshtein import (
    LevenshteinConfig,
    LevenshteinTransformer,
    build_round_example,
    collate_rounds,
)
from nanochat.levenshtein_data import iterative_levenshtein_pair_loader_with_state
from nanochat.tokenizer import (
    LEVENSHTEIN_BOS,
    LEVENSHTEIN_EOS,
    LEVENSHTEIN_MASK,
    LEVENSHTEIN_PAD,
    get_extended_indel_tokeniser,
    get_levenshtein_tokeniser,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-type", default="")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-insertions", type=int, default=32)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--total-batch-size", type=int, default=32768)
    parser.add_argument("--num-iterations", type=int, default=-1)
    parser.add_argument("--target-param-data-ratio", type=float, default=8.0)
    parser.add_argument("--stop-after-step", type=int, default=-1)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--warmdown-ratio", type=float, default=0.2)
    parser.add_argument("--final-lr-frac", type=float, default=0.05)
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--head-lr", type=float, default=0.008)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.28)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8-recipe", choices=("tensorwise", "rowwise"), default="tensorwise")
    parser.add_argument("--edit-raw-data-dir", required=True)
    parser.add_argument("--edit-generated", nargs="*", default=())
    parser.add_argument("--edit-copy-scan-dir", default=None)
    parser.add_argument("--edit-forward-direction-prob", type=float, default=1.0)
    parser.add_argument("--edit-random-corruption-prob", type=float, default=1.0)
    parser.add_argument("--edit-clean-row-prob", type=float, default=0.5)
    parser.add_argument("--edit-tokenizer-batch-size", type=int, default=128)
    parser.add_argument("--edit-reservoir-chunk-size", type=int, default=2048)
    parser.add_argument("--model-tag", default="iterative_levenshtein_d12")
    parser.add_argument("--resume-from-step", type=int, default=-1)
    parser.add_argument("--resume-model-tag", default=None)
    parser.add_argument("--resume-reset-optimizer", action="store_true")
    parser.add_argument("--resume-reset-dataloader", action="store_true")
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--delete-resume-checkpoint-after-load", action="store_true")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if args.edit_reservoir_chunk_size < 1:
        parser.error("--edit-reservoir-chunk-size must be positive")
    if not 0 <= args.edit_clean_row_prob <= 1:
        parser.error("--edit-clean-row-prob must be in [0, 1]")
    if not 0 <= args.edit_forward_direction_prob <= 1:
        parser.error("--edit-forward-direction-prob must be in [0, 1]")
    if args.max_insertions != 32:
        parser.error("This baseline is defined with exactly 32 insertions per gap/round")
    if args.max_rounds != 4:
        parser.error("This baseline is defined with exactly four iterative rounds")
    return args


def convert_fp8(model, recipe):
    from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training

    def module_filter(module, _fqn):
        return (
            isinstance(module, torch.nn.Linear)
            and module.in_features % 16 == 0
            and module.out_features % 16 == 0
            and min(module.in_features, module.out_features) >= 128
        )

    configuration = Float8LinearConfig.from_recipe_name(recipe)
    total = sum(isinstance(module, torch.nn.Linear) for module in model.modules())
    convert_to_float8_training(model, config=configuration, module_filter_fn=module_filter)
    converted = sum("Float8" in type(module).__name__ for module in model.modules())
    print0(f"FP8 converted {converted}/{total} linear layers ({recipe})")


def main():
    args = parse_args()
    print_banner()
    device_type = args.device_type or autodetect_device_type()
    distributed, rank, _local_rank, world_size, device = compute_init(device_type)
    torch.manual_seed(args.seed + rank)
    rng = np.random.default_rng(args.seed + rank)
    print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

    tokenizer = get_levenshtein_tokeniser()
    model_dim = math.ceil(args.depth * args.aspect_ratio / args.head_dim) * args.head_dim
    config = LevenshteinConfig(
        sequence_len=args.max_seq_len,
        vocab_size=tokenizer.get_vocab_size(),
        pad_id=tokenizer.encode_special(LEVENSHTEIN_PAD),
        mask_id=tokenizer.encode_special(LEVENSHTEIN_MASK),
        bos_id=tokenizer.encode_special(LEVENSHTEIN_BOS),
        eos_id=tokenizer.encode_special(LEVENSHTEIN_EOS),
        n_layer=args.depth,
        n_head=model_dim // args.head_dim,
        n_embd=model_dim,
        max_insertions=args.max_insertions,
        max_rounds=args.max_rounds,
    )
    with torch.device("meta"):
        model = LevenshteinTransformer(config)
    model.to_empty(device=device)
    model.init_weights()

    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.model_tag)
    resuming = args.resume_from_step >= 0
    optimizer_state = None
    metadata = None
    resume_checkpoint_dir = checkpoint_dir
    if resuming:
        resume_checkpoint_dir = os.path.join(
            base_dir,
            "base_checkpoints",
            args.resume_model_tag or args.model_tag,
        )
        model_state, optimizer_state, metadata = load_checkpoint(
            resume_checkpoint_dir,
            args.resume_from_step,
            device,
            load_optimizer=not args.resume_reset_optimizer,
            rank=rank,
        )
        model.load_state_dict(model_state, strict=True, assign=True)
        del model_state
        if metadata.get("model_type") != "iterative_levenshtein_style":
            raise ValueError("Resume checkpoint is not an iterative Levenshtein model")
        if metadata["model_config"] != asdict(config):
            raise ValueError("Resume model config does not match current CLI")

    if args.fp8:
        if device_type != "cuda":
            raise ValueError("--fp8 requires CUDA")
        convert_fp8(model, args.fp8_recipe)

    counts = model.num_scaling_params()
    print0("Parameter counts:\n" + json.dumps(counts, indent=2))
    if args.num_iterations > 0:
        num_iterations = args.num_iterations
    else:
        target_tokens = int(args.target_param_data_ratio * counts["scaling"])
        num_iterations = target_tokens // args.total_batch_size
    stop_after_step = num_iterations if args.stop_after_step < 0 else args.stop_after_step
    if not 0 < stop_after_step <= num_iterations:
        raise ValueError("stop-after-step must be within the full training horizon")
    logical_tokens = num_iterations * args.total_batch_size
    # Training always runs deletion, gap-count, and mask-token encoders.
    true_encoder_tokens = 3 * logical_tokens
    print0(
        f"Horizon: {num_iterations:,} steps, {logical_tokens:,} logical tokens, "
        f"{true_encoder_tokens:,} full-attention encoder-pass tokens"
    )
    print0(f"This invocation stops at absolute step {stop_after_step:,}")

    reference_batch = 524288
    batch_scale = math.sqrt(args.total_batch_size / reference_batch)
    optimizer = model.setup_optimizer(
        embedding_lr=args.embedding_lr * batch_scale,
        head_lr=args.head_lr * batch_scale,
        matrix_lr=args.matrix_lr * batch_scale,
        weight_decay=args.weight_decay,
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        del optimizer_state

    original_model = model
    compiled_model = model if args.no_compile else torch.compile(model, dynamic=False)

    sampling_config = EditSamplingConfig(
        raw_data_dir=args.edit_raw_data_dir,
        generated_edit_paths=tuple(args.edit_generated),
        copy_scan_dir=args.edit_copy_scan_dir,
        direction_probs={
            "forward": args.edit_forward_direction_prob,
            "inverse": 1.0 - args.edit_forward_direction_prob,
        },
        random_corruption_prob_when_no_candidates=args.edit_random_corruption_prob,
        stream_clean_row_prob=args.edit_clean_row_prob,
        stream_partition_row_prob=0.0,
        random_source_token_reserve=128,
        max_edits_per_example=4,
        seed=args.seed,
    )
    edit_token_ops = TokensOperations(get_extended_indel_tokeniser().enc)
    dataloader_state = (
        None
        if metadata is None or args.resume_reset_dataloader
        else metadata.get("dataloader_state_dict")
    )
    pair_loader = iterative_levenshtein_pair_loader_with_state(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        split="train",
        cfg=sampling_config,
        token_ops=edit_token_ops,
        tokenizer_batch_size=args.edit_tokenizer_batch_size,
        resume_state_dict=dataloader_state,
        reservoir_chunk_size=args.edit_reservoir_chunk_size,
    )
    pairs, dataloader_state = next(pair_loader)

    tokens_per_micro = args.device_batch_size * args.max_seq_len * world_size
    if args.total_batch_size % tokens_per_micro:
        raise ValueError(
            f"total batch {args.total_batch_size} is not divisible by world microbatch {tokens_per_micro}"
        )
    grad_accumulation = args.total_batch_size // tokens_per_micro
    print0(f"Gradient accumulation: {grad_accumulation} microbatches")

    if resuming:
        step = metadata["step"]
        total_training_time = metadata.get("loop_state", {}).get("total_training_time", 0.0)
        rng_state = (
            None
            if args.resume_reset_dataloader
            else metadata.get("loop_state", {}).get("round_rng_state")
        )
        if rng_state is not None:
            rng.bit_generator.state = rng_state
    else:
        step = 0
        total_training_time = 0.0

    def lr_multiplier(current_step):
        if current_step < args.warmup_steps:
            return (current_step + 1) / args.warmup_steps
        warmdown = round(args.warmdown_ratio * num_iterations)
        if current_step <= num_iterations - warmdown:
            return 1.0
        progress = (num_iterations - current_step) / max(warmdown, 1)
        return args.final_lr_frac + (1 - args.final_lr_frac) * max(0.0, progress)

    while step < stop_after_step:
        started = time.time()
        accumulated = {}
        original_model.zero_grad(set_to_none=True)
        for _micro_step in range(grad_accumulation):
            oracle_rounds = [
                build_round_example(
                    source,
                    target,
                    round_index=int(rng.integers(0, args.max_rounds)),
                    config=config,
                )
                for source, target in pairs
            ]
            batch = collate_rounds(
                oracle_rounds,
                pad_id=config.pad_id,
                device=device,
                sequence_width=config.sequence_len,
            )
            loss, metrics = compiled_model(batch)
            (loss / grad_accumulation).backward()
            for key, value in metrics.items():
                accumulated[key] = accumulated.get(key, 0.0) + value.detach() / grad_accumulation
            pairs, dataloader_state = next(pair_loader)

        multiplier = lr_multiplier(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * multiplier
            if group["kind"] == "muon":
                group["momentum"] = 0.85 + 0.12 * min(step / 400, 1.0)
        optimizer.step()
        step += 1
        elapsed = time.time() - started
        total_training_time += elapsed

        if step % args.log_every == 0 or step == 1 or step == stop_after_step:
            accumulated = {key: float(value) for key, value in accumulated.items()}
            delete_acc = accumulated["delete_correct"] / max(accumulated["delete_count"], 1)
            gap_acc = accumulated["gap_correct"] / max(accumulated["gap_count"], 1)
            token_acc = accumulated["token_correct"] / max(accumulated["token_count"], 1)
            print0(
                f"step {step:05d}/{num_iterations:05d} | loss {accumulated['loss']:.4f} "
                f"(del {accumulated['delete_loss']:.3f}, gap {accumulated['gap_loss']:.3f}, "
                f"tok {accumulated['token_loss']:.3f}) | acc {delete_acc:.3f}/{gap_acc:.3f}/{token_acc:.3f} "
                f"| 3-pass tok/s {int(3 * args.total_batch_size / max(elapsed, 1e-9)):,} | {elapsed:.2f}s"
            )

    if distributed:
        dist.barrier()
    if resuming and args.delete_resume_checkpoint_after_load:
        delete_checkpoint(resume_checkpoint_dir, args.resume_from_step, rank=rank)
        if distributed:
            dist.barrier()
        print0(
            f"Deleted loaded checkpoint {args.resume_model_tag or args.model_tag} "
            f"step {args.resume_from_step} "
            "before writing its replacement milestone"
        )
    save_checkpoint(
        checkpoint_dir,
        step,
        original_model.state_dict(),
        optimizer.state_dict() if args.save_optimizer else None,
        {
            "step": step,
            "model_type": "iterative_levenshtein_style",
            "model_config": asdict(config),
            "tokenizer": "levenshtein",
            "training_horizon": num_iterations,
            "logical_training_tokens": logical_tokens,
            "encoder_passes_per_example": 3,
            "true_encoder_pass_tokens": true_encoder_tokens,
            "dataloader_state_dict": dataloader_state,
            "loop_state": {
                "total_training_time": total_training_time,
                "round_rng_state": rng.bit_generator.state,
            },
            "user_config": vars(args),
        },
        rank=rank,
    )
    if distributed:
        dist.barrier()
    print0(f"Saved iterative Levenshtein checkpoint {args.model_tag} step {step}")
    compute_cleanup()


if __name__ == "__main__":
    main()
