"""End-to-end controlled evaluation for AR-DSL and flat-relative baselines."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nanochat.causal_serialisation_extended import resolve
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.edit_dsl import (
    EDIT_SCRIPT_CLOSE,
    EDIT_SCRIPT_OPEN,
    ar_dsl_to_sequence_edit,
)
from nanochat.engine import Engine
from nanochat.flat_relative_dsl import (
    flat_relative_tokens_to_sequence_edit,
    flat_relative_tokens_to_text,
    flat_relative_vocabulary_metadata,
)
from ces.workflows.controlled_edit_eval import (
    _decode_without_bos,
    _edit_signature,
    _gold_signature,
    _op_name,
    _read_jsonl,
    _similarity,
    _source_length_bin,
    _text_similarity,
    _write_jsonl,
    summarise_records,
)


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _extract_ar_dsl(program_text: str) -> str:
    start = program_text.find(EDIT_SCRIPT_OPEN)
    if start < 0:
        raise ValueError("Generated program is missing <edit-script>")
    close = program_text.find(EDIT_SCRIPT_CLOSE, start + len(EDIT_SCRIPT_OPEN))
    if close < 0:
        raise ValueError("Generated program is missing </edit-script>")
    return program_text[start:close + len(EDIT_SCRIPT_CLOSE)]


def _parse_program(
    mode: str,
    source_tokens: list[int],
    program_tokens: list[int],
    tokenizer,
):
    raw_text = tokenizer.decode(program_tokens)
    if mode == "ar_dsl":
        dsl_text = _extract_ar_dsl(raw_text)
        return ar_dsl_to_sequence_edit(source_tokens, dsl_text, tokenizer), raw_text, dsl_text
    resolved_text = flat_relative_tokens_to_text(len(source_tokens), program_tokens, tokenizer)
    generated_edit = flat_relative_tokens_to_sequence_edit(
        source_tokens,
        program_tokens,
        tokenizer,
    )
    return generated_edit, raw_text, _extract_ar_dsl(resolved_text)


def generate_records(
    benchmark_rows: list[dict[str, Any]],
    *,
    mode: str,
    model,
    tokenizer,
    device: torch.device,
    model_context: int,
    generation_seeds: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    prefill_open_tag: bool = True,
) -> list[dict[str, Any]]:
    engine = Engine(model, tokenizer)
    output: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for row_index, benchmark in enumerate(benchmark_rows):
            source_tokens = [int(token) for token in benchmark["source_tokens"]]
            target_tokens = [int(token) for token in benchmark["target_tokens"]]
            source_text = str(benchmark["source_text"])
            target_text = str(benchmark["target_text"])
            prompt_suffix = (
                [int(token) for token in tokenizer.encode(f"\n{EDIT_SCRIPT_OPEN}")]
                if prefill_open_tag
                else []
            )
            prompt_tokens = [*source_tokens, *prompt_suffix]
            prompt_positions = list(range(len(prompt_tokens)))
            source_char_similarity = _text_similarity(source_text, target_text)
            source_token_similarity = _similarity(source_tokens, target_tokens)
            gold_signature = _gold_signature(benchmark["gold_edits"])

            for generation_seed in generation_seeds:
                budget = min(max_new_tokens, model_context - len(prompt_tokens))
                status = "prompt_too_long"
                sample_tokens = prompt_tokens
                generated_edit = None
                resolved_tokens = None
                raw_program_text = tokenizer.decode(prompt_suffix)
                resolved_program_text = None
                generation_seconds = 0.0
                apply_seconds = 0.0
                if budget > 0:
                    _synchronise(device)
                    generation_start = time.perf_counter()
                    samples, _masks, _positions = engine.generate_batch(
                        prompt_tokens,
                        num_samples=1,
                        max_tokens=budget,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=int(generation_seed) + row_index,
                        no_edits=True,
                    )
                    _synchronise(device)
                    generation_seconds = time.perf_counter() - generation_start
                    sample_tokens = [int(token) for token in samples[0]]
                    program_tokens = sample_tokens[len(source_tokens):]
                    apply_start = time.perf_counter()
                    try:
                        generated_edit, raw_program_text, resolved_program_text = _parse_program(
                            mode,
                            source_tokens,
                            program_tokens,
                            tokenizer,
                        )
                        resolved_tokens = [int(token) for token in resolve(generated_edit)]
                        status = "valid"
                    except Exception as exc:
                        status = f"parse_or_resolve_error:{type(exc).__name__}"
                    apply_seconds = time.perf_counter() - apply_start

                valid = status == "valid" and resolved_tokens is not None
                final_tokens = resolved_tokens if valid else source_tokens
                final_text = _decode_without_bos(tokenizer, final_tokens)
                generated_signature = _edit_signature(generated_edit) if generated_edit is not None else ()
                generated_op_counts = (
                    Counter(_op_name(edit) for edit in generated_edit.edits)
                    if generated_edit is not None
                    else Counter()
                )
                final_char_similarity = _text_similarity(final_text, target_text)
                final_token_similarity = _similarity(final_tokens, target_tokens)
                char_delta = final_char_similarity - source_char_similarity
                token_delta = final_token_similarity - source_token_similarity
                recoverable_char_gap = max(1.0 - source_char_similarity, 0.0)
                recovery_fraction = (
                    char_delta / recoverable_char_gap
                    if recoverable_char_gap > 0
                    else float(final_text == target_text)
                )
                all_positions = list(range(len(sample_tokens)))

                output.append({
                    "evaluation_id": f"{benchmark['benchmark_id']}:decode_seed_{generation_seed}",
                    "benchmark_id": benchmark["benchmark_id"],
                    "generation_seed": int(generation_seed),
                    "baseline_mode": mode,
                    "stratum": benchmark["stratum"],
                    "repair_op_mode": benchmark["repair_op_mode"],
                    "repair_num_ops": int(benchmark["repair_num_ops"]),
                    "density_regime": benchmark["density_regime"],
                    "source_length_bin": _source_length_bin(len(source_tokens)),
                    "source_shard_id": benchmark["source_shard_id"],
                    "source_row_index": int(benchmark["source_row_index"]),
                    "source_token_offset": int(benchmark["source_token_offset"]),
                    "construction_seed": int(benchmark["construction_seed"]),
                    "status": status,
                    "valid": bool(valid),
                    "source_text": source_text,
                    "target_text": target_text,
                    "resolved_text": final_text,
                    "source_tokens": source_tokens,
                    "target_tokens": target_tokens,
                    "resolved_tokens": final_tokens,
                    "gold_edits": benchmark["gold_edits"],
                    "gold_op_counts": benchmark["gold_op_counts"],
                    "generated_op_counts": dict(sorted(generated_op_counts.items())),
                    "prompt_tokens": prompt_tokens,
                    "prompt_positions": prompt_positions,
                    "generated_full_tokens": sample_tokens,
                    "generated_full_positions": all_positions,
                    "generated_program_tokens": sample_tokens[len(source_tokens):],
                    "generated_program_positions": all_positions[len(source_tokens):],
                    "generated_program_text": raw_program_text,
                    "resolved_program_text": resolved_program_text,
                    "exact_target": bool(final_tokens == target_tokens),
                    "exact_program": bool(valid and generated_signature == gold_signature),
                    "source_char_similarity": source_char_similarity,
                    "final_char_similarity": final_char_similarity,
                    "char_similarity_delta": char_delta,
                    "source_token_similarity": source_token_similarity,
                    "final_token_similarity": final_token_similarity,
                    "token_similarity_delta": token_delta,
                    "target_similarity_improved": bool(char_delta > 0),
                    "char_gap_recovery_fraction": recovery_fraction,
                    "generation_seconds": generation_seconds,
                    "apply_seconds": apply_seconds,
                    "diagnostics": {},
                    "source_avg_logprob": None,
                    "target_avg_logprob": None,
                    "resolved_avg_logprob": None,
                    "target_minus_source_avg_logprob": None,
                    "target_likelihood_improved": None,
                    "likelihood_delta": None,
                    "likelihood_improved": None,
                })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ar_dsl", "flat_relative"), required=True)
    parser.add_argument("--benchmark-jsonl", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=[12345])
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--prefill-open-tag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefill the edit-script opening tag. Disable to match training exactly.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    parser.add_argument("--device-type", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")
    if not args.overwrite and (output_path.exists() or summary_path.exists()):
        raise FileExistsError("Output exists; use --overwrite or choose new output paths")
    benchmark = _read_jsonl(Path(args.benchmark_jsonl))
    if args.limit is not None:
        benchmark = benchmark[:args.limit]
    if not benchmark:
        raise ValueError("No benchmark rows selected")

    device_type = autodetect_device_type() if not args.device_type else args.device_type
    _ddp, rank, _local_rank, world_size, device = compute_init(device_type)
    if rank != 0 or world_size != 1:
        raise RuntimeError("Autoregressive edit evaluation expects a single process")
    model, tokenizer, meta = load_model(
        "base", device, phase="eval", model_tag=args.model_tag, step=args.step
    )
    if args.mode == "flat_relative" and meta.get("tokenizer") != "flat_relative_2048":
        raise ValueError("flat_relative evaluation requires a flat_relative_2048 checkpoint")
    max_manifest_token = max(
        int(token)
        for row in benchmark
        for field in ("source_tokens", "target_tokens")
        for token in row[field]
    )
    if max_manifest_token >= tokenizer.get_vocab_size():
        raise ValueError("Frozen manifest tokens exceed the checkpoint vocabulary")

    records = generate_records(
        benchmark,
        mode=args.mode,
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_context=int(meta["model_config"]["sequence_len"]),
        generation_seeds=args.generation_seeds,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        prefill_open_tag=args.prefill_open_tag,
    )
    summary = {
        "benchmark_jsonl": str(Path(args.benchmark_jsonl).resolve()),
        "model_tag": args.model_tag,
        "step": int(meta["step"]),
        "baseline_mode": args.mode,
        "tokenizer": meta.get("tokenizer"),
        "flat_relative_vocabulary": (
            flat_relative_vocabulary_metadata(tokenizer)
            if args.mode == "flat_relative"
            else None
        ),
        "selected_benchmark_examples": len(benchmark),
        "generated_records": len(records),
        "generation_config": {
            "generation_seeds": args.generation_seeds,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "prefill_open_tag": args.prefill_open_tag,
        },
        "uncertainty": {
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "note": "Generation seeds are decoding replicates, not training seeds.",
        },
        "groups": summarise_records(
            records,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
    }
    _write_jsonl(output_path, records)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_jsonl": str(output_path),
        "summary_json": str(summary_path),
        "overall": summary["groups"]["overall"],
    }, indent=2))
    compute_cleanup()


if __name__ == "__main__":
    main()
