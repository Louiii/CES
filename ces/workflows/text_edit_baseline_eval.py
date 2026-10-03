"""Evaluate full-regeneration or oracle-span FIM on the controlled benchmark."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch

from nanochat.causal_serialisation_extended import TokensOperations
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.engine import Engine
from nanochat.predicted_fim import generate_predicted_fim
from nanochat.text_edit_baselines import build_baseline_example
from ces.workflows.benchmark_tokenization import prepare_benchmark_for_tokenizer
from ces.workflows.controlled_edit_eval import (
    _read_jsonl,
    _source_length_bin,
    _text_similarity,
    _similarity,
    _write_jsonl,
    summarise_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("regeneration", "fim", "fim_predicted"), required=True)
    parser.add_argument("--benchmark-jsonl", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--generation-seed", type=int, default=12345)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--device-type", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_jsonl)
    summary_output = Path(args.summary_json) if args.summary_json else output.with_suffix(".summary.json")
    if not args.overwrite and (output.exists() or summary_output.exists()):
        raise FileExistsError("Output exists; use --overwrite or choose new output paths")
    benchmark = _read_jsonl(Path(args.benchmark_jsonl))
    if args.limit is not None:
        benchmark = benchmark[:args.limit]
    if not benchmark:
        raise ValueError("No benchmark examples selected")

    device_type = autodetect_device_type() if not args.device_type else args.device_type
    _ddp, rank, _local_rank, world_size, device = compute_init(device_type)
    if rank != 0 or world_size != 1:
        raise RuntimeError("Run one text-edit baseline evaluator process per GPU")
    model, tokenizer, meta = load_model(
        "base", device, phase="eval", model_tag=args.model_tag, step=args.step
    )
    benchmark, benchmark_tokenization = prepare_benchmark_for_tokenizer(
        benchmark,
        tokenizer=tokenizer,
        token_ops=TokensOperations(tokenizer.enc),
    )
    if benchmark_tokenization["mode"] == "retokenized_text":
        print(
            "Frozen benchmark token IDs are incompatible with this checkpoint tokenizer; "
            "rebuilt tokenizer-dependent fields from frozen source/target text.",
            flush=True,
        )
    engine = Engine(model, tokenizer)
    context = int(model.config.sequence_len)
    bos = tokenizer.get_bos_token_id()
    records = []
    model.eval()
    with torch.inference_mode():
        for row_index, row in enumerate(benchmark):
            source_tokens = [int(token) for token in row["source_tokens"]]
            target_tokens = [int(token) for token in row["target_tokens"]]
            example = build_baseline_example(
                args.mode,
                source_tokens,
                target_tokens,
                tokenizer,
            )
            predicted = None
            generation_seconds = 0.0
            if args.mode == "fim_predicted":
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                predicted = generate_predicted_fim(
                    model,
                    tokenizer,
                    source_tokens,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    seed=args.generation_seed + row_index,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                generation_seconds = time.perf_counter() - start
                prompt = [*predicted.locator_tokens, *predicted.appended_prompt_tokens]
                budget = predicted.generation_budget
                generated_completion = predicted.generated_completion_tokens
                raw_tokens = [*prompt, *generated_completion]
                terminated = predicted.terminated
                status = predicted.status
                candidate_tokens = predicted.resolved_tokens
                source_prefix = predicted.source_prefix
                source_suffix = predicted.source_suffix
            else:
                prompt = example.prompt_tokens
                budget = min(args.max_new_tokens, context - len(prompt))
                generated_completion: list[int] = []
                raw_tokens = prompt
                terminated = False
                status = "prompt_too_long"
                if budget > 0:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    samples, _masks, _pointers = engine.generate_batch(
                        prompt,
                        num_samples=1,
                        max_tokens=budget,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        seed=args.generation_seed + row_index,
                        no_edits=True,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    generation_seconds = time.perf_counter() - start
                    raw_tokens = [int(token) for token in samples[0]]
                    generated_completion = raw_tokens[len(prompt):]
                    terminated = len(generated_completion) < budget
                    status = "valid" if terminated else "max_tokens_without_terminator"
                if args.mode == "regeneration":
                    candidate_tokens = [bos, *generated_completion]
                else:
                    candidate_tokens = [
                        *example.source_prefix,
                        *generated_completion,
                        *example.source_suffix,
                    ]
                source_prefix = example.source_prefix
                source_suffix = example.source_suffix
            valid = status == "valid"
            resolved_tokens = candidate_tokens if valid else source_tokens
            source_text = str(row["source_text"])
            target_text = str(row["target_text"])
            resolved_text = tokenizer.decode([
                token for token in resolved_tokens if token != bos
            ])
            source_char_similarity = _text_similarity(source_text, target_text)
            final_char_similarity = _text_similarity(resolved_text, target_text)
            source_token_similarity = _similarity(source_tokens, target_tokens)
            final_token_similarity = _similarity(resolved_tokens, target_tokens)
            char_delta = final_char_similarity - source_char_similarity
            predicted_span = (
                None if predicted is None
                else [predicted.predicted_start, predicted.predicted_end]
            )
            location_exact = predicted_span == list(example.source_span) if predicted_span else None
            token_delta = final_token_similarity - source_token_similarity
            records.append({
                "evaluation_id": f"{row['benchmark_id']}:{args.mode}",
                "benchmark_id": row["benchmark_id"],
                "baseline_mode": args.mode,
                "oracle_span": args.mode == "fim",
                "generation_seed": args.generation_seed,
                "stratum": row["stratum"],
                "repair_op_mode": row["repair_op_mode"],
                "repair_num_ops": int(row["repair_num_ops"]),
                "density_regime": row["density_regime"],
                "source_length_bin": _source_length_bin(len(source_tokens)),
                "status": status,
                "valid": valid,
                "source_text": source_text,
                "target_text": target_text,
                "resolved_text": resolved_text,
                "source_tokens": source_tokens,
                "target_tokens": target_tokens,
                "resolved_tokens": resolved_tokens,
                "prompt_tokens": prompt,
                "generated_full_tokens": raw_tokens,
                "generated_completion_tokens": generated_completion,
                "generated_completion_text": tokenizer.decode(generated_completion),
                "gold_middle_tokens": example.gold_middle,
                "source_prefix_tokens": source_prefix,
                "source_suffix_tokens": source_suffix,
                "gold_source_span": list(example.source_span) if example.source_span else None,
                "predicted_source_span": predicted_span,
                "location_exact": location_exact,
                "exact_target": resolved_tokens == target_tokens,
                "exact_program": (
                    valid
                    and generated_completion == example.gold_middle
                    and (predicted is None or location_exact is True)
                ),
                "source_char_similarity": source_char_similarity,
                "final_char_similarity": final_char_similarity,
                "char_similarity_delta": char_delta,
                "source_token_similarity": source_token_similarity,
                "final_token_similarity": final_token_similarity,
                "token_similarity_delta": token_delta,
                "target_similarity_improved": char_delta > 0,
                "generation_seconds": generation_seconds,
                "apply_seconds": 0.0,
                "source_avg_logprob": None,
                "target_avg_logprob": None,
                "resolved_avg_logprob": None,
                "target_minus_source_avg_logprob": None,
                "target_likelihood_improved": None,
                "likelihood_delta": None,
                "likelihood_improved": None,
                "diagnostics": {
                    "terminated": terminated,
                    "generation_budget": budget,
                    "locator_logprob": None if predicted is None else predicted.locator_logprob,
                    "start_pointer": None if predicted is None else predicted.predicted_start_pointer,
                    "end_pointer": None if predicted is None else predicted.predicted_end_pointer,
                    "gold_op_counts": dict(Counter(edit["op"] for edit in row["gold_edits"])),
                },
            })
            if args.progress_every > 0 and (
                len(records) % args.progress_every == 0 or len(records) == len(benchmark)
            ):
                elapsed = sum(float(record["generation_seconds"]) for record in records)
                valid = sum(bool(record["valid"]) for record in records)
                print(
                    f"{args.mode}: {len(records)}/{len(benchmark)} "
                    f"valid={valid / len(records):.1%} generation_s={elapsed:.1f}",
                    flush=True,
                )

    localization = None
    if args.mode == "fim_predicted":
        located = [record for record in records if record["predicted_source_span"] is not None]
        exact = sum(record["location_exact"] is True for record in located)
        localization = {
            "examples": len(located),
            "exact": exact,
            "exact_rate": exact / len(located) if located else None,
            "mean_start_absolute_error": (
                sum(abs(record["predicted_source_span"][0] - record["gold_source_span"][0]) for record in located)
                / len(located)
                if located
                else None
            ),
            "mean_end_absolute_error": (
                sum(abs(record["predicted_source_span"][1] - record["gold_source_span"][1]) for record in located)
                / len(located)
                if located
                else None
            ),
        }


    summary = {
        "mode": args.mode,
        "localization": localization,
        "oracle_span": args.mode == "fim",
        "specialized_locator": args.mode == "fim_predicted",
        "benchmark_jsonl": str(Path(args.benchmark_jsonl).resolve()),
        "model_tag": args.model_tag,
        "step": int(meta["step"]),
        "records": len(records),
        "benchmark_tokenization": benchmark_tokenization,
        "generation": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.generation_seed,
            "max_new_tokens": args.max_new_tokens,
        },
        "groups": summarise_records(
            records,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
    }
    _write_jsonl(output, records)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_jsonl": str(output),
        "summary_json": str(summary_output),
        "overall": summary["groups"]["overall"],
    }, indent=2))
    compute_cleanup()


if __name__ == "__main__":
    main()
