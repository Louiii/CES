"""Evaluate CES with gold pointer locations or an oracle FIM-matched edit span."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nanochat.causal_serialisation_extended import (
    Copy,
    Delete,
    Insert,
    Move,
    Replace,
    SequenceEdit,
    SerialisedData,
    TokensOperations,
)
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.engine_extended import Engine
from nanochat.oracle_ces import (
    OraclePointerStructureMismatch,
    PointerOracleProvider,
    build_oracle_span_prompt,
)
from nanochat.oracle_edit_likelihood import canonical_sequence_edit_from_tokens
from ces.workflows.benchmark_tokenization import prepare_benchmark_for_tokenizer
from ces.workflows.controlled_edit_eval import (
    _gold_signature,
    _similarity,
    _source_length_bin,
    _text_similarity,
    _tokens_hash,
    score_unique_sequences,
    summarise_records,
)
from ces.workflows.human_edit_eval import _deserialise_generated, _edit_signature, _op_name


def _read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows if limit is None else rows[:limit]


def _edit_record(edit: Any) -> dict[str, Any]:
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
    raise TypeError(f"Unsupported edit type: {type(edit).__name__}")


def _text_pair_rows(
    manifest_rows: list[dict[str, Any]],
    *,
    tokenizer,
    token_ops: TokensOperations,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bos = int(tokenizer.get_bos_token_id())
    output: list[dict[str, Any]] = []
    for index, row in enumerate(manifest_rows):
        source_text = str(row["source_text"])
        target_text = str(row["target_text"])
        source_tokens = [bos, *[int(token) for token in tokenizer.encode(source_text)]]
        target_tokens = [bos, *[int(token) for token in tokenizer.encode(target_text)]]
        sequence_edit = canonical_sequence_edit_from_tokens(source_tokens, target_tokens)
        serialised = token_ops.serialise(sequence_edit, indel_bos_pos=0)
        tokens = [int(token) for token in serialised.tokens]
        positions = [int(position) for position in serialised.positions]
        starts = [
            i for i, token in enumerate(tokens)
            if token == int(token_ops.edit_start_token)
        ]
        if len(starts) != 1:
            raise ValueError(f"Expected one edit-start token, found {len(starts)}")
        edit_start = starts[0]
        gold_edits = [_edit_record(edit) for edit in sequence_edit.edits]
        output.append({
            **row,
            "benchmark_id": str(row.get("benchmark_id") or f"gec:{index}"),
            "source_text": source_text,
            "target_text": target_text,
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
            "gold_edits": gold_edits,
            "gold_op_counts": dict(sorted(Counter(edit["op"] for edit in gold_edits).items())),
            "prompt_tokens": tokens[:edit_start + 1],
            "prompt_positions": positions[:edit_start + 1],
            "gold_program_tokens": tokens[edit_start + 1:],
            "gold_program_positions": positions[edit_start + 1:],
            "stratum": "human_gec",
            "repair_op_mode": "human",
            "repair_num_ops": len(gold_edits),
            "density_regime": "human",
            "source_shard_id": "coedit",
            "source_row_index": index,
            "source_token_offset": 0,
            "construction_seed": 0,
        })
    return output, {
        "mode": "retokenized_text_pairs",
        "tokenizer_vocab_size": int(tokenizer.get_vocab_size()),
        "examples": len(output),
    }


def _decode_without_bos(tokenizer, tokens: list[int]) -> str:
    bos = int(tokenizer.get_bos_token_id())
    return tokenizer.decode([int(token) for token in tokens if int(token) != bos])


def _oracle_span_signature(span_prompt) -> tuple[tuple[Any, ...], ...]:
    return _edit_signature(span_prompt.sequence_edit)


def _generate_record(
    *,
    mode: str,
    row: dict[str, Any],
    row_index: int,
    model,
    tokenizer,
    token_ops: TokensOperations,
    engine: Engine,
    device: torch.device,
    max_edit_tokens: int,
    max_edit_ops: int,
    max_payload_tokens: int,
    temperature: float,
    generation_seed: int,
) -> dict[str, Any]:
    source_tokens = [int(token) for token in row["source_tokens"]]
    target_tokens = [int(token) for token in row["target_tokens"]]
    benchmark_signature = _gold_signature(row["gold_edits"])
    pointer_provider = None
    oracle_operation = None
    oracle_source_span = None
    oracle_target_span = None
    target_payload_tokens = 0

    if mode == "pointer_oracle":
        prompt_tokens = [int(token) for token in row["prompt_tokens"]]
        prompt_positions = [int(position) for position in row["prompt_positions"]]
        pointer_provider = PointerOracleProvider(
            [int(token) for token in row["gold_program_tokens"]],
            [int(position) for position in row["gold_program_positions"]],
            token_ops,
        )
        protocol_signature = benchmark_signature
        generation_max_edit_ops = max_edit_ops
    else:
        span_prompt = build_oracle_span_prompt(source_tokens, target_tokens, token_ops)
        if span_prompt is None:
            return _no_change_record(
                mode=mode,
                row=row,
                row_index=row_index,
                source_tokens=source_tokens,
                target_tokens=target_tokens,
                generation_seed=generation_seed,
            )
        prompt_tokens = list(span_prompt.prompt_tokens)
        prompt_positions = list(span_prompt.prompt_positions)
        protocol_signature = _oracle_span_signature(span_prompt)
        generation_max_edit_ops = 1
        oracle_operation = span_prompt.operation
        oracle_source_span = list(span_prompt.source_span)
        oracle_target_span = list(span_prompt.target_span)
        target_payload_tokens = len(span_prompt.target_payload_tokens)

    context = int(model.config.sequence_len)
    budget = min(max_edit_tokens, context - len(prompt_tokens))
    sample_tokens = prompt_tokens
    sample_positions = prompt_positions
    generated_edit = None
    resolved_tokens = None
    status = "prompt_too_long"
    generation_seconds = 0.0
    apply_seconds = 0.0
    diagnostics: dict[str, Any] = {}
    if budget > 0:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        try:
            samples, _masks, pointers = engine.generate_batch(
                prompt_tokens,
                tix=prompt_positions,
                num_samples=1,
                max_tokens=budget,
                temperature=temperature,
                pointer_temperature=temperature,
                seed=generation_seed + row_index,
                max_payload_tokens=max_payload_tokens,
                max_edit_ops=generation_max_edit_ops,
                stop_after_edit_end=True,
                pointer_provider=pointer_provider,
            )
            sample_tokens = [int(token) for token in samples[0]]
            sample_positions = [int(position) for position in pointers[0]]
            diagnostics = dict(engine.last_generation_diagnostics[0])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            generation_seconds = time.perf_counter() - start
            apply_start = time.perf_counter()
            generated_edit, status = _deserialise_generated(
                sample_tokens,
                sample_positions,
                token_ops,
            )
            if generated_edit is not None:
                try:
                    resolved = token_ops.resolve(SerialisedData(
                        tokens=np.asarray(sample_tokens, dtype=np.int64),
                        positions=np.asarray(sample_positions, dtype=np.int64),
                    ))
                    resolved_tokens = [int(token) for token in resolved]
                except Exception as exc:
                    status = f"apply_error:{type(exc).__name__}"
                    generated_edit = None
            apply_seconds = time.perf_counter() - apply_start
        except OraclePointerStructureMismatch as exc:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            generation_seconds = time.perf_counter() - start
            status = "oracle_pointer_structure_mismatch"
            diagnostics = {"oracle_pointer_error": str(exc)}

    valid = status == "valid" and resolved_tokens is not None
    final_tokens = resolved_tokens if valid else source_tokens
    source_text = str(row["source_text"])
    target_text = str(row["target_text"])
    final_text = _decode_without_bos(tokenizer, final_tokens)
    source_char_similarity = _text_similarity(source_text, target_text)
    final_char_similarity = _text_similarity(final_text, target_text)
    source_token_similarity = _similarity(source_tokens, target_tokens)
    final_token_similarity = _similarity(final_tokens, target_tokens)
    char_delta = final_char_similarity - source_char_similarity
    token_delta = final_token_similarity - source_token_similarity
    recoverable_gap = max(1.0 - source_char_similarity, 0.0)
    generated_signature = _edit_signature(generated_edit) if generated_edit is not None else ()
    pointer_events_total = len(pointer_provider.events) if pointer_provider is not None else 0
    pointer_events_consumed = pointer_provider.consumed if pointer_provider is not None else 0

    return {
        "evaluation_id": f"{row['benchmark_id']}:{mode}:seed_{generation_seed}",
        "benchmark_id": str(row["benchmark_id"]),
        "method": f"ces_{mode}",
        "oracle_mode": mode,
        "generation_seed": generation_seed,
        "stratum": str(row["stratum"]),
        "repair_op_mode": str(row["repair_op_mode"]),
        "repair_num_ops": int(row["repair_num_ops"]),
        "density_regime": str(row["density_regime"]),
        "source_length_bin": _source_length_bin(len(source_tokens)),
        "source_shard_id": str(row["source_shard_id"]),
        "source_row_index": int(row["source_row_index"]),
        "source_token_offset": int(row["source_token_offset"]),
        "construction_seed": int(row["construction_seed"]),
        "status": status,
        "valid": bool(valid),
        "source_text": source_text,
        "target_text": target_text,
        "prediction_text": final_text,
        "ungated_prediction_text": final_text,
        "resolved_text": final_text,
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "resolved_tokens": final_tokens,
        "gold_edits": row["gold_edits"],
        "gold_op_counts": row["gold_op_counts"],
        "prompt_tokens": prompt_tokens,
        "prompt_positions": prompt_positions,
        "generated_full_tokens": sample_tokens,
        "generated_full_positions": sample_positions,
        "generated_suffix_tokens": sample_tokens[len(prompt_tokens):],
        "generated_suffix_positions": sample_positions[len(prompt_positions):],
        "exact_target": bool(final_tokens == target_tokens),
        "exact_program": bool(valid and generated_signature == protocol_signature),
        "exact_benchmark_program": bool(valid and generated_signature == benchmark_signature),
        "source_char_similarity": source_char_similarity,
        "final_char_similarity": final_char_similarity,
        "char_similarity_delta": char_delta,
        "source_token_similarity": source_token_similarity,
        "final_token_similarity": final_token_similarity,
        "token_similarity_delta": token_delta,
        "target_similarity_improved": bool(char_delta > 0),
        "char_gap_recovery_fraction": (
            char_delta / recoverable_gap
            if recoverable_gap > 0
            else float(final_text == target_text)
        ),
        "generation_seconds": generation_seconds,
        "apply_seconds": apply_seconds,
        "source_avg_logprob": None,
        "candidate_avg_logprob": None,
        "candidate_minus_source_avg_logprob": None,
        "likelihood_delta": None,
        "likelihood_improved": None,
        "target_minus_source_avg_logprob": None,
        "target_likelihood_improved": None,
        "edit_confidence": None,
        "oracle_operation": oracle_operation,
        "oracle_source_span": oracle_source_span,
        "oracle_target_span": oracle_target_span,
        "oracle_target_payload_tokens": target_payload_tokens,
        "oracle_pointer_events_total": pointer_events_total,
        "oracle_pointer_events_consumed": pointer_events_consumed,
        "oracle_pointer_events_complete": (
            pointer_provider.complete if pointer_provider is not None else None
        ),
        "target_blind": False,
        "target_payload_exposed": False,
        "diagnostics": diagnostics,
    }


def _no_change_record(
    *,
    mode: str,
    row: dict[str, Any],
    row_index: int,
    source_tokens: list[int],
    target_tokens: list[int],
    generation_seed: int,
) -> dict[str, Any]:
    source_text = str(row["source_text"])
    target_text = str(row["target_text"])
    return {
        "evaluation_id": f"{row['benchmark_id']}:{mode}:seed_{generation_seed}",
        "benchmark_id": str(row["benchmark_id"]),
        "method": f"ces_{mode}",
        "oracle_mode": mode,
        "generation_seed": generation_seed,
        "stratum": str(row["stratum"]),
        "repair_op_mode": str(row["repair_op_mode"]),
        "repair_num_ops": int(row["repair_num_ops"]),
        "density_regime": str(row["density_regime"]),
        "source_length_bin": _source_length_bin(len(source_tokens)),
        "source_shard_id": str(row["source_shard_id"]),
        "source_row_index": int(row.get("source_row_index", row_index)),
        "source_token_offset": int(row.get("source_token_offset", 0)),
        "construction_seed": int(row.get("construction_seed", 0)),
        "status": "oracle_no_change",
        "valid": True,
        "source_text": source_text,
        "target_text": target_text,
        "prediction_text": source_text,
        "ungated_prediction_text": source_text,
        "resolved_text": source_text,
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "resolved_tokens": source_tokens,
        "gold_edits": row["gold_edits"],
        "gold_op_counts": row["gold_op_counts"],
        "prompt_tokens": [],
        "prompt_positions": [],
        "generated_full_tokens": [],
        "generated_full_positions": [],
        "generated_suffix_tokens": [],
        "generated_suffix_positions": [],
        "exact_target": True,
        "exact_program": True,
        "exact_benchmark_program": True,
        "source_char_similarity": 1.0,
        "final_char_similarity": 1.0,
        "char_similarity_delta": 0.0,
        "source_token_similarity": 1.0,
        "final_token_similarity": 1.0,
        "token_similarity_delta": 0.0,
        "target_similarity_improved": False,
        "char_gap_recovery_fraction": 1.0,
        "generation_seconds": 0.0,
        "apply_seconds": 0.0,
        "source_avg_logprob": None,
        "candidate_avg_logprob": None,
        "candidate_minus_source_avg_logprob": None,
        "likelihood_delta": None,
        "likelihood_improved": None,
        "target_minus_source_avg_logprob": None,
        "target_likelihood_improved": None,
        "edit_confidence": None,
        "oracle_operation": "none",
        "oracle_source_span": [len(source_tokens), len(source_tokens)],
        "oracle_target_span": [len(target_tokens), len(target_tokens)],
        "oracle_target_payload_tokens": 0,
        "oracle_pointer_events_total": 0,
        "oracle_pointer_events_consumed": 0,
        "oracle_pointer_events_complete": True,
        "target_blind": False,
        "target_payload_exposed": False,
        "diagnostics": {},
    }


def _attach_likelihoods(model, device: torch.device, records: list[dict[str, Any]]) -> None:
    sequences: list[list[int]] = []
    for record in records:
        sequences.append(record["source_tokens"])
        if record["valid"]:
            sequences.append(record["resolved_tokens"])
    scores = score_unique_sequences(model, sequences, device=device, batch_size=8)
    for record in records:
        source_score = scores.get(_tokens_hash(record["source_tokens"]))
        candidate_score = (
            scores.get(_tokens_hash(record["resolved_tokens"]))
            if record["valid"]
            else None
        )
        record["source_avg_logprob"] = source_score
        record["candidate_avg_logprob"] = candidate_score
        record["candidate_minus_source_avg_logprob"] = (
            candidate_score - source_score
            if candidate_score is not None and source_score is not None
            else None
        )
        record["likelihood_delta"] = record[
            "candidate_minus_source_avg_logprob"
        ]
        record["likelihood_improved"] = (
            candidate_score > source_score
            if candidate_score is not None and source_score is not None
            else None
        )
        record["edit_confidence"] = record["candidate_minus_source_avg_logprob"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pointer_oracle", "oracle_span"), required=True)
    parser.add_argument("--input-format", choices=("controlled", "text_pairs"), required=True)
    parser.add_argument("--manifest-jsonl", required=True, type=Path)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--generation-seed", type=int, default=12345)
    parser.add_argument("--max-edit-tokens", type=int, default=192)
    parser.add_argument("--max-payload-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--skip-likelihood-scoring", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-edit-ops", type=int, default=64)
    parser.add_argument("--device-type", default="")
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.overwrite and (args.output_jsonl.exists() or args.summary_json.exists()):
        raise FileExistsError("Output exists; use --overwrite or choose new paths")
    manifest_rows = _read_jsonl(args.manifest_jsonl, args.limit)
    if not manifest_rows:
        raise ValueError("No evaluation rows selected")

    device_type = args.device_type or autodetect_device_type()
    _distributed, rank, _local_rank, world_size, device = compute_init(device_type)
    if rank != 0 or world_size != 1:
        raise RuntimeError("Run oracle CES evaluation with one process on one device")
    model, tokenizer, meta = load_model(
        "base", device, phase="eval", model_tag=args.model_tag, step=args.step
    )
    token_ops = TokensOperations(getattr(tokenizer, "enc", tokenizer))
    if args.input_format == "controlled":
        rows, tokenization = prepare_benchmark_for_tokenizer(
            manifest_rows,
            tokenizer=tokenizer,
            token_ops=token_ops,
        )
    else:
        rows, tokenization = _text_pair_rows(
            manifest_rows,
            tokenizer=tokenizer,
            token_ops=token_ops,
        )

    model.eval()
    engine = Engine(model, tokenizer)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row_index, row in enumerate(rows):
            records.append(_generate_record(
                mode=args.mode,
                row=row,
                row_index=row_index,
                model=model,
                tokenizer=tokenizer,
                token_ops=token_ops,
                engine=engine,
                device=device,
                max_edit_tokens=args.max_edit_tokens,
                max_payload_tokens=args.max_payload_tokens,
                max_edit_ops=args.max_edit_ops,
                temperature=args.temperature,
                generation_seed=args.generation_seed,
            ))
            if args.progress_every > 0 and (
                len(records) % args.progress_every == 0 or len(records) == len(rows)
            ):
                valid = sum(bool(record["valid"]) for record in records)
                exact = sum(bool(record["exact_target"]) for record in records)
                print(
                    f"{args.mode}: {len(records)}/{len(rows)} "
                    f"valid={valid / len(records):.1%} exact={exact / len(records):.1%}",
                    flush=True,
                )

    if not args.skip_likelihood_scoring:
        _attach_likelihoods(model, device, records)

    summary = {
        "mode": args.mode,
        "method": f"ces_{args.mode}",
        "input_format": args.input_format,
        "manifest_jsonl": str(args.manifest_jsonl.resolve()),
        "model_tag": args.model_tag,
        "step": int(meta["step"]),
        "records": len(records),
        "benchmark_tokenization": tokenization,
        "protocol": {
            "target_blind": False,
            "target_payload_exposed": False,
            "pointer_head_used": False,
            "gold_operation_tokens_forced": args.mode == "oracle_span",
            "gold_operation_tokens_used_for_alignment": args.mode == "pointer_oracle",
            "oracle_span_matches_fim_envelope": args.mode == "oracle_span",
            "exact_program_reference": "protocol_conditioning_program",
        },
        "generation": {
            "temperature": args.temperature,
            "seed": args.generation_seed,
            "max_edit_tokens": args.max_edit_tokens,
            "max_payload_tokens": args.max_payload_tokens,
            "pointer_oracle_max_edit_ops": args.max_edit_ops,
        },
        "confidence_field": (
            None if args.skip_likelihood_scoring
            else "candidate_minus_source_avg_logprob"
        ),
        "groups": summarise_records(
            records,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
    }
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_jsonl": str(args.output_jsonl),
        "summary_json": str(args.summary_json),
        "overall": summary["groups"]["overall"],
    }, indent=2))
    compute_cleanup()


if __name__ == "__main__":
    main()
