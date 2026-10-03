"""Generate normalized CoEdit records for non-CES editing baselines.

Every record contains source, reference, final prediction, structural validity,
and an inference confidence. Causal editors use the candidate-minus-source
average language-model log probability. The iterative Levenshtein editor uses
the mean log probability of its greedy delete/gap/token decisions. Oracle-span
FIM discloses that its boundary selection is reference-derived.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from nanochat.causal_serialisation_extended import Delete, Insert, Replace
from nanochat.checkpoint_manager import load_checkpoint, load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.edit_dsl import (
    ar_dsl_coordinate_token_spans,
    sequence_edit_to_ar_dsl,
)
from nanochat.engine import Engine
from nanochat.flat_relative_dsl import (
    flat_relative_token_maps,
    sequence_edit_to_flat_relative_tokens,
)
from nanochat.levenshtein import LevenshteinConfig, LevenshteinTransformer
from nanochat.oracle_edit_likelihood import canonical_sequence_edit_from_tokens
from nanochat.predicted_fim import generate_predicted_fim
from nanochat.text_edit_baselines import build_baseline_example
from nanochat.tokenizer import get_levenshtein_tokeniser
from ces.workflows.ar_dsl_human_edit_eval import _teacher_stats
from ces.workflows.autoregressive_edit_eval import generate_records as generate_ar_records
from ces.workflows.controlled_edit_eval import _tokens_hash, score_unique_sequences


def _read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows if limit is None else rows[:limit]


def _gold_edits(source_tokens: list[int], target_tokens: list[int]) -> list[dict[str, Any]]:
    sequence_edit = canonical_sequence_edit_from_tokens(source_tokens, target_tokens)
    output = []
    for edit in sequence_edit.edits:
        if isinstance(edit, Insert):
            output.append({"op": "insert", "pos": int(edit.pos), "content": [int(x) for x in edit.content]})
        elif isinstance(edit, Delete):
            output.append({"op": "delete", "start": int(edit.start), "end": int(edit.end)})
        elif isinstance(edit, Replace):
            output.append({
                "op": "replace",
                "start": int(edit.start),
                "end": int(edit.end),
                "content": [int(x) for x in edit.content],
            })
        else:  # pragma: no cover - canonical alignment emits only these types
            raise TypeError(type(edit))
    return output


def _benchmark_rows(manifest_rows: list[dict[str, Any]], tokenizer) -> list[dict[str, Any]]:
    bos = tokenizer.get_bos_token_id()
    output = []
    for index, row in enumerate(manifest_rows):
        source_tokens = [bos, *map(int, tokenizer.encode(str(row["source_text"])))]
        target_tokens = [bos, *map(int, tokenizer.encode(str(row["target_text"])))]
        edits = _gold_edits(source_tokens, target_tokens)
        output.append({
            **row,
            "benchmark_id": str(row.get("benchmark_id") or f"gec:{index}"),
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
            "gold_edits": edits,
            "gold_op_counts": {
                op: sum(edit["op"] == op for edit in edits)
                for op in ("insert", "delete", "replace", "copy", "move")
            },
            "stratum": "human_gec",
            "repair_op_mode": "human",
            "repair_num_ops": len(edits),
            "density_regime": "human",
            "source_shard_id": "coedit",
            "source_row_index": index,
            "source_token_offset": 0,
            "construction_seed": 0,
        })
    return output


def _causal_likelihoods(model, device, records: list[dict[str, Any]]) -> None:
    sequences = []
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
        record["edit_confidence"] = (
            candidate_score - source_score
            if candidate_score is not None and source_score is not None
            else None
        )


def _normalize_ar_records(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    output = []
    for record in records:
        valid = bool(record["valid"])
        prediction = str(record["resolved_text"]) if valid else str(record["source_text"])
        output.append({
            "benchmark_id": record["benchmark_id"],
            "method": mode,
            "status": record["status"],
            "valid": valid,
            "source_text": record["source_text"],
            "target_text": record["target_text"],
            "prediction_text": prediction,
            "ungated_prediction_text": prediction,
            "source_tokens": record["source_tokens"],
            "resolved_tokens": record["resolved_tokens"],
            "exact_target": bool(record["exact_target"]),
            "generated_program_text": record["generated_program_text"],
            "generation_seconds": record["generation_seconds"],
        })
    return output


def _coordinate_teacher_summary(
    mode: str,
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    device,
    model_context: int,
) -> dict[str, Any]:
    totals = {
        "teacher_serialized_examples": 0,
        "teacher_skipped_too_long": 0,
        "coordinate_fields_correct": 0,
        "coordinate_fields": 0,
        "coordinate_tokens_correct": 0,
        "coordinate_tokens": 0,
        "coordinate_loss_sum": 0.0,
    }
    _distance_to_id, relative_ids = flat_relative_token_maps(tokenizer) if mode == "flat_relative" else ({}, {})
    model.eval()
    with torch.inference_mode():
        for row in rows:
            sequence_edit = canonical_sequence_edit_from_tokens(
                row["source_tokens"], row["target_tokens"]
            )
            if mode == "ar_dsl":
                dsl_text = "\n" + sequence_edit_to_ar_dsl(sequence_edit, tokenizer) + "\n"
                program_tokens = [int(token) for token in tokenizer.encode(dsl_text)]
                spans = ar_dsl_coordinate_token_spans(
                    tokenizer, dsl_text, len(row["source_tokens"])
                )
            else:
                program_tokens = sequence_edit_to_flat_relative_tokens(sequence_edit, tokenizer)
                source_len = len(row["source_tokens"])
                spans = [
                    (source_len + index, source_len + index + 1)
                    for index, token in enumerate(program_tokens)
                    if int(token) in relative_ids
                ]
            serialized = [*row["source_tokens"], *program_tokens]
            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            stats = _teacher_stats(
                model, serialized, model_context, device, autocast_ctx, spans
            )
            if stats["token_loss_sum"] is None:
                totals["teacher_skipped_too_long"] += 1
                continue
            totals["teacher_serialized_examples"] += 1
            totals["coordinate_fields_correct"] += int(stats["position_correct"])
            totals["coordinate_fields"] += int(stats["position_targets"])
            totals["coordinate_tokens_correct"] += int(stats["position_token_correct"])
            totals["coordinate_tokens"] += int(stats["position_token_targets"])
            totals["coordinate_loss_sum"] += float(stats["position_loss_sum"])

    fields = totals["coordinate_fields"]
    tokens = totals["coordinate_tokens"]
    return {
        **totals,
        "coordinate_field_accuracy": (
            totals["coordinate_fields_correct"] / fields if fields else None
        ),
        "coordinate_token_accuracy": (
            totals["coordinate_tokens_correct"] / tokens if tokens else None
        ),
        "coordinate_token_nll": (
            totals["coordinate_loss_sum"] / tokens if tokens else None
        ),
    }


def _eval_autoregressive(
    mode: str,
    manifest_rows: list[dict[str, Any]],
    *,
    model_tag: str,
    step: int | None,
    device,
    generation_seed: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    coordinate_teacher: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag, step=step)
    rows = _benchmark_rows(manifest_rows, tokenizer)
    generated = generate_ar_records(
        rows,
        mode=mode,
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_context=int(meta["model_config"]["sequence_len"]),
        generation_seeds=[generation_seed],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    records = _normalize_ar_records(generated, mode)
    _causal_likelihoods(model, device, records)
    coordinate_summary = (
        _coordinate_teacher_summary(
            mode,
            model,
            tokenizer,
            rows,
            device=device,
            model_context=int(meta["model_config"]["sequence_len"]),
        )
        if coordinate_teacher
        else {"coordinate_teacher_evaluation": "skipped_for_sampling_sweep"}
    )
    return records, {
        "model_tag": model_tag,
        "step": int(meta["step"]),
        "tokenizer": meta.get("tokenizer"),
        "generation_config": {
            "generation_seed": generation_seed,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        },
        **coordinate_summary,
    }


def _eval_text_baseline(
    mode: str,
    manifest_rows: list[dict[str, Any]],
    *,
    model_tag: str,
    step: int | None,
    device,
    generation_seed: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode not in {"regeneration", "fim", "fim_predicted"}:
        raise ValueError(f"Unsupported text baseline: {mode}")
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag, step=step)
    engine = Engine(model, tokenizer)
    bos = tokenizer.get_bos_token_id()
    context = int(meta["model_config"]["sequence_len"])
    records = []
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(manifest_rows):
            source_tokens = [bos, *map(int, tokenizer.encode(str(row["source_text"])))]
            target_tokens = [bos, *map(int, tokenizer.encode(str(row["target_text"])))]
            example = build_baseline_example(mode, source_tokens, target_tokens, tokenizer)
            predicted = None
            if mode == "fim_predicted":
                predicted = generate_predicted_fim(
                    model,
                    tokenizer,
                    source_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=generation_seed + index,
                )
                completion = predicted.generated_completion_tokens
                status = predicted.status
                candidate_tokens = predicted.resolved_tokens
            else:
                budget = min(max_new_tokens, context - len(example.prompt_tokens))
                completion: list[int] = []
                status = "prompt_too_long"
                if budget > 0:
                    samples, _masks, _pointers = engine.generate_batch(
                        example.prompt_tokens,
                        num_samples=1,
                        max_tokens=budget,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=generation_seed + index,
                        no_edits=True,
                    )
                    completion = [int(token) for token in samples[0][len(example.prompt_tokens):]]
                    status = "valid" if len(completion) < budget else "max_tokens_without_terminator"
                if mode == "regeneration":
                    candidate_tokens = [bos, *completion]
                else:
                    candidate_tokens = [
                        *example.source_prefix,
                        *completion,
                        *example.source_suffix,
                    ]
            valid = status == "valid"
            resolved_tokens = candidate_tokens if valid else source_tokens
            prediction = (
                tokenizer.decode([token for token in resolved_tokens if token != bos])
                if valid
                else str(row["source_text"])
            )
            records.append({
                "benchmark_id": str(row.get("benchmark_id") or f"gec:{index}"),
                "method": mode,
                "oracle_span": mode == "fim",
                "status": status,
                "valid": valid,
                "source_text": str(row["source_text"]),
                "target_text": str(row["target_text"]),
                "prediction_text": prediction,
                "ungated_prediction_text": prediction,
                "source_tokens": source_tokens,
                "resolved_tokens": resolved_tokens,
                "exact_target": prediction == str(row["target_text"]),
                "generated_completion_tokens": completion,
                "gold_source_span": list(example.source_span) if example.source_span else None,
                "predicted_source_span": (
                    None if predicted is None
                    else [predicted.predicted_start, predicted.predicted_end]
                ),
                "locator_logprob": None if predicted is None else predicted.locator_logprob,
            })
    localization = None
    if mode == "fim_predicted":
        located = [record for record in records if record["predicted_source_span"] is not None]
        exact = sum(
            record["predicted_source_span"] == record["gold_source_span"]
            for record in located
        )
        localization = {
            "examples": len(located),
            "exact": exact,
            "exact_rate": exact / len(located) if located else None,
        }

    _causal_likelihoods(model, device, records)
    return records, {
        "method": mode,
        "oracle_span": mode == "fim",
        "specialized_locator": mode == "fim_predicted",
        "localization": localization,
        "model_tag": model_tag,
        "step": int(meta["step"]),
        "tokenizer": meta.get("tokenizer"),
        "generation_config": {
            "generation_seed": generation_seed,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        },
    }


def _load_levenshtein(checkpoint_dir: Path, step: int, device):
    state, _optimizer, metadata = load_checkpoint(
        str(checkpoint_dir), step, device, load_optimizer=False
    )
    config = LevenshteinConfig(**metadata["model_config"])
    with torch.device("meta"):
        model = LevenshteinTransformer(config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(
        {key.removeprefix("_orig_mod."): value for key, value in state.items()},
        strict=True,
        assign=True,
    )
    return model, config, metadata


def _eval_levenshtein(
    manifest_rows: list[dict[str, Any]],
    *,
    checkpoint_dir: Path,
    step: int,
    device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, config, metadata = _load_levenshtein(checkpoint_dir, step, device)
    tokenizer = get_levenshtein_tokeniser()
    controls = {config.pad_id, config.mask_id, config.bos_id, config.eos_id}
    records = []
    with torch.inference_mode():
        for index, row in enumerate(manifest_rows):
            source_tokens = [int(token) for token in tokenizer.encode(str(row["source_text"]))]
            predicted, stats = model.iterative_decode(source_tokens, max_length=config.sequence_len)
            leaked = sum(token in controls for token in predicted)
            predicted = [token for token in predicted if token not in controls]
            valid = leaked == 0
            prediction = tokenizer.decode(predicted) if valid else str(row["source_text"])
            records.append({
                "benchmark_id": str(row.get("benchmark_id") or f"gec:{index}"),
                "method": "levenshtein",
                "status": "valid" if valid else "control_token_leak",
                "valid": valid,
                "source_text": str(row["source_text"]),
                "target_text": str(row["target_text"]),
                "prediction_text": prediction,
                "ungated_prediction_text": prediction,
                "exact_target": prediction == str(row["target_text"]),
                "edit_confidence": stats.get("mean_decision_logprob"),
                "decode_stats": stats,
            })
    return records, {"checkpoint_dir": str(checkpoint_dir), "step": int(metadata["step"]), "tokenizer": "levenshtein"}


def _summary(records: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    total = len(records)
    valid = sum(bool(record["valid"]) for record in records)
    changed = sum(record["prediction_text"] != record["source_text"] for record in records)
    exact = sum(bool(record["exact_target"]) for record in records)
    with_confidence = sum(isinstance(record.get("edit_confidence"), (int, float)) for record in records)
    return {
        **metadata,
        "examples": total,
        "valid": valid,
        "validity_rate": valid / total if total else None,
        "changed": changed,
        "change_rate": changed / total if total else None,
        "exact_target": exact,
        "exact_target_rate": exact / total if total else None,
        "confidence_field": "edit_confidence",
        "records_with_confidence": with_confidence,
        "confidence_is_target_blind": not bool(metadata.get("oracle_span", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("ar_dsl", "flat_relative", "regeneration", "fim", "fim_predicted", "levenshtein"),
        required=True,
    )
    parser.add_argument("--manifest-jsonl", required=True, type=Path)
    parser.add_argument("--model-tag")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--generation-seed", type=int, default=12345)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--skip-coordinate-teacher",
        action="store_true",
        help="Skip the sampling-independent AR coordinate teacher-forcing diagnostic.",
    )
    parser.add_argument("--device-type", default="")
    parser.add_argument("--output-records-jsonl", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "levenshtein" and args.checkpoint_dir is None:
        parser.error("--checkpoint-dir is required for Levenshtein")
    if args.mode != "levenshtein" and not args.model_tag:
        parser.error("--model-tag is required for causal methods")

    manifest_rows = _read_jsonl(args.manifest_jsonl, args.limit)
    device_type = args.device_type or autodetect_device_type()
    _distributed, rank, _local_rank, world_size, device = compute_init(device_type)
    if rank != 0 or world_size != 1:
        raise RuntimeError("method_gec_eval expects one process on one device")
    if args.mode in {"ar_dsl", "flat_relative"}:
        records, metadata = _eval_autoregressive(
            args.mode,
            manifest_rows,
            model_tag=str(args.model_tag),
            step=args.step,
            device=device,
            generation_seed=args.generation_seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            coordinate_teacher=not args.skip_coordinate_teacher,
        )
    elif args.mode in {"regeneration", "fim", "fim_predicted"}:
        records, metadata = _eval_text_baseline(
            args.mode,
            manifest_rows,
            model_tag=str(args.model_tag),
            step=args.step,
            device=device,
            generation_seed=args.generation_seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    else:
        records, metadata = _eval_levenshtein(
            manifest_rows,
            checkpoint_dir=args.checkpoint_dir,
            step=args.step,
            device=device,
        )
    summary = _summary(records, metadata)
    args.output_records_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_records_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    compute_cleanup()


if __name__ == "__main__":
    main()
