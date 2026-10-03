"""Score canonical target-resolving CoEdit programs under a CES checkpoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from nanochat.causal_serialisation_extended import SequenceEdit, TokensOperations
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.oracle_edit_likelihood import (
    canonical_sequence_edit_from_text,
    conditional_program_training_data,
    score_conditional_program,
)


def _load_manifest(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("source_text"), str) or not isinstance(
                row.get("target_text"), str
            ):
                raise ValueError(f"{path}:{line_number}: invalid source/target")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _bucket_length(length: int, maximum: int) -> int | None:
    for bucket in (256, 512, 1024, 1536, 2048):
        if length <= bucket and bucket <= maximum:
            return bucket
    return maximum if length <= maximum else None


def _program_dict(score) -> dict:
    return {
        "token_nll": score.token_nll,
        "pointer_nll": score.pointer_nll,
        "total_nll": score.total_nll,
        "token_decisions": score.token_decisions,
        "pointer_decisions": score.pointer_decisions,
        "decisions": score.decisions,
        "bits_per_decision": score.bits_per_decision,
        "serialized_input_tokens": score.serialized_input_tokens,
    }


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--manifest-jsonl", required=True, type=Path)
    parser.add_argument("--output-records-jsonl", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--position-buffer-size", type=int, default=64)
    args = parser.parse_args()

    device_type = autodetect_device_type()
    _ddp, rank, _local_rank, _world_size, device = compute_init(device_type)
    if rank != 0:
        raise RuntimeError("Oracle likelihood evaluation is single-process only")
    model, tokenizer, metadata = load_model(
        "base",
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    model.eval()
    token_ops = TokensOperations(tokenizer.enc)
    maximum = int(metadata["model_config"]["sequence_len"])
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    records = []
    status_counts: dict[str, int] = {}
    for index, row in enumerate(_load_manifest(args.manifest_jsonl, args.limit)):
        status = "ok"
        output = {
            "benchmark_id": str(
                row.get("benchmark_id") or row.get("example_id") or index
            ),
            "source_text": row["source_text"],
            "target_text": row["target_text"],
        }
        try:
            oracle_edit = canonical_sequence_edit_from_text(
                tokenizer, row["source_text"], row["target_text"]
            )
            no_edit = SequenceEdit(
                sequence=np.asarray(oracle_edit.sequence, dtype=np.int64),
                edits=[],
            )
            oracle_data = conditional_program_training_data(
                token_ops,
                oracle_edit,
                position_buffer_size=args.position_buffer_size,
            )
            noop_data = conditional_program_training_data(
                token_ops,
                no_edit,
                position_buffer_size=args.position_buffer_size,
            )
            if bool(oracle_data.target.overflow):
                status = "position_buffer_overflow"
            else:
                bucket = _bucket_length(
                    max(
                        len(oracle_data.input_tokens),
                        len(noop_data.input_tokens),
                    ),
                    maximum,
                )
                if bucket is None:
                    status = "too_long"
                else:
                    oracle_score = score_conditional_program(
                        model,
                        oracle_data,
                        device=device,
                        padded_length=bucket,
                        autocast_ctx=autocast_ctx,
                    )
                    noop_score = score_conditional_program(
                        model,
                        noop_data,
                        device=device,
                        padded_length=bucket,
                        autocast_ctx=autocast_ctx,
                    )
                    output.update(
                        {
                            "operation_count": len(oracle_edit.edits),
                            "oracle": _program_dict(oracle_score),
                            "noop": _program_dict(noop_score),
                            "correct_vs_noop_log_odds": (
                                noop_score.total_nll - oracle_score.total_nll
                            ),
                            "correct_vs_noop_bits_per_decision_delta": (
                                (noop_score.bits_per_decision or 0.0)
                                - (oracle_score.bits_per_decision or 0.0)
                            ),
                        }
                    )
        except Exception as exc:
            status = f"error:{type(exc).__name__}"
            output["error"] = str(exc)
        output["status"] = status
        status_counts[status] = status_counts.get(status, 0) + 1
        records.append(output)
        if (index + 1) % 50 == 0:
            print(f"Scored {index + 1}/{len(records) if args.limit else '?'}", flush=True)

    ok = [row for row in records if row["status"] == "ok"]
    flat = []
    for row in ok:
        flat.append(
            {
                "oracle_total_nll": row["oracle"]["total_nll"],
                "oracle_token_nll": row["oracle"]["token_nll"],
                "oracle_pointer_nll": row["oracle"]["pointer_nll"],
                "oracle_bits_per_decision": row["oracle"]["bits_per_decision"],
                "noop_total_nll": row["noop"]["total_nll"],
                "correct_vs_noop_log_odds": row["correct_vs_noop_log_odds"],
                "correct_vs_noop_bits_per_decision_delta": row[
                    "correct_vs_noop_bits_per_decision_delta"
                ],
                "operation_count": row["operation_count"],
            }
        )
    summary = {
        "model_tag": args.model_tag,
        "step": args.step,
        "manifest_jsonl": str(args.manifest_jsonl.resolve()),
        "loaded_examples": len(records),
        "scored_examples": len(ok),
        "coverage": len(ok) / len(records) if records else 0.0,
        "status_counts": status_counts,
        "canonicalizer": {
            "alignment": "minimum unit-cost token Levenshtein",
            "operations": ["insert", "delete", "replace"],
            "marginalization": "one deterministic canonical program",
        },
        "likelihood_definition": (
            "Teacher-forced CES edit suffix only; token NLL uses training token "
            "masks and pointer NLL uses the gathered causal position-head loss."
        ),
        "metrics": {
            key: _mean(flat, key)
            for key in (
                "oracle_total_nll",
                "oracle_token_nll",
                "oracle_pointer_nll",
                "oracle_bits_per_decision",
                "noop_total_nll",
                "correct_vs_noop_log_odds",
                "correct_vs_noop_bits_per_decision_delta",
                "operation_count",
            )
        },
    }
    args.output_records_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_records_jsonl.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    compute_cleanup()


if __name__ == "__main__":
    main()
