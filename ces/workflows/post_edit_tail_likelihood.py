"""Measure clean-tail likelihood after a supplied CES correction program.

The CES evaluation compares six causal histories for the same held-out tail:

* clean source text;
* corrupted source text;
* clean source followed by an empty edit program;
* corrupted source followed by an empty edit program;
* corrupted source followed by a valid, non-correcting sham program; and
* corrupted source followed by the gold correcting program.

The standard-AR mode evaluates the two plain-text histories on the exact same
case manifest. Tail tokens are always teacher forced and excluded from the
prefix; only their conditional likelihood is accumulated.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from nanochat.causal_serialisation_extended import (
    Delete,
    Insert,
    Replace,
    SequenceEdit,
    TokensOperations,
    resolve,
)
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.tokenizer import get_extended_indel_tokeniser


CES_CONDITIONS = (
    "clean_plain",
    "corrupt_plain",
    "clean_noop_program",
    "corrupt_noop_program",
    "corrupt_sham_program",
    "corrupt_gold_program",
)
STANDARD_CONDITIONS = ("clean_plain", "corrupt_plain")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def edit_from_spec(spec: dict[str, Any]):
    content = np.asarray(spec.get("content", []), dtype=np.int64)
    if spec["type"] == "replace":
        return Replace(int(spec["start"]), int(spec["end"]), content)
    if spec["type"] == "insert":
        return Insert(int(spec["position"]), content)
    if spec["type"] == "delete":
        return Delete(int(spec["start"]), int(spec["end"]))
    raise ValueError(f"Unknown edit type: {spec['type']!r}")


def resolved_tokens(source: list[int], spec: dict[str, Any]) -> list[int]:
    sequence_edit = SequenceEdit(
        sequence=np.asarray(source, dtype=np.int64),
        edits=[edit_from_spec(spec)],
    )
    return [int(token) for token in resolve(sequence_edit).tolist()]


def _donor_span(
    tokens: list[int],
    length: int,
    *,
    rng: random.Random,
    forbidden: list[list[int]],
    forbidden_interval: tuple[int, int],
) -> list[int]:
    low, high = 1, len(tokens) - length
    if high < low:
        raise ValueError("Prefix is too short for donor sampling")
    for _ in range(256):
        start = rng.randint(low, high)
        end = start + length
        fstart, fend = forbidden_interval
        if max(start, fstart) < min(end, fend):
            continue
        candidate = [int(token) for token in tokens[start:end]]
        if candidate and all(candidate != previous for previous in forbidden):
            return candidate
    raise ValueError("Could not sample a distinct non-overlapping donor span")


def make_corruption(
    clean_prefix: list[int],
    *,
    operation: str,
    span_length: int,
    edit_distance: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Create one corruption plus gold and same-family sham repair programs."""

    if span_length <= 0 or edit_distance < 0:
        raise ValueError("Span length must be positive and distance non-negative")
    edit_end = len(clean_prefix) - edit_distance
    edit_start = edit_end - span_length
    if edit_start <= 0 or edit_end > len(clean_prefix):
        raise ValueError("Requested edit would touch BOS or leave the prefix")
    clean_span = [int(token) for token in clean_prefix[edit_start:edit_end]]
    donor = _donor_span(
        clean_prefix,
        span_length,
        rng=rng,
        forbidden=[clean_span],
        forbidden_interval=(edit_start, edit_end),
    )
    donor_two = _donor_span(
        clean_prefix,
        span_length,
        rng=rng,
        forbidden=[clean_span, donor],
        forbidden_interval=(edit_start, edit_end),
    )

    if operation == "replace":
        corrupt = clean_prefix[:edit_start] + donor + clean_prefix[edit_end:]
        gold = {
            "type": "replace", "start": edit_start, "end": edit_end,
            "content": clean_span,
        }
        sham = {
            "type": "replace", "start": edit_start, "end": edit_end,
            "content": donor_two,
        }
    elif operation == "delete":
        corrupt = clean_prefix[:edit_start] + clean_prefix[edit_end:]
        gold = {"type": "insert", "position": edit_start, "content": clean_span}
        sham = {"type": "insert", "position": edit_start, "content": donor_two}
    elif operation == "insert":
        insert_at = edit_end
        corrupt = clean_prefix[:insert_at] + donor + clean_prefix[insert_at:]
        gold = {"type": "delete", "start": insert_at, "end": insert_at + span_length}
        sham_start = insert_at - span_length
        if sham_start <= 0:
            raise ValueError("No room for the sham deletion before the inserted span")
        sham = {"type": "delete", "start": sham_start, "end": insert_at}
    else:
        raise ValueError(f"Unsupported operation: {operation!r}")

    gold_target = resolved_tokens(corrupt, gold)
    sham_target = resolved_tokens(corrupt, sham)
    if gold_target != clean_prefix:
        raise AssertionError("Gold program does not restore the clean prefix")
    if sham_target == clean_prefix or sham_target == corrupt:
        raise AssertionError("Sham program must be valid but non-correcting")
    return {
        "corrupt_prefix": corrupt,
        "gold_edit": gold,
        "sham_edit": sham,
        "gold_target": gold_target,
        "sham_target": sham_target,
        "clean_edit_start": edit_start if operation != "insert" else edit_end,
        "clean_edit_end": edit_end,
    }


def build_case_manifest(
    *,
    validation_parquet: Path,
    prefix_tokens: int,
    tail_tokens: int,
    operations: list[str],
    span_lengths: list[int],
    edit_distances: list[int],
    cases_per_stratum: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    if prefix_tokens <= 1 or tail_tokens <= 0 or cases_per_stratum <= 0:
        raise ValueError("Prefix, tail, and cases-per-stratum must be positive")
    tokenizer = get_extended_indel_tokeniser()
    bos = int(tokenizer.get_bos_token_id())
    strata = list(itertools.product(operations, span_lengths, edit_distances))
    assignments = strata * cases_per_stratum
    rng = random.Random(seed)
    rng.shuffle(assignments)

    table = pq.read_table(validation_parquet, columns=["text"])
    row_indices = list(range(table.num_rows))
    rng.shuffle(row_indices)
    rows: list[dict[str, Any]] = []
    attempted = 0
    for row_index in row_indices:
        if len(rows) >= len(assignments):
            break
        attempted += 1
        text = str(table.column("text")[row_index].as_py())
        ordinary = [int(token) for token in tokenizer.encode(text)]
        required = prefix_tokens - 1 + tail_tokens
        if len(ordinary) < required:
            continue
        window_start = rng.randint(0, len(ordinary) - required)
        window = ordinary[window_start:window_start + required]
        clean_prefix = [bos, *window[:prefix_tokens - 1]]
        tail = window[prefix_tokens - 1:]
        operation, span_length, edit_distance = assignments[len(rows)]
        try:
            corruption = make_corruption(
                clean_prefix,
                operation=operation,
                span_length=int(span_length),
                edit_distance=int(edit_distance),
                rng=rng,
            )
        except (ValueError, AssertionError):
            continue
        tail_byte_lengths = [
            len(tokenizer.enc.decode_single_token_bytes(int(token))) for token in tail
        ]
        rows.append({
            "case_id": f"val:{row_index}:{window_start}:{operation}:s{span_length}:d{edit_distance}",
            "validation_row": row_index,
            "window_start": window_start,
            "document_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "operation": operation,
            "repair_operation": corruption["gold_edit"]["type"],
            "span_length": int(span_length),
            "edit_distance": int(edit_distance),
            "clean_prefix": clean_prefix,
            "corrupt_prefix": corruption["corrupt_prefix"],
            "tail": tail,
            "tail_byte_lengths": tail_byte_lengths,
            "gold_edit": corruption["gold_edit"],
            "sham_edit": corruption["sham_edit"],
            "sham_target": corruption["sham_target"],
        })
    if len(rows) != len(assignments):
        raise RuntimeError(
            f"Built {len(rows)} of {len(assignments)} requested cases after {attempted} rows"
        )
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[f"{row['operation']}:s{row['span_length']}:d{row['edit_distance']}"] += 1
    audit = {
        "validation_parquet": str(validation_parquet),
        "validation_parquet_sha256": hashlib.sha256(validation_parquet.read_bytes()).hexdigest(),
        "prefix_tokens": prefix_tokens,
        "tail_tokens": tail_tokens,
        "operations": operations,
        "span_lengths": span_lengths,
        "edit_distances": edit_distances,
        "cases_per_stratum": cases_per_stratum,
        "cases": len(rows),
        "validation_rows_attempted": attempted,
        "seed": seed,
        "stratum_counts": dict(sorted(counts.items())),
    }
    return rows, audit


def plain_prefix(tokens: list[int]) -> tuple[list[int], list[int]]:
    return tokens, list(range(len(tokens)))


def program_prefix(
    token_ops: TokensOperations,
    source: list[int],
    edits: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    sequence_edit = SequenceEdit(
        sequence=np.asarray(source, dtype=np.int64),
        edits=[edit_from_spec(spec) for spec in edits],
    )
    serialised = token_ops.serialise(sequence_edit, indel_bos_pos=1)
    return (
        [int(token) for token in serialised.tokens],
        [int(position) for position in serialised.positions],
    )


def build_condition_prefixes(
    case: dict[str, Any], token_ops: TokensOperations | None, *, role: str
) -> dict[str, tuple[list[int], list[int]]]:
    clean = [int(token) for token in case["clean_prefix"]]
    corrupt = [int(token) for token in case["corrupt_prefix"]]
    output = {
        "clean_plain": plain_prefix(clean),
        "corrupt_plain": plain_prefix(corrupt),
    }
    if role == "standard":
        return output
    if token_ops is None:
        raise ValueError("CES conditions require TokensOperations")
    output.update({
        "clean_noop_program": program_prefix(token_ops, clean, []),
        "corrupt_noop_program": program_prefix(token_ops, corrupt, []),
        "corrupt_sham_program": program_prefix(token_ops, corrupt, [case["sham_edit"]]),
        "corrupt_gold_program": program_prefix(token_ops, corrupt, [case["gold_edit"]]),
    })
    return output


@torch.inference_mode()
def score_tail(
    model,
    *,
    prefix_tokens: list[int],
    prefix_positions: list[int],
    tail_tokens: list[int],
    tail_byte_lengths: list[int],
) -> dict[str, Any]:
    if len(prefix_tokens) != len(prefix_positions):
        raise ValueError("Prefix token and position lengths differ")
    if len(tail_tokens) != len(tail_byte_lengths):
        raise ValueError("Tail token and byte-length counts differ")
    full_tokens = prefix_tokens + tail_tokens
    if len(full_tokens) > int(model.config.sequence_len):
        raise ValueError(
            f"Serialized prefix and tail use {len(full_tokens)} tokens, exceeding "
            f"context {model.config.sequence_len}"
        )
    full_positions = prefix_positions + list(range(len(prefix_tokens), len(full_tokens)))
    device = model.get_device()
    inputs = torch.as_tensor([full_tokens[:-1]], dtype=torch.long, device=device)
    positions = torch.as_tensor([full_positions[:-1]], dtype=torch.long, device=device)
    targets = torch.as_tensor(tail_tokens, dtype=torch.long, device=device)
    logits = model(inputs, tix=positions)
    start = len(prefix_tokens) - 1
    tail_logits = logits[0, start:start + len(tail_tokens)].float()
    log_probs = torch.log_softmax(tail_logits, dim=-1)
    nll = -log_probs.gather(1, targets[:, None]).squeeze(1)
    values = [float(value) for value in nll.cpu().tolist()]
    total_nll = float(sum(values))
    total_bytes = int(sum(tail_byte_lengths))
    return {
        "prefix_tokens": len(prefix_tokens),
        "tail_tokens": len(tail_tokens),
        "tail_bytes": total_bytes,
        "tail_nll": total_nll,
        "tail_mean_nll": total_nll / len(tail_tokens),
        "tail_bits_per_byte": total_nll / math.log(2.0) / total_bytes,
        "tail_token_nll": values,
    }


def evaluate_cases(
    model,
    tokenizer,
    cases: list[dict[str, Any]],
    *,
    role: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    conditions = STANDARD_CONDITIONS if role == "standard" else CES_CONDITIONS
    token_ops = TokensOperations(tokenizer.enc) if role == "ces" else None
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for case_index, case in enumerate(cases):
        try:
            prefixes = build_condition_prefixes(case, token_ops, role=role)
            scored = {
                condition: score_tail(
                    model,
                    prefix_tokens=prefixes[condition][0],
                    prefix_positions=prefixes[condition][1],
                    tail_tokens=[int(token) for token in case["tail"]],
                    tail_byte_lengths=[int(value) for value in case["tail_byte_lengths"]],
                )
                for condition in conditions
            }
        except (ValueError, RuntimeError) as exc:
            skipped.append({"case_id": case["case_id"], "reason": str(exc)})
            continue
        records.append({
            "case_id": case["case_id"],
            "validation_row": case["validation_row"],
            "operation": case["operation"],
            "repair_operation": case["repair_operation"],
            "span_length": case["span_length"],
            "edit_distance": case["edit_distance"],
            "conditions": scored,
        })
        print(f"{case_index + 1:04d}/{len(cases):04d} {case['case_id']}", flush=True)
    return records, skipped


def aggregate_condition(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    values = [row["conditions"][condition] for row in rows]
    total_nll = sum(float(value["tail_nll"]) for value in values)
    total_tokens = sum(int(value["tail_tokens"]) for value in values)
    total_bytes = sum(int(value["tail_bytes"]) for value in values)
    per_position = np.asarray([value["tail_token_nll"] for value in values], dtype=np.float64)
    return {
        "tail_nll": total_nll,
        "tail_tokens": total_tokens,
        "tail_bytes": total_bytes,
        "mean_nll": total_nll / total_tokens,
        "bits_per_byte": total_nll / math.log(2.0) / total_bytes,
        "mean_token_nll_by_tail_position": per_position.mean(axis=0).tolist(),
    }


def summarise_group(rows: list[dict[str, Any]], conditions: tuple[str, ...]) -> dict[str, Any]:
    condition_summary = {
        condition: aggregate_condition(rows, condition) for condition in conditions
    }
    output: dict[str, Any] = {
        "examples": len(rows),
        "conditions": condition_summary,
    }
    if "corrupt_gold_program" in conditions:
        clean = condition_summary["clean_noop_program"]["mean_nll"]
        corrupt = condition_summary["corrupt_noop_program"]["mean_nll"]
        sham = condition_summary["corrupt_sham_program"]["mean_nll"]
        repair = condition_summary["corrupt_gold_program"]["mean_nll"]
        damage = corrupt - clean
        output.update({
            "corruption_excess_mean_nll": damage,
            "repair_mean_nll_gain_vs_corrupt_noop": corrupt - repair,
            "repair_mean_nll_gain_vs_sham": sham - repair,
            "aggregate_recovery_fraction": (corrupt - repair) / damage if damage > 0 else None,
        })
    return output


def summarise_records(
    records: list[dict[str, Any]], *, role: str
) -> dict[str, Any]:
    conditions = STANDARD_CONDITIONS if role == "standard" else CES_CONDITIONS
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups["overall"].append(row)
        groups[f"operation:{row['operation']}"].append(row)
        groups[f"span:{row['span_length']}"].append(row)
        groups[f"distance:{row['edit_distance']}"].append(row)
        groups[
            f"stratum:{row['operation']}:s{row['span_length']}:d{row['edit_distance']}"
        ].append(row)
    return {
        name: summarise_group(rows, conditions) for name, rows in sorted(groups.items())
    }


def render_report(summary: dict[str, Any], audit: dict[str, Any], *, role: str) -> str:
    overall = summary["overall"]
    lines = [
        "# Held-out clean-tail likelihood after CES programs",
        "",
        f"Checkpoint: `{audit['model_tag']}` at step `{audit['step']}`; role: `{role}`.",
        "",
    ]
    if role == "ces":
        lines.extend([
            "| Group | N | Clean no-op BPB | Corrupt no-op BPB | Sham BPB | Repair BPB | Recovery |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        selected = ["overall"]
        selected += [name for name in summary if name.startswith("operation:")]
        selected += [name for name in summary if name.startswith("span:")]
        selected += [name for name in summary if name.startswith("distance:")]
        for name in selected:
            group = summary[name]
            conditions = group["conditions"]
            recovery = group["aggregate_recovery_fraction"]
            recovery_text = f"{100 * recovery:.1f}%" if recovery is not None else "n/a"
            lines.append(
                f"| {name} | {group['examples']} | "
                f"{conditions['clean_noop_program']['bits_per_byte']:.4f} | "
                f"{conditions['corrupt_noop_program']['bits_per_byte']:.4f} | "
                f"{conditions['corrupt_sham_program']['bits_per_byte']:.4f} | "
                f"{conditions['corrupt_gold_program']['bits_per_byte']:.4f} | "
                f"{recovery_text} |"
            )
    else:
        conditions = overall["conditions"]
        lines.extend([
            "| Group | N | Clean BPB | Corrupt BPB |",
            "|---|---:|---:|---:|",
            f"| overall | {overall['examples']} | "
            f"{conditions['clean_plain']['bits_per_byte']:.4f} | "
            f"{conditions['corrupt_plain']['bits_per_byte']:.4f} |",
        ])
    lines.extend([
        "",
        "All tails are held-out ClimbMix validation tokens and are identical across",
        "conditions. The sham is a valid same-operation-family program of matched",
        "span size that does not restore the clean prefix. Recovery is computed from",
        "dataset-level mean token NLL, not by averaging unstable per-example ratios.",
        "",
        "```json",
        json.dumps(audit, indent=2),
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-jsonl", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--validation-parquet",
        type=Path,
        default=Path("data/climbmix/shard_06542.parquet"),
    )
    parser.add_argument("--prefix-tokens", type=int, default=384)
    parser.add_argument("--tail-tokens", type=int, default=128)
    parser.add_argument("--operations", nargs="+", default=["replace", "delete", "insert"])
    parser.add_argument("--span-lengths", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--edit-distances", nargs="+", type=int, default=[16, 64, 128])
    parser.add_argument("--cases-per-stratum", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--model-tag")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--role", choices=("ces", "standard"), default="ces")
    parser.add_argument("--device-type", default="")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.cases_jsonl.exists():
        cases, case_audit = build_case_manifest(
            validation_parquet=args.validation_parquet,
            prefix_tokens=args.prefix_tokens,
            tail_tokens=args.tail_tokens,
            operations=args.operations,
            span_lengths=args.span_lengths,
            edit_distances=args.edit_distances,
            cases_per_stratum=args.cases_per_stratum,
            seed=args.seed,
        )
        write_jsonl(args.cases_jsonl, cases)
        args.cases_jsonl.with_suffix(".meta.json").write_text(
            json.dumps(case_audit, indent=2), encoding="utf-8"
        )
    if args.prepare_only:
        print(f"Prepared {args.cases_jsonl}")
        return
    if not args.model_tag or args.output_dir is None:
        parser.error("--model-tag and --output-dir are required unless --prepare-only")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "records": args.output_dir / "records.jsonl",
        "summary": args.output_dir / "summary.json",
        "report": args.output_dir / "report.md",
    }
    if not args.overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("Output exists; pass --overwrite or choose a new directory")

    cases = read_jsonl(args.cases_jsonl)
    if args.limit > 0:
        cases = cases[:args.limit]
    device_type = args.device_type or autodetect_device_type()
    _ddp, rank, _local_rank, world_size, device = compute_init(device_type)
    if rank != 0 or world_size != 1:
        raise RuntimeError("Run this evaluator with one process")
    model, tokenizer, meta = load_model(
        "base", device, phase="eval", model_tag=args.model_tag, step=args.step
    )
    model.eval()
    if args.role == "ces" and tokenizer.get_vocab_size() < 32768:
        raise ValueError("CES role requires the extended edit tokenizer")
    records, skipped = evaluate_cases(model, tokenizer, cases, role=args.role)
    if not records:
        raise RuntimeError("No cases were evaluated")
    summary = summarise_records(records, role=args.role)
    audit = {
        "model_tag": args.model_tag,
        "step": int(meta["step"]),
        "role": args.role,
        "model_config": meta["model_config"],
        "cases_jsonl": str(args.cases_jsonl),
        "cases_sha256": hashlib.sha256(args.cases_jsonl.read_bytes()).hexdigest(),
        "requested_cases": len(cases),
        "evaluated_cases": len(records),
        "skipped_cases": skipped,
        "conditions": list(STANDARD_CONDITIONS if args.role == "standard" else CES_CONDITIONS),
    }
    write_jsonl(paths["records"], records)
    paths["summary"].write_text(
        json.dumps({"audit": audit, "groups": summary}, indent=2), encoding="utf-8"
    )
    paths["report"].write_text(render_report(summary, audit, role=args.role), encoding="utf-8")
    print(json.dumps({"audit": audit, "overall": summary["overall"]}, indent=2))
    compute_cleanup()


if __name__ == "__main__":
    main()
