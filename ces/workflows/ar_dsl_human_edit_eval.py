"""Evaluate plain autoregressive edit-DSL checkpoints on human edit pairs."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanochat.causal_serialisation_extended import Delete, Insert, Replace, SequenceEdit, resolve
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.edit_dsl import (
    EDIT_SCRIPT_CLOSE,
    EDIT_SCRIPT_OPEN,
    ar_dsl_coordinate_token_spans,
    ar_dsl_to_sequence_edit,
    sequence_edit_to_ar_dsl,
)
from nanochat.engine import Engine


@dataclass(frozen=True)
class HumanEditExample:
    dataset: str
    split: str
    task: str
    example_id: str
    source_text: str
    target_text: str


def _iter_dataset_files(data_dir: Path, datasets: list[str], splits: list[str]) -> list[Path]:
    dataset_dirs = [p for p in sorted(data_dir.iterdir()) if p.is_dir()] if datasets == ["all"] else [data_dir / d for d in datasets]
    paths = []
    for dataset_dir in dataset_dirs:
        for split in splits:
            path = dataset_dir / f"{split}.parquet"
            if path.exists():
                paths.append(path)
    return paths


def _load_examples(paths: list[Path], limit: int | None) -> list[HumanEditExample]:
    examples: list[HumanEditExample] = []
    for path in paths:
        for row in pq.read_table(path).to_pylist():
            source = row.get("source_text")
            target = row.get("target_text")
            if not isinstance(source, str) or not isinstance(target, str):
                continue
            examples.append(
                HumanEditExample(
                    dataset=str(row.get("dataset") or path.parent.name),
                    split=str(row.get("split") or path.stem),
                    task=str(row.get("task") or "atomic"),
                    example_id=str(row.get("example_id") or len(examples)),
                    source_text=source,
                    target_text=target,
                )
            )
            if limit is not None and len(examples) >= limit:
                return examples
    return examples


def _token_edit_from_pair(tokenizer, source_text: str, target_text: str) -> SequenceEdit | None:
    bos = tokenizer.get_bos_token_id()
    source_tokens = [bos] + [int(t) for t in tokenizer.encode(source_text)]
    target_tokens = [bos] + [int(t) for t in tokenizer.encode(target_text)]
    matcher = difflib.SequenceMatcher(a=source_tokens, b=target_tokens, autojunk=False)
    edits = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            edits.append(Insert(i1, np.array(target_tokens[j1:j2], dtype=np.int64)))
        elif tag == "delete":
            edits.append(Delete(i1, i2))
        elif tag == "replace":
            edits.append(Replace(i1, i2, np.array(target_tokens[j1:j2], dtype=np.int64)))
    if not edits:
        return None
    return SequenceEdit(sequence=np.array(source_tokens, dtype=np.int64), edits=edits)


def _decode_without_bos(tokenizer, tokens: list[int] | np.ndarray) -> str:
    bos = tokenizer.get_bos_token_id()
    return tokenizer.decode([int(t) for t in np.array(tokens).tolist() if int(t) != bos])


def _similarity(a: str, b: str) -> float:
    return float(difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio())


def _empty_group() -> dict[str, Any]:
    return {
        "examples": 0,
        "serialised": 0,
        "skipped_too_long": 0,
        "teacher_token_loss_sum": 0.0,
        "teacher_token_count": 0,
        "position_correct": 0.0,
        "position_targets": 0.0,
        "position_loss_sum": 0.0,
        "position_token_correct": 0.0,
        "position_token_targets": 0.0,
        "position_token_loss_sum": 0.0,
        "generated": 0,
        "valid_generations": 0,
        "exact_matches": 0,
        "exact_edit_scripts": 0,
        "source_similarity_sum": 0.0,
        "generated_similarity_sum": 0.0,
        "improved_similarity": 0,
        "generation_status_counts": defaultdict(int),
        "gold_op_counts": defaultdict(int),
        "generated_op_counts": defaultdict(int),
    }


def _finalise_group(group: dict[str, Any]) -> dict[str, Any]:
    out = dict(group)
    status_counts = dict(out.pop("generation_status_counts"))
    out["gold_op_counts"] = dict(out.pop("gold_op_counts"))
    out["generated_op_counts"] = dict(out.pop("generated_op_counts"))
    generated = out["generated"]
    valid = out["valid_generations"]
    token_count = out["teacher_token_count"]
    pos_targets = out["position_targets"]
    pos_token_targets = out["position_token_targets"]
    out["teacher_token_loss"] = out["teacher_token_loss_sum"] / token_count if token_count else None
    out["position_accuracy"] = out["position_correct"] / pos_targets if pos_targets else None
    out["position_loss"] = out["position_loss_sum"] / pos_token_targets if pos_token_targets else None
    out["position_field_accuracy"] = out["position_accuracy"]
    out["position_field_targets"] = pos_targets
    out["position_token_accuracy"] = out["position_token_correct"] / pos_token_targets if pos_token_targets else None
    out["position_token_loss"] = out["position_loss"]
    out["valid_generation_rate"] = valid / generated if generated else None
    out["exact_match_rate"] = out["exact_matches"] / valid if valid else None
    out["exact_edit_script_rate"] = out["exact_edit_scripts"] / valid if valid else None
    out["mean_source_similarity"] = out["source_similarity_sum"] / generated if generated else None
    out["mean_generated_similarity"] = out["generated_similarity_sum"] / valid if valid else None
    out["similarity_improvement_rate"] = out["improved_similarity"] / valid if valid else None
    out["generation_status_counts"] = status_counts
    return out


def _group_keys(example: HumanEditExample) -> list[str]:
    return [
        "overall",
        f"dataset:{example.dataset}",
        f"split:{example.split}",
        f"task:{example.task}",
        f"dataset_split:{example.dataset}/{example.split}",
        f"dataset_task:{example.dataset}/{example.task}",
    ]


def _add_to_groups(groups: dict[str, dict[str, Any]], keys: list[str], update) -> None:
    for key in keys:
        update(groups[key])


def _extract_dsl(generated_text: str) -> tuple[str | None, str]:
    start = generated_text.find(EDIT_SCRIPT_OPEN)
    if start < 0:
        return None, "missing_open"
    close = generated_text.find(EDIT_SCRIPT_CLOSE, start + len(EDIT_SCRIPT_OPEN))
    if close < 0:
        return None, "missing_close"
    end = close + len(EDIT_SCRIPT_CLOSE)
    return generated_text[start:end], "valid"


def _edit_signature(seq_edit: SequenceEdit) -> tuple[tuple[Any, ...], ...]:
    out = []
    for edit in seq_edit.edits:
        if isinstance(edit, Insert):
            out.append(("insert", int(edit.pos), tuple(int(t) for t in np.array(edit.content).tolist())))
        elif isinstance(edit, Delete):
            out.append(("delete", int(edit.start), int(edit.end)))
        elif isinstance(edit, Replace):
            out.append(("replace", int(edit.start), int(edit.end), tuple(int(t) for t in np.array(edit.content).tolist())))
        else:
            out.append((type(edit).__name__, repr(edit)))
    return tuple(out)


def _op_name(edit: Any) -> str:
    if isinstance(edit, Insert):
        return "insert"
    if isinstance(edit, Delete):
        return "delete"
    if isinstance(edit, Replace):
        return "replace"
    return type(edit).__name__


def _teacher_stats(
    model,
    row: list[int],
    T: int,
    device: torch.device,
    autocast_ctx,
    coordinate_spans: list[tuple[int, int]],
) -> dict[str, float | int | None]:
    if len(row) < 2 or len(row) > T + 1:
        return {"token_loss_sum": None, "token_count": 0}
    x = torch.tensor(row[:-1], dtype=torch.long, device=device)[None, :]
    y = torch.tensor(row[1:], dtype=torch.long, device=device)[None, :]
    tix = torch.arange(x.shape[1], dtype=torch.long, device=device)[None, :]
    with autocast_ctx:
        logits = model(x, tix=tix)
    losses = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")
    pred = torch.argmax(logits[0], dim=-1)

    position_correct = 0
    position_targets = 0
    position_token_correct = 0
    position_token_targets = 0
    position_loss_sum = 0.0
    for start, end in coordinate_spans:
        target_indices = [ix - 1 for ix in range(start, end) if 0 < ix < len(row)]
        if not target_indices:
            continue
        target_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)
        token_matches = pred.index_select(0, target_tensor).eq(y[0].index_select(0, target_tensor))
        position_correct += int(bool(torch.all(token_matches).item()))
        position_targets += 1
        position_token_correct += int(token_matches.sum().item())
        position_token_targets += int(token_matches.numel())
        position_loss_sum += float(losses.index_select(0, target_tensor).sum().item())

    return {
        "token_loss_sum": float(losses.sum().item()),
        "token_count": int(y.numel()),
        "position_correct": position_correct,
        "position_targets": position_targets,
        "position_loss_sum": position_loss_sum,
        "position_token_correct": position_token_correct,
        "position_token_targets": position_token_targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an AR-DSL checkpoint on human source/target edit pairs.")
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--data-dir", default="data/human_edits")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval", default="teacher,generate", help="Comma-separated: teacher,generate.")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device-type", default="")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--include-examples", type=int, default=0)
    args = parser.parse_args()

    eval_modes = {part.strip() for part in args.eval.split(",") if part.strip()}
    unknown_modes = eval_modes - {"teacher", "generate"}
    if unknown_modes:
        raise ValueError(f"Unknown eval modes: {sorted(unknown_modes)}")

    data_dir = Path(args.data_dir)
    paths = _iter_dataset_files(data_dir, args.datasets, args.splits)
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _ddp, ddp_rank, _ddp_local_rank, _ddp_world_size, device = compute_init(device_type)
    if ddp_rank != 0:
        raise RuntimeError("ar_dsl_human_edit_eval.py is intended to run as a single process.")

    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    T = int(meta["model_config"]["sequence_len"])
    engine = Engine(model, tokenizer) if "generate" in eval_modes else None
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    examples = _load_examples(paths, args.limit)
    groups: dict[str, dict[str, Any]] = defaultdict(_empty_group)
    sample_outputs: list[dict[str, Any]] = []

    model.eval()
    with torch.inference_mode():
        for ix, example in enumerate(examples):
            keys = _group_keys(example)
            _add_to_groups(groups, keys, lambda g: g.__setitem__("examples", g["examples"] + 1))
            seq_edit = _token_edit_from_pair(tokenizer, example.source_text, example.target_text)
            if seq_edit is None:
                continue
            gold_signature = _edit_signature(seq_edit)
            for edit in seq_edit.edits:
                op = _op_name(edit)
                _add_to_groups(groups, keys, lambda g, op=op: g["gold_op_counts"].__setitem__(op, g["gold_op_counts"][op] + 1))
            gold_dsl = "\n" + sequence_edit_to_ar_dsl(seq_edit, tokenizer) + "\n"
            source_tokens = [int(t) for t in np.array(seq_edit.sequence).tolist()]
            gold_row = source_tokens + [int(t) for t in tokenizer.encode(gold_dsl)]
            if len(gold_row) > T + 1:
                _add_to_groups(groups, keys, lambda g: g.__setitem__("skipped_too_long", g["skipped_too_long"] + 1))
                continue
            _add_to_groups(groups, keys, lambda g: g.__setitem__("serialised", g["serialised"] + 1))

            if "teacher" in eval_modes:
                stats = _teacher_stats(
                    model,
                    gold_row,
                    T,
                    device,
                    autocast_ctx,
                    ar_dsl_coordinate_token_spans(tokenizer, gold_dsl, len(source_tokens)),
                )
                if stats["token_loss_sum"] is not None:
                    for key in keys:
                        groups[key]["teacher_token_loss_sum"] += float(stats["token_loss_sum"])
                        groups[key]["teacher_token_count"] += int(stats["token_count"])
                        groups[key]["position_correct"] += float(stats["position_correct"])
                        groups[key]["position_targets"] += float(stats["position_targets"])
                        groups[key]["position_loss_sum"] += float(stats["position_loss_sum"])
                        groups[key]["position_token_correct"] += float(stats["position_token_correct"])
                        groups[key]["position_token_targets"] += float(stats["position_token_targets"])

            if "generate" in eval_modes:
                assert engine is not None
                prompt = source_tokens + [int(t) for t in tokenizer.encode(f"\n{EDIT_SCRIPT_OPEN}")]
                budget = min(args.max_new_tokens, T - len(prompt))
                source_sim = _similarity(example.source_text, example.target_text)
                generated_text = None
                generated_seq_edit = None
                if budget <= 0:
                    status = "prompt_too_long"
                else:
                    sample, _, _ = engine.generate_batch(
                        prompt,
                        num_samples=1,
                        max_tokens=budget,
                        temperature=args.temperature,
                        seed=args.seed + ix,
                        no_edits=True,
                    )
                    text = tokenizer.decode(sample[0][len(source_tokens):])
                    dsl, status = _extract_dsl(text)
                    if dsl is not None:
                        try:
                            generated_seq_edit = ar_dsl_to_sequence_edit(source_tokens, dsl, tokenizer)
                            generated_text = _decode_without_bos(tokenizer, resolve(generated_seq_edit))
                        except Exception as exc:
                            status = f"parse_or_resolve_error:{type(exc).__name__}"

                for key in keys:
                    group = groups[key]
                    group["generated"] += 1
                    group["source_similarity_sum"] += source_sim
                    group["generation_status_counts"][status] += 1
                    if generated_text is None:
                        continue
                    generated_similarity = _similarity(generated_text, example.target_text)
                    group["valid_generations"] += 1
                    group["generated_similarity_sum"] += generated_similarity
                    group["exact_matches"] += int(generated_text == example.target_text)
                    group["improved_similarity"] += int(generated_similarity > source_sim)
                    group["exact_edit_scripts"] += int(
                        generated_seq_edit is not None and _edit_signature(generated_seq_edit) == gold_signature
                    )
                    if generated_seq_edit is not None:
                        for edit in generated_seq_edit.edits:
                            op = _op_name(edit)
                            group["generated_op_counts"][op] += 1

                if len(sample_outputs) < args.include_examples:
                    sample_outputs.append({
                        "dataset": example.dataset,
                        "split": example.split,
                        "task": example.task,
                        "example_id": example.example_id,
                        "status": status,
                        "source_text": example.source_text,
                        "target_text": example.target_text,
                        "generated_text": generated_text,
                    })

    result = {
        "model_tag": args.model_tag,
        "step": meta["step"],
        "data_dir": str(data_dir),
        "files": [str(p) for p in paths],
        "eval_modes": sorted(eval_modes),
        "max_seq_len": T,
        "num_loaded_examples": len(examples),
        "groups": {key: _finalise_group(value) for key, value in sorted(groups.items())},
    }
    if sample_outputs:
        result["examples"] = sample_outputs

    print(json.dumps(result, indent=2))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}")

    compute_cleanup()


if __name__ == "__main__":
    main()
