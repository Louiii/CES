"""Evaluate causal edit checkpoints on normalized human edit pair datasets."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanochat.causal_serialisation_extended import (
    Copy,
    Delete,
    Insert,
    Move,
    Replace,
    SequenceEdit,
    SerialisedData,
    TokensOperations,
    TrainingData,
    TrainingTarget,
)
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, print0
from nanochat.ces_rl import replay_policy_logprobs
from nanochat.engine_extended import Engine


@dataclass(frozen=True)
class HumanEditExample:
    dataset: str
    split: str
    task: str
    example_id: str
    source_text: str
    target_text: str


def _iter_dataset_files(data_dir: Path, datasets: list[str], splits: list[str]) -> list[Path]:
    paths = []
    dataset_dirs = [p for p in sorted(data_dir.iterdir()) if p.is_dir()] if datasets == ["all"] else [data_dir / d for d in datasets]
    for dataset_dir in dataset_dirs:
        for split in splits:
            path = dataset_dir / f"{split}.parquet"
            if path.exists():
                paths.append(path)
    return paths


def _load_examples(paths: list[Path], limit: int | None) -> list[HumanEditExample]:
    examples = []
    for path in paths:
        table = pq.read_table(path)
        for row in table.to_pylist():
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


def _load_manifest_examples(path: Path, limit: int | None) -> list[HumanEditExample]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source_text")
            target = row.get("target_text")
            if not isinstance(source, str) or not isinstance(target, str):
                raise ValueError(
                    f"{path}:{line_number}: source_text and target_text must be strings"
                )
            examples.append(
                HumanEditExample(
                    dataset=str(row.get("dataset") or "manifest"),
                    split=str(row.get("role") or row.get("split") or path.stem),
                    task=str(row.get("task") or "gec"),
                    example_id=str(
                        row.get("benchmark_id")
                        or row.get("example_id")
                        or f"{path.name}:{line_number}"
                    ),
                    source_text=source,
                    target_text=target,
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def _token_edit_from_pair(tokenizer, source_text: str, target_text: str) -> SequenceEdit:
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
    return SequenceEdit(sequence=np.array(source_tokens, dtype=np.int64), edits=edits)


def _pad_training_data(training_data: TrainingData, T: int) -> TrainingData:
    n = len(training_data.input_tokens)
    if n > T:
        raise ValueError(f"serialised edit example has length {n}, exceeds T={T}")
    pad = T - n
    return TrainingData(
        input_tokens=np.pad(training_data.input_tokens, (0, pad), constant_values=0),
        input_positions=np.pad(training_data.input_positions, (0, pad), constant_values=0),
        target=TrainingTarget(
            tokens=np.pad(training_data.target.tokens, (0, pad), constant_values=-1),
            gathered_pos_target=training_data.target.gathered_pos_target,
            token_loss_weights=np.pad(training_data.target.token_loss_weights, (0, pad), constant_values=False),
            gather_indices=training_data.target.gather_indices,
            gather_valid=training_data.target.gather_valid,
            overflow=training_data.target.overflow,
            perplexity_masks=training_data.target.perplexity_masks,
        ),
    )


def _stack_training_batch(rows: list[TrainingData], device: torch.device) -> TrainingData:
    use_cuda = device.type == "cuda"

    def ten(values):
        arr = np.stack(values)
        dtype = torch.long if np.issubdtype(arr.dtype, np.integer) else torch.bool
        tensor = torch.as_tensor(arr, dtype=dtype)
        if use_cuda:
            tensor = tensor.pin_memory()
        return tensor.to(device=device, non_blocking=use_cuda)

    return TrainingData(
        input_tokens=ten([r.input_tokens for r in rows]),
        input_positions=ten([r.input_positions for r in rows]),
        target=TrainingTarget(
            tokens=ten([r.target.tokens for r in rows]),
            gathered_pos_target=ten([r.target.gathered_pos_target for r in rows]),
            token_loss_weights=ten([r.target.token_loss_weights for r in rows]),
            gather_indices=ten([r.target.gather_indices for r in rows]),
            gather_valid=ten([r.target.gather_valid for r in rows]),
            overflow=ten([r.target.overflow for r in rows]),
            perplexity_masks=None,
        ),
    )


def _decode_without_bos(tokenizer, tokens: list[int]) -> str:
    bos = tokenizer.get_bos_token_id()
    return tokenizer.decode([int(t) for t in tokens if int(t) != bos])


def _similarity(a: str, b: str) -> float:
    return float(difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio())


def _resolved_change_features(source_text: str, candidate_text: str) -> tuple[tuple[str, ...], int]:
    matcher = difflib.SequenceMatcher(a=source_text, b=candidate_text, autojunk=False)
    changes = [opcode for opcode in matcher.get_opcodes() if opcode[0] != "equal"]
    signature = tuple(opcode[0] for opcode in changes)
    # Count removed and inserted characters separately.  This matches the
    # post-hoc acceptance analysis used to select the deployment gate: a
    # two-character replacement has mass four, not two.
    char_mass = sum(
        (i2 - i1) + (j2 - j1) for _tag, i1, i2, j1, j2 in changes
    )
    return signature, char_mass


def _resolved_change_signature(source_text: str, candidate_text: str) -> tuple[str, ...]:
    return _resolved_change_features(source_text, candidate_text)[0]


@torch.no_grad()
def _seq_avg_logprob(model, tokens: list[int], device: torch.device, autocast_ctx) -> float | None:
    """Average ordinary autoregressive log-probability for a decoded sequence."""
    if len(tokens) < 3 or len(tokens) - 1 > model.config.sequence_len:
        return None
    x = torch.tensor(tokens[:-1], device=device, dtype=torch.long)[None, :]
    y = torch.tensor(tokens[1:], device=device, dtype=torch.long)[None, :]
    tix = torch.arange(x.shape[1], device=device, dtype=torch.long)[None, :]
    with autocast_ctx:
        logits = model(x, tix=tix)
    logp = F.log_softmax(logits, dim=-1)
    return float(logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).mean().item())


def _resolve_generated(tokens: list[int], positions: list[int], token_ops: TokensOperations) -> tuple[list[int] | None, str]:
    tokens_arr = np.array(tokens, dtype=np.int64)
    positions_arr = np.array(positions, dtype=np.int64)
    edit_end_count = int(np.sum(tokens_arr == token_ops.edit_end_token))
    if edit_end_count != 1:
        return None, f"invalid_edit_end_count_{edit_end_count}"
    try:
        resolved = token_ops.resolve(SerialisedData(tokens=tokens_arr, positions=positions_arr))
    except Exception as exc:
        return None, f"resolve_error:{type(exc).__name__}"
    return [int(t) for t in resolved], "valid"


def _empty_group() -> dict[str, Any]:
    return {
        "examples": 0,
        "serialised": 0,
        "skipped_too_long": 0,
        "position_correct": 0.0,
        "position_targets": 0.0,
        "position_loss_sum": 0.0,
        "token_loss_sum": 0.0,
        "teacher_batches": 0,
        "generated": 0,
        "valid_generations": 0,
        "exact_matches": 0,
        "source_similarity_sum": 0.0,
        "generated_similarity_sum": 0.0,
        "improved_similarity": 0,
        "exact_edit_scripts": 0,
        "final_similarity_with_source_fallback_sum": 0.0,
        "similarity_delta_with_source_fallback_sum": 0.0,
        "generated_edit_tokens": 0,
        "constraint_interventions": 0,
        "likelihood_scored_generations": 0,
        "likelihood_accepted_generations": 0,
        "likelihood_rejected_generations": 0,
        "forced_payload_closures": 0,
        "forced_program_closures": 0,
        "gold_op_counts": defaultdict(int),
        "generated_op_counts": defaultdict(int),
        "generation_status_counts": defaultdict(int),
    }


def _finalise_group(group: dict[str, Any]) -> dict[str, Any]:
    out = dict(group)
    status_counts = dict(out.pop("generation_status_counts"))
    out["gold_op_counts"] = dict(out.pop("gold_op_counts"))
    out["generated_op_counts"] = dict(out.pop("generated_op_counts"))
    pos_targets = out["position_targets"]
    generated = out["generated"]
    valid = out["valid_generations"]
    out["position_accuracy"] = out["position_correct"] / pos_targets if pos_targets else None
    out["position_loss"] = out["position_loss_sum"] / pos_targets if pos_targets else None
    out["mean_token_loss_per_batch"] = out["token_loss_sum"] / out["teacher_batches"] if out["teacher_batches"] else None
    out["valid_generation_rate"] = valid / generated if generated else None
    out["exact_match_rate"] = out["exact_matches"] / valid if valid else None
    out["unconditional_exact_match_rate"] = out["exact_matches"] / generated if generated else None
    out["exact_edit_script_rate"] = out["exact_edit_scripts"] / valid if valid else None
    out["mean_source_similarity"] = out["source_similarity_sum"] / generated if generated else None
    out["mean_generated_similarity"] = out["generated_similarity_sum"] / valid if valid else None
    out["similarity_improvement_rate"] = out["improved_similarity"] / valid if valid else None
    out["unconditional_similarity_improvement_rate"] = out["improved_similarity"] / generated if generated else None
    out["mean_final_similarity_with_source_fallback"] = (
        out["final_similarity_with_source_fallback_sum"] / generated if generated else None
    )
    out["mean_similarity_delta_with_source_fallback"] = (
        out["similarity_delta_with_source_fallback_sum"] / generated if generated else None
    )
    out["mean_generated_edit_tokens"] = out["generated_edit_tokens"] / generated if generated else None
    out["constraint_intervention_rate"] = out["constraint_interventions"] / generated if generated else None
    scored = out["likelihood_scored_generations"]
    out["likelihood_acceptance_rate"] = (
        out["likelihood_accepted_generations"] / scored if scored else None
    )
    out["likelihood_rejection_rate"] = (
        out["likelihood_rejected_generations"] / scored if scored else None
    )
    out["generation_status_counts"] = status_counts
    return out


def _add_to_groups(groups: dict[str, dict[str, Any]], keys: list[str], update) -> None:
    for key in keys:
        update(groups[key])


def _op_name(edit: Any) -> str:
    if isinstance(edit, Insert):
        return "insert"
    if isinstance(edit, Delete):
        return "delete"
    if isinstance(edit, Replace):
        return "replace"
    if isinstance(edit, Copy):
        return "copy"
    if isinstance(edit, Move):
        return "move"
    return type(edit).__name__


def _edit_signature(seq_edit: SequenceEdit) -> tuple[tuple[Any, ...], ...]:
    out = []
    for edit in seq_edit.edits:
        if isinstance(edit, Insert):
            out.append(("insert", int(edit.pos), tuple(int(t) for t in np.array(edit.content).tolist())))
        elif isinstance(edit, Delete):
            out.append(("delete", int(edit.start), int(edit.end)))
        elif isinstance(edit, Replace):
            out.append(("replace", int(edit.start), int(edit.end), tuple(int(t) for t in np.array(edit.content).tolist())))
        elif isinstance(edit, Copy):
            out.append(("copy", int(edit.start), int(edit.end), int(edit.dest)))
        elif isinstance(edit, Move):
            out.append(("move", int(edit.start), int(edit.end), int(edit.dest)))
        else:
            out.append((type(edit).__name__, repr(edit)))
    return tuple(out)


def _deserialise_generated(tokens: list[int], positions: list[int], token_ops: TokensOperations) -> tuple[SequenceEdit | None, str]:
    tokens_arr = np.array(tokens, dtype=np.int64)
    positions_arr = np.array(positions, dtype=np.int64)
    edit_end_count = int(np.sum(tokens_arr == token_ops.edit_end_token))
    if edit_end_count != 1:
        return None, f"invalid_edit_end_count_{edit_end_count}"
    try:
        seq_edit = token_ops.deserialise(SerialisedData(tokens=tokens_arr, positions=positions_arr))
    except Exception as exc:
        return None, f"resolve_error:{type(exc).__name__}"
    return seq_edit, "valid"


def _confidence_summary(token_logprobs, pointer_logprobs) -> dict[str, float | int | None]:
    """Summarise the selected program's stochastic policy decisions."""

    token_values = [float(value.detach().float().item()) for value in token_logprobs]
    pointer_values = [float(value.detach().float().item()) for value in pointer_logprobs]
    values = token_values + pointer_values
    return {
        "program_mean_logprob": sum(values) / len(values) if values else None,
        "program_min_logprob": min(values) if values else None,
        "program_total_logprob": sum(values) if values else None,
        "program_token_mean_logprob": sum(token_values) / len(token_values) if token_values else None,
        "program_pointer_mean_logprob": sum(pointer_values) / len(pointer_values) if pointer_values else None,
        "program_decision_count": len(values),
        "program_token_decision_count": len(token_values),
        "program_pointer_decision_count": len(pointer_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a causal edit checkpoint on human source/target edit pairs.")
    parser.add_argument("--model-tag", required=True, help="Checkpoint tag or absolute checkpoint directory.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step. Defaults to latest.")
    parser.add_argument(
        "--ces-pointer-token-mode",
        choices=("role_specific", "shared"),
        default=None,
        help="Override pointer serialization; default reads checkpoint metadata.",
    )
    parser.add_argument("--data-dir", default="data/human_edits")
    parser.add_argument(
        "--manifest-jsonl",
        default=None,
        help="Frozen source/target JSONL manifest. Ignores --data-dir/--datasets/--splits.",
    )
    parser.add_argument("--datasets", nargs="+", default=["all"], help="Dataset directories, or 'all'.")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--tasks", nargs="+", default=None, help="Optional task labels to retain before applying --limit.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum examples across all selected files.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--pos-buffer-size", type=int, default=64)
    parser.add_argument("--eval", default="teacher,generate", help="Comma-separated: teacher,generate.")
    parser.add_argument("--max-edit-tokens", type=int, default=128)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Samples per prompt; selects the first valid program without looking at the target (valid pass@k).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--pointer-temperature",
        type=float,
        default=None,
        help="Position-head sampling temperature; defaults to token temperature. Use 0 for greedy localization.",
    )
    parser.add_argument("--closure-bias", type=float, default=0.0)
    parser.add_argument("--closure-bias-min-payload-tokens", type=int, default=0)
    parser.add_argument("--max-payload-tokens", type=int, default=None)
    parser.add_argument("--max-edit-ops", type=int, default=None)
    parser.add_argument(
        "--first-decision-edit-threshold",
        type=float,
        default=None,
        help=(
            "Optional hierarchical GEC gate using logsumexp(delete, insert, "
            "replace) minus EDIT_END before operation selection."
        ),
    )
    parser.add_argument(
        "--accept-if-ar-logprob-improves",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply a target-blind deployment gate: keep the edit only when its ordinary autoregressive average log-probability improves.",
    )
    parser.add_argument(
        "--score-candidate-likelihood",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record target-blind candidate-minus-source AR log-probability without gating.",
    )
    parser.add_argument(
        "--acceptance-logprob-margin",
        type=float,
        default=0.0,
        help="Required edited-minus-source average AR log-probability for the likelihood gate.",
    )
    parser.add_argument(
        "--accept-resolved-signature",
        choices=("insert", "delete", "replace"),
        default=None,
        help="Target-blind abstention gate requiring exactly one primitive resolved source-to-output change of this type.",
    )
    parser.add_argument(
        "--accept-resolved-max-chars",
        type=int,
        default=None,
        help="Target-blind abstention gate limiting total resolved source-to-output changed characters.",
    )
    parser.add_argument(
        "--accept-resolved-max-regions",
        type=int,
        default=None,
        help="Target-blind abstention gate limiting primitive source-to-output change regions.",
    )
    parser.add_argument(
        "--stop-after-edit-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop terminal human-edit generation after the first complete edit program.",
    )
    parser.add_argument(
        "--score-program-confidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replay each selected program and record token/pointer policy confidence.",
    )
    parser.add_argument(
        "--accept-program-mean-logprob-min",
        type=float,
        default=None,
        help="Target-blind gate requiring the selected program's mean policy log-probability to meet this floor.",
    )
    parser.add_argument(
        "--confidence-temperature",
        type=float,
        default=1.0,
        help="Untruncated policy temperature used only to score program confidence.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device-type", default="", help="cuda|cpu|mps; empty = autodetect.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--output-records-jsonl",
        default=None,
        help="Write every generated source/target/prediction record for external scoring.",
    )
    parser.add_argument("--include-examples", type=int, default=0)
    args = parser.parse_args()
    if args.num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if args.accept_resolved_max_chars is not None and args.accept_resolved_max_chars < 1:
        raise ValueError("accept-resolved-max-chars must be >= 1")
    if args.accept_resolved_max_regions is not None and args.accept_resolved_max_regions < 1:
        raise ValueError("accept-resolved-max-regions must be >= 1")
    if args.confidence_temperature <= 0:
        raise ValueError("--confidence-temperature must be positive")
    if args.accept_program_mean_logprob_min is not None and args.accept_program_mean_logprob_min > 0:
        raise ValueError("--accept-program-mean-logprob-min cannot exceed zero")

    eval_modes = {part.strip() for part in args.eval.split(",") if part.strip()}
    unknown_modes = eval_modes - {"teacher", "generate"}
    if unknown_modes:
        raise ValueError(f"Unknown eval modes: {sorted(unknown_modes)}")

    data_dir = Path(args.data_dir)
    manifest_path = Path(args.manifest_jsonl) if args.manifest_jsonl else None
    paths = [] if manifest_path else _iter_dataset_files(data_dir, args.datasets, args.splits)
    if manifest_path is not None and not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if manifest_path is None and not paths:
        raise FileNotFoundError(
            f"No parquet files found under {data_dir} for datasets={args.datasets} splits={args.splits}"
        )
    if args.output_records_jsonl and "generate" not in eval_modes:
        raise ValueError("--output-records-jsonl requires generate evaluation")

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _ddp, ddp_rank, _ddp_local_rank, _ddp_world_size, device = compute_init(device_type)
    if ddp_rank != 0:
        raise RuntimeError("human_edit_eval.py is intended to run as a single process, not under torchrun.")

    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    T = args.max_seq_len or int(meta["model_config"]["sequence_len"])
    pointer_token_mode = (
        args.ces_pointer_token_mode
        or meta.get("user_config", {}).get("ces_pointer_token_mode", "role_specific")
    )
    token_ops = TokensOperations(
        tokenizer.enc, pointer_token_mode=pointer_token_mode
    )
    engine = (
        Engine(model, tokenizer, pointer_token_mode=pointer_token_mode)
        if "generate" in eval_modes else None
    )
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    examples = (
        _load_manifest_examples(manifest_path, None if args.tasks else args.limit)
        if manifest_path is not None
        else _load_examples(paths, None if args.tasks else args.limit)
    )
    if args.tasks:
        task_filter = set(args.tasks)
        examples = [example for example in examples if example.task in task_filter]
    if args.limit is not None:
        examples = examples[: args.limit]
    groups: dict[str, dict[str, Any]] = defaultdict(_empty_group)
    teacher_rows: list[tuple[HumanEditExample, TrainingData]] = []
    sample_outputs: list[dict[str, Any]] = []
    generation_records: list[dict[str, Any]] = []

    def group_keys(example: HumanEditExample) -> list[str]:
        return [
            "overall",
            f"dataset:{example.dataset}",
            f"split:{example.split}",
            f"task:{example.task}",
            f"dataset_split:{example.dataset}/{example.split}",
            f"dataset_task:{example.dataset}/{example.task}",
        ]

    model.eval()
    with torch.inference_mode():
        for ix, example in enumerate(examples):
            keys = group_keys(example)
            _add_to_groups(groups, keys, lambda g: g.__setitem__("examples", g["examples"] + 1))
            seq_edit = _token_edit_from_pair(tokenizer, example.source_text, example.target_text)
            try:
                serialised = token_ops.serialise(seq_edit, indel_bos_pos=0)
            except ValueError:
                continue
            if len(serialised.tokens) > T + 1:
                _add_to_groups(groups, keys, lambda g: g.__setitem__("skipped_too_long", g["skipped_too_long"] + 1))
                continue
            _add_to_groups(groups, keys, lambda g: g.__setitem__("serialised", g["serialised"] + 1))
            gold_signature = _edit_signature(seq_edit)
            for edit in seq_edit.edits:
                op = _op_name(edit)
                _add_to_groups(groups, keys, lambda g, op=op: g["gold_op_counts"].__setitem__(op, g["gold_op_counts"][op] + 1))

            if "teacher" in eval_modes:
                training_data = _pad_training_data(token_ops.prepare_training_example(serialised, args.pos_buffer_size), T)
                teacher_rows.append((example, training_data))
                if len(teacher_rows) >= args.batch_size:
                    _run_teacher_batch(model, teacher_rows, groups, group_keys, device, autocast_ctx)
                    teacher_rows.clear()

            if "generate" in eval_modes:
                assert engine is not None
                edit_start_ix = int(np.where(serialised.tokens == token_ops.edit_start_token)[0][0])
                prompt_tokens = [int(t) for t in serialised.tokens[: edit_start_ix + 1]]
                prompt_positions = [int(p) for p in serialised.positions[: edit_start_ix + 1]]
                budget = min(args.max_edit_tokens, T - len(prompt_tokens))
                source_sim = _similarity(example.source_text, example.target_text)
                generated_seq_edit = None
                source_avg_logprob = None
                candidate_avg_logprob = None
                likelihood_accepted = None
                candidate_generated_text = None
                candidate_status = None
                candidate_resolved_signature = None
                candidate_resolved_char_mass = None
                diagnostics = None
                selected_sample_ix = None
                candidate_statuses = None
                candidate_generated_texts = None
                generated_edit_token_count = 0
                confidence = _confidence_summary((), ())
                if budget <= 0:
                    status = "prompt_too_long"
                    generated_text = None
                    generated_tokens = None
                else:
                    generation_kwargs = dict(
                        tix=prompt_positions,
                        num_samples=args.num_samples,
                        max_tokens=budget,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        pointer_temperature=args.pointer_temperature,
                        closure_bias=args.closure_bias,
                        closure_bias_min_payload_tokens=args.closure_bias_min_payload_tokens,
                        seed=args.seed + ix,
                        max_payload_tokens=args.max_payload_tokens,
                        max_edit_ops=args.max_edit_ops,
                        stop_after_edit_end=args.stop_after_edit_end,
                        first_decision_edit_threshold=(
                            args.first_decision_edit_threshold
                        ),
                    )
                    rollouts = None
                    if (
                        args.score_program_confidence
                        or args.accept_program_mean_logprob_min is not None
                    ):
                        rollouts, _ = engine.generate_rollout_batch(
                            prompt_tokens, **generation_kwargs
                        )
                        samples = [list(rollout.tokens) for rollout in rollouts]
                        pointers = [list(rollout.pointers) for rollout in rollouts]
                    else:
                        samples, _, pointers = engine.generate_batch(
                            prompt_tokens, **generation_kwargs
                        )
                    candidates = []
                    for sample_ix, (sample_tokens, sample_pointers) in enumerate(zip(samples, pointers)):
                        candidate_edit, candidate_status = _deserialise_generated(
                            sample_tokens, sample_pointers, token_ops
                        )
                        candidate_tokens = None
                        if candidate_edit is not None:
                            try:
                                resolved = token_ops.resolve(
                                    SerialisedData(
                                        tokens=np.array(sample_tokens),
                                        positions=np.array(sample_pointers),
                                    )
                                )
                                candidate_tokens = [int(t) for t in resolved]
                            except Exception as exc:
                                candidate_status = f"resolve_error:{type(exc).__name__}"
                        candidate_text = (
                            _decode_without_bos(tokenizer, candidate_tokens)
                            if candidate_tokens is not None
                            else None
                        )
                        candidates.append(
                            {
                                "sample_ix": sample_ix,
                                "status": candidate_status,
                                "seq_edit": candidate_edit,
                                "text": candidate_text,
                                "tokens": sample_tokens,
                                "diagnostics": engine.last_generation_diagnostics[sample_ix],
                            }
                        )
                    selected = next(
                        (candidate for candidate in candidates if candidate["status"] == "valid"),
                        candidates[0],
                    )
                    candidate_generated_texts = [
                        candidate["text"]
                        if candidate["status"] == "valid" and candidate["text"] is not None
                        else example.source_text
                        for candidate in candidates
                    ]
                    selected_sample_ix = int(selected["sample_ix"])
                    if rollouts is not None:
                        scoring_rollout = replace(
                            rollouts[selected_sample_ix],
                            temperature=args.confidence_temperature,
                            pointer_temperature=args.confidence_temperature,
                            top_k=None,
                            top_p=None,
                        )
                        replayed = replay_policy_logprobs(model, tokenizer, scoring_rollout)
                        confidence = _confidence_summary(
                            replayed.token_logprobs, replayed.pointer_logprobs
                        )
                    status = str(selected["status"])
                    candidate_status = status
                    generated_seq_edit = selected["seq_edit"]
                    generated_text = selected["text"]
                    diagnostics = selected["diagnostics"]
                    candidate_statuses = [candidate["status"] for candidate in candidates]
                    generated_edit_token_count = len(selected["tokens"]) - len(prompt_tokens)
                    candidate_generated_text = generated_text
                    if status == "valid" and generated_text is not None:
                        candidate_resolved_signature, candidate_resolved_char_mass = (
                            _resolved_change_features(example.source_text, generated_text)
                        )
                    if (
                        args.accept_resolved_signature is not None
                        and status == "valid"
                        and candidate_resolved_signature != (args.accept_resolved_signature,)
                    ):
                        status = "valid_rejected_resolved_signature"
                        generated_text = example.source_text
                        generated_seq_edit = None
                    if (
                        args.accept_resolved_max_chars is not None
                        and status == "valid"
                        and (
                            candidate_resolved_char_mass is None
                            or candidate_resolved_char_mass > args.accept_resolved_max_chars
                        )
                    ):
                        status = "valid_rejected_resolved_span"
                        generated_text = example.source_text
                        generated_seq_edit = None
                    if (
                        args.accept_resolved_max_regions is not None
                        and status == "valid"
                        and (
                            candidate_resolved_signature is None
                            or len(candidate_resolved_signature) > args.accept_resolved_max_regions
                        )
                    ):
                        status = "valid_rejected_resolved_regions"
                        generated_text = example.source_text
                        generated_seq_edit = None
                    if (
                        args.accept_program_mean_logprob_min is not None
                        and status == "valid"
                        and (
                            confidence["program_mean_logprob"] is None
                            or confidence["program_mean_logprob"]
                            < args.accept_program_mean_logprob_min
                        )
                    ):
                        status = "valid_rejected_program_confidence"
                        generated_text = example.source_text
                        generated_seq_edit = None
                    if (
                        (args.score_candidate_likelihood or args.accept_if_ar_logprob_improves)
                        and candidate_status == "valid"
                        and generated_text is not None
                    ):
                        bos = tokenizer.get_bos_token_id()
                        source_lm_tokens = [bos] + [int(t) for t in tokenizer.encode(example.source_text)]
                        candidate_lm_tokens = [bos] + [int(t) for t in tokenizer.encode(candidate_generated_text)]
                        source_avg_logprob = _seq_avg_logprob(
                            model, source_lm_tokens, device, autocast_ctx
                        )
                        candidate_avg_logprob = _seq_avg_logprob(
                            model, candidate_lm_tokens, device, autocast_ctx
                        )
                        likelihood_accepted = (
                            source_avg_logprob is not None
                            and candidate_avg_logprob is not None
                            and candidate_avg_logprob - source_avg_logprob
                            >= args.acceptance_logprob_margin
                        )
                        if args.accept_if_ar_logprob_improves and status == "valid" and not likelihood_accepted:
                            status = "valid_rejected_ar_logprob"
                            generated_text = example.source_text
                            generated_seq_edit = None
                _record_generation(
                    groups,
                    keys,
                    status=status,
                    source_similarity=source_sim,
                    generated_text=generated_text,
                    target_text=example.target_text,
                    generated_seq_edit=generated_seq_edit,
                    gold_signature=gold_signature,
                    generated_edit_tokens=generated_edit_token_count,
                    diagnostics=diagnostics,
                    likelihood_accepted=likelihood_accepted,
                )
                generated_similarity = (
                    _similarity(generated_text, example.target_text)
                    if generated_text is not None
                    else None
                )
                candidate_generated_similarity = (
                    _similarity(candidate_generated_text, example.target_text)
                    if candidate_generated_text is not None
                    else None
                )
                record = {
                    "dataset": example.dataset,
                    "split": example.split,
                    "task": example.task,
                    "example_id": example.example_id,
                    "status": status,
                    "source_text": example.source_text,
                    "target_text": example.target_text,
                    "generated_text": generated_text,
                    "prediction_text": generated_text or example.source_text,
                    "candidate_status": candidate_status,
                    "candidate_generated_text": candidate_generated_text,
                    "ungated_prediction_text": (
                        candidate_generated_text
                        if candidate_status == "valid" and candidate_generated_text is not None
                        else example.source_text
                    ),
                    "candidate_generated_similarity": candidate_generated_similarity,
                    "candidate_resolved_signature": candidate_resolved_signature,
                    "candidate_resolved_char_mass": candidate_resolved_char_mass,
                    "candidate_similarity_delta": (
                        candidate_generated_similarity - source_sim
                        if candidate_generated_similarity is not None
                        else 0.0
                    ),
                    "source_similarity": source_sim,
                    "generated_similarity": generated_similarity,
                    "similarity_delta": (
                        generated_similarity - source_sim
                        if generated_similarity is not None
                        else 0.0
                    ),
                    "source_avg_logprob": source_avg_logprob,
                    "candidate_avg_logprob": candidate_avg_logprob,
                    "candidate_minus_source_avg_logprob": (
                        candidate_avg_logprob - source_avg_logprob
                        if source_avg_logprob is not None and candidate_avg_logprob is not None
                        else None
                    ),
                    "likelihood_accepted": likelihood_accepted,
                    "generated_edit_tokens": generated_edit_token_count,
                    "selected_sample_ix": selected_sample_ix,
                    "candidate_statuses": candidate_statuses,
                    "candidate_generated_texts": candidate_generated_texts,
                    "diagnostics": diagnostics,
                    **confidence,
                }
                generation_records.append(record)
                if len(sample_outputs) < args.include_examples:
                    sample_outputs.append(record)

        if teacher_rows:
            _run_teacher_batch(model, teacher_rows, groups, group_keys, device, autocast_ctx)
            teacher_rows.clear()

    result = {
        "model_tag": args.model_tag,
        "step": meta["step"],
        "data_dir": str(data_dir),
        "manifest_jsonl": str(manifest_path) if manifest_path is not None else None,
        "files": [str(p) for p in paths],
        "tasks": args.tasks,
        "eval_modes": sorted(eval_modes),
        "max_seq_len": T,
        "num_loaded_examples": len(examples),
        "generation_config": {
            "max_edit_tokens": args.max_edit_tokens,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "pointer_temperature": args.pointer_temperature,
            "closure_bias": args.closure_bias,
            "closure_bias_min_payload_tokens": args.closure_bias_min_payload_tokens,
            "max_payload_tokens": args.max_payload_tokens,
            "max_edit_ops": args.max_edit_ops,
            "first_decision_edit_threshold": args.first_decision_edit_threshold,
            "accept_if_ar_logprob_improves": args.accept_if_ar_logprob_improves,
            "score_candidate_likelihood": args.score_candidate_likelihood,
            "acceptance_logprob_margin": args.acceptance_logprob_margin,
            "accept_resolved_signature": args.accept_resolved_signature,
            "accept_resolved_max_chars": args.accept_resolved_max_chars,
            "accept_resolved_max_regions": args.accept_resolved_max_regions,
            "stop_after_edit_end": args.stop_after_edit_end,
            "score_program_confidence": args.score_program_confidence,
            "confidence_temperature": args.confidence_temperature,
            "accept_program_mean_logprob_min": args.accept_program_mean_logprob_min,
        },
        "groups": {key: _finalise_group(value) for key, value in sorted(groups.items())},
    }
    if sample_outputs:
        result["examples"] = sample_outputs

    print(json.dumps(result, indent=2))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.output_records_jsonl:
        records_path = Path(args.output_records_jsonl)
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("w", encoding="utf-8") as handle:
            for record in generation_records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        print0(f"Wrote {records_path} ({len(generation_records)} records)")

        print0(f"Wrote {path}")

    compute_cleanup()


def _run_teacher_batch(
    model,
    rows: list[tuple[HumanEditExample, TrainingData]],
    groups: dict[str, dict[str, Any]],
    group_keys,
    device: torch.device,
    autocast_ctx,
) -> None:
    for example, training_data in rows:
        batch = _stack_training_batch([training_data], device)
        with autocast_ctx:
            _loss, aux = model(
                batch.input_tokens,
                tix=batch.input_positions,
                targets=batch.target,
                loss_reduction="mean",
            )
        correct, count = aux["position_correct"]
        count_f = float(count.item())
        correct_f = float(correct.item())
        pos_loss_sum = float(aux["position_loss"].item()) * count_f
        token_loss = float(aux["token_loss"].item())
        for key in group_keys(example):
            group = groups[key]
            group["position_correct"] += correct_f
            group["position_targets"] += count_f
            group["position_loss_sum"] += pos_loss_sum
            group["token_loss_sum"] += token_loss
            group["teacher_batches"] += 1


def _record_generation(
    groups: dict[str, dict[str, Any]],
    keys: list[str],
    *,
    status: str,
    source_similarity: float,
    generated_text: str | None,
    target_text: str,
    generated_seq_edit: SequenceEdit | None = None,
    gold_signature: tuple[tuple[Any, ...], ...] | None = None,
    generated_edit_tokens: int = 0,
    diagnostics: dict[str, Any] | None = None,
    likelihood_accepted: bool | None = None,
) -> None:
    generated_similarity = _similarity(generated_text, target_text) if generated_text is not None else None
    forced_payload = int((diagnostics or {}).get("forced_payload_closures", 0))
    forced_program = int((diagnostics or {}).get("forced_program_closures", 0))
    for key in keys:
        group = groups[key]
        group["generated"] += 1
        group["source_similarity_sum"] += source_similarity
        group["generated_edit_tokens"] += generated_edit_tokens
        group["forced_payload_closures"] += forced_payload
        group["forced_program_closures"] += forced_program
        group["constraint_interventions"] += int(forced_payload > 0 or forced_program > 0)
        group["likelihood_scored_generations"] += int(likelihood_accepted is not None)
        group["likelihood_accepted_generations"] += int(likelihood_accepted is True)
        group["likelihood_rejected_generations"] += int(likelihood_accepted is False)
        group["generation_status_counts"][status] += 1
        if generated_similarity is None:
            group["final_similarity_with_source_fallback_sum"] += source_similarity
            continue
        group["valid_generations"] += 1
        group["generated_similarity_sum"] += generated_similarity
        group["final_similarity_with_source_fallback_sum"] += generated_similarity
        group["similarity_delta_with_source_fallback_sum"] += generated_similarity - source_similarity
        group["exact_matches"] += int(generated_text == target_text)
        group["improved_similarity"] += int(generated_similarity > source_similarity)
        if generated_seq_edit is not None:
            for edit in generated_seq_edit.edits:
                op = _op_name(edit)
                group["generated_op_counts"][op] += 1
            if gold_signature is not None:
                group["exact_edit_scripts"] += int(_edit_signature(generated_seq_edit) == gold_signature)


if __name__ == "__main__":
    main()
