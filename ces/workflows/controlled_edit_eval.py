"""Run complete CES generation→parse→apply evaluation on a frozen benchmark.

Unlike the factorized position/payload probes, every requested example remains
in the denominator. Invalid programs resolve to an unchanged-source no-op.
Every generated token and pointer is saved, and aggregate results are stratified
by gold program complexity, operation mode, density regime, and decoding seed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from collections import Counter, defaultdict, deque
from contextlib import nullcontext
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.causal_serialisation_extended import SerialisedData, TokensOperations
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.engine_extended import Engine
from ces.workflows.benchmark_tokenization import prepare_benchmark_for_tokenizer
from ces.workflows.human_edit_eval import _deserialise_generated, _edit_signature, _op_name


PRIMARY_METRICS = (
    "valid",
    "exact_target",
    "exact_program",
    "char_similarity_delta",
    "token_similarity_delta",
    "likelihood_delta",
    "likelihood_improved",
    "target_similarity_improved",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def select_benchmark_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int | None,
    selection: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Choose an exact deterministic evaluation subset.

    ``stratified`` round-robins seeded per-stratum queues, so represented
    strata differ by at most one row whenever every stratum has enough rows.
    """

    if limit is None or limit >= len(rows):
        return list(rows)
    if limit < 1:
        raise ValueError("--limit must be positive")
    if selection == "head":
        return list(rows[:limit])
    if selection != "stratified":
        raise ValueError(f"Unknown benchmark selection: {selection!r}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("stratum", "<missing>"))].append(row)
    rng = random.Random(seed)
    strata = sorted(grouped)
    rng.shuffle(strata)
    queues: dict[str, deque[dict[str, Any]]] = {}
    for stratum in strata:
        candidates = list(grouped[stratum])
        rng.shuffle(candidates)
        queues[stratum] = deque(candidates)

    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        progress = False
        for stratum in strata:
            if queues[stratum]:
                selected.append(queues[stratum].popleft())
                progress = True
                if len(selected) == limit:
                    break
        if not progress:
            break
    return selected


def _tokens_hash(tokens: Iterable[int]) -> str:
    payload = json.dumps([int(token) for token in tokens], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decode_without_bos(tokenizer, tokens: Iterable[int]) -> str:
    bos = tokenizer.get_bos_token_id()
    return tokenizer.decode([int(token) for token in tokens if int(token) != bos])


def _gold_signature(edits: list[dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    signature: list[tuple[Any, ...]] = []
    for edit in edits:
        op = edit["op"]
        if op == "insert":
            signature.append(("insert", int(edit["pos"]), tuple(int(token) for token in edit["content"])))
        elif op == "delete":
            signature.append(("delete", int(edit["start"]), int(edit["end"])))
        elif op == "replace":
            signature.append((
                "replace",
                int(edit["start"]),
                int(edit["end"]),
                tuple(int(token) for token in edit["content"]),
            ))
        elif op in {"copy", "move"}:
            signature.append((op, int(edit["start"]), int(edit["end"]), int(edit["dest"])))
        else:
            raise ValueError(f"Unknown gold operation: {op}")
    return tuple(signature)


def _similarity(left: Iterable[Any], right: Iterable[Any]) -> float:
    return float(SequenceMatcher(a=list(left), b=list(right), autojunk=False).ratio())


def _text_similarity(left: str, right: str) -> float:
    return float(SequenceMatcher(a=left, b=right, autojunk=False).ratio())


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device: torch.device):
    return (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def score_unique_sequences(
    model,
    sequences: list[list[int]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | None]:
    """Return average token log-probability keyed by deterministic token hash."""

    unique: dict[str, list[int]] = {}
    for sequence in sequences:
        unique.setdefault(_tokens_hash(sequence), [int(token) for token in sequence])
    ordered = sorted(unique.items(), key=lambda item: len(item[1]))
    scores: dict[str, float | None] = {}
    context = int(model.config.sequence_len)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ordered), batch_size):
            chunk = ordered[start:start + batch_size]
            scoreable = [(key, sequence) for key, sequence in chunk if 1 < len(sequence) <= context + 1]
            for key, sequence in chunk:
                if not (1 < len(sequence) <= context + 1):
                    scores[key] = None
            if not scoreable:
                continue
            max_len = max(len(sequence) for _key, sequence in scoreable)
            # GPT.forward requires at least two input positions. For a two-token
            # sequence, add one causally later padding position and leave it
            # masked so the one real next-token likelihood is unchanged.
            input_width = max(2, max_len - 1)
            inputs = torch.zeros((len(scoreable), input_width), dtype=torch.long, device=device)
            targets = torch.zeros_like(inputs)
            mask = torch.zeros_like(inputs, dtype=torch.bool)
            for row_index, (_key, sequence) in enumerate(scoreable):
                length = len(sequence) - 1
                inputs[row_index, :length] = torch.tensor(sequence[:-1], dtype=torch.long, device=device)
                targets[row_index, :length] = torch.tensor(sequence[1:], dtype=torch.long, device=device)
                mask[row_index, :length] = True
            positions = torch.arange(input_width, device=device, dtype=torch.long)[None, :].expand(len(scoreable), -1)
            with _autocast(device):
                logits = model(inputs, tix=positions)
                token_logp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            for row_index, (key, _sequence) in enumerate(scoreable):
                row_mask = mask[row_index]
                scores[key] = float(token_logp[row_index][row_mask].mean().item())
    return scores


def _source_length_bin(num_tokens: int) -> str:
    if num_tokens <= 128:
        return "le128"
    if num_tokens <= 256:
        return "129_256"
    if num_tokens <= 512:
        return "257_512"
    return "gt512"


def generate_records(
    benchmark_rows: list[dict[str, Any]],
    *,
    model,
    tokenizer,
    token_ops: TokensOperations,
    device: torch.device,
    model_context: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    engine = Engine(
        model, tokenizer, pointer_token_mode=args.ces_pointer_token_mode
    )
    output: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for row_index, benchmark in enumerate(benchmark_rows):
            prompt_tokens = [int(token) for token in benchmark["prompt_tokens"]]
            prompt_positions = [int(position) for position in benchmark["prompt_positions"]]
            source_tokens = [int(token) for token in benchmark["source_tokens"]]
            target_tokens = [int(token) for token in benchmark["target_tokens"]]
            source_text = str(benchmark["source_text"])
            target_text = str(benchmark["target_text"])
            source_char_similarity = _text_similarity(source_text, target_text)
            source_token_similarity = _similarity(source_tokens, target_tokens)
            gold_signature = _gold_signature(benchmark["gold_edits"])

            for generation_seed in args.generation_seeds:
                budget = min(args.max_edit_tokens, model_context - len(prompt_tokens))
                status = "prompt_too_long"
                sample_tokens = prompt_tokens
                sample_positions = prompt_positions
                generated_edit = None
                resolved_tokens = None
                diagnostics: dict[str, Any] = {}
                generation_seconds = 0.0
                apply_seconds = 0.0
                if budget > 0:
                    _synchronise(device)
                    generation_start = time.perf_counter()
                    samples, _masks, pointers = engine.generate_batch(
                        prompt_tokens,
                        tix=prompt_positions,
                        num_samples=1,
                        max_tokens=budget,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        pointer_temperature=args.pointer_temperature,
                        seed=int(generation_seed) + row_index,
                        closure_bias=args.closure_bias,
                        closure_bias_min_payload_tokens=args.closure_bias_min_payload_tokens,
                        max_payload_tokens=args.max_payload_tokens,
                        max_edit_ops=args.max_edit_ops,
                        stop_after_edit_end=args.stop_after_edit_end,
                        first_decision_edit_threshold=(
                            args.first_decision_edit_threshold
                        ),
                    )
                    _synchronise(device)
                    generation_seconds = time.perf_counter() - generation_start
                    sample_tokens = [int(token) for token in samples[0]]
                    sample_positions = [int(position) for position in pointers[0]]
                    diagnostics = dict(engine.last_generation_diagnostics[0])
                    apply_start = time.perf_counter()
                    generated_edit, status = _deserialise_generated(sample_tokens, sample_positions, token_ops)
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

                valid = status == "valid" and resolved_tokens is not None
                final_tokens = resolved_tokens if valid else source_tokens
                final_text = _decode_without_bos(tokenizer, final_tokens)
                generated_signature = _edit_signature(generated_edit) if generated_edit is not None else ()
                generated_op_counts = Counter(
                    _op_name(edit) for edit in generated_edit.edits
                ) if generated_edit is not None else Counter()
                final_char_similarity = _text_similarity(final_text, target_text)
                final_token_similarity = _similarity(final_tokens, target_tokens)
                char_delta = final_char_similarity - source_char_similarity
                token_delta = final_token_similarity - source_token_similarity
                recoverable_char_gap = max(1.0 - source_char_similarity, 0.0)
                recovery_fraction = (
                    char_delta / recoverable_char_gap if recoverable_char_gap > 0 else float(final_text == target_text)
                )

                output.append({
                    "evaluation_id": f"{benchmark['benchmark_id']}:decode_seed_{generation_seed}",
                    "benchmark_id": benchmark["benchmark_id"],
                    "generation_seed": int(generation_seed),
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
                    "generated_full_positions": sample_positions,
                    "generated_program_tokens": sample_tokens[len(prompt_tokens):],
                    "generated_program_positions": sample_positions[len(prompt_positions):],
                    "generated_program_text": tokenizer.decode(sample_tokens[len(prompt_tokens):]),
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
                    "diagnostics": diagnostics,
                })
    return output


def attach_likelihood_scores(
    records: list[dict[str, Any]],
    *,
    scorer,
    device: torch.device,
    batch_size: int,
) -> None:
    sequences: list[list[int]] = []
    for record in records:
        sequences.extend((record["source_tokens"], record["target_tokens"], record["resolved_tokens"]))
    scores = score_unique_sequences(scorer, sequences, device=device, batch_size=batch_size)
    for record in records:
        source_score = scores.get(_tokens_hash(record["source_tokens"]))
        target_score = scores.get(_tokens_hash(record["target_tokens"]))
        final_score = scores.get(_tokens_hash(record["resolved_tokens"]))
        record["source_avg_logprob"] = source_score
        record["target_avg_logprob"] = target_score
        record["resolved_avg_logprob"] = final_score
        record["target_minus_source_avg_logprob"] = (
            target_score - source_score if target_score is not None and source_score is not None else None
        )
        record["target_likelihood_improved"] = (
            target_score > source_score if target_score is not None and source_score is not None else None
        )
        record["likelihood_delta"] = (
            final_score - source_score if final_score is not None and source_score is not None else None
        )
        record["likelihood_improved"] = (
            final_score > source_score if final_score is not None and source_score is not None else None
        )


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1 or samples <= 0:
        mean = float(array.mean())
        return [mean, mean]
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    remaining = samples
    while remaining:
        chunk_size = min(1000, remaining)
        indices = rng.integers(0, len(array), size=(chunk_size, len(array)))
        means.append(array[indices].mean(axis=1))
        remaining -= chunk_size
    distribution = np.concatenate(means)
    return [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))]


def _group_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int((base_seed + int.from_bytes(digest[:4], "little")) % (2**32))


def summarise_group(
    records: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    valid_rows = [record for record in records if record["valid"]]
    status_counts = Counter(str(record["status"]) for record in records)
    out: dict[str, Any] = {
        "examples": len(records),
        "valid_programs": len(valid_rows),
        "valid_program_rate": len(valid_rows) / len(records) if records else None,
        "exact_targets": sum(bool(record["exact_target"]) for record in records),
        "unconditional_exact_target_rate": (
            sum(bool(record["exact_target"]) for record in records) / len(records) if records else None
        ),
        "exact_programs": sum(bool(record["exact_program"]) for record in records),
        "unconditional_exact_program_rate": (
            sum(bool(record["exact_program"]) for record in records) / len(records) if records else None
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "mean_generation_seconds": (
            float(np.mean([record["generation_seconds"] for record in records])) if records else None
        ),
        "mean_apply_seconds": (
            float(np.mean([record["apply_seconds"] for record in records])) if records else None
        ),
    }
    metric_values: dict[str, list[float]] = {
        "valid": [float(record["valid"]) for record in records],
        "exact_target": [float(record["exact_target"]) for record in records],
        "exact_program": [float(record["exact_program"]) for record in records],
        "char_similarity_delta": [float(record["char_similarity_delta"]) for record in records],
        "token_similarity_delta": [float(record["token_similarity_delta"]) for record in records],
        "likelihood_delta": [float(record["likelihood_delta"]) for record in records if record.get("likelihood_delta") is not None],
        "likelihood_improved": [float(record["likelihood_improved"]) for record in records if record.get("likelihood_improved") is not None],
        "target_minus_source_avg_logprob": [
            float(record["target_minus_source_avg_logprob"])
            for record in records
            if record.get("target_minus_source_avg_logprob") is not None
        ],
        "target_likelihood_improved": [
            float(record["target_likelihood_improved"])
            for record in records
            if record.get("target_likelihood_improved") is not None
        ],
        "target_similarity_improved": [float(record["target_similarity_improved"]) for record in records],
    }
    for metric, values in metric_values.items():
        out[f"mean_{metric}"] = float(np.mean(values)) if values else None
        out[f"mean_{metric}_ci95"] = _bootstrap_mean_ci(
            values,
            samples=bootstrap_samples,
            seed=_group_seed(bootstrap_seed, metric),
        )
    out["valid_exact_target_rate"] = (
        sum(bool(record["exact_target"]) for record in valid_rows) / len(valid_rows) if valid_rows else None
    )
    out["valid_target_similarity_improvement_rate"] = (
        sum(bool(record["target_similarity_improved"]) for record in valid_rows) / len(valid_rows)
        if valid_rows else None
    )
    valid_likelihood = [
        float(record["likelihood_delta"])
        for record in valid_rows
        if record.get("likelihood_delta") is not None
    ]
    out["mean_valid_likelihood_delta"] = float(np.mean(valid_likelihood)) if valid_likelihood else None
    return out


def summarise_records(
    records: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        keys = (
            "overall",
            f"stratum:{record['stratum']}",
            f"complexity:{record['repair_num_ops']}",
            f"operation:{record['repair_op_mode']}",
            f"density:{record['density_regime']}",
            f"source_length:{record['source_length_bin']}",
            f"generation_seed:{record['generation_seed']}",
        )
        for key in keys:
            groups[key].append(record)
    return {
        key: summarise_group(
            rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=_group_seed(bootstrap_seed, key),
        )
        for key, rows in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-jsonl", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument(
        "--ces-pointer-token-mode",
        choices=("role_specific", "shared"),
        default=None,
        help="Override pointer serialization; default reads checkpoint metadata.",
    )
    parser.add_argument("--scorer-model-tag", default=None, help="Frozen clean AR checkpoint; defaults to the edit model.")
    parser.add_argument("--scorer-step", type=int, default=None)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--selection",
        choices=("head", "stratified"),
        default="head",
        help="Subset rule used when --limit is smaller than the benchmark",
    )
    parser.add_argument("--selection-seed", type=int, default=20260813)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=[12345])
    parser.add_argument("--max-edit-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--pointer-temperature", type=float, default=None)
    parser.add_argument("--closure-bias", type=float, default=0.0)
    parser.add_argument("--closure-bias-min-payload-tokens", type=int, default=0)
    parser.add_argument("--max-payload-tokens", type=int, default=None)
    parser.add_argument("--max-edit-ops", type=int, default=None)
    parser.add_argument("--first-decision-edit-threshold", type=float, default=None)
    parser.add_argument("--stop-after-edit-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--skip-likelihood-scoring", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    parser.add_argument("--device-type", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.score_batch_size < 1:
        raise ValueError("--score-batch-size must be positive")
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")
    if not args.overwrite and (output_path.exists() or summary_path.exists()):
        raise FileExistsError("Output exists; use --overwrite or choose a new output path")

    benchmark = _read_jsonl(Path(args.benchmark_jsonl))
    benchmark = [row for index, row in enumerate(benchmark) if index % args.num_shards == args.shard_index]
    benchmark = select_benchmark_rows(
        benchmark,
        limit=args.limit,
        selection=args.selection,
        seed=args.selection_seed,
    )
    if not benchmark:
        raise ValueError("No benchmark rows selected")

    device_type = autodetect_device_type() if not args.device_type else args.device_type
    _ddp, rank, _local_rank, world_size, device = compute_init(device_type)
    if rank != 0 or world_size != 1:
        raise RuntimeError("Use one process per shard; launch separate CUDA_VISIBLE_DEVICES jobs for multi-GPU evaluation")
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    args.ces_pointer_token_mode = (
        args.ces_pointer_token_mode
        or meta.get("user_config", {}).get("ces_pointer_token_mode", "role_specific")
    )
    token_ops = TokensOperations(
        tokenizer.enc, pointer_token_mode=args.ces_pointer_token_mode
    )
    benchmark, benchmark_tokenization = prepare_benchmark_for_tokenizer(
        benchmark,
        tokenizer=tokenizer,
        token_ops=token_ops,
    )
    if benchmark_tokenization["mode"] == "retokenized_text":
        print(
            "Frozen benchmark token IDs are incompatible with this checkpoint tokenizer; "
            "rebuilt tokenizer-dependent fields from frozen source/target text.",
            flush=True,
        )
    records = generate_records(
        benchmark,
        model=model,
        tokenizer=tokenizer,
        token_ops=token_ops,
        device=device,
        model_context=int(meta["model_config"]["sequence_len"]),
        args=args,
    )

    scorer_label = "none"
    if not args.skip_likelihood_scoring:
        if args.scorer_model_tag is None:
            scorer = model
            scorer_label = f"editor:{args.model_tag}@{meta['step']}"
        else:
            del token_ops
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            scorer, scorer_tokenizer, scorer_meta = load_model(
                "base",
                device,
                phase="eval",
                model_tag=args.scorer_model_tag,
                step=args.scorer_step,
            )
            if scorer_tokenizer.get_vocab_size() != tokenizer.get_vocab_size():
                raise ValueError("Editor and scorer tokenizers have different vocabulary sizes")
            scorer_label = f"frozen:{args.scorer_model_tag}@{scorer_meta['step']}"
        attach_likelihood_scores(records, scorer=scorer, device=device, batch_size=args.score_batch_size)
    else:
        for record in records:
            record.update({
                "source_avg_logprob": None,
                "target_avg_logprob": None,
                "resolved_avg_logprob": None,
                "target_minus_source_avg_logprob": None,
                "target_likelihood_improved": None,
                "likelihood_delta": None,
                "likelihood_improved": None,
            })

    summary = {
        "benchmark_jsonl": str(Path(args.benchmark_jsonl).resolve()),
        "model_tag": args.model_tag,
        "step": int(meta["step"]),
        "scorer": scorer_label,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected_benchmark_examples": len(benchmark),
        "selection": args.selection,
        "selection_seed": args.selection_seed,
        "generated_records": len(records),
        "benchmark_tokenization": benchmark_tokenization,
        "generation_config": {
            "generation_seeds": args.generation_seeds,
            "max_edit_tokens": args.max_edit_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "pointer_temperature": args.pointer_temperature,
            "closure_bias": args.closure_bias,
            "closure_bias_min_payload_tokens": args.closure_bias_min_payload_tokens,
            "max_payload_tokens": args.max_payload_tokens,
            "max_edit_ops": args.max_edit_ops,
            "first_decision_edit_threshold": args.first_decision_edit_threshold,
            "stop_after_edit_end": args.stop_after_edit_end,
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
