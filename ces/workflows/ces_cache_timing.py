"""Benchmark CES KV-cache rebuilding for terminal and edit-resume workflows.

The benchmark consumes either the frozen controlled-edit manifest (source ->
gold target) or records emitted by ``ces.workflows.controlled_edit_eval`` (source ->
resolved model output). It measures synchronized model latency for:

* source prefill;
* no-edit continuation from an already populated source cache;
* rebuilding the complete edited prefix and resuming generation; and
* reusing the unchanged prefix before the earliest changed token, rebuilding
  only the affected suffix, and resuming generation.

Cache allocation, checkpoint loading, tokenization, and edit-program generation
are deliberately outside the timed regions. Terminal CES editing requires no
post-edit cache rebuild and is represented in the summary as zero cache cost.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from nanochat.causal_serialisation_extended import INDEL_SPECIALS
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.engine_extended import KVCache
from ces.workflows.benchmark_tokenization import prepare_benchmark_for_tokenizer


SCENARIOS = (
    "source_prefill",
    "no_edit_resume",
    "full_rebuild_resume",
    "prefix_reuse_resume",
)
TIMING_METRICS = ("rebuild_ms", "resume_decode_ms", "total_ms")


@dataclass(frozen=True)
class TimingCase:
    case_id: str
    source_tokens: tuple[int, ...]
    edited_tokens: tuple[int, ...]
    stable_prefix_tokens: int
    source_length_bin: str
    edit_location_bin: str
    input_kind: str
    edit_mass_tokens: int | None = None

    @property
    def source_num_tokens(self) -> int:
        return len(self.source_tokens)

    @property
    def edited_num_tokens(self) -> int:
        return len(self.edited_tokens)

    @property
    def recomputed_suffix_tokens(self) -> int:
        return len(self.edited_tokens) - self.stable_prefix_tokens

    @property
    def stable_prefix_fraction(self) -> float:
        content_tokens = max(len(self.source_tokens) - 1, 1)
        content_prefix = max(self.stable_prefix_tokens - 1, 0)
        return min(content_prefix / content_tokens, 1.0)


@dataclass
class CachePair:
    model: KVCache
    position: KVCache | None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    prefix = 0
    for left_token, right_token in zip(left, right):
        if int(left_token) != int(right_token):
            break
        prefix += 1
    return prefix


def source_length_bin(num_tokens: int) -> str:
    if num_tokens <= 128:
        return "0001-0128"
    if num_tokens <= 256:
        return "0129-0256"
    if num_tokens <= 512:
        return "0257-0512"
    if num_tokens <= 1024:
        return "0513-1024"
    return "1025+"


def edit_location_bin(prefix_tokens: int, source_tokens: int) -> str:
    content_tokens = max(source_tokens - 1, 1)
    content_prefix = max(prefix_tokens - 1, 0)
    fraction = min(content_prefix / content_tokens, 1.0)
    if fraction < 0.25:
        return "start"
    if fraction < 0.75:
        return "middle"
    return "end"


def balanced_select(cases: Sequence[TimingCase], limit: int | None) -> list[TimingCase]:
    """Deterministically round-robin length/location cells."""
    groups: dict[tuple[str, str], list[TimingCase]] = defaultdict(list)
    for case in sorted(cases, key=lambda item: item.case_id):
        groups[(case.source_length_bin, case.edit_location_bin)].append(case)
    keys = sorted(groups)
    selected: list[TimingCase] = []
    offset = 0
    while keys and (limit is None or len(selected) < limit):
        remaining_keys: list[tuple[str, str]] = []
        for key in keys:
            rows = groups[key]
            if offset < len(rows):
                selected.append(rows[offset])
                if limit is not None and len(selected) >= limit:
                    break
            if offset + 1 < len(rows):
                remaining_keys.append(key)
        else:
            offset += 1
            keys = remaining_keys
            continue
        break
    return selected


def _tokens_from_text(tokenizer, text: str) -> tuple[int, ...]:
    return (int(tokenizer.get_bos_token_id()), *map(int, tokenizer.encode(text)))


def _detect_input_kind(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Input JSONL is empty")
    if "resolved_tokens" in rows[0] or "resolved_text" in rows[0]:
        return "controlled_records"
    if "target_tokens" in rows[0] and "gold_edits" in rows[0]:
        return "benchmark"
    if "source_tokens" in rows[0] and "edited_tokens" in rows[0]:
        return "token_pairs"
    raise ValueError("Could not infer input kind; use --input-kind explicitly")


def prepare_cases(
    rows: list[dict[str, Any]],
    *,
    input_kind: str,
    tokenizer,
    token_ops,
    context_tokens: int,
    resume_tokens: int,
    include_invalid_records: bool,
) -> tuple[list[TimingCase], dict[str, Any]]:
    metadata: dict[str, Any] = {"input_kind": input_kind}
    cases: list[TimingCase] = []
    if input_kind == "benchmark":
        rows, portability = prepare_benchmark_for_tokenizer(
            rows,
            tokenizer=tokenizer,
            token_ops=token_ops,
        )
        metadata["benchmark_tokenization"] = portability
        candidates = (
            (
                str(row["benchmark_id"]),
                tuple(map(int, row["source_tokens"])),
                tuple(map(int, row["target_tokens"])),
                None,
            )
            for row in rows
        )
    elif input_kind == "controlled_records":
        candidate_rows: list[
            tuple[str, tuple[int, ...], tuple[int, ...], int | None]
        ] = []
        for index, row in enumerate(rows):
            if not include_invalid_records and not bool(row.get("valid", True)):
                continue
            case_id = str(row.get("evaluation_id") or row.get("benchmark_id") or index)
            if row.get("source_text") is not None and row.get("resolved_text") is not None:
                source = _tokens_from_text(tokenizer, str(row["source_text"]))
                edited = _tokens_from_text(tokenizer, str(row["resolved_text"]))
            else:
                source = tuple(map(int, row["source_tokens"]))
                edited = tuple(map(int, row["resolved_tokens"]))
            candidate_rows.append((case_id, source, edited, None))
        candidates = iter(candidate_rows)
    elif input_kind == "token_pairs":
        candidates = (
            (
                str(row.get("case_id", index)),
                tuple(map(int, row["source_tokens"])),
                tuple(map(int, row["edited_tokens"])),
                (
                    int(row["edit_mass_tokens"])
                    if row.get("edit_mass_tokens") is not None
                    else None
                ),
            )
            for index, row in enumerate(rows)
        )
    else:
        raise ValueError(f"Unsupported input kind: {input_kind}")

    skipped_unchanged = 0
    skipped_context = 0
    skipped_empty = 0
    for case_id, source, edited, edit_mass_tokens in candidates:
        if not source or not edited:
            skipped_empty += 1
            continue
        prefix = longest_common_prefix(source, edited)
        if prefix == len(source) == len(edited):
            skipped_unchanged += 1
            continue
        if max(len(source), len(edited)) + resume_tokens > context_tokens:
            skipped_context += 1
            continue
        cases.append(TimingCase(
            case_id=case_id,
            source_tokens=source,
            edited_tokens=edited,
            stable_prefix_tokens=prefix,
            source_length_bin=source_length_bin(len(source)),
            edit_location_bin=edit_location_bin(prefix, len(source)),
            input_kind=input_kind,
            edit_mass_tokens=edit_mass_tokens,
        ))
    metadata["candidate_rows"] = len(rows)
    metadata["eligible_changed_cases"] = len(cases)
    metadata["skipped_unchanged"] = skipped_unchanged
    metadata["skipped_context"] = skipped_context
    metadata["skipped_empty"] = skipped_empty
    return cases, metadata


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cache_dtype(model, device: torch.device) -> torch.dtype:
    return model.cos.dtype if hasattr(model, "cos") else (torch.bfloat16 if device.type == "cuda" else torch.float32)


def make_cache_pair(model, *, capacity: int, device: torch.device) -> CachePair:
    config = model.config
    head_dim = (
        model.get_kv_cache_head_dim()
        if hasattr(model, "get_kv_cache_head_dim")
        else config.n_embd // config.n_head
    )
    dtype = _cache_dtype(model, device)
    model_cache = KVCache(
        batch_size=1,
        num_heads=config.n_kv_head,
        seq_len=capacity,
        head_dim=head_dim,
        num_layers=config.n_layer,
        device=device,
        dtype=dtype,
    )
    position_cache = None
    if getattr(model, "position_head", None) is not None:
        position_cache = KVCache(
            batch_size=1,
            num_heads=1,
            seq_len=capacity,
            head_dim=config.n_embd,
            num_layers=1,
            device=device,
            dtype=dtype,
        )
    return CachePair(model=model_cache, position=position_cache)


def _reset_cache_pair(caches: CachePair) -> None:
    caches.model.reset()
    if caches.position is not None:
        caches.position.reset()


def _set_cache_boundary(
    caches: CachePair,
    *,
    length: int,
    previous_embedding: torch.Tensor | None,
) -> None:
    caches.model.cache_seqlens.fill_(length)
    caches.model.prev_embedding = previous_embedding
    if caches.position is not None:
        caches.position.cache_seqlens.fill_(length)
        caches.position.prev_embedding = None


@torch.inference_mode()
def _prefill(
    model,
    caches: CachePair,
    tokens: Sequence[int],
    *,
    start_position: int = 0,
) -> torch.Tensor:
    if not tokens:
        raise ValueError("Cannot prefill an empty token sequence")
    device = model.get_device()
    ids = torch.tensor([list(map(int, tokens))], dtype=torch.long, device=device)
    tix = torch.arange(
        start_position,
        start_position + len(tokens),
        dtype=torch.long,
        device=device,
    )[None, :]
    logits, _position_logits = model.forward(
        ids,
        tix=tix,
        kv_cache=caches.model,
        pos_k_cache=caches.position,
        return_pos_logits=True,
    )
    # Clone the last-token logits so the full [T,V] tensor can be released.
    return logits[:, -1, :].clone()


def _masked_greedy(logits: torch.Tensor, banned_token_ids: Sequence[int]) -> torch.Tensor:
    if banned_token_ids:
        logits = logits.clone()
        logits[:, list(banned_token_ids)] = float("-inf")
    return torch.argmax(logits, dim=-1, keepdim=True)


@torch.inference_mode()
def _resume_decode(
    model,
    caches: CachePair,
    logits: torch.Tensor,
    *,
    num_tokens: int,
    banned_token_ids: Sequence[int],
) -> None:
    device = model.get_device()
    for _ in range(num_tokens):
        next_id = _masked_greedy(logits, banned_token_ids)
        position = caches.model.get_pos()
        tix = torch.full((1, 1), position, dtype=torch.long, device=device)
        logits_full, _position_logits = model.forward(
            next_id,
            tix=tix,
            kv_cache=caches.model,
            pos_k_cache=caches.position,
            return_pos_logits=True,
        )
        logits = logits_full[:, -1, :]


def _time_prefill(
    model,
    caches: CachePair,
    tokens: Sequence[int],
    *,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    _reset_cache_pair(caches)
    _synchronise(device)
    started = time.perf_counter()
    logits = _prefill(model, caches, tokens)
    _synchronise(device)
    return (time.perf_counter() - started) * 1000.0, logits


def _time_resume(
    model,
    caches: CachePair,
    logits: torch.Tensor,
    *,
    device: torch.device,
    resume_tokens: int,
    banned_token_ids: Sequence[int],
) -> float:
    _synchronise(device)
    started = time.perf_counter()
    _resume_decode(
        model,
        caches,
        logits,
        num_tokens=resume_tokens,
        banned_token_ids=banned_token_ids,
    )
    _synchronise(device)
    return (time.perf_counter() - started) * 1000.0


def _run_source_prefill(
    model,
    caches: CachePair,
    case: TimingCase,
    *,
    device: torch.device,
) -> dict[str, float]:
    rebuild_ms, _logits = _time_prefill(
        model,
        caches,
        case.source_tokens,
        device=device,
    )
    return {"rebuild_ms": rebuild_ms, "resume_decode_ms": 0.0, "total_ms": rebuild_ms}


def _materialise_state(
    model,
    caches: CachePair,
    tokens: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    _reset_cache_pair(caches)
    logits = _prefill(model, caches, tokens)
    previous_embedding = (
        caches.model.prev_embedding.clone()
        if caches.model.prev_embedding is not None
        else None
    )
    return logits, previous_embedding


def _run_no_edit_resume(
    model,
    caches: CachePair,
    *,
    source_length: int,
    source_logits: torch.Tensor,
    source_previous_embedding: torch.Tensor | None,
    device: torch.device,
    resume_tokens: int,
    banned_token_ids: Sequence[int],
) -> dict[str, float]:
    _set_cache_boundary(
        caches,
        length=source_length,
        previous_embedding=source_previous_embedding,
    )
    resume_ms = _time_resume(
        model,
        caches,
        source_logits,
        device=device,
        resume_tokens=resume_tokens,
        banned_token_ids=banned_token_ids,
    )
    return {"rebuild_ms": 0.0, "resume_decode_ms": resume_ms, "total_ms": resume_ms}


def _run_full_rebuild_resume(
    model,
    caches: CachePair,
    case: TimingCase,
    *,
    device: torch.device,
    resume_tokens: int,
    banned_token_ids: Sequence[int],
) -> dict[str, float]:
    rebuild_ms, logits = _time_prefill(
        model,
        caches,
        case.edited_tokens,
        device=device,
    )
    resume_ms = _time_resume(
        model,
        caches,
        logits,
        device=device,
        resume_tokens=resume_tokens,
        banned_token_ids=banned_token_ids,
    )
    return {
        "rebuild_ms": rebuild_ms,
        "resume_decode_ms": resume_ms,
        "total_ms": rebuild_ms + resume_ms,
    }


def _run_prefix_reuse_resume(
    model,
    caches: CachePair,
    case: TimingCase,
    *,
    prefix_logits: torch.Tensor | None,
    prefix_previous_embedding: torch.Tensor | None,
    device: torch.device,
    resume_tokens: int,
    banned_token_ids: Sequence[int],
) -> dict[str, float]:
    prefix_length = case.stable_prefix_tokens
    _set_cache_boundary(
        caches,
        length=prefix_length,
        previous_embedding=prefix_previous_embedding,
    )
    suffix = case.edited_tokens[prefix_length:]
    if suffix:
        _synchronise(device)
        started = time.perf_counter()
        logits = _prefill(
            model,
            caches,
            suffix,
            start_position=prefix_length,
        )
        _synchronise(device)
        rebuild_ms = (time.perf_counter() - started) * 1000.0
    else:
        if prefix_logits is None:
            raise AssertionError("A zero-length suffix requires prefix logits")
        logits = prefix_logits
        rebuild_ms = 0.0
    resume_ms = _time_resume(
        model,
        caches,
        logits,
        device=device,
        resume_tokens=resume_tokens,
        banned_token_ids=banned_token_ids,
    )
    return {
        "rebuild_ms": rebuild_ms,
        "resume_decode_ms": resume_ms,
        "total_ms": rebuild_ms + resume_ms,
    }


def _measurement_row(
    case: TimingCase,
    *,
    scenario: str,
    repeat: int,
    timing: dict[str, float],
    resume_tokens: int,
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "scenario": scenario,
        "repeat": repeat,
        "source_num_tokens": case.source_num_tokens,
        "edited_num_tokens": case.edited_num_tokens,
        "stable_prefix_tokens": case.stable_prefix_tokens,
        "stable_prefix_fraction": case.stable_prefix_fraction,
        "recomputed_suffix_tokens": case.recomputed_suffix_tokens,
        "source_length_bin": case.source_length_bin,
        "edit_location_bin": case.edit_location_bin,
        "edit_mass_tokens": case.edit_mass_tokens,
        "resume_tokens": resume_tokens,
        **timing,
    }


def benchmark_case(
    model,
    case: TimingCase,
    *,
    device: torch.device,
    warmups: int,
    repetitions: int,
    resume_tokens: int,
    banned_token_ids: Sequence[int],
) -> list[dict[str, Any]]:
    capacity = max(case.source_num_tokens, case.edited_num_tokens) + resume_tokens
    source_caches = make_cache_pair(model, capacity=capacity, device=device)
    full_caches = make_cache_pair(model, capacity=capacity, device=device)
    prefix_caches = make_cache_pair(model, capacity=capacity, device=device)

    source_logits, source_previous_embedding = _materialise_state(
        model,
        source_caches,
        case.source_tokens,
    )
    prefix_logits: torch.Tensor | None = None
    prefix_previous_embedding: torch.Tensor | None = None
    if case.stable_prefix_tokens:
        prefix_logits, prefix_previous_embedding = _materialise_state(
            model,
            prefix_caches,
            case.edited_tokens[:case.stable_prefix_tokens],
        )

    runners = {
        "source_prefill": lambda: _run_source_prefill(
            model,
            full_caches,
            case,
            device=device,
        ),
        "no_edit_resume": lambda: _run_no_edit_resume(
            model,
            source_caches,
            source_length=case.source_num_tokens,
            source_logits=source_logits,
            source_previous_embedding=source_previous_embedding,
            device=device,
            resume_tokens=resume_tokens,
            banned_token_ids=banned_token_ids,
        ),
        "full_rebuild_resume": lambda: _run_full_rebuild_resume(
            model,
            full_caches,
            case,
            device=device,
            resume_tokens=resume_tokens,
            banned_token_ids=banned_token_ids,
        ),
        "prefix_reuse_resume": lambda: _run_prefix_reuse_resume(
            model,
            prefix_caches,
            case,
            prefix_logits=prefix_logits,
            prefix_previous_embedding=prefix_previous_embedding,
            device=device,
            resume_tokens=resume_tokens,
            banned_token_ids=banned_token_ids,
        ),
    }

    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for _ in range(warmups):
            runners[scenario]()
        for repeat in range(repetitions):
            timing = runners[scenario]()
            rows.append(_measurement_row(
                case,
                scenario=scenario,
                repeat=repeat,
                timing=timing,
                resume_tokens=resume_tokens,
            ))
    del source_caches, full_caches, prefix_caches
    return rows


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile)) if values else None


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1 or samples <= 0:
        value = float(array.mean())
        return [value, value]
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    remaining = samples
    while remaining:
        chunk = min(1000, remaining)
        indices = rng.integers(0, len(array), size=(chunk, len(array)))
        means.append(array[indices].mean(axis=1))
        remaining -= chunk
    distribution = np.concatenate(means)
    return [
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    ]


def _summary_stats(
    values: Sequence[float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            "cases": 0,
            "mean": None,
            "mean_ci95": None,
            "median": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
            "stdev": None,
        }
    return {
        "cases": len(clean),
        "mean": float(statistics.fmean(clean)),
        "mean_ci95": _bootstrap_mean_ci(
            clean,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "median": float(statistics.median(clean)),
        "p90": _percentile(clean, 0.90),
        "p95": _percentile(clean, 0.95),
        "min": min(clean),
        "max": max(clean),
        "stdev": float(statistics.stdev(clean)) if len(clean) > 1 else 0.0,
    }


def case_median_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case_id"]), str(row["scenario"]))].append(row)
    output: list[dict[str, Any]] = []
    for (_case_id, _scenario), measurements in sorted(grouped.items()):
        base = dict(measurements[0])
        base.pop("repeat", None)
        base["measurements"] = len(measurements)
        for metric in TIMING_METRICS:
            base[metric] = float(statistics.median(float(row[metric]) for row in measurements))
        output.append(base)
    return output


def build_case_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    medians = case_median_rows(rows)
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in medians:
        by_case[str(row["case_id"])][str(row["scenario"])] = row
    output: list[dict[str, Any]] = []
    for case_id, scenarios in sorted(by_case.items()):
        if set(scenarios) != set(SCENARIOS):
            raise ValueError(f"Case {case_id} is missing timing scenarios")
        exemplar = scenarios["full_rebuild_resume"]
        row: dict[str, Any] = {
            key: exemplar[key]
            for key in (
                "case_id",
                "source_num_tokens",
                "edited_num_tokens",
                "stable_prefix_tokens",
                "stable_prefix_fraction",
                "recomputed_suffix_tokens",
                "source_length_bin",
                "edit_location_bin",
                "edit_mass_tokens",
                "resume_tokens",
            )
        }
        for scenario in SCENARIOS:
            for metric in TIMING_METRICS:
                row[f"{scenario}_{metric}"] = scenarios[scenario][metric]
        full_rebuild = float(scenarios["full_rebuild_resume"]["rebuild_ms"])
        prefix_rebuild = float(scenarios["prefix_reuse_resume"]["rebuild_ms"])
        full_total = float(scenarios["full_rebuild_resume"]["total_ms"])
        prefix_total = float(scenarios["prefix_reuse_resume"]["total_ms"])
        no_edit_total = float(scenarios["no_edit_resume"]["total_ms"])
        source_prefill = float(scenarios["source_prefill"]["total_ms"])
        row.update({
            "prefix_rebuild_fraction_of_full": (
                prefix_rebuild / full_rebuild if full_rebuild > 0 else None
            ),
            "prefix_total_fraction_of_full": (
                prefix_total / full_total if full_total > 0 else None
            ),
            "full_total_over_no_edit_resume": (
                full_total / no_edit_total if no_edit_total > 0 else None
            ),
            "full_rebuild_over_source_prefill": (
                full_rebuild / source_prefill if source_prefill > 0 else None
            ),
        })
        output.append(row)
    return output


def summarise_case_rows(
    cases: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        groups["overall"].append(row)
        groups[f"source_length:{row['source_length_bin']}"].append(row)
        groups[f"source_tokens:{row['source_num_tokens']}"].append(row)
        groups[f"edit_location:{row['edit_location_bin']}"].append(row)
        if row.get("edit_mass_tokens") is not None:
            groups[f"edit_mass:{row['edit_mass_tokens']}"].append(row)
        groups[
            f"cell:{row['source_length_bin']}:{row['edit_location_bin']}"
        ].append(row)

    summary: dict[str, Any] = {}
    metrics = (
        "source_prefill_total_ms",
        "no_edit_resume_total_ms",
        "full_rebuild_resume_rebuild_ms",
        "full_rebuild_resume_resume_decode_ms",
        "full_rebuild_resume_total_ms",
        "prefix_reuse_resume_rebuild_ms",
        "prefix_reuse_resume_resume_decode_ms",
        "prefix_reuse_resume_total_ms",
        "prefix_rebuild_fraction_of_full",
        "prefix_total_fraction_of_full",
        "full_total_over_no_edit_resume",
        "full_rebuild_over_source_prefill",
    )
    for group_name, group_rows in sorted(groups.items()):
        group_summary: dict[str, Any] = {"cases": len(group_rows)}
        for metric_index, metric in enumerate(metrics):
            values = [
                float(row[metric])
                for row in group_rows
                if row.get(metric) is not None
            ]
            group_summary[metric] = _summary_stats(
                values,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + metric_index,
            )
        summary[group_name] = group_summary
    return summary


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write empty CSV")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata.update({
            "device_name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": int(properties.total_memory),
        })
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument(
        "--input-kind",
        choices=("auto", "benchmark", "controlled_records", "token_pairs"),
        default="auto",
    )
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--resume-tokens", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument("--include-invalid-records", action="store_true")
    parser.add_argument("--device-type", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for name in ("max_cases", "warmups", "repetitions", "resume_tokens"):
        value = getattr(args, name)
        minimum = 0 if name in {"warmups", "resume_tokens"} else 1
        if value < minimum:
            raise ValueError(f"--{name.replace('_', '-')} must be >= {minimum}")

    output_dir = Path(args.output_dir)
    raw_path = output_dir / "raw_timings.jsonl"
    case_path = output_dir / "case_summary.csv"
    summary_path = output_dir / "summary.json"
    if not args.overwrite and any(path.exists() for path in (raw_path, case_path, summary_path)):
        raise FileExistsError("Timing output exists; use --overwrite or choose another directory")

    device_type = autodetect_device_type() if not args.device_type else args.device_type
    _ddp, rank, _local_rank, world_size, device = compute_init(device_type)
    if rank != 0 or world_size != 1:
        raise RuntimeError("The cache timing benchmark requires a single process/GPU")
    model, tokenizer, meta = load_model(
        "base",
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )

    # Deferred import avoids constructing tokenizer-dependent edit operations in
    # tests that only exercise the pure selection/summary helpers.
    from nanochat.causal_serialisation_extended import TokensOperations

    token_ops = TokensOperations(tokenizer.enc)
    input_path = Path(args.input_jsonl)
    rows = _read_jsonl(input_path)
    input_kind = _detect_input_kind(rows) if args.input_kind == "auto" else args.input_kind
    cases, input_metadata = prepare_cases(
        rows,
        input_kind=input_kind,
        tokenizer=tokenizer,
        token_ops=token_ops,
        context_tokens=int(model.config.sequence_len),
        resume_tokens=args.resume_tokens,
        include_invalid_records=args.include_invalid_records,
    )
    selected = balanced_select(cases, args.max_cases)
    if not selected:
        raise ValueError("No changed timing cases fit the model context")

    vocab_size = int(model.config.vocab_size)
    banned_token_ids = sorted({
        int(tokenizer.get_bos_token_id()),
        *(
            int(tokenizer.encode_special(token))
            for token in INDEL_SPECIALS
            if 0 <= int(tokenizer.encode_special(token)) < vocab_size
        ),
    })

    raw_rows: list[dict[str, Any]] = []
    for index, case in enumerate(selected, start=1):
        raw_rows.extend(benchmark_case(
            model,
            case,
            device=device,
            warmups=args.warmups,
            repetitions=args.repetitions,
            resume_tokens=args.resume_tokens,
            banned_token_ids=banned_token_ids,
        ))
        print(
            f"cache timing: {index}/{len(selected)} "
            f"source={case.source_num_tokens} edited={case.edited_num_tokens} "
            f"prefix={case.stable_prefix_tokens}",
            flush=True,
        )

    case_rows = build_case_summary(raw_rows)
    aggregate = summarise_case_rows(
        case_rows,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(raw_path, raw_rows)
    _write_csv(case_path, case_rows)
    summary = {
        "schema_version": 1,
        "git_commit": _git_commit(),
        "input_jsonl": str(input_path.resolve()),
        "input_sha256": _file_sha256(input_path),
        "input": input_metadata,
        "selection": {
            "eligible_cases": len(cases),
            "selected_cases": len(selected),
            "selection": "deterministic round-robin over source-length/edit-location cells",
        },
        "model": {
            "model_tag": args.model_tag,
            "step": int(meta["step"]),
            "depth": int(model.config.n_layer),
            "context_tokens": int(model.config.sequence_len),
            "vocab_size": vocab_size,
            "use_dual_rope": bool(model.use_dual_rope),
            "dual_rope_style": str(model.dual_rope_style),
            "has_position_head": model.position_head is not None,
        },
        "device": _device_metadata(device),
        "timing_protocol": {
            "warmups_per_case_scenario": args.warmups,
            "repetitions_per_case_scenario": args.repetitions,
            "resume_tokens": args.resume_tokens,
            "cuda_synchronised": device.type == "cuda",
            "cache_allocation_timed": False,
            "tokenization_timed": False,
            "checkpoint_loading_timed": False,
            "edit_program_generation_timed": False,
            "edit_application_timed": False,
            "full_rebuild_definition": "prefill the complete edited prefix into an empty preallocated cache",
            "prefix_reuse_definition": (
                "retain cache entries and boundary embedding before the earliest changed token; "
                "recompute the edited suffix and then resume"
            ),
            "terminal_edit_cache_rebuild_ms": 0.0,
            "uncertainty_unit": (
                "aggregate statistics use each case's median across repetitions; "
                "bootstrap intervals resample cases, not individual repeats"
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "selected_case_counts": {
            f"{length_bin}:{location_bin}": count
            for (length_bin, location_bin), count in sorted(
                {
                    key: sum(
                        case.source_length_bin == key[0]
                        and case.edit_location_bin == key[1]
                        for case in selected
                    )
                    for key in {
                        (case.source_length_bin, case.edit_location_bin)
                        for case in selected
                    }
                }.items()
            )
        },
        "groups": aggregate,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "raw_timings": str(raw_path),
        "case_summary": str(case_path),
        "summary": str(summary_path),
        "overall": summary["groups"]["overall"],
    }, indent=2))
    compute_cleanup()


if __name__ == "__main__":
    main()
