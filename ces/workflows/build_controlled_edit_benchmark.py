"""Build a frozen controlled CES source/target manifest.

By default the manifest uses only the final ClimbMix validation shard. Training
manifests can instead draw from non-final ClimbMix shards. Each row contains
the corrupted source visible at inference, the clean target, the exact inverse
repair program, construction provenance, and a serialized prompt ending at
``<|edit-start|>``. The manifest is model-independent but tokenizer-specific.

Examples are balanced across requested repair complexity and operation modes.
The one/two/four-operation strata retain the training span/payload sampler;
eight-operation rows are explicitly labelled as a dense stress test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

from nanochat.causal_serialisation import ChunkReservoir
from nanochat.causal_serialisation_extended import (
    Copy,
    Delete,
    Insert,
    Move,
    Replace,
    SequenceEdit,
    TokensOperations,
    resolve,
)
from nanochat.dataset import list_parquet_files
from nanochat.edit_dataloader import EditSamplingConfig, _sample_random_inverse_edit
from nanochat.tokenizer import get_extended_indel_tokeniser


OP_MODES = ("mixed", "insert", "delete", "replace", "move")


@dataclass(frozen=True)
class BenchmarkStratum:
    repair_op_mode: str
    repair_num_ops: int

    @property
    def name(self) -> str:
        return f"{self.repair_op_mode}_k{self.repair_num_ops}"


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
    raise TypeError(f"Unsupported edit type: {type(edit)!r}")


def _edit_record(edit: Any) -> dict[str, Any]:
    if isinstance(edit, Insert):
        return {"op": "insert", "pos": int(edit.pos), "content": np.asarray(edit.content).astype(int).tolist()}
    if isinstance(edit, Delete):
        return {"op": "delete", "start": int(edit.start), "end": int(edit.end)}
    if isinstance(edit, Replace):
        return {
            "op": "replace",
            "start": int(edit.start),
            "end": int(edit.end),
            "content": np.asarray(edit.content).astype(int).tolist(),
        }
    if isinstance(edit, Copy):
        return {"op": "copy", "start": int(edit.start), "end": int(edit.end), "dest": int(edit.dest)}
    if isinstance(edit, Move):
        return {"op": "move", "start": int(edit.start), "end": int(edit.end), "dest": int(edit.dest)}
    raise TypeError(f"Unsupported edit type: {type(edit)!r}")


def _token_hash(tokens: Iterable[int]) -> str:
    payload = json.dumps([int(token) for token in tokens], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decode_without_special_bos(tokenizer, tokens: Iterable[int]) -> str:
    bos = tokenizer.get_bos_token_id()
    return tokenizer.decode([int(token) for token in tokens if int(token) != bos])


def _forced_sampler_config(
    base: EditSamplingConfig,
    *,
    stratum: BenchmarkStratum,
    source_len: int,
) -> EditSamplingConfig:
    # For exact-count rows, use the original span/payload lengths but shorten
    # the inter-edit gap enough that the requested number normally fits. k=8 is
    # intentionally out-of-distribution density stress.
    mean_gap = min(
        float(base.random_no_del_mean),
        max(6.0, (max(source_len - 1, 1) / max(stratum.repair_num_ops * 1.5, 1))),
    )
    kwargs: dict[str, Any] = {
        "max_edits_per_example": int(stratum.repair_num_ops),
        "random_no_del_mean": mean_gap,
        "random_no_ins_mean": mean_gap,
    }

    # _sample_random_inverse_edit samples a clean->corrupt operation and then
    # complements it. Insert/delete therefore exchange names in the repair.
    if stratum.repair_op_mode == "insert":
        kwargs.update(random_replace_prob=0.0, random_delete_prob=1.0, random_insert_prob=0.0, random_move_prob=0.0)
    elif stratum.repair_op_mode == "delete":
        kwargs.update(random_replace_prob=0.0, random_delete_prob=0.0, random_insert_prob=1.0, random_move_prob=0.0)
    elif stratum.repair_op_mode == "replace":
        kwargs.update(random_replace_prob=1.0, random_delete_prob=0.0, random_insert_prob=0.0, random_move_prob=0.0)
    elif stratum.repair_op_mode == "move":
        kwargs.update(random_replace_prob=0.0, random_delete_prob=0.0, random_insert_prob=0.0, random_move_prob=1.0)
    elif stratum.repair_op_mode != "mixed":
        raise ValueError(f"Unknown repair operation mode: {stratum.repair_op_mode}")
    return replace(base, **kwargs)


def sample_exact_repair(
    clean_tokens: list[int],
    *,
    token_ops: TokensOperations,
    stratum: BenchmarkStratum,
    seed: int,
    max_attempts: int = 256,
) -> SequenceEdit | None:
    """Sample an exact-count inverse repair using the live training sampler."""

    rng = np.random.default_rng(seed)
    base = EditSamplingConfig()
    cfg = _forced_sampler_config(base, stratum=stratum, source_len=len(clean_tokens))
    reservoir = ChunkReservoir(max_chunks=1, chunk_size=32, rng=rng)
    clean = np.asarray(clean_tokens, dtype=np.int64)
    bos = token_ops.tokeniser._special_tokens.get("<|bos|>", -1)
    for _ in range(max_attempts):
        repair = _sample_random_inverse_edit(
            clean,
            cfg=cfg,
            token_ops=token_ops,
            reservoir=reservoir,
            rng=rng,
        )
        if repair is None or len(repair.edits) != stratum.repair_num_ops:
            continue
        names = [_op_name(edit) for edit in repair.edits]
        if stratum.repair_op_mode != "mixed" and any(name != stratum.repair_op_mode for name in names):
            continue
        if int(np.sum(np.asarray(repair.sequence) == bos)) != 1:
            continue
        if not np.array_equal(resolve(repair), clean):
            continue
        return repair
    return None


def _select_source_paths(
    raw_data_dir: str,
    *,
    source_split: str,
    max_source_shards: int | None,
    source_shard_selection: str,
    seed: int,
) -> list[str]:
    paths = list_parquet_files(data_dir=raw_data_dir)
    if not paths:
        raise FileNotFoundError(f"No parquet files beneath {raw_data_dir}")
    if source_split == "validation":
        selected_paths = paths[-1:]
    elif source_split == "train":
        selected_paths = paths[:-1]
        if not selected_paths:
            raise ValueError("Need at least two shards to reserve the final shard for validation")
    else:
        raise ValueError(f"Unknown source split: {source_split!r}")
    if max_source_shards is not None:
        if max_source_shards < 1:
            raise ValueError("--max-source-shards must be positive")
        count = min(max_source_shards, len(selected_paths))
        if source_shard_selection == "seeded_interleaved":
            rng = np.random.default_rng(seed)
            indices = sorted(
                int(index)
                for index in rng.choice(len(selected_paths), size=count, replace=False)
            )
            selected_paths = [selected_paths[index] for index in indices]
        elif source_shard_selection == "sequential":
            selected_paths = selected_paths[:count]
        else:
            raise ValueError(
                f"Unknown source shard selection: {source_shard_selection!r}"
            )
    return selected_paths


def _iter_path_documents(path: str) -> Iterable[tuple[str, int, str]]:
    source_path = Path(path)
    parquet = pq.ParquetFile(source_path)
    row_offset = 0
    for row_group in range(parquet.num_row_groups):
        texts = parquet.read_row_group(row_group, columns=["text"]).column("text").to_pylist()
        for local_index, text in enumerate(texts):
            yield source_path.stem, row_offset + local_index, str(text)
        row_offset += len(texts)


def _iter_documents(
    raw_data_dir: str,
    *,
    source_split: str,
    max_source_shards: int | None,
    source_shard_selection: str = "sequential",
    seed: int = 0,
) -> Iterable[tuple[str, int, str]]:
    selected_paths = _select_source_paths(
        raw_data_dir,
        source_split=source_split,
        max_source_shards=max_source_shards,
        source_shard_selection=source_shard_selection,
        seed=seed,
    )

    if source_shard_selection == "seeded_interleaved" and len(selected_paths) > 1:
        active = deque(iter(_iter_path_documents(path)) for path in selected_paths)
        while active:
            iterator = active.popleft()
            try:
                yield next(iterator)
            except StopIteration:
                continue
            active.append(iterator)
        return

    for path in selected_paths:
        yield from _iter_path_documents(path)


def _document_window(
    text: str,
    *,
    tokenizer,
    seed: int,
    min_source_tokens: int,
    max_source_tokens: int,
) -> tuple[list[int], int] | None:
    body = [int(token) for token in tokenizer.encode(text)]
    if len(body) + 1 < min_source_tokens:
        return None
    width = max_source_tokens - 1
    offset = 0
    if len(body) > width:
        rng = np.random.default_rng(seed)
        offset = int(rng.integers(0, len(body) - width + 1))
        body = body[offset:offset + width]
    return [tokenizer.get_bos_token_id(), *body], offset


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.examples_per_stratum < 1:
        raise ValueError("--examples-per-stratum must be positive")
    if args.min_source_tokens < 4 or args.max_source_tokens < args.min_source_tokens:
        raise ValueError("invalid source token limits")

    tokenizer = get_extended_indel_tokeniser()
    token_ops = TokensOperations(tokenizer.enc)
    strata = [
        BenchmarkStratum(op_mode, int(complexity))
        for complexity in args.complexities
        for op_mode in args.op_modes
    ]
    remaining = {stratum.name: args.examples_per_stratum for stratum in strata}
    records: list[dict[str, Any]] = []
    stratum_cursor = 0

    for shard_id, row_index, text in _iter_documents(
        args.raw_data_dir,
        source_split=args.source_split,
        max_source_shards=args.max_source_shards,
        source_shard_selection=args.source_shard_selection,
        seed=args.seed,
    ):
        if not any(remaining.values()):
            break
        candidates = [strata[(stratum_cursor + i) % len(strata)] for i in range(len(strata))]
        stratum = next((candidate for candidate in candidates if remaining[candidate.name] > 0), None)
        if stratum is None:
            break
        stratum_cursor = (strata.index(stratum) + 1) % len(strata)
        construction_seed = int(np.random.SeedSequence([args.seed, row_index, strata.index(stratum)]).generate_state(1)[0])
        window = _document_window(
            text,
            tokenizer=tokenizer,
            seed=construction_seed,
            min_source_tokens=args.min_source_tokens,
            max_source_tokens=args.max_source_tokens,
        )
        if window is None:
            continue
        clean_tokens, source_token_offset = window
        repair = sample_exact_repair(
            clean_tokens,
            token_ops=token_ops,
            stratum=stratum,
            seed=construction_seed,
            max_attempts=args.max_sampling_attempts,
        )
        if repair is None:
            continue
        try:
            serialised = token_ops.serialise(repair, indel_bos_pos=0)
        except ValueError:
            continue
        edit_start_indices = np.where(serialised.tokens == token_ops.edit_start_token)[0]
        if len(edit_start_indices) != 1:
            continue
        edit_start_index = int(edit_start_indices[0])
        prompt_tokens = np.asarray(serialised.tokens[: edit_start_index + 1]).astype(int).tolist()
        prompt_positions = np.asarray(serialised.positions[: edit_start_index + 1]).astype(int).tolist()
        gold_program_tokens = np.asarray(serialised.tokens[edit_start_index + 1:]).astype(int).tolist()
        gold_program_positions = np.asarray(serialised.positions[edit_start_index + 1:]).astype(int).tolist()
        if len(prompt_tokens) + min(len(gold_program_tokens) + args.generation_token_margin, args.max_edit_tokens) > args.model_context_tokens:
            continue

        source_tokens = np.asarray(repair.sequence).astype(int).tolist()
        target_tokens = [int(token) for token in clean_tokens]
        op_counts = Counter(_op_name(edit) for edit in repair.edits)
        benchmark_id = f"{shard_id}:{row_index}:{source_token_offset}:{stratum.name}:{construction_seed}"
        records.append({
            "benchmark_id": benchmark_id,
            "benchmark_version": 1,
            "split": args.manifest_split,
            "source_shard_id": shard_id,
            "source_row_index": row_index,
            "source_token_offset": source_token_offset,
            "construction_seed": construction_seed,
            "stratum": stratum.name,
            "repair_op_mode": stratum.repair_op_mode,
            "repair_num_ops": stratum.repair_num_ops,
            "density_regime": "dense_stress" if stratum.repair_num_ops > 4 else "training_span_distribution",
            "source_num_tokens": len(source_tokens),
            "target_num_tokens": len(target_tokens),
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
            "source_token_sha256": _token_hash(source_tokens),
            "target_token_sha256": _token_hash(target_tokens),
            "source_text": _decode_without_special_bos(tokenizer, source_tokens),
            "target_text": _decode_without_special_bos(tokenizer, target_tokens),
            "gold_edits": [_edit_record(edit) for edit in repair.edits],
            "gold_op_counts": dict(sorted(op_counts.items())),
            "prompt_tokens": prompt_tokens,
            "prompt_positions": prompt_positions,
            "gold_program_tokens": gold_program_tokens,
            "gold_program_positions": gold_program_positions,
        })
        remaining[stratum.name] -= 1

    missing = {name: count for name, count in remaining.items() if count}
    if missing:
        raise RuntimeError(f"Source shards exhausted before filling manifest strata: {missing}")
    meta = {
        "benchmark_version": 1,
        "seed": args.seed,
        "raw_data_dir": str(Path(args.raw_data_dir).resolve()),
        "source_split": args.source_split,
        "manifest_split": args.manifest_split,
        "max_source_shards": args.max_source_shards,
        "source_shard_selection": args.source_shard_selection,
        "source_shard_counts": dict(sorted(Counter(record["source_shard_id"] for record in records).items())),
        "tokenizer": "extended_indel",
        "examples": len(records),
        "examples_per_stratum": args.examples_per_stratum,
        "complexities": [int(value) for value in args.complexities],
        "op_modes": list(args.op_modes),
        "min_source_tokens": args.min_source_tokens,
        "max_source_tokens": args.max_source_tokens,
        "model_context_tokens": args.model_context_tokens,
        "max_edit_tokens": args.max_edit_tokens,
        "stratum_counts": dict(sorted(Counter(record["stratum"] for record in records).items())),
    }
    return records, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--source-split", choices=("train", "validation"), default="validation"
    )
    parser.add_argument(
        "--manifest-split",
        choices=("training", "validation"),
        default=None,
        help="Row split label (defaults from --source-split)",
    )
    parser.add_argument(
        "--max-source-shards",
        type=int,
        default=None,
        help="Limit selected shards; useful when exporting an RL training subset",
    )
    parser.add_argument(
        "--source-shard-selection",
        choices=("sequential", "seeded_interleaved"),
        default="sequential",
        help=(
            "How to choose/traverse source shards. seeded_interleaved samples "
            "shards deterministically and round-robins documents across them."
        ),
    )
    parser.add_argument("--examples-per-stratum", type=int, default=50)
    parser.add_argument("--complexities", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--op-modes", nargs="+", choices=OP_MODES, default=list(OP_MODES))
    parser.add_argument("--min-source-tokens", type=int, default=96)
    parser.add_argument("--max-source-tokens", type=int, default=512)
    parser.add_argument("--model-context-tokens", type=int, default=2048)
    parser.add_argument("--max-edit-tokens", type=int, default=128)
    parser.add_argument("--generation-token-margin", type=int, default=16)
    parser.add_argument("--max-sampling-attempts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    if args.manifest_split is None:
        args.manifest_split = (
            "training" if args.source_split == "train" else "validation"
        )
    if args.source_split == "validation" and args.manifest_split == "training":
        parser.error("A training manifest cannot draw from the reserved validation shard")

    records, meta = build_manifest(args)
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_jsonl": str(output), "meta_json": str(meta_path), **meta}, indent=2))


if __name__ == "__main__":
    main()
