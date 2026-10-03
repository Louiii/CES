"""Streaming source/target pairs for on-policy CES post-training.

This is deliberately an adapter over :mod:`nanochat.edit_dataloader`, not an
independent corruption implementation.  It reuses the pretraining candidate
index, inverse-direction sampling, token conversion, long-document windowing,
and random inverse-edit sampler, but returns complete ``SequenceEdit`` repair
problems instead of padded supervised batches.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from nanochat.causal_serialisation import ChunkReservoir
from nanochat.causal_serialisation_extended import SequenceEdit, TokensOperations
from nanochat.dataset import list_parquet_files
from nanochat.edit_dataloader import (
    EditCandidateStore,
    EditSamplingConfig,
    _PreparedDoc,
    _prepare_doc,
    _sample_random_inverse_edit,
    _window_candidate_sequence_edit,
)


@dataclass(frozen=True)
class StreamingEditPair:
    """One verifier-ready repair problem sampled from the pretraining stream."""

    example_id: str
    kind: str
    source_shard_id: str
    source_row_index: int
    epoch: int
    sequence_edit: SequenceEdit


class StreamingEditPairSampler:
    """Sample GPT, random, and identity repair pairs without prompt repetition.

    Every source document is consumed once before the key schedule reshuffles.
    Mixture decisions are deterministic for a given seed, so data-parallel
    ranks independently reconstruct identical global prompts for RLOO.
    """

    def __init__(
        self,
        *,
        tokenizer,
        token_ops: TokensOperations,
        cfg: EditSamplingConfig,
        gpt_probability: float,
        random_probability: float,
        noop_probability: float,
        max_source_tokens: int = 512,
        max_serialised_tokens: int = 768,
        seed: int = 42,
    ) -> None:
        probabilities = np.asarray(
            [gpt_probability, random_probability, noop_probability],
            dtype=np.float64,
        )
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError(
                "GPT/random/no-op stream probabilities must be non-negative "
                "and sum to one"
            )
        if max_source_tokens < 8:
            raise ValueError("max_source_tokens must be at least eight")
        if max_serialised_tokens <= max_source_tokens:
            raise ValueError(
                "max_serialised_tokens must exceed max_source_tokens"
            )
        if cfg.raw_data_dir is None:
            raise ValueError("The streaming pair sampler requires raw_data_dir")
        if not cfg.generated_edit_paths:
            raise ValueError(
                "The streaming pair sampler requires generated_edit_paths"
            )

        self.tokenizer = tokenizer
        self.token_ops = token_ops
        self.cfg = replace(
            cfg,
            direction_probs={"inverse": 1.0},
            max_source_tokens=max_source_tokens,
            max_serialised_tokens=max_serialised_tokens,
        )
        self.store = EditCandidateStore(self.cfg)
        raw_paths = [Path(path) for path in list_parquet_files(cfg.raw_data_dir)]
        if len(raw_paths) < 2:
            raise ValueError(
                "Streaming edit pairs require train shards plus a final "
                "reserved validation shard"
            )
        train_shards = {path.stem for path in raw_paths[:-1]}
        self.keys = sorted(
            key for key in self.store.generated if key[0] in train_shards
        )
        if not self.keys:
            raise ValueError(
                "No generated edit candidates join the raw training shards"
            )
        self.probabilities = probabilities / probabilities.sum()
        self.max_source_tokens = int(max_source_tokens)
        self.max_serialised_tokens = int(max_serialised_tokens)
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.keys)
        self.cursor = 0
        self.epoch = 1
        self.consumed = 0
        self.skipped = 0
        self.kind_counts = {"gpt": 0, "random": 0, "noop": 0}
        self.bos_token = int(tokenizer.get_bos_token_id())
        self.reservoir = ChunkReservoir(
            max_chunks=8192,
            chunk_size=min(128, self.max_source_tokens),
            rng=self.rng,
        )

    @property
    def examples_per_epoch(self) -> int:
        return len(self.keys)

    def state_dict(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor,
            "epoch": self.epoch,
            "consumed": self.consumed,
            "skipped": self.skipped,
            "kind_counts": dict(self.kind_counts),
            "examples_per_epoch": self.examples_per_epoch,
        }

    def _next_key(self) -> tuple[str, int]:
        if self.cursor >= len(self.keys):
            self.epoch += 1
            self.cursor = 0
            self.rng.shuffle(self.keys)
        key = self.keys[self.cursor]
        self.cursor += 1
        self.consumed += 1
        return key

    def _clean_tokens(self, text: str) -> list[int] | None:
        body = [int(token) for token in self.tokenizer.encode(text)]
        if not body:
            return None
        width = self.max_source_tokens - 1
        if len(body) > width:
            start = int(self.rng.integers(0, len(body) - width + 1))
            body = body[start : start + width]
        tokens = [self.bos_token, *body]
        if len(tokens) >= self.reservoir.chunk_size:
            self.reservoir.offer(tokens[: self.reservoir.chunk_size])
        return tokens

    def _fits(self, sequence_edit: SequenceEdit) -> bool:
        try:
            serialised = self.token_ops.serialise(
                sequence_edit, indel_bos_pos=0
            )
        except ValueError:
            return False
        return (
            len(sequence_edit.sequence) <= self.max_source_tokens
            and len(serialised.tokens) <= self.max_serialised_tokens
        )

    def _gpt_edit(
        self, shard_id: str, row_index: int, original_text: str
    ) -> SequenceEdit | None:
        prepared = _prepare_doc(
            shard_id=shard_id,
            row_index=row_index,
            raw_text=original_text,
            tokenizer=self.tokenizer,
            bos_token=self.bos_token,
            store=self.store,
            cfg=self.cfg,
            rng=self.rng,
        )
        sequence_edit = prepared.token_edit
        if sequence_edit is None or not sequence_edit.edits:
            return None
        if self._fits(sequence_edit):
            return sequence_edit
        windowed = _window_candidate_sequence_edit(
            _PreparedDoc(
                shard_id=prepared.shard_id,
                row_index=prepared.row_index,
                source_text=prepared.source_text,
                tokens=prepared.tokens,
                token_edit=sequence_edit,
            ),
            full_token_budget=self.max_serialised_tokens,
            token_ops=self.token_ops,
            bos_token=self.bos_token,
        )
        return windowed if windowed is not None and self._fits(windowed) else None

    def _random_edit(self, original_text: str) -> SequenceEdit | None:
        clean = self._clean_tokens(original_text)
        if clean is None:
            return None
        for _ in range(16):
            sequence_edit = _sample_random_inverse_edit(
                np.asarray(clean, dtype=np.int64),
                cfg=self.cfg,
                token_ops=self.token_ops,
                reservoir=self.reservoir,
                rng=self.rng,
            )
            if sequence_edit is not None and self._fits(sequence_edit):
                return sequence_edit
        return None

    def _noop_edit(self, original_text: str) -> SequenceEdit | None:
        clean = self._clean_tokens(original_text)
        if clean is None:
            return None
        sequence_edit = SequenceEdit(
            sequence=np.asarray(clean, dtype=np.int64), edits=[]
        )
        return sequence_edit if self._fits(sequence_edit) else None

    def next(self, *, kind: str | None = None) -> StreamingEditPair:
        kinds = ("gpt", "random", "noop")
        if kind is None:
            kind = kinds[
                int(self.rng.choice(len(kinds), p=self.probabilities))
            ]
        elif kind not in kinds:
            raise ValueError(f"Unknown stream pair kind: {kind}")
        for _ in range(max(10_000, len(self.keys))):
            shard_id, row_index = self._next_key()
            candidates = self.store.generated[(shard_id, row_index)]
            original_text = candidates[0].source_text
            if kind == "gpt":
                sequence_edit = self._gpt_edit(
                    shard_id, row_index, original_text
                )
            elif kind == "random":
                sequence_edit = self._random_edit(original_text)
            else:
                sequence_edit = self._noop_edit(original_text)
            if sequence_edit is None:
                self.skipped += 1
                continue
            self.kind_counts[kind] += 1
            return StreamingEditPair(
                example_id=(
                    f"stream:{shard_id}:{row_index}:{self.epoch}:{kind}"
                ),
                kind=kind,
                source_shard_id=shard_id,
                source_row_index=int(row_index),
                epoch=self.epoch,
                sequence_edit=sequence_edit,
            )
        raise RuntimeError("Could not construct a streaming CES repair pair")

    def next_batch(self, count: int) -> list[StreamingEditPair]:
        """Return a shuffled largest-remainder mixture for one optimizer step."""

        if count < 1:
            raise ValueError("Streaming pair batch size must be positive")
        expected = self.probabilities * count
        counts = np.floor(expected).astype(np.int64)
        remaining = count - int(counts.sum())
        fractional_order = np.argsort(-(expected - counts), kind="stable")
        for index in fractional_order[:remaining]:
            counts[int(index)] += 1
        kinds = [
            kind
            for kind, kind_count in zip(("gpt", "random", "noop"), counts)
            for _ in range(int(kind_count))
        ]
        self.rng.shuffle(kinds)
        return [self.next(kind=kind) for kind in kinds]
