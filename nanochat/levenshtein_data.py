"""Source/target pair stream for the iterative Levenshtein baseline."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from nanochat.causal_serialisation import ChunkReservoir
from nanochat.causal_serialisation_extended import SequenceEdit, TokensOperations, resolve
from nanochat.common import get_dist_info
from nanochat.edit_dataloader import (
    EditCandidateStore,
    EditSamplingConfig,
    _PreparedDoc,
    _document_rows,
    _prepare_doc,
    _sample_random_inverse_edit,
    _window_candidate_sequence_edit,
    sequence_edit_mass,
)


def iterative_levenshtein_pair_loader_with_state(
    tokenizer,
    batch_size: int,
    sequence_len: int,
    *,
    split: str,
    cfg: EditSamplingConfig,
    token_ops: TokensOperations,
    tokenizer_batch_size: int = 128,
    resume_state_dict=None,
    reservoir_max_chunks: int = 8192,
    reservoir_chunk_size: int = 2048,
) -> Iterator[tuple[list[tuple[list[int], list[int]]], dict]]:
    """Yield resolved pairs from the same candidate/random edit machinery as CES.

    The downstream alignment rewrites replace, copy, and move effects into pure
    deletion plus insertion targets. Clean rows are sampled directly from the
    configured CES clean-row probability; no extra replay stream is introduced.
    """

    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    _distributed, rank, _local_rank, _world_size = get_dist_info()
    rng = np.random.default_rng(cfg.seed + rank)
    store = EditCandidateStore(cfg)
    rows = _document_rows(split, cfg, resume_state_dict, tokenizer_batch_size)
    documents: list[_PreparedDoc] = []
    pq_idx, rg_idx, epoch = 0, 0, 1
    bos_token = tokenizer.get_bos_token_id()
    max_content = sequence_len - 2
    reservoir = ChunkReservoir(
        max_chunks=reservoir_max_chunks,
        chunk_size=reservoir_chunk_size,
        rng=rng,
    )

    def refill():
        nonlocal pq_idx, rg_idx, epoch
        row_batch, (pq_idx, rg_idx, epoch) = next(rows)
        for shard_id, row_index, raw_text in row_batch:
            documents.append(
                _prepare_doc(
                    shard_id=shard_id,
                    row_index=row_index,
                    raw_text=raw_text,
                    tokenizer=tokenizer,
                    bos_token=bos_token,
                    store=store,
                    cfg=cfg,
                    rng=rng,
                )
            )

    while True:
        pairs: list[tuple[list[int], list[int]]] = []
        candidate_rows = 0
        random_rows = 0
        clean_rows = 0
        fallback_rows = 0
        raw_examples = 0
        raw_tokens = 0
        edit_mass = 0

        for _ in range(batch_size):
            while not documents:
                refill()
            document = documents.pop(0)
            raw_examples += 1
            raw_tokens += len(document.tokens)
            content_start = int(bool(document.tokens) and int(document.tokens[0]) == bos_token)
            clean_source = [int(token) for token in document.tokens[content_start:content_start + max_content]]
            if len(clean_source) >= reservoir_chunk_size:
                reservoir.offer(clean_source[:reservoir_chunk_size])

            row_kind = "clean"
            sequence_edit = None
            force_clean = rng.random() < cfg.stream_clean_row_prob
            if not force_clean and document.token_edit is not None:
                if len(document.token_edit.sequence) <= max_content:
                    sequence_edit = document.token_edit
                else:
                    sequence_edit = _window_candidate_sequence_edit(
                        document,
                        full_token_budget=max_content,
                        token_ops=token_ops,
                        bos_token=bos_token,
                    )
                if sequence_edit is not None:
                    row_kind = "candidate"
            elif not force_clean and rng.random() < cfg.random_corruption_prob_when_no_candidates:
                sequence_edit = _sample_random_inverse_edit(
                    np.asarray(clean_source, dtype=np.int64),
                    cfg=cfg,
                    token_ops=token_ops,
                    reservoir=reservoir,
                    rng=rng,
                )
                if sequence_edit is not None:
                    row_kind = "random"
            if sequence_edit is None:
                sequence_edit = SequenceEdit(
                    sequence=np.asarray(clean_source, dtype=np.int64), edits=[]
                )

            source = [int(token) for token in np.asarray(sequence_edit.sequence).tolist()]
            target = [int(token) for token in np.asarray(resolve(sequence_edit)).tolist()]
            if source and source[0] == bos_token:
                source = source[1:]
            if target and target[0] == bos_token:
                target = target[1:]
            if len(source) > max_content or len(target) > max_content:
                source = clean_source
                target = list(source)
                row_kind = "fallback"

            candidate_rows += int(row_kind == "candidate")
            random_rows += int(row_kind == "random")
            clean_rows += int(row_kind == "clean")
            fallback_rows += int(row_kind == "fallback")
            if row_kind in {"candidate", "random"}:
                edit_mass += sequence_edit_mass(sequence_edit)
            pairs.append((source, target))

        yield pairs, {
            "pq_idx": pq_idx,
            "rg_idx": rg_idx,
            "epoch": epoch,
            "candidate_rows": candidate_rows,
            "random_rows": random_rows,
            "clean_rows": clean_rows,
            "fallback_rows": fallback_rows,
            "raw_examples_consumption": raw_examples,
            "raw_token_consumption": raw_tokens,
            "edit_mass": edit_mass,
        }
