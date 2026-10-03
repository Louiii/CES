"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

from typing import Callable

import numpy as np
import pyarrow.parquet as pq
import torch

from nanochat.causal_serialisation import (
    ChunkReservoir,
    TrainingData,
    TrainingTarget,
    get_training_data_from_stream,
)
from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files


def training_data_identity(x, y):
    use_cuda = x.device == "cuda"
    pos = torch.arange(x.shape[-1], dtype=torch.long, pin_memory=use_cuda)
    pos = pos.to(device=x.device, non_blocking=use_cuda)
    return TrainingData(x, pos, TrainingTarget(y, None, None, None, None, None, None))


def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state(
    tokenizer, B, T, split, indel_config, token_ops,
    perplexity_masks: Callable | None = None,
    tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None,
    pos_buffer_size: int = 64,
    reservoir_max_chunks: int = 8192,
    reservoir_chunk_size: int = 2048,
    reservoir_seed: int = 1234,
):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This is the original dataloader that streams tokens into a flat buffer and reshapes.
    Rows may start mid-document (no guaranteed BOS at position 0).

    Supports approximate resume via state_dict.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    needed_tokens = B * T + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()
    token_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    # Per-rank RNG for reservoir behavior/sampling
    rng = np.random.default_rng(reservoir_seed + ddp_rank)
    reservoir = ChunkReservoir(
        max_chunks=reservoir_max_chunks,
        chunk_size=reservoir_chunk_size,
        rng=rng,
    )

    use_cuda = device == "cuda"
    def ten(x):
        return torch.tensor(
            np.stack(x), dtype=torch.long, pin_memory=use_cuda
        ).to(device=device, non_blocking=use_cuda)

    while True:

        # Accumulate enough tokens
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)

        batch = []
        for _ in range(B):
            advance, training_data = get_training_data_from_stream(
                cfg=indel_config,
                token_buffer=token_buffer,  # Ensure this is at least B*T+1
                reservoir=reservoir,  # Adapt to deal with an empty buffer.
                token_ops=token_ops,
                token_budget=T,
                pos_buffer_size=pos_buffer_size,
                perplexity_masks=perplexity_masks,
                rng=rng,
            )
            token_buffer = token_buffer[advance:]
            batch.append(training_data)

        # # Package tokens into inputs and targets, yield
        # sh = [x[0].shape[0] for x in batch]
        # if not all(s == sh[0] for s in sh[1:]):
        #     from nanochat.causal_serialisation import print_training_examples
        #     print_training_examples([batch[34]], token_ops.tokeniser)
        #     breakpoint()
        training_batch = TrainingData(
            input_positions=ten([x.input_positions for x in batch]),
            input_tokens=ten([x.input_tokens for x in batch]),
            target=TrainingTarget(
                *[ten([x.target[i] for x in batch]) for i in range(len(TrainingTarget._fields))]
            ),
        )
        yield training_batch, {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}


def tokenizing_distributed_data_loader(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets


def tokenizing_distributed_indel_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    indel_config, token_ops,
    perplexity_masks: Callable | None = None,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000,
    pos_buffer_size: int = 64,
    reservoir_max_chunks: int = 8192,
    reservoir_chunk_size: int = 2048,
    reservoir_seed: int = 1234,
):
    """
    BOS-aligned loader using nanochat's bestfit packing,
    but producing your indel TrainingData per row.

    Fairness goal:
    - Same BOS-bestfit packing as nanochat baseline (row starts with BOS, row length T+1).
    - No cross-row streaming, no overlap, no offset tricks.
    - One training example per row.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    rng = np.random.default_rng(reservoir_seed + ddp_rank)
    reservoir = ChunkReservoir(
        max_chunks=reservoir_max_chunks,
        chunk_size=reservoir_chunk_size,
        rng=rng,
    )

    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1
    use_cuda = device == "cuda"

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(
            doc_batch, prepend=bos_token, num_threads=tokenizer_threads
        )
        for tokens in token_lists:
            doc_buffer.append(tokens)

    def pack_one_row_bestfit():
        """
        Replicates nanochat bos_bestfit row packing:
        - Fill row to exactly row_capacity with full docs when possible (largest that fits)
        - Otherwise crop shortest doc to fill remaining exactly
        """
        row = []
        while len(row) < row_capacity:
            while len(doc_buffer) < buffer_size:
                refill_buffer()

            remaining = row_capacity - len(row)

            # largest doc that fits
            best_idx = -1
            best_len = 0
            for i, doc in enumerate(doc_buffer):
                L = len(doc)
                if L <= remaining and L > best_len:
                    best_idx = i
                    best_len = L

            if best_idx >= 0:
                doc = doc_buffer.pop(best_idx)
                row.extend(doc)
            else:
                # crop shortest doc to fill remaining
                shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                doc = doc_buffer.pop(shortest_idx)
                row.extend(doc[:remaining])

        # ensure exact length
        return row[:row_capacity]

    def ten(x_list):
        # x_list is list of np arrays (or lists) of equal length
        x = np.stack(x_list)

        # if np.issubdtype(x.dtype, np.integer) or x.dtype == bool:
        if np.issubdtype(x.dtype, np.integer):
            dtype = torch.long
        elif np.issubdtype(x.dtype, np.floating):
            dtype = torch.float
        elif x.dtype == bool:
            dtype = bool
        else:
            raise TypeError(f"Unsupported dtype: {x.dtype}")
        return torch.tensor(x, dtype=dtype, pin_memory=use_cuda).to(device=device, non_blocking=use_cuda)

    while True:
        raw_token_consumption = 0
        batch = []
        for _ in range(B):
            row = pack_one_row_bestfit()

            # IMPORTANT:
            # Feed your corruption code exactly T "budget" tokens of source context.
            # The row has length T+1. You can pass the full row; your function slices.
            # We IGNORE 'advance' because we are not doing streaming across rows.
            advance, training_data = get_training_data_from_stream(
                cfg=indel_config,
                token_buffer=row,          # BOS-aligned packed row
                reservoir=reservoir,
                token_ops=token_ops,
                token_budget=T,            # match baseline crop length
                pos_buffer_size=pos_buffer_size,
                perplexity_masks=perplexity_masks,
                rng=rng,
            )
            batch.append(training_data)
            raw_token_consumption += advance

        # Have all this annoying code because there is no easy way to tree map.
        kwgs = {
            f: ten([getattr(x.target, f) for x in batch])
            for f in [
                "tokens",
                "gathered_pos_target",
                "token_loss_weights",
                "gather_indices",
                "gather_valid",
                "overflow",
            ]
        }
        b0pm = batch[0].target.perplexity_masks
        if b0pm is not None:
            kwgs["perplexity_masks"] = {
                k: ten([x.target.perplexity_masks[k] for x in batch]) for k in b0pm
            }

        training_batch = TrainingData(
            input_positions=ten([x.input_positions for x in batch]),
            input_tokens=ten([x.input_tokens for x in batch]),
            target=TrainingTarget(**kwgs),
        )
        yield training_batch, {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch, "raw_token_consumption": raw_token_consumption}


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
