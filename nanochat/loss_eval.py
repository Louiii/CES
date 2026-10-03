"""
A number of functions that help with evaluating a base model.
"""
import math

import torch
import torch.distributed as dist

from nanochat.dataloader import training_data_identity


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes, edit_model: bool = True):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(*(training_data_identity(x, y) if edit_model else (x, y)), loss_reduction='none') # (B, T)
        loss2d = loss2d.view(-1) # flatten
        y = y.view(-1) # flatten
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb

@torch.no_grad()
def evaluate_ces_joint_bpb(model, batches, steps, token_bytes):
    """Evaluate token and causal-pointer surprise on deterministic CES batches."""
    device = model.get_device()
    totals = {
        "token_nats": torch.tensor(0.0, device=device),
        "position_nats": torch.tensor(0.0, device=device),
        "bytes": torch.tensor(0, dtype=torch.int64, device=device),
        "pointers": torch.tensor(0, dtype=torch.int64, device=device),
        "position_correct": torch.tensor(0, dtype=torch.int64, device=device),
        "log_support": torch.tensor(0.0, device=device),
        "edit_mass": torch.tensor(0, dtype=torch.int64, device=device),
        "raw_tokens": torch.tensor(0, dtype=torch.int64, device=device),
    }
    batch_iter = iter(batches)
    for _ in range(steps):
        batch, metadata = next(batch_iter)
        _loss, aux = model(*batch, loss_reduction="sum")
        target = batch.target
        y = target.tokens.reshape(-1)
        weights = target.token_loss_weights.reshape(-1).bool()
        valid_y = y >= 0
        y_safe = torch.where(valid_y, y, torch.zeros_like(y))
        byte_counts = token_bytes[y_safe]
        totals["bytes"] += torch.where(
            valid_y & weights, byte_counts, torch.zeros_like(byte_counts)
        ).sum()
        totals["token_nats"] += aux["token_loss"]
        totals["position_nats"] += aux["position_loss"]
        valid_pointer = target.gather_valid.bool() & (target.gathered_pos_target != -1)
        support = (target.gather_indices + 1).clamp_min(1)
        totals["pointers"] += valid_pointer.sum()
        totals["log_support"] += torch.where(
            valid_pointer, support.float().log(), torch.zeros_like(support, dtype=torch.float32)
        ).sum()
        correct, _count = aux["position_correct"]
        totals["position_correct"] += correct
        totals["edit_mass"] += int(metadata.get("edit_mass", 0))
        totals["raw_tokens"] += int(metadata.get("raw_token_consumption", 0))
    if dist.is_initialized():
        for value in totals.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    values = {key: value.item() for key, value in totals.items()}
    byte_denom = math.log(2) * values["bytes"]
    pointer_denom = max(values["pointers"], 1)
    return {
        "token_bpb": values["token_nats"] / byte_denom if byte_denom else float("inf"),
        "position_bits_per_byte": values["position_nats"] / byte_denom if byte_denom else float("inf"),
        "joint_bpb": (values["token_nats"] + values["position_nats"]) / byte_denom if byte_denom else float("inf"),
        "position_bits_per_pointer": values["position_nats"] / (math.log(2) * pointer_denom),
        "position_accuracy": values["position_correct"] / pointer_denom,
        "position_normalized_nll": values["position_nats"] / values["log_support"] if values["log_support"] else float("nan"),
        "pointers": int(values["pointers"]),
        "bytes": int(values["bytes"]),
        "edit_mass_rho": values["edit_mass"] / max(values["raw_tokens"], 1),
    }
