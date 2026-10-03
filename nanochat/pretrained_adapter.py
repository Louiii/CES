"""Explicit, auditable import of a standard NanoChat checkpoint into CES."""

from __future__ import annotations

import math
from typing import Any

import torch


VOCAB_PARAMETER_NAMES = {
    "transformer.wte.weight",
    "lm_head.weight",
}


def _token_identity_map(tokenizer) -> dict[tuple[str, bytes | str], int]:
    """Return exact token identities without lossy text decoding."""

    enc = getattr(tokenizer, "enc", None)
    if enc is not None:
        out: dict[tuple[str, bytes | str], int] = {
            ("bytes", bytes(piece)): int(token_id)
            for piece, token_id in enc._mergeable_ranks.items()
        }
        out.update(
            {("special", str(piece)): int(token_id) for piece, token_id in enc._special_tokens.items()}
        )
        return out

    return {
        ("token", str(tokenizer.id_to_token(token_id))): token_id
        for token_id in range(tokenizer.get_vocab_size())
    }


def token_label(identity: tuple[str, bytes | str]) -> str:
    kind, value = identity
    if isinstance(value, bytes):
        return f"{kind}:{value.hex()}"
    return f"{kind}:{value}"


def _is_vocab_parameter(name: str) -> bool:
    return name in VOCAB_PARAMETER_NAMES or (
        name.startswith("value_embeds.") and name.endswith(".weight")
    )


def _is_allowed_new_parameter(name: str) -> bool:
    return (
        name.startswith("position_head.")
        or ".attn.c_q_pointer." in name
        or ".attn.c_k_pointer." in name
        or name.endswith(".attn.pointer_attn_gate")
    )


@torch.no_grad()
def _initialize_new_vocab_rows(
    model,
    new_token_ids: list[int],
    *,
    seed: int,
) -> None:
    """Initialize activated vocabulary rows with the model's normal rules."""

    if not new_token_ids:
        return
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    n_embd = model.config.n_embd
    uniform_bound = math.sqrt(3.0) * n_embd**-0.5

    def normal_rows(weight: torch.Tensor, std: float) -> None:
        values = torch.randn(
            (len(new_token_ids), weight.size(1)), generator=generator,
            dtype=torch.float32, device="cpu",
        ) * std
        weight.index_copy_(0, torch.tensor(new_token_ids, device=weight.device), values.to(device=weight.device, dtype=weight.dtype))

    def uniform_rows(weight: torch.Tensor) -> None:
        values = torch.empty(
            (len(new_token_ids), weight.size(1)), dtype=torch.float32, device="cpu"
        )
        values.uniform_(-uniform_bound, uniform_bound, generator=generator)
        weight.index_copy_(0, torch.tensor(new_token_ids, device=weight.device), values.to(device=weight.device, dtype=weight.dtype))

    normal_rows(model.transformer.wte.weight, 0.8)
    normal_rows(model.lm_head.weight, 0.001)
    for value_embed in model.value_embeds.values():
        uniform_rows(value_embed.weight)


@torch.no_grad()
def adapt_pretrained_state(
    *,
    target_model,
    target_tokenizer,
    source_state: dict[str, torch.Tensor],
    source_tokenizer,
    source_metadata: dict[str, Any],
    seed: int = 20260726,
) -> dict[str, Any]:
    """Copy a standard checkpoint into an initialized CES target model.

    Ordinary tensor mismatches fail loudly. Vocabulary tensors are remapped by
    exact byte/special-token identity. Only the position head and appended
    pointer-attention parameters may be absent from the source checkpoint.
    """

    source_state = {key.removeprefix("_orig_mod."): value for key, value in source_state.items()}
    if target_model.position_head is None or not all(
        block.attn.append_pointer_rope for block in target_model.transformer.h
    ):
        raise ValueError("Pretrained CES adaptation requires an appended-RoPE target model")
    unexpected_source_modules = sorted(
        name for name in source_state
        if name.startswith("position_head.") or _is_allowed_new_parameter(name)
    )
    if unexpected_source_modules:
        raise ValueError("Source checkpoint is not a standard model; found: " + ", ".join(unexpected_source_modules))

    target_state = target_model.state_dict()
    source_vocab = _token_identity_map(source_tokenizer)
    target_vocab = _token_identity_map(target_tokenizer)
    shared = sorted(set(source_vocab) & set(target_vocab), key=lambda item: target_vocab[item])
    new_identities = sorted(set(target_vocab) - set(source_vocab), key=lambda item: target_vocab[item])
    new_token_ids = [target_vocab[item] for item in new_identities]

    _initialize_new_vocab_rows(target_model, new_token_ids, seed=seed)
    target_state = target_model.state_dict()

    copied_exact: list[str] = []
    remapped_vocab: dict[str, int] = {}
    initialized_new: list[str] = []
    rejected: list[str] = []

    for name, target_tensor in target_state.items():
        if _is_vocab_parameter(name):
            source_tensor = source_state.get(name)
            if source_tensor is None:
                rejected.append(f"missing vocabulary parameter: {name}")
                continue
            if source_tensor.ndim != 2 or target_tensor.ndim != 2:
                rejected.append(f"vocabulary parameter is not a matrix: {name}")
                continue
            if source_tensor.size(1) != target_tensor.size(1):
                rejected.append(
                    f"vocabulary width mismatch for {name}: "
                    f"{tuple(source_tensor.shape)} != {tuple(target_tensor.shape)}"
                )
                continue
            for identity in shared:
                source_id = source_vocab[identity]
                target_id = target_vocab[identity]
                if source_id >= source_tensor.size(0) or target_id >= target_tensor.size(0):
                    rejected.append(f"token row out of range for {name}: {token_label(identity)}")
                    continue
                target_tensor[target_id].copy_(
                    source_tensor[source_id].to(device=target_tensor.device, dtype=target_tensor.dtype)
                )
            remapped_vocab[name] = len(shared)
            continue

        source_tensor = source_state.get(name)
        if source_tensor is None:
            if _is_allowed_new_parameter(name):
                initialized_new.append(name)
                continue
            rejected.append(f"missing non-vocabulary parameter: {name}")
            continue
        if source_tensor.shape != target_tensor.shape:
            rejected.append(
                f"shape mismatch for {name}: {tuple(source_tensor.shape)} != {tuple(target_tensor.shape)}"
            )
            continue
        target_tensor.copy_(source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype))
        copied_exact.append(name)

    target_only = set(target_state) - set(source_state)
    source_only = set(source_state) - set(target_state)
    unexpected_target_only = sorted(name for name in target_only if not _is_allowed_new_parameter(name))
    if unexpected_target_only:
        rejected.extend(f"unexpected target-only parameter: {name}" for name in unexpected_target_only)
    if source_only:
        rejected.extend(f"unexpected source-only parameter: {name}" for name in sorted(source_only))

    # Copy the learned sequence projections into the new address projections;
    # their zero scalar gate guarantees no initial change to attention logits.
    for layer_index, block in enumerate(target_model.transformer.h):
        if block.attn.append_pointer_rope:
            block.attn.c_q_pointer.weight.copy_(block.attn.c_q.weight)
            block.attn.c_k_pointer.weight.copy_(block.attn.c_k.weight)
            block.attn.pointer_attn_gate.zero_()

    if rejected:
        raise ValueError("Pretrained adaptation rejected incompatible state:\n- " + "\n- ".join(rejected))

    return {
        "kind": "pretrained_to_ces",
        "source_step": source_metadata.get("step"),
        "source_model_config": source_metadata.get("model_config"),
        "source_tokenizer": source_metadata.get("tokenizer"),
        "target_vocab_size": getattr(target_model.config, "vocab_size", None),
        "seed": seed,
        "shared_token_count": len(shared),
        "new_token_count": len(new_identities),
        "new_tokens": [token_label(item) for item in new_identities],
        "copied_exact_keys": sorted(copied_exact),
        "remapped_vocab_keys": remapped_vocab,
        "initialized_new_keys": sorted(initialized_new),
        "rejected_keys": [],
        "optimizer_reset": True,
        "dataloader_reset": True,
    }


def new_token_ids(source_tokenizer, target_tokenizer) -> list[int]:
    source_vocab = _token_identity_map(source_tokenizer)
    target_vocab = _token_identity_map(target_tokenizer)
    return sorted(target_vocab[item] for item in set(target_vocab) - set(source_vocab))
