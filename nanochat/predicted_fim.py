"""Target-blind span localization followed by append-only FIM generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nanochat.engine import EditController, KVCache, sample_next_token
from nanochat.text_edit_baselines import (
    build_predicted_fim_infill_prompt,
    build_predicted_fim_locator,
)


@dataclass(frozen=True)
class PredictedFIMResult:
    locator_tokens: list[int]
    appended_prompt_tokens: list[int]
    generated_completion_tokens: list[int]
    source_prefix: list[int]
    source_suffix: list[int]
    predicted_start_pointer: int
    predicted_end_pointer: int
    predicted_start: int
    predicted_end: int
    start_logprob: float
    end_logprob: float
    status: str
    terminated: bool
    generation_budget: int

    @property
    def resolved_tokens(self) -> list[int]:
        return [
            *self.source_prefix,
            *self.generated_completion_tokens,
            *self.source_suffix,
        ]

    @property
    def locator_logprob(self) -> float:
        return 0.5 * (self.start_logprob + self.end_logprob)


def _cache_kwargs(model) -> tuple[dict, dict]:
    config = model.config
    head_dim = (
        model.get_kv_cache_head_dim()
        if hasattr(model, "get_kv_cache_head_dim")
        else config.n_embd // config.n_head
    )
    return (
        {
            "num_heads": config.n_kv_head,
            "head_dim": head_dim,
            "num_layers": config.n_layer,
        },
        {"num_heads": 1, "head_dim": config.n_embd, "num_layers": 1},
    )


def _empty_result(
    *,
    locator: list[int],
    appended: list[int],
    prefix: list[int],
    suffix: list[int],
    start_pointer: int,
    end_pointer: int,
    start_logprob: float,
    end_logprob: float,
    status: str,
) -> PredictedFIMResult:
    return PredictedFIMResult(
        locator_tokens=locator,
        appended_prompt_tokens=appended,
        generated_completion_tokens=[],
        source_prefix=prefix,
        source_suffix=suffix,
        predicted_start_pointer=start_pointer,
        predicted_end_pointer=end_pointer,
        predicted_start=start_pointer + 1,
        predicted_end=end_pointer + 1,
        start_logprob=start_logprob,
        end_logprob=end_logprob,
        status=status,
        terminated=False,
        generation_budget=0,
    )


@torch.inference_mode()
def generate_predicted_fim(
    model,
    tokenizer,
    source_tokens: list[int],
    *,
    max_new_tokens: int = 640,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int = 12345,
) -> PredictedFIMResult:
    """Locate one source envelope, append its FIM prompt, and generate its middle.

    The original source is prefetched exactly once. The predicted prefix/suffix
    request is appended to those cached activations; no token is retroactively
    inserted into the source.
    """

    if model.position_head is None:
        raise ValueError("Predicted-span FIM requires a model position head")
    if model.config.use_dual_rope:
        raise ValueError("Predicted-span FIM is defined here for the single-RoPE locator")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    source = [int(token) for token in source_tokens]
    locator, query_indices = build_predicted_fim_locator(source, tokenizer)
    context = int(model.config.sequence_len)
    if len(locator) >= context:
        last = max(len(source) - 1, 0)
        return _empty_result(
            locator=locator,
            appended=[],
            prefix=source,
            suffix=[],
            start_pointer=last,
            end_pointer=last,
            start_logprob=float("-inf"),
            end_logprob=float("-inf"),
            status="locator_too_long",
        )

    device = model.get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_cache_kwargs, pos_cache_kwargs = _cache_kwargs(model)
    kv_cache = KVCache(
        batch_size=1,
        seq_len=context,
        device=device,
        dtype=dtype,
        **model_cache_kwargs,
    )
    pos_cache = KVCache(
        batch_size=1,
        seq_len=context,
        device=device,
        dtype=dtype,
        **pos_cache_kwargs,
    )
    ids = torch.tensor([locator], dtype=torch.long, device=device)
    tix = torch.arange(len(locator), dtype=torch.long, device=device)[None, :]
    queries = torch.tensor([query_indices], dtype=torch.long, device=device)
    _token_logits, position_logits = model.forward(
        ids,
        tix=tix,
        kv_cache=kv_cache,
        pos_k_cache=pos_cache,
        position_queries=queries,
        return_pos_logits=True,
    )
    if position_logits is None or position_logits.shape[1] != 2:
        raise RuntimeError("Locator must return start and end pointer distributions")

    source_support = position_logits[0, 0, :len(source)]
    start_logprobs = F.log_softmax(source_support.float(), dim=-1)
    start_pointer = int(source_support.argmax().item())
    end_support = position_logits[0, 1, start_pointer:len(source)]
    end_logprobs = F.log_softmax(end_support.float(), dim=-1)
    end_offset = int(end_support.argmax().item())
    end_pointer = start_pointer + end_offset
    start_logprob = float(start_logprobs[start_pointer].item())
    end_logprob = float(end_logprobs[end_offset].item())
    start, end = start_pointer + 1, end_pointer + 1
    appended, prefix, suffix = build_predicted_fim_infill_prompt(
        source, start, end, tokenizer
    )
    prompt_length = len(locator) + len(appended)
    if prompt_length >= context:
        return _empty_result(
            locator=locator,
            appended=appended,
            prefix=prefix,
            suffix=suffix,
            start_pointer=start_pointer,
            end_pointer=end_pointer,
            start_logprob=start_logprob,
            end_logprob=end_logprob,
            status="infill_prompt_too_long",
        )

    append_ids = torch.tensor([appended], dtype=torch.long, device=device)
    append_tix = torch.arange(
        len(locator), prompt_length, dtype=torch.long, device=device
    )[None, :]
    logits, _position_logits = model.forward(
        append_ids,
        tix=append_tix,
        kv_cache=kv_cache,
        pos_k_cache=pos_cache,
        return_pos_logits=True,
    )
    logits = logits[:, -1, :]
    budget = min(max_new_tokens, context - prompt_length)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    bos = tokenizer.get_bos_token_id()
    edit_controller = EditController(tokenizer)
    banned = (
        set(edit_controller.indel_specials)
        | set(edit_controller.pointer_tokens)
        | set(edit_controller.structural_ptr)
        | {edit_controller.tok["INDEL_BOS"]}
    )
    completion: list[int] = []
    terminated = False
    for _ in range(budget):
        logits[:, list(banned)] = float("-inf")
        token = int(sample_next_token(logits, rng, temperature, top_k, top_p)[0, 0].item())
        if token in {assistant_end, bos}:
            terminated = True
            break
        completion.append(token)
        token_ids = torch.tensor([[token]], dtype=torch.long, device=device)
        token_tix = torch.tensor([[kv_cache.get_pos()]], dtype=torch.long, device=device)
        logits, _position_logits = model.forward(
            token_ids,
            tix=token_tix,
            kv_cache=kv_cache,
            pos_k_cache=pos_cache,
            return_pos_logits=True,
        )
        logits = logits[:, -1, :]
    status = "valid" if terminated else "max_tokens_without_terminator"
    return PredictedFIMResult(
        locator_tokens=locator,
        appended_prompt_tokens=appended,
        generated_completion_tokens=completion,
        source_prefix=prefix,
        source_suffix=suffix,
        predicted_start_pointer=start_pointer,
        predicted_end_pointer=end_pointer,
        predicted_start=start,
        predicted_end=end,
        start_logprob=start_logprob,
        end_logprob=end_logprob,
        status=status,
        terminated=terminated,
        generation_budget=budget,
    )
