"""Independent iterative Levenshtein-style full-attention baseline."""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, get_dist_info, print0
from nanochat.gpt import Linear, apply_rotary_emb, compute_cos_sin, norm
from nanochat.optim import DistMuonAdamW, MuonAdamW


@dataclass
class LevenshteinConfig:
    sequence_len: int = 1024
    vocab_size: int = 32752
    pad_id: int = 32748
    mask_id: int = 32749
    bos_id: int = 32750
    eos_id: int = 32751
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    rope_base: float = 10000.0
    max_insertions: int = 32
    max_rounds: int = 4


class LevenshteinBatch(NamedTuple):
    current_ids: torch.Tensor
    current_padding_mask: torch.Tensor
    delete_labels: torch.Tensor
    kept_ids: torch.Tensor
    kept_padding_mask: torch.Tensor
    gap_labels: torch.Tensor
    masked_ids: torch.Tensor
    masked_padding_mask: torch.Tensor
    token_labels: torch.Tensor


@dataclass
class OracleRound:
    current: list[int]
    delete_labels: list[int]
    kept: list[int]
    gap_labels: list[int]
    masked: list[int]
    token_labels: list[int]
    completed: list[int]


class BidirectionalSelfAttention(nn.Module):
    def __init__(self, config: LevenshteinConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        if self.head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos_sin, padding_mask):
        batch_size, sequence_len, _ = x.shape
        q = self.c_q(x).view(batch_size, sequence_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch_size, sequence_len, self.n_head, self.head_dim)
        v = self.c_v(x).view(batch_size, sequence_len, self.n_head, self.head_dim)
        cos, sin = cos_sin
        q = norm(apply_rotary_emb(q, cos, sin)) * 1.2
        k = norm(apply_rotary_emb(k, cos, sin)) * 1.2
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        key_mask = padding_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=key_mask,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_len, self.n_embd)
        return self.c_proj(y)


class LevenshteinMLP(nn.Module):
    def __init__(self, config: LevenshteinConfig):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class LevenshteinBlock(nn.Module):
    def __init__(self, config: LevenshteinConfig):
        super().__init__()
        self.attn = BidirectionalSelfAttention(config)
        self.mlp = LevenshteinMLP(config)

    def forward(self, x, cos_sin, padding_mask):
        x = x + self.attn(norm(x), cos_sin, padding_mask)
        x = x + self.mlp(norm(x))
        return x


class LevenshteinTransformer(nn.Module):
    def __init__(self, config: LevenshteinConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        padded_vocab = math.ceil(config.vocab_size / pad_vocab_size_to) * pad_vocab_size_to
        self.token_embedding = nn.Embedding(padded_vocab, config.n_embd)
        self.blocks = nn.ModuleList(
            [LevenshteinBlock(config) for _ in range(config.n_layer)]
        )
        self.delete_head = Linear(config.n_embd, 2, bias=True)
        # The distributed AdamW path reduce-scatters along dimension zero, so
        # every AdamW parameter must have a leading dimension divisible by the
        # two-rank baseline world size. Counts 0..32 require 33 logical classes;
        # keep one optimizer-only padding row and slice it from public logits.
        gap_classes = config.max_insertions + 1
        padded_gap_classes = math.ceil(gap_classes / 2) * 2
        self.gap_head = Linear(2 * config.n_embd, padded_gap_classes, bias=True)
        self.token_head = Linear(config.n_embd, padded_vocab, bias=False)
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

    @torch.no_grad()
    def init_weights(self):
        config = self.config
        scale = (config.n_embd ** -0.5) * 3**0.5
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.8)
        nn.init.normal_(self.token_head.weight, mean=0.0, std=0.001)
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -scale, scale)
            nn.init.uniform_(block.attn.c_k.weight, -scale, scale)
            nn.init.uniform_(block.attn.c_v.weight, -scale, scale)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * scale, 0.4 * scale)
            nn.init.zeros_(block.mlp.c_proj.weight)
        nn.init.normal_(self.delete_head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.delete_head.bias)
        nn.init.normal_(self.gap_head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.gap_head.bias)
        positions = torch.arange(config.sequence_len, device=self.token_embedding.weight.device)
        cos, sin = compute_cos_sin(
            positions,
            config.n_embd // config.n_head,
            config.rope_base,
            positions.device,
            dtype=COMPUTE_DTYPE,
        )
        self.cos = cos[None, :, None, :]
        self.sin = sin[None, :, None, :]

    def encode(self, token_ids: torch.Tensor, padding_mask: torch.Tensor):
        sequence_len = token_ids.size(1)
        if sequence_len > self.config.sequence_len:
            raise ValueError(
                f"Sequence length {sequence_len} exceeds {self.config.sequence_len}"
            )
        x = norm(self.token_embedding(token_ids).to(COMPUTE_DTYPE))
        cos_sin = self.cos[:, :sequence_len], self.sin[:, :sequence_len]
        for block in self.blocks:
            x = block(x, cos_sin, padding_mask)
        x = norm(x)
        return x.masked_fill(~padding_mask.unsqueeze(-1), 0.0)

    def delete_logits(self, hidden):
        return F.linear(
            hidden,
            self.delete_head.weight.to(dtype=hidden.dtype),
            self.delete_head.bias.to(dtype=hidden.dtype),
        ).float()

    def gap_logits(self, hidden):
        gap_classes = self.config.max_insertions + 1
        if hidden.size(1) < 2:
            return hidden.new_zeros(hidden.size(0), 0, gap_classes)
        features = torch.cat([hidden[:, :-1], hidden[:, 1:]], dim=-1)
        padded_logits = F.linear(
            features,
            self.gap_head.weight.to(dtype=features.dtype),
            self.gap_head.bias.to(dtype=features.dtype),
        )
        return padded_logits[..., :gap_classes].float()

    def token_logits(self, hidden):
        return self.token_head(hidden)[..., : self.config.vocab_size].float()

    @staticmethod
    def _masked_cross_entropy(logits, labels):
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-1,
            reduction="sum",
        )
        return losses / (labels != -1).sum().clamp_min(1)

    def compute_loss(self, batch: LevenshteinBatch):
        current_hidden = self.encode(batch.current_ids, batch.current_padding_mask)
        delete_logits = self.delete_logits(current_hidden)
        delete_loss = self._masked_cross_entropy(delete_logits, batch.delete_labels)

        kept_hidden = self.encode(batch.kept_ids, batch.kept_padding_mask)
        gap_logits = self.gap_logits(kept_hidden)
        gap_loss = self._masked_cross_entropy(gap_logits, batch.gap_labels)

        masked_hidden = self.encode(batch.masked_ids, batch.masked_padding_mask)
        token_logits = self.token_logits(masked_hidden)
        token_loss = self._masked_cross_entropy(token_logits, batch.token_labels)
        loss = delete_loss + gap_loss + token_loss

        with torch.no_grad():
            delete_valid = batch.delete_labels != -1
            gap_valid = batch.gap_labels != -1
            token_valid = batch.token_labels != -1
            metrics = {
                "loss": loss.detach(),
                "delete_loss": delete_loss.detach(),
                "gap_loss": gap_loss.detach(),
                "token_loss": token_loss.detach(),
                "delete_correct": ((delete_logits.argmax(-1) == batch.delete_labels) & delete_valid).sum(),
                "delete_count": delete_valid.sum(),
                "gap_correct": ((gap_logits.argmax(-1) == batch.gap_labels) & gap_valid).sum(),
                "gap_count": gap_valid.sum(),
                "token_correct": ((token_logits.argmax(-1) == batch.token_labels) & token_valid).sum(),
                "token_count": token_valid.sum(),
                "encoder_passes": torch.tensor(3, device=loss.device),
            }
        return loss, metrics

    def forward(self, batch: LevenshteinBatch):
        return self.compute_loss(batch)

    def num_scaling_params(self):
        transformer = sum(parameter.numel() for parameter in self.blocks.parameters())
        token_head = self.token_head.weight.numel()
        return {
            "embedding": self.token_embedding.weight.numel(),
            "transformer_matrices": transformer,
            "token_head": token_head,
            "delete_head": sum(p.numel() for p in self.delete_head.parameters()),
            "gap_head": sum(p.numel() for p in self.gap_head.parameters()),
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "scaling": transformer + token_head,
        }

    def setup_optimizer(
        self,
        *,
        embedding_lr=0.3,
        head_lr=0.008,
        matrix_lr=0.02,
        weight_decay=0.28,
    ):
        dmodel_scale = (self.config.n_embd / 768) ** -0.5
        head_params = [
            *self.token_head.parameters(),
            *self.delete_head.parameters(),
            *self.gap_head.parameters(),
        ]
        matrix_params = list(self.blocks.parameters())
        groups = [
            dict(
                kind="adamw",
                params=list(self.token_embedding.parameters()),
                lr=embedding_lr * dmodel_scale,
                betas=(0.8, 0.995),
                eps=1e-10,
                weight_decay=0.001,
            ),
            dict(
                kind="adamw",
                params=head_params,
                lr=head_lr * dmodel_scale,
                betas=(0.8, 0.96),
                eps=1e-10,
                weight_decay=0.01,
            ),
        ]
        for shape in sorted({parameter.shape for parameter in matrix_params}):
            groups.append(
                dict(
                    kind="muon",
                    params=[p for p in matrix_params if p.shape == shape],
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.9,
                    weight_decay=weight_decay,
                )
            )
        distributed, *_ = get_dist_info()
        optimizer_cls = DistMuonAdamW if distributed else MuonAdamW
        optimizer = optimizer_cls(groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        print0(f"Levenshtein optimizer groups: {len(optimizer.param_groups)}")
        return optimizer

    @torch.no_grad()
    def iterative_decode(self, source_tokens: Sequence[int], max_length: int | None = None):
        self.eval()
        device = self.token_embedding.weight.device
        max_length = max_length or self.config.sequence_len
        tokens = [self.config.bos_id, *map(int, source_tokens), self.config.eos_id]
        tokens = tokens[: max_length - 1] + [self.config.eos_id] if len(tokens) > max_length else tokens
        stats = {
            "rounds": 0,
            "encoder_passes": 0,
            "deleted": 0,
            "inserted": 0,
            "decision_logprob_sum": 0.0,
            "decision_count": 0,
        }

        for round_index in range(self.config.max_rounds):
            ids = torch.tensor(tokens, dtype=torch.long, device=device)[None, :]
            mask = torch.ones_like(ids, dtype=torch.bool)
            hidden = self.encode(ids, mask)
            stats["encoder_passes"] += 1
            delete_logits = self.delete_logits(hidden)
            delete_choice = delete_logits.argmax(-1)
            delete_logprob = delete_logits.log_softmax(-1).gather(
                -1, delete_choice.unsqueeze(-1)
            ).squeeze(-1)[0, 1:-1]
            stats["decision_logprob_sum"] += float(delete_logprob.sum().item())
            stats["decision_count"] += int(delete_logprob.numel())
            delete = delete_choice[0].bool().tolist()
            delete[0] = False
            delete[-1] = False
            kept = [token for token, remove in zip(tokens, delete) if not remove]
            deleted = len(tokens) - len(kept)

            kept_ids = torch.tensor(kept, dtype=torch.long, device=device)[None, :]
            kept_mask = torch.ones_like(kept_ids, dtype=torch.bool)
            kept_hidden = self.encode(kept_ids, kept_mask)
            stats["encoder_passes"] += 1
            gap_logits = self.gap_logits(kept_hidden)
            gap_choice = gap_logits.argmax(-1)
            gap_logprob = gap_logits.log_softmax(-1).gather(
                -1, gap_choice.unsqueeze(-1)
            ).squeeze(-1)[0]
            stats["decision_logprob_sum"] += float(gap_logprob.sum().item())
            stats["decision_count"] += int(gap_logprob.numel())
            counts = gap_choice[0].tolist()

            masked: list[int] = []
            inserted = 0
            for gap_index, token in enumerate(kept[:-1]):
                masked.append(token)
                available = max_length - len(masked) - (len(kept) - gap_index - 1)
                count = min(int(counts[gap_index]), self.config.max_insertions, max(available, 0))
                masked.extend([self.config.mask_id] * count)
                inserted += count
            masked.append(kept[-1])

            if inserted:
                masked_ids = torch.tensor(masked, dtype=torch.long, device=device)[None, :]
                masked_mask = torch.ones_like(masked_ids, dtype=torch.bool)
                masked_hidden = self.encode(masked_ids, masked_mask)
                stats["encoder_passes"] += 1
                token_logits = self.token_logits(masked_hidden)
                token_choice = token_logits.argmax(-1)
                token_logprob = token_logits.log_softmax(-1).gather(
                    -1, token_choice.unsqueeze(-1)
                ).squeeze(-1)[0]
                inserted_mask = masked_ids[0].eq(self.config.mask_id)
                stats["decision_logprob_sum"] += float(
                    token_logprob[inserted_mask].sum().item()
                )
                stats["decision_count"] += int(inserted_mask.sum().item())
                predictions = token_choice[0].tolist()
                tokens = [
                    predictions[index] if token == self.config.mask_id else token
                    for index, token in enumerate(masked)
                ]
            else:
                tokens = kept

            stats["rounds"] = round_index + 1
            stats["deleted"] += deleted
            stats["inserted"] += inserted
            if deleted == 0 and inserted == 0:
                break

        stats["mean_decision_logprob"] = (
            stats["decision_logprob_sum"] / stats["decision_count"]
            if stats["decision_count"]
            else None
        )
        return tokens[1:-1], stats


def _alignment(source: Sequence[int], target: Sequence[int]):
    return difflib.SequenceMatcher(a=list(source), b=list(target), autojunk=False).get_opcodes()


def build_oracle_round(
    current: Sequence[int],
    target: Sequence[int],
    *,
    mask_id: int,
    max_insertions: int,
) -> OracleRound:
    """Rewrite every non-equal opcode as source deletion plus gap insertion."""

    current = list(map(int, current))
    target = list(map(int, target))
    delete_labels = [1] * len(current)
    target_positions: dict[int, int] = {}
    for tag, source_start, source_end, target_start, target_end in _alignment(current, target):
        if tag != "equal":
            continue
        for offset in range(source_end - source_start):
            source_index = source_start + offset
            delete_labels[source_index] = 0
            target_positions[source_index] = target_start + offset
    delete_labels[0] = 0
    delete_labels[-1] = 0
    target_positions.setdefault(0, 0)
    target_positions.setdefault(len(current) - 1, len(target) - 1)

    kept_indices = [index for index, label in enumerate(delete_labels) if label == 0]
    kept = [current[index] for index in kept_indices]
    gap_payloads: list[list[int]] = []
    for left, right in zip(kept_indices[:-1], kept_indices[1:]):
        left_target = target_positions[left]
        right_target = target_positions[right]
        gap_payloads.append(target[left_target + 1 : right_target])

    gap_labels = [min(len(payload), max_insertions) for payload in gap_payloads]
    masked: list[int] = []
    token_labels: list[int] = []
    completed: list[int] = []
    for index, token in enumerate(kept[:-1]):
        masked.append(token)
        token_labels.append(-1)
        completed.append(token)
        payload = gap_payloads[index][:max_insertions]
        masked.extend([mask_id] * len(payload))
        token_labels.extend(payload)
        completed.extend(payload)
    masked.append(kept[-1])
    token_labels.append(-1)
    completed.append(kept[-1])
    return OracleRound(
        current=current,
        delete_labels=delete_labels,
        kept=kept,
        gap_labels=gap_labels,
        masked=masked,
        token_labels=token_labels,
        completed=completed,
    )


def build_round_example(
    source_tokens: Sequence[int],
    target_tokens: Sequence[int],
    *,
    round_index: int,
    config: LevenshteinConfig,
) -> OracleRound:
    current = [config.bos_id, *map(int, source_tokens), config.eos_id]
    target = [config.bos_id, *map(int, target_tokens), config.eos_id]
    for _ in range(round_index):
        previous = build_oracle_round(
            current,
            target,
            mask_id=config.mask_id,
            max_insertions=config.max_insertions,
        )
        current = previous.completed
        if current == target:
            break
    return build_oracle_round(
        current,
        target,
        mask_id=config.mask_id,
        max_insertions=config.max_insertions,
    )


def collate_rounds(
    rounds: Sequence[OracleRound],
    *,
    pad_id: int,
    device: torch.device | str,
    sequence_width: int | None = None,
) -> LevenshteinBatch:
    def pad(sequences, value, width=None):
        width = width or max(len(sequence) for sequence in sequences)
        if any(len(sequence) > width for sequence in sequences):
            raise ValueError(f"Oracle sequence exceeds collate width {width}")
        out = torch.full((len(sequences), width), value, dtype=torch.long)
        for row, sequence in enumerate(sequences):
            out[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        return out.to(device)

    current_ids = pad([item.current for item in rounds], pad_id, sequence_width)
    kept_ids = pad([item.kept for item in rounds], pad_id, sequence_width)
    masked_ids = pad([item.masked for item in rounds], pad_id, sequence_width)
    delete_labels = pad([item.delete_labels for item in rounds], -1, current_ids.size(1))
    gap_labels = pad([item.gap_labels for item in rounds], -1, max(kept_ids.size(1) - 1, 1))
    token_labels = pad([item.token_labels for item in rounds], -1, masked_ids.size(1))
    return LevenshteinBatch(
        current_ids=current_ids,
        current_padding_mask=current_ids != pad_id,
        delete_labels=delete_labels,
        kept_ids=kept_ids,
        kept_padding_mask=kept_ids != pad_id,
        gap_labels=gap_labels[:, : max(kept_ids.size(1) - 1, 0)],
        masked_ids=masked_ids,
        masked_padding_mask=masked_ids != pad_id,
        token_labels=token_labels,
    )
