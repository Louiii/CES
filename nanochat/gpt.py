"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    rope_base: float = 10000
    use_dual_rope: bool = True
    dual_rope_style: str = "split"  # split (legacy CES) or append (pretrained CES adaptation)
    use_position_head: bool = True
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def compute_cos_sin(t, head_dim, base, device, dtype=None):
    # stride the channels
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    # calculate the rotation frequencies at each (time, channel) pair
    freqs = torch.einsum("...t,f->...tf", t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self.use_dual_rope = config.use_dual_rope
        self.dual_rope_style = config.dual_rope_style
        if self.dual_rope_style not in {"split", "append"}:
            raise ValueError(f"Unsupported dual_rope_style: {self.dual_rope_style!r}")
        self.append_pointer_rope = self.use_dual_rope and self.dual_rope_style == "append"
        if self.append_pointer_rope:
            self.c_q_pointer = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
            self.c_k_pointer = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.pointer_attn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, ve, cos_sin, cos2_sin2, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values.
        # The appended CES mode retains these pretrained projections unchanged
        # and adds an independently rotated pointer-address subspace below.
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys. The legacy split mode
        # divides an existing head between generation and address coordinates.
        # The append mode instead preserves the complete pretrained head and
        # adds a gated full-width address head. At gate=0 its attention logits
        # are exactly the standard single-RoPE logits.
        softmax_scale = None
        if self.append_pointer_rope:
            cos1, sin1 = cos_sin
            q = norm(apply_rotary_emb(q, cos1, sin1)) * 1.2
            k = norm(apply_rotary_emb(k, cos1, sin1)) * 1.2
            q_pointer = self.c_q_pointer(x).view(B, T, self.n_head, self.head_dim)
            k_pointer = self.c_k_pointer(x).view(B, T, self.n_kv_head, self.head_dim)
            cos2, sin2 = cos2_sin2
            q_pointer = norm(apply_rotary_emb(q_pointer, cos2, sin2)) * 1.2
            k_pointer = norm(apply_rotary_emb(k_pointer, cos2, sin2)) * 1.2
            gate = self.pointer_attn_gate.to(q.dtype)
            q = torch.cat([q, gate * q_pointer], dim=-1)
            k = torch.cat([k, k_pointer], dim=-1)
            # FA3 requires Q/K/V to use the same head dimension. The padded
            # value half stays exactly zero and is removed after attention.
            v = torch.cat([v, torch.zeros_like(v)], dim=-1)
            softmax_scale = self.head_dim ** -0.5
        elif self.use_dual_rope:
            h = self.head_dim // 2
            q1, q2 = torch.split(q, h, dim=-1)
            k1, k2 = torch.split(k, h, dim=-1)
            cos1, sin1 = cos_sin
            q1, k1 = apply_rotary_emb(q1, cos1, sin1), apply_rotary_emb(k1, cos1, sin1)
            cos2, sin2 = cos2_sin2
            q2, k2 = apply_rotary_emb(q2, cos2, sin2), apply_rotary_emb(k2, cos2, sin2)
            q, k = torch.cat([q1, q2], dim=-1), torch.cat([k1, k2], dim=-1)
            q, k = norm(q), norm(k)
            q, k = q * 1.2, k * 1.2
        else:
            cos, sin = cos_sin
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)
            q, k = q * 1.2, k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(
                q, k, v, causal=True, window_size=window_size,
                softmax_scale=softmax_scale,
            )
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
                softmax_scale=softmax_scale,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Remove the zero-padded value half used only for FA3 shape
        # compatibility, then re-assemble the original value heads.
        if self.append_pointer_rope:
            y = y[..., :self.head_dim]
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class SequencePositionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kv_dim = config.n_embd
        self.c_q = Linear(config.n_embd, self.kv_dim, bias=False)
        self.c_k = Linear(config.n_embd, self.kv_dim, bias=False)

    def forward(self, x, pos_k_cache, queries_gather=None):
        B, T, C = x.size()
        if queries_gather is not None:
            _B, G = queries_gather.size()
            # Training.
            idx = queries_gather.unsqueeze(-1).expand(-1, -1, C)   # (B, G, C)
            xg = x.gather(dim=1, index=idx)                        # (B, G, C)
        else:
            # Inference.
            # The query must only be size one at inference, we only predict one step
            # into the future.
            xg = x[:, -1:, :]
            G = 1

        # Project the input to get queries, keys, and values
        q = self.c_q(xg)  #.view(B, G, self.kv_dim)
        k = self.c_k(x)  #.view(B, T, self.kv_dim)

        q, k = norm(q), norm(k) # QK norm

        # with torch.no_grad():
        #     q = self.c_q(xg)
        #     k = self.c_k(x)
        #     qn, kn = norm(q), norm(k)
        #     print("q mean/std", q.float().mean().item(), q.float().std().item())
        #     print("k mean/std", k.float().mean().item(), k.float().std().item())
        #     print("qn mean/std", qn.float().mean().item(), qn.float().std().item())
        #     print("kn mean/std", kn.float().mean().item(), kn.float().std().item())


        # Apply KV cache: insert current k,v into cache, get the full view so far
        if pos_k_cache is not None:
            k_cache, _ = pos_k_cache.get_layer_cache(0)  # (B, max_T, 1, C)
            assert k_cache.size(0) == B
            assert k_cache.size(2) == 1
            assert k_cache.size(3) == C

            pos = pos_k_cache.get_pos()
            Tnew = k.size(1)

            # bounds check
            assert pos + Tnew <= pos_k_cache.max_seq_len, "pos_k_cache overflow"

            # append new keys
            k_cache[:, pos:pos+Tnew, 0, :] = k  # k: (B, Tnew, C)

            pos_k_cache.advance(Tnew)

            # use only the filled prefix as attention context
            k = k_cache[:, :pos+Tnew, 0, :]     # (B, T_total, C)

        scale_factor = 1 / math.sqrt(q.size(-1))
        if queries_gather is not None:
            t = torch.arange(T, dtype=queries_gather.dtype, device=queries_gather.device)
            valid_context = t[None, None, :] <= queries_gather[:, :, None]

            context_bias = torch.zeros(B, G, T, dtype=q.dtype, device=q.device)
            context_bias.masked_fill_(valid_context.logical_not(), float("-inf"))
            position_logits = q @ k.transpose(-2, -1) * scale_factor  # (B, G, T)
            position_logits += context_bias
        else:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            position_logits = q @ k.transpose(-2, -1) * scale_factor  # (B, G, T)

        return position_logits


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, cos2_sin2, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, cos2_sin2, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        self.rope_base = config.rope_base
        self.use_dual_rope = config.use_dual_rope
        self.dual_rope_style = config.dual_rope_style
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.position_head = SequencePositionHead(config) if config.use_position_head else None
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        self.half_head = head_dim // 2
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        rotary_head_dim = (
            self.half_head
            if self.use_dual_rope and self.dual_rope_style == "split"
            else head_dim
        )
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, rotary_head_dim, config.rope_base)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Appended address projections start as copies of the pretrained-style
        # sequence projections. Their zero gate makes this function-preserving.
        for block in self.transformer.h:
            if block.attn.append_pointer_rope:
                block.attn.c_q_pointer.weight.copy_(block.attn.c_q.weight)
                block.attn.c_k_pointer.weight.copy_(block.attn.c_k.weight)
                torch.nn.init.zeros_(block.attn.pointer_attn_gate)

        # Position head (init like other Q/K projections)
        if self.position_head is not None:
            torch.nn.init.uniform_(self.position_head.c_q.weight, -s, s)
            torch.nn.init.uniform_(self.position_head.c_k.weight, -s, s)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        rotary_head_dim = (
            self.half_head
            if self.use_dual_rope and self.dual_rope_style == "split"
            else head_dim
        )
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, rotary_head_dim, self.config.rope_base)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        cos, sin = compute_cos_sin(t, head_dim, base, device, dtype=COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def get_kv_cache_head_dim(self):
        head_dim = self.config.n_embd // self.config.n_head
        if self.use_dual_rope and self.dual_rope_style == "append":
            return 2 * head_dim
        return head_dim

    def estimate_flops(self, position_queries: int = 0):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        ``position_queries`` adds the position head's gathered-query-by-sequence
        matmul. Training pads this dimension, so use the configured buffer size,
        not the number of valid pointer targets in a particular batch.
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel() +
                          sum(block.attn.pointer_attn_gate.numel() for block in self.transformer.h if block.attn.append_pointer_rope))
        h = self.config.n_head
        q = self.get_kv_cache_head_dim()
        t = self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        position_addressing_flops = 6 * max(int(position_queries), 0) * self.config.n_embd
        num_flops_per_token = (
            6 * (nparams - nparams_exclude) + attn_flops + position_addressing_flops
        )
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(
            p.numel() for name, p in self.transformer.h.named_parameters()
            if not name.endswith("pointer_attn_gate")
        )
        position_head = 0 if self.position_head is None else sum(p.numel() for p in self.position_head.parameters())
        pointer_gates = sum(
            block.attn.pointer_attn_gate.numel()
            for block in self.transformer.h
            if block.attn.append_pointer_rope
        )
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel() + pointer_gates
        total = wte + value_embeds + lm_head + transformer_matrices + position_head + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'position_head': position_head,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        scalar_lr=0.5,
        distributed_optimizer_overlap=True,
        distributed_muon_group_size=None,
    ):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate matrix parameters from scalar address gates. New adaptation
        # matrices are kept in distinct optimizer groups even when their shapes
        # match backbone matrices, allowing a staged LR schedule.
        new_matrix_params = []
        backbone_matrix_params = []
        pointer_gate_params = []
        for name, param in self.transformer.h.named_parameters():
            if name.endswith("pointer_attn_gate"):
                pointer_gate_params.append(param)
            elif ".c_q_pointer." in name or ".c_k_pointer." in name:
                new_matrix_params.append(param)
            else:
                backbone_matrix_params.append(param)
        if self.position_head is not None:
            new_matrix_params.extend(self.position_head.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        matrix_params = [*backbone_matrix_params, *new_matrix_params]
        accounted = [
            *matrix_params, *pointer_gate_params, *embedding_params,
            *lm_head_params, *value_embeds_params, *resid_params,
            *x0_params, *smear_params,
        ]
        assert len(list(self.parameters())) == len(accounted)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', adapt_role='vocab', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', adapt_role='vocab', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', adapt_role='vocab', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', adapt_role='backbone', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', adapt_role='backbone', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', adapt_role='backbone', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if pointer_gate_params:
            param_groups.append(dict(kind='adamw', adapt_role='new', params=pointer_gate_params, lr=scalar_lr * 0.1, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        # Muon groups (matrix params, grouped by shape and adaptation role).
        if distributed_muon_group_size is not None and distributed_muon_group_size < 1:
            raise ValueError("distributed_muon_group_size must be positive or None")
        for role, params in (("backbone", backbone_matrix_params), ("new", new_matrix_params)):
            for shape in sorted({p.shape for p in params}):
                group_params = [p for p in params if p.shape == shape]
                chunk_size = (
                    distributed_muon_group_size
                    if ddp and distributed_muon_group_size is not None
                    else len(group_params)
                )
                for start in range(0, len(group_params), chunk_size):
                    param_groups.append(dict(
                        kind='muon', adapt_role=role,
                        params=group_params[start:start + chunk_size], lr=matrix_lr,
                        momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
                    ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = (
            Factory(
                param_groups,
                overlap_communication=distributed_optimizer_overlap,
            )
            if ddp
            else Factory(param_groups)
        )
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def setup_optimizers(self, *args, **kwargs):
        return self.setup_optimizer(*args, **kwargs)

    def forward(
        self, idx, tix=None, targets=None, kv_cache=None, pos_k_cache=None,
        loss_reduction='mean', return_pos_logits=False, return_probe=False,
        position_queries=None, return_hidden=False,
    ):
        B, T = idx.size()
        if tix is None:
            use_cuda = idx.device == "cuda"
            tix = torch.arange(idx.shape[-1], dtype=torch.long, pin_memory=use_cuda)
            tix = tix.to(device=idx.device, non_blocking=use_cuda)
            tix = tix[None, :].repeat(B, 1)

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        if self.use_dual_rope:
            address_head_dim = (
                self.half_head if self.dual_rope_style == "split"
                else self.config.n_embd // self.config.n_head
            )
            cos2, sin2 = compute_cos_sin(
                tix, address_head_dim, self.rope_base, self.cos.device, dtype=self.cos.dtype
            )
            if tix.ndim == 1:
                cos2_sin2 = cos2[None, :, None, :], sin2[None, :, None, :]
            elif tix.ndim == 2:
                cos2_sin2 = cos2[:, :, None, :], sin2[:, :, None, :]
            else:
                raise ValueError(f"{tix.shape} is not supported - it must be 1 or 2 dimensional")
        else:
            cos2_sin2 = None

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, cos2_sin2, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Frozen-backbone probes and reward models only need the contextual
        # representation. Returning here avoids materialising the much larger
        # vocabulary logits tensor. Keep this deliberately incompatible with
        # the ordinary training/inference auxiliary outputs so callers cannot
        # accidentally bypass a requested language-model loss.
        if return_hidden:
            if targets is not None or return_pos_logits or position_queries is not None:
                raise ValueError(
                    "return_hidden cannot be combined with targets, position logits, "
                    "or explicit position queries"
                )
            return x

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        # By default the position is set to next from each point
        # But sometimes we will provide data which has an insertion / deletion
        # The model is tasked with predicting the token position where the insertion
        # should be placed after. Or the position which should be deleted.
        if targets is not None and position_queries is not None:
            raise ValueError("Pass either training targets or explicit position queries, not both")
        queries_gather = (
            targets.gather_indices if targets is not None else position_queries
        )
        position_logits = (
            None
            if self.position_head is None
            else self.position_head(x, pos_k_cache, queries_gather)
        )

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            tloss_raw = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.tokens.view(-1),
                ignore_index=-1,
                reduction='none',
            )

            if self.position_head is None or targets.gathered_pos_target is None:
                if loss_reduction == 'none':
                    loss = tloss_raw.view(B, T)   # per-token loss
                elif loss_reduction == 'mean':
                    loss = tloss_raw.mean()
                else:
                    loss = tloss_raw.sum()
                return loss
            else:
                tloss = tloss_raw * targets.token_loss_weights.view(-1)
                tloss = tloss.mean() if loss_reduction == 'mean' else tloss.sum()

                pos_softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
                position_logits = position_logits.float() # switch to fp32 for logit softcap and loss computation
                position_logits = pos_softcap * torch.tanh(position_logits / pos_softcap) # squash the logits

                ploss_raw = F.cross_entropy(
                    position_logits.view(-1, position_logits.size(-1)),
                    targets.gathered_pos_target.view(-1),
                    ignore_index=-1,
                    reduction='none',
                ).view_as(targets.gathered_pos_target)
                ploss = (targets.gather_valid * ploss_raw).sum(-1)  # Sum seq gathered dim
                ploss = ploss.sum()  # Sum batch dim
                if loss_reduction == 'mean':
                    pos_denom = targets.gather_valid.sum()
                    if pos_denom > 0:
                        ploss /= pos_denom
                    else:
                        ploss = ploss * 0.0
                loss = tloss + ploss
                valid = targets.gather_valid.bool() & (targets.gathered_pos_target != -1)
                correct = ((position_logits.argmax(-1) == targets.gathered_pos_target) & valid).sum()
                aux = {
                    "position_loss": ploss,
                    "token_loss": tloss,
                    "position_correct": (correct.detach(), valid.sum().detach())
                }
                if targets.perplexity_masks:
                    for k, m in targets.perplexity_masks.items():
                        masked_nll_sum = (tloss_raw * m.view(-1)).sum()
                        denom = m.sum()
                        aux[k] = (masked_nll_sum.detach(), denom.detach())
                if return_probe:
                    aux["probe"] = x
                return loss, aux
        else:
            # inference: just return the logits directly
            return (logits, position_logits) if return_pos_logits else logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
