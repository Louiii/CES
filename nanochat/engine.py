"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import signal
import warnings
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto

import torch
import torch.nn.functional as F

from nanochat.causal_serialisation import (
    DEL_END,
    DEL_OPEN,
    DEL_START,
    EDIT_END,
    EDIT_START,
    INDEL_SPECIALS,
    INS_LOC,
    INS_OPEN,
    INDL_BOS,
)
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous token's normalized embedding for smear (set by model forward pass)
        self.prev_embedding = None

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        # Copy smear state: expand batch=1 prev_embedding to num_samples
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()

# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None, top_p=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1] or None")
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, scaled.size(-1))
        threshold = torch.topk(scaled, k, dim=-1).values[:, -1:]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Retain the first token whose inclusion crosses the nucleus threshold.
        sorted_remove = cumulative - sorted_probs >= top_p
        remove = torch.zeros_like(sorted_remove).scatter(1, sorted_indices, sorted_remove)
        scaled = scaled.masked_fill(remove, float("-inf"))
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation


class Mode(Enum):
    NORMAL = auto()
    AFTER_EDIT = auto()
    AFTER_DEL_OPEN = auto()
    AFTER_DEL_START = auto()
    AFTER_INS_OPEN = auto()
    IN_INS = auto()


@dataclass
class Row:
    mode: Mode = Mode.NORMAL
    edit_start_ix: int | None = None
    last_ptr: int = -1
    deleted_upto: int = -1
    pending_del_start: int | None = None
    active_ins_loc: int | None = None
    used_ins_locs: set[int] = field(default_factory=set)
    ins_len: int = 0
    base_ix: int = 0   # index of the active <|indel-bos|> anchor for this edit span


class EditController:
    def __init__(self, tokenizer):
        sp = tokenizer.encode_special
        self.tok = {
            "EDIT_START": sp(EDIT_START),
            "EDIT_END":   sp(EDIT_END),
            "DEL_OPEN":   sp(DEL_OPEN),
            "DEL_START":  sp(DEL_START),
            "DEL_END":    sp(DEL_END),
            "INS_OPEN":   sp(INS_OPEN),
            "INS_LOC":    sp(INS_LOC),
            "INDEL_BOS":   sp(INDL_BOS),
        }
        self.indel_specials = {sp(s) for s in INDEL_SPECIALS}
        self.pointer_tokens = {self.tok["DEL_START"], self.tok["DEL_END"], self.tok["INS_LOC"]}
        self.structural_ptr = {self.tok["EDIT_START"], self.tok["EDIT_END"], self.tok["DEL_OPEN"], self.tok["INS_OPEN"]}
        self.rows: list[Row] = []

        # Precompute a tensor list for fast indexing (optional)
        self._indel_specials_list = None  # set later when device known

    def reset(self, B: int):
        self.rows = [Row() for _ in range(B)]

    def prime_from_prompt(self, prompt_tokens: list[int], prompt_tix: list[int], batch_size: int):
        """
        Replay a prompt through the FSM using *known* pointers (tix) for each token.
        This is required if the prompt contains pointer-bearing tokens (del-start/del-end/ins-loc)
        or inserted content that should inherit ins-loc.
        """
        assert len(prompt_tokens) == len(prompt_tix), "prompt_tokens and prompt_tix must match in length"

        if len(self.rows) != batch_size:
            self.reset(batch_size)

        for write_ix, (tok, ptr) in enumerate(zip(prompt_tokens, prompt_tix)):
            for i in range(batch_size):
                self._prime_step(i=i, tok=tok, write_ix=write_ix, ptr=int(ptr))

    def _has_any_valid_ptr(self, r: Row, lo: int, hi: int) -> bool:
        return lo < hi  # because pointers are integer positions

    def allowed(self, r: Row) -> list[int] | None:
        t = self.tok
        match r.mode:
            case Mode.AFTER_EDIT:
                out = [t["DEL_OPEN"], t["INS_OPEN"], t["EDIT_END"]]

                # If we know edit_start_ix, we can prune impossible actions.
                if r.edit_start_ix is not None:
                    hi = r.edit_start_ix
                    base = r.base_ix

                    # Deletion feasibility: del-start must be > max(last_ptr, deleted_upto)
                    # del_lo = max(r.last_ptr, r.deleted_upto) + 1
                    del_lo = max(base + 1, r.last_ptr + 1, r.deleted_upto + 1)
                    if del_lo >= hi:
                        out = [x for x in out if x != t["DEL_OPEN"]]

                    # Insertion feasibility: ins-loc must be > last_ptr and unused
                    # ins_lo = r.last_ptr + 1
                    ins_lo = max(base, r.last_ptr + 1)
                    if ins_lo >= hi:
                        out = [x for x in out if x != t["INS_OPEN"]]
                    else:
                        # if everything in [ins_lo, hi) has been used, disallow
                        if r.used_ins_locs and len(r.used_ins_locs) >= (hi - ins_lo):
                            # conservative: only safe if used_ins_locs are all in-range, but good enough
                            out = [x for x in out if x != t["INS_OPEN"]]

                return out

            case Mode.AFTER_DEL_OPEN:
                return [t["DEL_START"]]
            case Mode.AFTER_DEL_START:
                return [t["DEL_END"]]
            case Mode.AFTER_INS_OPEN:
                return [t["INS_LOC"]]
            case _:
                return None

    def apply_mask_(self, logits, forced_mask=None):
        # logits (B,V)
        B, V = logits.shape
        for i in range(B):
            if forced_mask is not None and forced_mask[i].item():
                continue
            r = self.rows[i]
            allow = self.allowed(r)
            if allow is not None:
                mask = torch.full((V,), float("-inf"), device=logits.device, dtype=logits.dtype)
                mask[allow] = 0.0
                logits[i] += mask
                continue
            elif r.mode == Mode.IN_INS and r.ins_len == 0:
                # disallow control until >=1 normal token
                logits[i, list(self.indel_specials | {self.tok["DEL_OPEN"], self.tok["INS_OPEN"], self.tok["EDIT_END"]})] = float("-inf")
            elif r.mode == Mode.IN_INS and r.ins_len >= 1:
                # We allow normal tokens and also allow control tokens, BUT prune impossible opens.
                if r.edit_start_ix is not None:
                    hi = r.edit_start_ix
                    base = getattr(r, "base_ix", 0)

                    # deletion feasible?
                    del_lo = max(base + 1, r.last_ptr + 1, r.deleted_upto + 1)
                    if del_lo >= hi:
                        logits[i, self.tok["DEL_OPEN"]] = float("-inf")

                    # insertion feasible?
                    ins_lo = max(base, r.last_ptr + 1)
                    if ins_lo >= hi:
                        logits[i, self.tok["INS_OPEN"]] = float("-inf")

    def valid_ptr_range(self, r: Row, tok: int, write_ix: int, device):
        if r.edit_start_ix is None:
            return torch.empty((0,), dtype=torch.long, device=device)

        # global upper bound: strictly before edit_start, strictly before write_ix
        hi = min(write_ix, r.edit_start_ix)
        if hi <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)

        # base lower bound depends on token type
        base = r.base_ix if hasattr(r, "base_ix") else 0

        # insertion can point to base itself; deletion cannot
        if tok == self.tok["INS_LOC"]:
            domain_lo = base
        else:
            # DEL_START / DEL_END
            domain_lo = base + 1

        # monotone pointer constraint
        lo = max(domain_lo, r.last_ptr + 1, 0)

        if lo >= hi:
            return torch.empty((0,), dtype=torch.long, device=device)

        valid = torch.arange(lo, hi, device=device, dtype=torch.long)

        if tok in (self.tok["DEL_START"], self.tok["DEL_END"]):
            valid = valid[valid > r.deleted_upto]
            if tok == self.tok["DEL_END"] and r.pending_del_start is not None:
                valid = valid[valid >= r.pending_del_start]

        if tok == self.tok["INS_LOC"] and r.used_ins_locs:
            used = torch.tensor(sorted(r.used_ins_locs), device=device, dtype=torch.long)
            valid = valid[torch.isin(valid, used, invert=True)]

        return valid

    def _fallback_ptr(self, r: Row, tok: int) -> int:
        # Requires we are inside an edit span
        if r.edit_start_ix is None:
            return 0

        base = getattr(r, "base_ix", 0)
        hi = r.edit_start_ix  # pointers must be < hi

        # If domain is degenerate, just return something safe
        if hi <= 0:
            return 0

        if tok == self.tok["INS_LOC"]:
            lo = base              # insertion may point to base
        else:
            lo = base + 1          # deletion may NOT point to base

        # choose the *latest* valid in-domain pointer, if any
        if lo < hi:
            return hi - 1

        # no valid domain at all
        return max(min(hi - 1, 0), 0)

    def sample_ptr(self, r: Row, tok: int, write_ix: int, position_logits_row, rng, temperature, top_k, top_p):
        # If we have no position logits (should be rare now), use safe fallback.
        if position_logits_row is None:
            return self._fallback_ptr(r, tok)

        device = position_logits_row.device
        valid = self.valid_ptr_range(r, tok, write_ix, device)

        # If empty, DO NOT crash in training; return safe fallback and let resolver crop.
        if valid.numel() == 0:
            return self._fallback_ptr(r, tok)

        row = position_logits_row
        masked = row.clone()
        masked.fill_(float("-inf"))
        masked[valid] = row[valid]

        return int(sample_next_token(
            masked.unsqueeze(0), rng, temperature, top_k, top_p
        )[0, 0].item())

    def _apply_fsm(self, r: Row, tok: int, write_ix: int, ptr: int):
        """Apply mode transitions + bookkeeping given already-decided ptr."""
        t = self.tok
        if tok == t["INDEL_BOS"]:
            r.base_ix = write_ix

        match r.mode:
            case Mode.NORMAL:
                if tok == t["EDIT_START"]:
                    r.mode = Mode.AFTER_EDIT
                    r.edit_start_ix = write_ix

                    # snapshot base for this edit region:
                    # base_ix was being updated as we scanned tokens; keep current value.
                    # If there might be no indel-bos before, you can default to 0.
                    r.base_ix = getattr(r, "base_ix", 0)

                    r.last_ptr = -1
                    r.deleted_upto = -1
                    r.pending_del_start = None
                    r.active_ins_loc = None
                    r.used_ins_locs.clear()
                    r.ins_len = 0

            case Mode.AFTER_EDIT:
                if tok == t["DEL_OPEN"]:
                    r.mode = Mode.AFTER_DEL_OPEN
                elif tok == t["INS_OPEN"]:
                    r.mode = Mode.AFTER_INS_OPEN
                elif tok == t["EDIT_END"]:
                    r.mode = Mode.NORMAL

            case Mode.AFTER_DEL_OPEN:
                if tok == t["DEL_START"]:
                    r.pending_del_start = ptr
                    r.last_ptr = ptr
                    r.mode = Mode.AFTER_DEL_START

            case Mode.AFTER_DEL_START:
                if tok == t["DEL_END"]:
                    r.last_ptr = ptr
                    r.deleted_upto = max(r.deleted_upto, ptr)
                    r.pending_del_start = None
                    r.mode = Mode.AFTER_EDIT

            case Mode.AFTER_INS_OPEN:
                if tok == t["INS_LOC"]:
                    r.last_ptr = ptr
                    r.used_ins_locs.add(ptr)
                    r.active_ins_loc = ptr
                    r.ins_len = 0
                    r.mode = Mode.IN_INS

            case Mode.IN_INS:
                # bookkeeping for inserted content length:
                if tok not in self.indel_specials and tok not in self.pointer_tokens and tok not in self.structural_ptr:
                    r.ins_len += 1

                # edit controls
                if tok == t["DEL_OPEN"]:
                    r.mode = Mode.AFTER_DEL_OPEN
                elif tok == t["INS_OPEN"]:
                    r.mode = Mode.AFTER_INS_OPEN
                elif tok == t["EDIT_END"]:
                    r.mode = Mode.NORMAL

    def _prime_step(self, *, i: int, tok: int, write_ix: int, ptr: int):
        r = self.rows[i]

        # clear insertion inheritance on any indel special
        if tok in self.indel_specials:
            r.active_ins_loc = None
            r.ins_len = 0

        # Trust ptr from prompt_tix; do NOT override it.
        self._apply_fsm(r, tok, write_ix, ptr)

    def step(self, i: int, tok: int, write_ix: int, position_logits_row, rng, temperature, top_k, top_p) -> int:
        r = self.rows[i]
        ptr = write_ix

        # pointer-bearing tokens sample a ptr from previous token's position_logits
        if tok in self.pointer_tokens:
            ptr = self.sample_ptr(r, tok, write_ix, position_logits_row, rng, temperature, top_k, top_p)

        # structural tokens always point to their stream position
        if tok in self.structural_ptr:
            ptr = write_ix

        # insertion inheritance for plain tokens during decode
        if r.mode == Mode.IN_INS:
            if tok not in self.indel_specials and tok not in self.pointer_tokens and tok not in self.structural_ptr:
                if r.active_ins_loc is not None:
                    ptr = r.active_ins_loc

        # clear insertion inheritance on any indel special
        if tok in self.indel_specials:
            r.active_ins_loc = None
            r.ins_len = 0

        self._apply_fsm(r, tok, write_ix, ptr)
        return ptr


class NullEditController:
    """No-op controller for ordinary tokenizers without the CES vocabulary."""

    enabled = False
    pointer_tokens: set[int] = set()
    indel_specials: set[int] = set()
    structural_ptr: set[int] = set()
    tok: dict[str, int] = {}

    def reset(self, _batch_size: int):
        return None

    def prime_from_prompt(self, _prompt_tokens, _prompt_tix, _batch_size: int):
        return None

    def apply_mask_(self, _logits, forced_mask=None):
        return None

    def step(self, *, write_ix: int, **_kwargs) -> int:
        return int(write_ix)


class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use
        try:
            self.edit_controller = EditController(tokenizer)
            self.edit_controller.enabled = True
        except (KeyError, ValueError):
            self.edit_controller = NullEditController()

    @torch.inference_mode()
    def generate(self, tokens, tix=None, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, top_p=None, seed=42, no_edits: bool = False):
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"

        # Build/validate prompt tix
        if tix is None:
            # If prompt contains pointer-bearing tokens, arange(tix) would be wrong.
            # (pointer-bearing tokens include del-start/del-end/ins-loc)
            if any(tok in self.edit_controller.pointer_tokens for tok in tokens):
                raise ValueError(
                    "Prompt contains pointer-bearing edit tokens; you must supply tix "
                    "(the per-token pointer stream positions)."
                )
            prompt_tix = list(range(len(tokens)))
        else:
            if len(tix) != len(tokens):
                raise ValueError(f"tix length must match tokens: {len(tix)} != {len(tokens)}")
            prompt_tix = list(map(int, tix))

        device = self.model.get_device()
        # NOTE: setting the dtype here and in this way is an ugly hack.
        # Currently the repo assumes that cuda -> bfloat16 and everything else -> float32.
        # We need to know the dtype here to call __init__ on KVCache and pre-allocate its tensors.
        # As a quick hack, we're making generate() function inherit and know about this repo-wise assumption.
        # I think there has to be a bigger refactor to deal with device/dtype tracking across the codebase.
        # In particular, the KVCache should allocate its tensors lazily
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # init controller state
        self.edit_controller.reset(num_samples)
        self.edit_controller.prime_from_prompt(tokens, prompt_tix, num_samples)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        cache_head_dim = (
            self.model.get_kv_cache_head_dim()
            if hasattr(self.model, "get_kv_cache_head_dim")
            else m.n_embd // m.n_head
        )
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": cache_head_dim, "num_layers": m.n_layer}
        kv_pos_kwargs = {"num_heads": 1, "head_dim": m.n_embd, "num_layers": 1}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        pos_k_cache_prefill = (
            KVCache(
                batch_size=1,
                seq_len=len(tokens),
                device=device,
                dtype=dtype,
                **kv_pos_kwargs,
            )
            if self.edit_controller.enabled
            else None
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        tix_prompt = torch.tensor([prompt_tix], dtype=torch.long, device=device)  # (1, T)

        if self.edit_controller.enabled:
            logits, position_logits = self.model.forward(
                ids,
                tix=tix_prompt,
                kv_cache=kv_cache_prefill,
                pos_k_cache=pos_k_cache_prefill,
                return_pos_logits=True,
            )
        else:
            logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
            position_logits = None

        # seed decode from last token
        logits = logits[:, -1, :].expand(num_samples, -1)                 # (B, V)
        if position_logits is not None:
            position_logits = position_logits[:, -1, :].expand(num_samples, -1)  # (B, Tctx)


        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        pos_k_cache_decode = (
            KVCache(
                batch_size=num_samples,
                seq_len=kv_length_hint,
                device=device,
                dtype=dtype,
                **kv_pos_kwargs,
            )
            if self.edit_controller.enabled
            else None
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        if pos_k_cache_decode is not None:
            pos_k_cache_decode.prefill(pos_k_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around
        del pos_k_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # Hmm... no I think tix needs to be exactly the same size as the tokens for
        # decode - i.e. 1 in the sequence dim.
        prompt_len = len(tokens)
        hist_len = prompt_len + (max_tokens if max_tokens is not None else 0) + 1
        tix_hist = torch.empty((num_samples, hist_len), dtype=torch.long, device=device)

        prompt_tix_tensor = torch.tensor(prompt_tix, device=device, dtype=torch.long)[None, :].expand(num_samples, -1)
        tix_hist[:, :prompt_len] = prompt_tix_tensor

        # 4) Main generation loop
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            forced_mask = torch.tensor([len(s.forced_tokens) > 0 for s in row_states],
                           dtype=torch.bool, device=logits.device)
            self.edit_controller.apply_mask_(logits, forced_mask=forced_mask)

            # Optionally ban all edit tokens (for normal autoregressive evals)
            if no_edits and self.edit_controller.enabled:
                ec = self.edit_controller
                banned = set(ec.indel_specials) | set(ec.pointer_tokens) | set(ec.structural_ptr)
                # safety: ensure anchor is included too
                banned.add(ec.tok["INDEL_BOS"])
                logits[:, list(banned)] = float("-inf")

            # Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k, top_p)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            write_ix = kv_cache_decode.get_pos()
            # print("pos", kv_cache_decode.get_pos(), "num_generated", num_generated)

            # print(write_ix)

            # position_logits at decode is (B, Tctx) after you slice; for first step it's None
            tix_next = torch.empty((num_samples,), dtype=torch.long, device=device)
            pointer_column = [None] * num_samples

            for i, tok in enumerate(token_column):
                # # DEBUG: confirm pointer sampling alignment
                # if True:  # flip to False to disable
                #     for i, tok in enumerate(token_column):
                #         if tok in self.edit_controller.pointer_tokens:
                #             # this token is pointer-bearing, so position_logits should have been produced by previous token
                #             prev_tok = row_states[i].current_tokens[-2] if len(row_states[i].current_tokens) >= 2 else None
                #             print(f"[row {i}] POINTER TOKEN={tok} prev_tok={prev_tok} "
                #                 f"pos_logits_shape={tuple(position_logits[i].shape)} write_ix={write_ix}")

                row_position_logits = None if position_logits is None else position_logits[i]
                ptr = self.edit_controller.step(
                    i=i,
                    tok=tok,
                    write_ix=write_ix,
                    position_logits_row=row_position_logits,
                    rng=rng,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                tix_next[i] = ptr
                pointer_column[i] = int(ptr)

            if write_ix >= tix_hist.size(1):
                raise RuntimeError(f"tix_hist too small: write_ix={write_ix}, hist_len={tix_hist.size(1)}")

            tix_hist[:, write_ix] = tix_next
            if self.edit_controller.enabled:
                yield token_column, token_masks, pointer_column
            else:
                yield token_column, token_masks
            num_generated += 1

            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            tix_step = tix_next[:, None]
            if self.edit_controller.enabled:
                logits, position_logits = self.model.forward(
                    ids,
                    tix=tix_step,
                    kv_cache=kv_cache_decode,
                    pos_k_cache=pos_k_cache_decode,
                    return_pos_logits=True,
                )
            else:
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)
                position_logits = None
            logits = logits[:, -1, :]
            if position_logits is not None:
                position_logits = position_logits[:, -1, :]   # (B, Tctx)


    def generate_batch(self, tokens, num_samples=1, tix=None, no_edits: bool = False, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        # TODO: resample a sub batch which has edits to be applied.
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        results  = [tokens.copy() for _ in range(num_samples)]
        masks    = [[0] * len(tokens) for _ in range(num_samples)]
        if tix is None:
            pointers = [list(range(len(tokens))) for _ in range(num_samples)]
        else:
            pointers = [list(tix) for _ in range(num_samples)]

        completed = [False] * num_samples

        for generated in self.generate(
            tokens, tix=tix, num_samples=num_samples, no_edits=no_edits, **kwargs
        ):
            if len(generated) == 3:
                token_column, token_masks, pointer_column = generated
            else:
                token_column, token_masks = generated
                pointer_column = [None] * num_samples
            for i, (token, mask, ptr) in enumerate(zip(token_column, token_masks, pointer_column)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
                        pointers[i].append(ptr)

            if all(completed):
                break

        return results, masks, pointers



if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time

    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks, pointer_column in stream:
        token = token_column[0] # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
