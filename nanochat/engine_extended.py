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

from nanochat.causal_serialisation_extended import (
    COPY_DEST,
    COPY_END,
    COPY_OPEN,
    COPY_START,
    DEL_END,
    DEL_OPEN,
    DEL_START,
    EDIT_END,
    EDIT_START,
    INDEL_SPECIALS,
    INS_LOC,
    INS_OPEN,
    INDL_BOS,
    MOVE_DEST,
    MOVE_END,
    MOVE_OPEN,
    MOVE_START,
    REPLACE_END,
    REPLACE_EOS,
    REPLACE_OPEN,
    REPLACE_START,
    SHARED_POINTER_TOKEN,
    validate_ces_pointer_token_mode,
)
from nanochat.checkpoint_manager import load_model
from nanochat.common import COMPUTE_DTYPE, autodetect_device_type, compute_init

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
def filtered_sampling_logits(logits, temperature=1.0, top_k=None, top_p=None):
    """Return the exact logits used by stochastic decoding.

    Keeping filtering in one function is important for RL: rollout-time and
    differentiable replay probabilities must describe the same policy.
    """
    assert temperature >= 0.0, "temperature must be non-negative"
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1] or None")
    if temperature == 0.0:
        return logits
    scaled = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, scaled.size(-1))
        threshold = torch.topk(scaled, k, dim=-1).values[:, -1:]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Keep the first token whose inclusion crosses top_p.
        sorted_remove = cumulative - sorted_probs >= top_p
        remove = torch.zeros_like(sorted_remove).scatter(1, sorted_indices, sorted_remove)
        scaled = scaled.masked_fill(remove, float("-inf"))
    return scaled


def selected_action_logprobs(logits, actions, temperature=1.0, top_k=None, top_p=None):
    """Log-probabilities of selected actions under the decoding distribution."""
    if temperature == 0.0:
        greedy = torch.argmax(logits, dim=-1, keepdim=True)
        return torch.where(
            actions == greedy,
            torch.zeros_like(actions, dtype=logits.dtype),
            torch.full_like(actions, float("-inf"), dtype=logits.dtype),
        )
    filtered = filtered_sampling_logits(logits, temperature, top_k, top_p)
    return F.log_softmax(filtered, dim=-1).gather(-1, actions)


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None, top_p=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    filtered = filtered_sampling_logits(logits, temperature, top_k, top_p)
    probs = F.softmax(filtered, dim=-1)
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
    AFTER_REPLACE_OPEN = auto()
    AFTER_REPLACE_START = auto()
    IN_REPLACE = auto()
    AFTER_COPY_OPEN = auto()
    AFTER_COPY_START = auto()
    AFTER_COPY_END = auto()
    AFTER_MOVE_OPEN = auto()
    AFTER_MOVE_START = auto()
    AFTER_MOVE_END = auto()


@dataclass
class Row:
    mode: Mode = Mode.NORMAL
    edit_start_ix: int | None = None
    last_ptr: int = -1
    deleted_upto: int = -1
    pending_del_start: int | None = None
    pending_copy_start: int | None = None
    pending_move_start: int | None = None
    pending_move_end: int | None = None
    active_payload_ptr: int | None = None
    payload_len: int = 0
    max_payload_len_seen: int = 0
    base_ix: int = 0   # index of the active <|indel-bos|> anchor for this edit span
    completed_ops: int = 0
    forced_payload_closures: int = 0
    forced_program_closures: int = 0
    first_decision_edit_margin: float | None = None
    hierarchical_edit_selected: bool | None = None


@dataclass(frozen=True)
class PolicyActionTrace:
    """One generated CES action and its rollout-time policy probability."""

    absolute_index: int
    token_id: int
    pointer: int
    token_sampled: bool
    pointer_sampled: bool
    token_logprob: float | None
    pointer_logprob: float | None
    valid_pointer_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class PolicyRollout:
    """A complete rollout sufficient for exact differentiable replay."""

    prompt_length: int
    tokens: tuple[int, ...]
    pointers: tuple[int, ...]
    actions: tuple[PolicyActionTrace, ...]
    temperature: float
    pointer_temperature: float
    top_k: int | None
    top_p: float | None
    closure_bias: float
    closure_bias_min_payload_tokens: int
    no_edits: bool
    max_payload_tokens: int | None
    max_edit_ops: int | None


class EditController:
    def __init__(self, tokenizer, pointer_token_mode: str = "role_specific"):
        sp = tokenizer.encode_special
        self.pointer_token_mode = validate_ces_pointer_token_mode(pointer_token_mode)
        self.tok = {
            "EDIT_START": sp(EDIT_START),
            "EDIT_END":   sp(EDIT_END),
            "DEL_OPEN":   sp(DEL_OPEN),
            "DEL_START":  sp(DEL_START),
            "DEL_END":    sp(DEL_END),
            "INS_OPEN":   sp(INS_OPEN),
            "INS_LOC":    sp(INS_LOC),
            "REPLACE_OPEN": sp(REPLACE_OPEN),
            "REPLACE_START": sp(REPLACE_START),
            "REPLACE_END": sp(REPLACE_END),
            "REPLACE_EOS": sp(REPLACE_EOS),
            "COPY_OPEN": sp(COPY_OPEN),
            "COPY_START": sp(COPY_START),
            "COPY_END": sp(COPY_END),
            "COPY_DEST": sp(COPY_DEST),
            "MOVE_OPEN": sp(MOVE_OPEN),
            "MOVE_START": sp(MOVE_START),
            "MOVE_END": sp(MOVE_END),
            "MOVE_DEST": sp(MOVE_DEST),
            "INDEL_BOS": sp(INDL_BOS),
        }
        if self.pointer_token_mode == "shared":
            shared_pointer_token = sp(SHARED_POINTER_TOKEN)
            for role in (
                "DEL_START",
                "DEL_END",
                "INS_LOC",
                "REPLACE_START",
                "REPLACE_END",
                "COPY_START",
                "COPY_END",
                "COPY_DEST",
                "MOVE_START",
                "MOVE_END",
                "MOVE_DEST",
            ):
                self.tok[role] = shared_pointer_token
        self.indel_specials = {sp(s) for s in INDEL_SPECIALS}
        self.pointer_tokens = {
            self.tok["DEL_START"],
            self.tok["DEL_END"],
            self.tok["INS_LOC"],
            self.tok["REPLACE_START"],
            self.tok["REPLACE_END"],
            self.tok["COPY_START"],
            self.tok["COPY_END"],
            self.tok["COPY_DEST"],
            self.tok["MOVE_START"],
            self.tok["MOVE_END"],
            self.tok["MOVE_DEST"],
        }
        self.structural_ptr = {
            self.tok["EDIT_START"],
            self.tok["EDIT_END"],
            self.tok["DEL_OPEN"],
            self.tok["INS_OPEN"],
            self.tok["REPLACE_OPEN"],
            self.tok["REPLACE_EOS"],
            self.tok["COPY_OPEN"],
            self.tok["MOVE_OPEN"],
        }
        self.rows: list[Row] = []

        # Precompute a tensor list for fast indexing (optional)
        self._indel_specials_list = None  # set later when device known
        self.max_payload_tokens: int | None = None
        self.max_edit_ops: int | None = None

    def reset(
        self,
        B: int,
        *,
        max_payload_tokens: int | None = None,
        max_edit_ops: int | None = None,
    ):
        if max_payload_tokens is not None and max_payload_tokens < 1:
            raise ValueError("max_payload_tokens must be >= 1 or None")
        if max_edit_ops is not None and max_edit_ops < 1:
            raise ValueError("max_edit_ops must be >= 1 or None")
        self.max_payload_tokens = max_payload_tokens
        self.max_edit_ops = max_edit_ops
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
                if self.max_edit_ops is not None and r.completed_ops >= self.max_edit_ops:
                    return [t["EDIT_END"]]
                return [
                    t["DEL_OPEN"],
                    t["INS_OPEN"],
                    t["REPLACE_OPEN"],
                    t["COPY_OPEN"],
                    t["MOVE_OPEN"],
                    t["EDIT_END"],
                ]

            case Mode.AFTER_DEL_OPEN:
                return [t["DEL_START"]]
            case Mode.AFTER_DEL_START:
                return [t["DEL_END"]]
            case Mode.AFTER_INS_OPEN:
                return [t["INS_LOC"]]
            case Mode.AFTER_REPLACE_OPEN:
                return [t["REPLACE_START"]]
            case Mode.AFTER_REPLACE_START:
                return [t["REPLACE_END"]]
            case Mode.AFTER_COPY_OPEN:
                return [t["COPY_START"]]
            case Mode.AFTER_COPY_START:
                return [t["COPY_END"]]
            case Mode.AFTER_COPY_END:
                return [t["COPY_DEST"]]
            case Mode.AFTER_MOVE_OPEN:
                return [t["MOVE_START"]]
            case Mode.AFTER_MOVE_START:
                return [t["MOVE_END"]]
            case Mode.AFTER_MOVE_END:
                return [t["MOVE_DEST"]]
            case Mode.IN_INS:
                if self.max_payload_tokens is not None and r.payload_len >= self.max_payload_tokens:
                    if self.max_edit_ops is not None and r.completed_ops + 1 >= self.max_edit_ops:
                        return [t["EDIT_END"]]
                    return [
                        t["DEL_OPEN"],
                        t["INS_OPEN"],
                        t["REPLACE_OPEN"],
                        t["COPY_OPEN"],
                        t["MOVE_OPEN"],
                        t["EDIT_END"],
                    ]
            case Mode.IN_REPLACE:
                if self.max_payload_tokens is not None and r.payload_len >= self.max_payload_tokens:
                    return [t["REPLACE_EOS"]]
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
            elif r.mode == Mode.IN_INS:
                banned = self.indel_specials - {
                    self.tok["DEL_OPEN"],
                    self.tok["INS_OPEN"],
                    self.tok["REPLACE_OPEN"],
                    self.tok["COPY_OPEN"],
                    self.tok["MOVE_OPEN"],
                    self.tok["EDIT_END"],
                }
                logits[i, list(banned)] = float("-inf")
            elif r.mode == Mode.IN_REPLACE:
                banned = self.indel_specials - {self.tok["REPLACE_EOS"]}
                logits[i, list(banned)] = float("-inf")

    def apply_closure_bias_(self, logits, bias: float, min_payload_tokens: int = 0, forced_mask=None):
        """Bias legal tokens that close the current payload or edit program."""
        if bias == 0.0:
            return
        for i, r in enumerate(self.rows):
            if forced_mask is not None and forced_mask[i].item():
                continue
            token = None
            if r.mode == Mode.IN_INS and r.payload_len >= min_payload_tokens:
                token = self.tok["EDIT_END"]
            elif r.mode == Mode.IN_REPLACE and r.payload_len >= min_payload_tokens:
                token = self.tok["REPLACE_EOS"]
            elif r.mode == Mode.AFTER_EDIT and r.completed_ops > 0:
                token = self.tok["EDIT_END"]
            if token is not None and torch.isfinite(logits[i, token]):
                logits[i, token] += bias

    def apply_first_decision_gate_(
        self,
        logits,
        threshold: float,
        forced_mask=None,
    ):
        """Choose edit versus no-op before choosing a GEC operation.

        The paired gate loss marginalises deletion, insertion, and replacement
        into one edit class. Flat greedy decoding compares each opener with
        EDIT_END separately; this restores the hierarchy used for training.
        """

        edit_tokens = [
            self.tok["DEL_OPEN"],
            self.tok["INS_OPEN"],
            self.tok["REPLACE_OPEN"],
        ]
        noop_token = self.tok["EDIT_END"]
        for i, row in enumerate(self.rows):
            if forced_mask is not None and forced_mask[i].item():
                continue
            if row.mode != Mode.AFTER_EDIT or row.completed_ops != 0:
                continue
            edit_logit = torch.logsumexp(logits[i, edit_tokens].float(), dim=0)
            margin = edit_logit - logits[i, noop_token].float()
            select_edit = bool(float(margin) >= threshold)
            row.first_decision_edit_margin = float(margin)
            row.hierarchical_edit_selected = select_edit
            allowed = edit_tokens if select_edit else [noop_token]
            mask = torch.full_like(logits[i], float("-inf"))
            mask[allowed] = 0.0
            logits[i] += mask

    def ban_edit_tokens_(self, logits):
        """Mask the complete CES vocabulary for ordinary AR decoding."""
        banned = set(self.indel_specials) | set(self.pointer_tokens) | set(self.structural_ptr)
        banned.add(self.tok["INDEL_BOS"])
        logits[:, list(banned)] = float("-inf")

    def _pointer_role(self, r: Row, tok: int) -> str | None:
        """Recover pointer semantics from the pre-token grammar state."""
        expected = {
            Mode.AFTER_DEL_OPEN: ("deletion_start", "DEL_START"),
            Mode.AFTER_DEL_START: ("deletion_end", "DEL_END"),
            Mode.AFTER_INS_OPEN: ("insertion_loc", "INS_LOC"),
            Mode.AFTER_REPLACE_OPEN: ("replace_start", "REPLACE_START"),
            Mode.AFTER_REPLACE_START: ("replace_end", "REPLACE_END"),
            Mode.AFTER_COPY_OPEN: ("copy_start", "COPY_START"),
            Mode.AFTER_COPY_START: ("copy_end", "COPY_END"),
            Mode.AFTER_COPY_END: ("copy_dest", "COPY_DEST"),
            Mode.AFTER_MOVE_OPEN: ("move_start", "MOVE_START"),
            Mode.AFTER_MOVE_START: ("move_end", "MOVE_END"),
            Mode.AFTER_MOVE_END: ("move_dest", "MOVE_DEST"),
        }.get(r.mode)
        if expected is None:
            return None
        role, token_name = expected
        return role if tok == self.tok[token_name] else None

    def valid_ptr_range(
        self,
        r: Row,
        tok: int,
        write_ix: int,
        device,
        *,
        role: str | None = None,
    ):
        role = self._pointer_role(r, tok) if role is None else role
        if r.edit_start_ix is None:
            return torch.empty((0,), dtype=torch.long, device=device)

        # global upper bound: strictly before edit_start, strictly before write_ix
        hi = min(write_ix, r.edit_start_ix)
        if hi <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)

        base = r.base_ix if hasattr(r, "base_ix") else 0

        insertion_like = {"insertion_loc", "copy_dest", "move_dest"}
        valid = torch.arange(0, hi, device=device, dtype=torch.long)

        if role not in insertion_like:
            valid = valid[valid != base]

        if role in {"deletion_end", "replace_end"} and r.pending_del_start is not None:
            valid = valid[valid >= r.pending_del_start]
        if role == "copy_end" and r.pending_copy_start is not None:
            valid = valid[valid >= r.pending_copy_start]
        if role == "move_end" and r.pending_move_start is not None:
            valid = valid[valid >= r.pending_move_start]

        if role == "move_dest" and r.pending_move_start is not None and r.pending_move_end is not None:
            interior_lo = r.pending_move_start + 1
            interior_hi = r.pending_move_end
            if interior_lo < interior_hi:
                mask = (valid < interior_lo) | (valid >= interior_hi)
                valid = valid[mask]

        return valid

    def _fallback_ptr(self, r: Row, tok: int, role: str | None = None) -> int:
        # Requires we are inside an edit span
        if r.edit_start_ix is None:
            return 0

        base = getattr(r, "base_ix", 0)
        hi = r.edit_start_ix  # pointers must be < hi

        # If domain is degenerate, just return something safe
        if hi <= 0:
            return 0

        role = self._pointer_role(r, tok) if role is None else role
        if role in {"insertion_loc", "copy_dest", "move_dest"}:
            return max(min(base, hi - 1), 0)
        if hi <= 1:
            return 0
        return max(min(hi - 1, 0), 0)

    def sample_ptr(
        self,
        r: Row,
        tok: int,
        write_ix: int,
        position_logits_row,
        rng,
        temperature,
        top_k,
        top_p,
        *,
        role: str | None = None,
    ):
        # If we have no position logits (should be rare now), use safe fallback.
        if position_logits_row is None:
            return self._fallback_ptr(r, tok, role)

        device = position_logits_row.device
        valid = self.valid_ptr_range(r, tok, write_ix, device, role=role)

        # If empty, DO NOT crash in training; return safe fallback and let resolver crop.
        if valid.numel() == 0:
            return self._fallback_ptr(r, tok, role)

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
                    r.pending_copy_start = None
                    r.pending_move_start = None
                    r.pending_move_end = None
                    r.active_payload_ptr = None
                    r.payload_len = 0

            case Mode.AFTER_EDIT:
                if tok == t["DEL_OPEN"]:
                    r.mode = Mode.AFTER_DEL_OPEN
                elif tok == t["INS_OPEN"]:
                    r.mode = Mode.AFTER_INS_OPEN
                elif tok == t["REPLACE_OPEN"]:
                    r.mode = Mode.AFTER_REPLACE_OPEN
                elif tok == t["COPY_OPEN"]:
                    r.mode = Mode.AFTER_COPY_OPEN
                elif tok == t["MOVE_OPEN"]:
                    r.mode = Mode.AFTER_MOVE_OPEN
                elif tok == t["EDIT_END"]:
                    if self.max_edit_ops is not None and r.completed_ops >= self.max_edit_ops:
                        r.forced_program_closures += 1
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
                    r.completed_ops += 1
                    r.mode = Mode.AFTER_EDIT

            case Mode.AFTER_INS_OPEN:
                if tok == t["INS_LOC"]:
                    r.last_ptr = ptr
                    r.active_payload_ptr = ptr
                    r.payload_len = 0
                    r.mode = Mode.IN_INS

            case Mode.IN_INS:
                if tok not in self.indel_specials and tok not in self.pointer_tokens and tok not in self.structural_ptr:
                    r.payload_len += 1
                    r.max_payload_len_seen = max(r.max_payload_len_seen, r.payload_len)

                closes_payload = tok in {
                    t["DEL_OPEN"],
                    t["INS_OPEN"],
                    t["REPLACE_OPEN"],
                    t["COPY_OPEN"],
                    t["MOVE_OPEN"],
                    t["EDIT_END"],
                }
                if closes_payload:
                    r.completed_ops += 1

                if tok == t["DEL_OPEN"]:
                    r.mode = Mode.AFTER_DEL_OPEN
                elif tok == t["INS_OPEN"]:
                    r.mode = Mode.AFTER_INS_OPEN
                elif tok == t["REPLACE_OPEN"]:
                    r.mode = Mode.AFTER_REPLACE_OPEN
                elif tok == t["COPY_OPEN"]:
                    r.mode = Mode.AFTER_COPY_OPEN
                elif tok == t["MOVE_OPEN"]:
                    r.mode = Mode.AFTER_MOVE_OPEN
                elif tok == t["EDIT_END"]:
                    r.mode = Mode.NORMAL

            case Mode.AFTER_REPLACE_OPEN:
                if tok == t["REPLACE_START"]:
                    r.pending_del_start = ptr
                    r.last_ptr = ptr
                    r.mode = Mode.AFTER_REPLACE_START

            case Mode.AFTER_REPLACE_START:
                if tok == t["REPLACE_END"]:
                    r.last_ptr = ptr
                    r.deleted_upto = max(r.deleted_upto, ptr)
                    r.active_payload_ptr = r.pending_del_start
                    r.payload_len = 0
                    r.pending_del_start = None
                    r.mode = Mode.IN_REPLACE

            case Mode.IN_REPLACE:
                if tok not in self.indel_specials and tok not in self.pointer_tokens and tok not in self.structural_ptr:
                    r.payload_len += 1
                    r.max_payload_len_seen = max(r.max_payload_len_seen, r.payload_len)
                elif tok == t["REPLACE_EOS"]:
                    r.completed_ops += 1
                    r.active_payload_ptr = None
                    r.payload_len = 0
                    r.mode = Mode.AFTER_EDIT

            case Mode.AFTER_COPY_OPEN:
                if tok == t["COPY_START"]:
                    r.pending_copy_start = ptr
                    r.last_ptr = ptr
                    r.mode = Mode.AFTER_COPY_START

            case Mode.AFTER_COPY_START:
                if tok == t["COPY_END"]:
                    r.last_ptr = ptr
                    r.mode = Mode.AFTER_COPY_END

            case Mode.AFTER_COPY_END:
                if tok == t["COPY_DEST"]:
                    r.pending_copy_start = None
                    r.completed_ops += 1
                    r.mode = Mode.AFTER_EDIT

            case Mode.AFTER_MOVE_OPEN:
                if tok == t["MOVE_START"]:
                    r.pending_move_start = ptr
                    r.last_ptr = ptr
                    r.mode = Mode.AFTER_MOVE_START

            case Mode.AFTER_MOVE_START:
                if tok == t["MOVE_END"]:
                    r.last_ptr = ptr
                    r.pending_move_end = ptr
                    r.mode = Mode.AFTER_MOVE_END

            case Mode.AFTER_MOVE_END:
                if tok == t["MOVE_DEST"]:
                    if r.pending_move_end is not None:
                        r.deleted_upto = max(r.deleted_upto, r.pending_move_end)
                    r.pending_move_start = None
                    r.pending_move_end = None
                    r.completed_ops += 1
                    r.mode = Mode.AFTER_EDIT

    def _prime_step(self, *, i: int, tok: int, write_ix: int, ptr: int):
        r = self.rows[i]

        if tok in self.indel_specials:
            r.active_payload_ptr = None
            r.payload_len = 0

        self._apply_fsm(r, tok, write_ix, ptr)

    def step(
        self,
        i: int,
        tok: int,
        write_ix: int,
        position_logits_row,
        rng,
        temperature,
        top_k,
        top_p,
        pointer_override: int | None = None,
    ) -> int:
        r = self.rows[i]
        ptr = write_ix
        forced_payload_close = (
            self.max_payload_tokens is not None
            and r.mode in (Mode.IN_INS, Mode.IN_REPLACE)
            and r.payload_len >= self.max_payload_tokens
        )

        # pointer-bearing tokens sample a ptr from previous token's position_logits
        pointer_role = self._pointer_role(r, tok)
        if tok in self.pointer_tokens:
            ptr = (
                self.sample_ptr(
                    r,
                    tok,
                    write_ix,
                    position_logits_row,
                    rng,
                    temperature,
                    top_k,
                    top_p,
                    role=pointer_role,
                )
                if pointer_override is None
                else int(pointer_override)
            )
            if pointer_override is not None and position_logits_row is not None:
                valid = self.valid_ptr_range(
                    r,
                    tok,
                    write_ix,
                    position_logits_row.device,
                    role=pointer_role,
                )
                if not bool((valid == ptr).any().item()):
                    raise ValueError(
                        f"Pointer override {ptr} is invalid for token {tok}"
                    )

        # structural tokens always point to their stream position
        if tok in self.structural_ptr:
            ptr = write_ix

        if r.mode in (Mode.IN_INS, Mode.IN_REPLACE):
            if tok not in self.indel_specials and tok not in self.pointer_tokens and tok not in self.structural_ptr:
                if r.active_payload_ptr is not None:
                    ptr = r.active_payload_ptr

        if tok in self.indel_specials:
            r.active_payload_ptr = None
            r.payload_len = 0

        self._apply_fsm(r, tok, write_ix, ptr)
        if forced_payload_close:
            r.forced_payload_closures += 1
        return ptr


class Engine:

    def __init__(
        self, model, tokenizer, pointer_token_mode: str = "role_specific"
    ):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use
        self.edit_controller = EditController(
            tokenizer, pointer_token_mode=pointer_token_mode
        )

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        tix=None,
        num_samples=1,
        max_tokens=None,
        temperature=1.0,
        top_k=None,
        top_p=None,
        pointer_temperature=None,
        closure_bias: float = 0.0,
        closure_bias_min_payload_tokens: int = 0,
        seed=42,
        no_edits: bool = False,
        max_payload_tokens: int | None = None,
        max_edit_ops: int | None = None,
        stop_after_edit_end: bool = False,
        first_decision_edit_threshold: float | None = None,
        pointer_provider=None,
        _policy_trace_sink: list[list[PolicyActionTrace]] | None = None,
    ):
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        if pointer_temperature is None:
            pointer_temperature = temperature
        if pointer_temperature < 0:
            raise ValueError("pointer_temperature must be non-negative")
        if closure_bias_min_payload_tokens < 0:
            raise ValueError("closure_bias_min_payload_tokens must be non-negative")

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
        # Attention projections use the model's compute dtype, which can differ
        # from the device default (for example, CPU inference on a GPU host).
        dtype = COMPUTE_DTYPE
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # init controller state
        self.edit_controller.reset(
            num_samples,
            max_payload_tokens=max_payload_tokens,
            max_edit_ops=max_edit_ops,
        )
        self.edit_controller.prime_from_prompt(tokens, prompt_tix, num_samples)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row
        edit_end = self.edit_controller.tok["EDIT_END"]

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
        pos_k_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_pos_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        tix_prompt = torch.tensor([prompt_tix], dtype=torch.long, device=device)  # (1, T)

        logits, position_logits = self.model.forward(
            ids,
            tix=tix_prompt,
            kv_cache=kv_cache_prefill,
            pos_k_cache=pos_k_cache_prefill,
            return_pos_logits=True,
        )

        # seed decode from last token
        logits = logits[:, -1, :].expand(num_samples, -1)                 # (B, V)
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
        pos_k_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_pos_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
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
            self.edit_controller.apply_closure_bias_(
                logits,
                closure_bias,
                min_payload_tokens=closure_bias_min_payload_tokens,
                forced_mask=forced_mask,
            )
            if first_decision_edit_threshold is not None:
                self.edit_controller.apply_first_decision_gate_(
                    logits,
                    first_decision_edit_threshold,
                    forced_mask=forced_mask,
                )

            # Optionally ban all edit tokens (for normal autoregressive evals)
            if no_edits:
                self.edit_controller.ban_edit_tokens_(logits)

            # Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k, top_p)  # (B, 1)
            sampled_token_logprobs = (
                selected_action_logprobs(
                    logits, next_ids, temperature, top_k, top_p
                )[:, 0]
                if _policy_trace_sink is not None
                else None
            )
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            active_before_step = (
                [not state.completed for state in row_states]
                if _policy_trace_sink is not None
                else None
            )
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
                elif stop_after_edit_end and next_token == edit_end:
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

                pointer_override = None
                if pointer_provider is not None and tok in self.edit_controller.pointer_tokens:
                    pointer_override = pointer_provider(
                        i,
                        tok,
                        self.edit_controller.rows[i],
                        write_ix,
                    )
                valid_pointer_indices: tuple[int, ...] = ()
                pointer_was_sampled = False
                if (
                    _policy_trace_sink is not None
                    and tok in self.edit_controller.pointer_tokens
                    and position_logits is not None
                ):
                    valid = self.edit_controller.valid_ptr_range(
                        self.edit_controller.rows[i], tok, write_ix, position_logits[i].device
                    )
                    valid_pointer_indices = tuple(int(value) for value in valid.tolist())
                    pointer_was_sampled = pointer_override is None and valid.numel() > 0
                ptr = self.edit_controller.step(
                    i=i,
                    tok=tok,
                    write_ix=write_ix,
                    position_logits_row=position_logits[i],  # now always available
                    rng=rng,
                    temperature=pointer_temperature,
                    top_k=top_k,
                    top_p=top_p,
                    pointer_override=pointer_override,
                )
                tix_next[i] = ptr
                pointer_column[i] = int(ptr)
                pointer_logprob = None
                if pointer_was_sampled:
                    masked_pointer_logits = torch.full_like(position_logits[i], float("-inf"))
                    masked_pointer_logits[list(valid_pointer_indices)] = position_logits[i, list(valid_pointer_indices)]
                    pointer_logprob = float(selected_action_logprobs(
                        masked_pointer_logits[None, :],
                        torch.tensor([[ptr]], dtype=torch.long, device=device),
                        pointer_temperature,
                        top_k,
                        top_p,
                    )[0, 0].item())
                if (
                    _policy_trace_sink is not None
                    and active_before_step is not None
                    and active_before_step[i]
                ):
                    assert sampled_token_logprobs is not None
                    token_was_sampled = bool(token_masks[i])
                    _policy_trace_sink[i].append(PolicyActionTrace(
                        absolute_index=write_ix,
                        token_id=int(tok),
                        pointer=int(ptr),
                        token_sampled=token_was_sampled,
                        pointer_sampled=pointer_was_sampled,
                        token_logprob=(
                            float(sampled_token_logprobs[i].item())
                            if token_was_sampled else None
                        ),
                        pointer_logprob=pointer_logprob,
                        valid_pointer_indices=valid_pointer_indices,
                    ))

            if write_ix >= tix_hist.size(1):
                raise RuntimeError(f"tix_hist too small: write_ix={write_ix}, hist_len={tix_hist.size(1)}")

            tix_hist[:, write_ix] = tix_next
            yield token_column, token_masks, pointer_column
            num_generated += 1

            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            tix_step = tix_next[:, None]
            logits, position_logits = self.model.forward(ids, tix=tix_step, kv_cache=kv_cache_decode, pos_k_cache=pos_k_cache_decode, return_pos_logits=True)
            logits = logits[:, -1, :]
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
        edit_end = self.edit_controller.tok["EDIT_END"]
        stop_after_edit_end = bool(kwargs.get("stop_after_edit_end", False))

        results  = [tokens.copy() for _ in range(num_samples)]
        masks    = [[0] * len(tokens) for _ in range(num_samples)]
        if tix is None:
            pointers = [list(range(len(tokens))) for _ in range(num_samples)]
        else:
            pointers = [list(tix) for _ in range(num_samples)]

        completed = [False] * num_samples

        for token_column, token_masks, pointer_column in self.generate(
            tokens, tix=tix, num_samples=num_samples, no_edits=no_edits, **kwargs
        ):
            for i, (token, mask, ptr) in enumerate(zip(token_column, token_masks, pointer_column)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
                        pointers[i].append(ptr)
                        if stop_after_edit_end and token == edit_end:
                            completed[i] = True

            if all(completed):
                break

        self.last_generation_diagnostics = [
            {
                "final_mode": row.mode.name,
                "payload_len": row.payload_len,
                "max_payload_len_seen": row.max_payload_len_seen,
                "completed_ops": row.completed_ops,
                "forced_payload_closures": row.forced_payload_closures,
                "forced_program_closures": row.forced_program_closures,
                "first_decision_edit_margin": row.first_decision_edit_margin,
                "hierarchical_edit_selected": row.hierarchical_edit_selected,
            }
            for row in self.edit_controller.rows
        ]
        return results, masks, pointers

    def generate_rollout_batch(self, tokens, num_samples=1, tix=None, no_edits: bool = False, **kwargs):
        """Generate CES samples plus policy traces for on-policy RL.

        The legacy ``generate_batch`` API is deliberately unchanged. Terminal
        assistant/BOS actions, which it omits from returned sequences, are also
        omitted from the corresponding rollout trace.
        """
        temperature = float(kwargs.get("temperature", 1.0))
        pointer_temperature_arg = kwargs.get("pointer_temperature")
        pointer_temperature = (
            temperature if pointer_temperature_arg is None else float(pointer_temperature_arg)
        )
        trace_sink: list[list[PolicyActionTrace]] = [[] for _ in range(num_samples)]
        results, masks, pointers = self.generate_batch(
            tokens,
            num_samples=num_samples,
            tix=tix,
            no_edits=no_edits,
            _policy_trace_sink=trace_sink,
            **kwargs,
        )
        prompt_length = len(tokens)
        rollouts = []
        for row_index, (row_tokens, row_pointers) in enumerate(zip(results, pointers)):
            kept_actions = tuple(
                action for action in trace_sink[row_index]
                if action.absolute_index < len(row_tokens)
            )
            rollouts.append(PolicyRollout(
                prompt_length=prompt_length,
                tokens=tuple(int(token) for token in row_tokens),
                pointers=tuple(int(pointer) for pointer in row_pointers),
                actions=kept_actions,
                temperature=temperature,
                pointer_temperature=pointer_temperature,
                top_k=kwargs.get("top_k"),
                top_p=kwargs.get("top_p"),
                closure_bias=float(kwargs.get("closure_bias", 0.0)),
                closure_bias_min_payload_tokens=int(
                    kwargs.get("closure_bias_min_payload_tokens", 0)
                ),
                no_edits=bool(no_edits),
                max_payload_tokens=kwargs.get("max_payload_tokens"),
                max_edit_ops=kwargs.get("max_edit_ops"),
            ))
        return rollouts, masks



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
