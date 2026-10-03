"""Insertion/deletion serialization and shared corruption utilities.

This module implements the two-operation insertion/deletion format used by the
legacy loader and decoder. The five-operation CES format used by the paper is
implemented in ``causal_serialisation_extended.py``. Their edit coordinates,
training-target types and special-token vocabularies are not interchangeable.

Edit representation
-------------------
``SequenceEdit`` contains a source sequence, deletions represented as
``(start, length)`` pairs in that source, and ``(index, payload)`` insertions.
Insertion indices are boundaries in the sequence *after deletion*: applying an
insertion at index ``i`` places its payload before the surviving element at
``i``. ``complement`` applies the edits and constructs their inverse;
``resolve`` returns the resulting sequence.

Serialized representation
-------------------------
``TokensOperations.serialise`` inserts an ``INDL_BOS`` marker into the source
(at index zero by default), appends ``EDIT_START``, orders edit blocks by source
location, and terminates with ``EDIT_END``. The block grammar is:

    insertion: ins-open, ins-loc, payload...
    deletion:  del-open, del-start, del-end

An insertion payload ends at the next edit opener or ``EDIT_END``. Parallel
``tokens`` and ``positions`` arrays carry the token IDs and source addresses.
Deletion endpoints address the first and last deleted tokens (inclusive);
insertion addresses identify the preceding source anchor, with ``INDL_BOS``
providing the initial anchor. Payload tokens inherit the insertion address;
source and control tokens otherwise use their serialized indices.

The stream index and address index are inputs to dual RoPE in ``gpt.py``.
They do not specify a sparse attention mask. ``prepare_training_example``
shifts tokens for next-token prediction and gathers address targets at the
preceding hidden state, with masks for token loss and padded pointer slots.

Shared utilities
----------------
The current CES data pipeline also imports ``ChunkReservoir``,
``InsertTypeProbs``, ``sample_insert`` and ``gamma_shaped_partition`` from this
module. Other consumers include the iterative-editor and RL pair loaders.
These dependencies are why this module remains alongside the extended
serializer. The dataframe/printing helpers support inspection of serialized
examples; the functions at the end of the file are standalone diagnostics.
"""
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from typing import Callable, Deque, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tiktoken

Index = int
Length = int
Sequence = list[int | str] | np.ndarray
InsertionList = list[tuple[Index, Sequence]]  # The insertion index is *post-deletion*.
DenseDeletionMask = np.ndarray[bool]
SparseDeletionMask = list[tuple[Index, Length]] | np.ndarray
Positions = np.ndarray
RNG = Union[np.random.Generator, np.random.RandomState]


DEL_OPEN = "<|del-open|>"
DEL_START = "<|del-start|>"
DEL_END = "<|del-end|>"
INS_OPEN = "<|ins-open|>"
INS_LOC = "<|ins-loc|>"
EDIT_START = "<|edit-start|>"
EDIT_END = "<|edit-end|>"
INDL_BOS = "<|indel-bos|>"

NUM_EDIT_TOKENS = 2

POSTION_TOKENS = [DEL_START, DEL_END, INS_LOC]
INDEL_SPECIALS = [
    DEL_OPEN,
    DEL_START,
    DEL_END,
    INS_OPEN,
    INS_LOC,
    EDIT_START,
    EDIT_END,
    INDL_BOS,
]


STYLE = {
    DEL_OPEN: "red",
    DEL_START: "bold #720202",
    DEL_END: "bold #BB1A28",
    INS_OPEN: "green",
    INS_LOC: "green bold",
    EDIT_START: "yellow",
    EDIT_END: "yellow",
    INDL_BOS: "blue",
}

########################################################################################
####                                      TYPE                                      ####
########################################################################################


class SequenceEdit(NamedTuple):
    sequence: Sequence
    deletions: SparseDeletionMask  # Corresponds to the sequence length above.
    insertions: InsertionList  # The insertion index is *post-deletion*.


def _alternating_green_colormap(n, hue=120, sat=0.65, lmin=0.25, lmax=0.75):
    """Continuous green ramp with alternating ordering: 0, 0.5, 1/n, 0.5+1/n, 0.5-1/n, ...
    Returns a list of HTML hex colours.
    """
    import colorsys

    step = (1 / n) if n > 0 else 0
    ts = [0.0, 0.5]

    k = 1
    while len(ts) < n:
        for t in (k*step, 0.5 + k*step, 0.5 - k*step):
            if 0 <= t <= 1:
                ts.append(t)
                if len(ts) == n:
                    break
        k += 1

    def hsl_hex(t):
        l = lmin + t * (lmax - lmin)
        r, g, b = colorsys.hls_to_rgb(hue/360, l, sat)
        return "#{:02X}{:02X}{:02X}".format(
            int(r*255), int(g*255), int(b*255)
        )

    return [hsl_hex(t) for t in ts[:n]]


class SerialisedData(NamedTuple):
    tokens: Sequence
    positions: Positions

    def print(self, tokeniser) -> None:
        import shutil

        from rich.console import Console
        from rich.text import Text


        toks = tokeniser.decode_batch([[t] for t in self.tokens])

        del_mask = np.zeros(np.where(np.array(toks) == EDIT_START)[0][0], dtype=bool)
        for ds, de in zip(
            self.positions[np.array(toks) == DEL_START],
            self.positions[np.array(toks) == DEL_END],
        ):
            del_mask[ds:de+1] = True
        del_ixs, = np.where(del_mask)

        ins_loc = self.positions[np.array(toks) == INS_LOC]
        index_to_colour = dict(zip(ins_loc, _alternating_green_colormap(len(ins_loc))))
        for i in del_ixs:
            index_to_colour[i] = "red"

        colours = []
        for j, (t, i) in enumerate(zip(toks, self.positions)):
            c = index_to_colour.get(i, None)
            if c is None:
                c = STYLE.get(t, None)
            colours.append(c)

        console = Console()
        columns = shutil.get_terminal_size().columns

        def is_newline_token(t: str) -> bool:
            return "\n" in t  # catches '\n' or tokens that include newlines

        def visible_token(t: str) -> str:
            # keep it single-line in the terminal
            return t.replace("\n", "↵")

        def token_width(t: str) -> int:
            # width you want to budget for the token in the row
            return len(visible_token(t))

        def chunk_by_columns_and_newlines(toks, columns):
            """
            Yield (start, end) ranges where:
            - total visible width <= columns
            - newline tokens force a break (the newline token is included at end of the chunk)
            """
            start = 0
            acc = 0
            for k, t in enumerate(toks):
                w = token_width(t)

                # If adding this token would exceed width, break before it.
                if acc + w > columns and k > start:
                    yield start, k
                    start = k
                    acc = 0

                acc += w

                # Hard break after any newline-containing token (include it in this chunk).
                if is_newline_token(t):
                    yield start, k + 1
                    start = k + 1
                    acc = 0

            if start < len(toks):
                yield start, len(toks)

        def build_ix_text(pos_chunk, tok_chunk, col_chunk):
            out = Text()
            for i, t, c in zip(pos_chunk, tok_chunk, col_chunk):
                t_vis = visible_token(t)
                cell = str(i).rjust(len(t_vis)) if len(str(i)) < len(t_vis) else ".".rjust(len(t_vis))
                out.append(cell, style=c)
            return out

        def build_tok_text(pos_chunk, tok_chunk, col_chunk, global_offset):
            out = Text()
            for j, (i, t, c) in enumerate(zip(pos_chunk, tok_chunk, col_chunk), start=global_offset):
                t_vis = visible_token(t)
                underline = (j % 2 == 1)  # flip to == 0 if you want first underlined
                style_parts = [c] if c is not None else []
                if underline:
                    style_parts.append("underline")
                out.append(t_vis, style=" ".join(style_parts) if style_parts else None)
            return out

        for start, end in chunk_by_columns_and_newlines(toks, columns):
            col_chunk = colours[start:end]
            tok_chunk = toks[start:end]
            pos_chunk = self.positions[start:end]

            print()
            console.print(
                build_ix_text(pos_chunk, tok_chunk, col_chunk),
                overflow="ignore",
                no_wrap=True,
            )
            console.print(
                build_tok_text(pos_chunk, tok_chunk, col_chunk, global_offset=start),
                overflow="ignore",
                no_wrap=True,
            )


class TrainingTarget(NamedTuple):
    tokens: Sequence
    gathered_pos_target: np.ndarray
    token_loss_weights: np.ndarray
    gather_indices: np.ndarray
    gather_valid: np.ndarray
    overflow: bool
    perplexity_masks: dict[str, np.ndarray] | None = None


class TrainingData(NamedTuple):
    input_tokens: np.ndarray
    input_positions: np.ndarray
    target: TrainingTarget

    def df(self, tokeniser, index: int | None = None):
        slc = slice(None) if index is None else index
        ix = self.target.gather_indices[slc][self.target.gather_valid[slc]]
        tg = self.target.gathered_pos_target[slc][self.target.gather_valid[slc]]
        ref = -np.ones(self.input_tokens[slc].shape, dtype=np.int32)
        ref[ix] = tg
        items = (
            self.input_tokens[slc],
            self.input_positions[slc],
            self.target.tokens[slc],
            self.target.token_loss_weights[slc],
            ref,
        )
        pms = self.target.perplexity_masks
        if pms is not None:
            pms = {k: np.array(v)[slc] for k, v in pms.items()}
            if not (sum(pms.values()) <= 1).all():
                print("Warning: this is not a partition - the latest masks take priority.")
            keys = list(pms.keys())
            naming_map = {
                i: k.replace("-", " ").split(" ")[0][:3]
                for i, k in enumerate(keys, start=1)
            }
            pm = np.zeros(pms[keys[0]].shape, dtype=np.int32)
            for i, m in enumerate(keys, start=1):
                pm[pms[m]] = i
            items = items + (pm,)
        rows = []
        for item in zip(*items):
            t, p, tr, w, r, *pm = item
            row = {
                "in-tok": tokeniser.decode([t]).replace("\n", "↵"),
                "in-pos": int(p),
                "targ": tokeniser.decode([tr]).replace("\n", "↵"),
                "ref": str(r) if r >= 0 else '',
                "mask": int(w),
            }
            if pm:
                row["reg"] = naming_map.get(pm[0], "")
            rows.append(row)
        return pd.DataFrame(rows)


def print_df_grid_rich(dfs, styles_list=None, *, max_rows=30, pad=1, max_col_w=20):
    import shutil

    from rich.console import Console
    from rich.text import Text

    console = Console()
    columns = shutil.get_terminal_size().columns


    def _clip_pad(s: str, w: int) -> str:
        s = "" if s is None else str(s)
        if len(s) > w:
            return (s[: max(0, w - 1)] + "…") if w > 1 else s[:w]
        return s.ljust(w)


    def df_to_rich_lines(df, styles=None, *, max_rows=30, max_col_w=20, header_style=None):
        df = df.iloc[:max_rows]
        styles = styles.reindex_like(df) if styles is not None else None

        cols = list(df.columns)
        idx = df.index.astype(str).tolist()

        # column widths from header+cells (+index)
        idx_w = min(max(len(x) for x in ([""] + idx)), max_col_w)
        col_w = []
        for c in cols:
            w = max(len(str(c)), *(len(str(v)) for v in df[c].tolist()))
            col_w.append(min(w, max_col_w))

        lines = []

        # header line
        h = Text(_clip_pad("", idx_w), style=header_style)
        for c, w in zip(cols, col_w):
            h.append(" ")
            h.append(_clip_pad(str(c), w), style=header_style)
        lines.append(h)

        # rows
        for r_i, (i, row) in enumerate(df.iterrows()):
            t = Text(_clip_pad(str(i), idx_w), style=header_style)
            for c, w in zip(cols, col_w):
                t.append(" ")
                cell = _clip_pad(row[c], w)
                st = "" if styles is None else (styles.at[i, c] or "")
                t.append(cell, style=st)
            lines.append(t)

        return lines  # list[Text]


    # render each df -> list[Text]
    cells = []
    for k, df in enumerate(dfs):
        st = None if styles_list is None else styles_list[k]
        lines = [Text(f"df[{k}]", style="bold")] + df_to_rich_lines(df, st, max_rows=max_rows, max_col_w=max_col_w)
        cells.append(lines)

    # decide a cell width (plain length) + how many fit
    cellw = min(max(max(len(line.plain) for line in cell) for cell in cells), columns - 2)
    ncols = max(1, columns // (cellw + 2 * pad + 1))
    ncols = min(ncols, len(cells))
    hline = Text("-" * columns)

    for r in range(0, len(cells), ncols):
        row = cells[r:r + ncols]
        h = max(len(c) for c in row)
        row = [c + [Text("")] * (h - len(c)) for c in row]

        for i in range(h):
            out = Text("|")
            for j, panel in enumerate(row):
                seg = panel[i]
                out.append(" " * pad)
                out.append(_clip_pad(seg.plain, cellw), style=None)  # base padding/clip
                # overlay styles by reusing the original styled segment (truncated) if it fits
                # (simple approach: replace the just-added plain text with the styled text if no truncation)
                if len(seg.plain) <= cellw:
                    out = out[:-cellw]  # remove the plain version we just appended
                    out.append(Text(seg.plain.ljust(cellw), style=seg.style, spans=seg.spans))
                out.append(" " * pad)
                out.append("|")
            console.print(out, overflow="ignore", no_wrap=True)

        console.print(hline, overflow="ignore", no_wrap=True)


def colour_df(df):
    sdf = pd.DataFrame(df)
    tok = df["in-tok"].to_numpy()
    pos = df["in-pos"].to_numpy()
    del_start = pos[tok == DEL_START]
    del_end = pos[tok == DEL_END]
    del_mask = np.zeros(len(tok), dtype=bool)
    for ds, de in zip(del_start, del_end):
        del_mask[ds:de+1] = True
    arng = np.arange(len(tok))
    ins_loc = pos[tok == INS_LOC]
    index_to_colour = dict(zip(ins_loc, _alternating_green_colormap(len(ins_loc))))
    in_tok = sdf["in-tok"].apply(lambda x: STYLE.get(x, None)).to_numpy()
    in_pos = sdf["in-pos"].apply(lambda x: index_to_colour.get(x, None))
    for i in ins_loc:
        in_tok[i] = index_to_colour[i]

    for i, t, p, dm in zip(arng, tok, sdf["in-pos"].to_numpy(), del_mask):
        in_pos[i] = STYLE.get(t, None)
        if p in ins_loc:
            in_tok[i] = index_to_colour[p]
            in_pos[i] = index_to_colour[p]
        if dm:
            in_tok[i] = STYLE[DEL_START]
            in_pos[i] = STYLE[DEL_START]
    sdf["in-tok"] = in_tok
    sdf["in-pos"] = in_pos
    sdf.targ = sdf.targ.apply(lambda x: None)
    sdf["mask"] = sdf["mask"].apply(lambda x: None if x else "#8B50EB")
    sdf["ref"] = sdf["ref"].apply(lambda x: "#EB50B7" if len(x) else None)
    return sdf


def print_training_examples(training_examples, tokeniser, max_rows=1025):
    dfs = [da.df(tokeniser) for da in training_examples]
    style_dfs = [colour_df(df) for df in dfs]
    print_df_grid_rich(dfs, style_dfs, max_rows=max_rows)


class DeletionMask:
    @classmethod
    def to_sparse(cls, dense_mask: DenseDeletionMask) -> SparseDeletionMask:
        """Find contiguous stretches of True values in *deletion_mask*.

        Args:
            dense_mask : array-like of bool
                1-D mask to analyse.

        Returns:
            list[tuple[int, int]]
                return (start, length) in the index space that remains after all earlier
                True elements have been deleted.
        """
        # ---- standard vectorised run detection ---------------------------------
        starts = np.flatnonzero(dense_mask & np.r_[True,  ~dense_mask[:-1]])
        ends   = np.flatnonzero(dense_mask & np.r_[~dense_mask[1:], True])
        lengths = ends - starts + 1                     # run lengths
        return np.stack([starts, lengths], axis=-1)

    @classmethod
    def to_dense(
        cls, sparse_deletion_mask: SparseDeletionMask, full_length: int
    ) -> DenseDeletionMask:
        mask = np.zeros(full_length, dtype=bool)
        for start, length in sparse_deletion_mask:
            mask[start:start+length] = True
        return mask


########################################################################################
####                                   OPERATIONS                                   ####
########################################################################################

def post_deletion_index(
    indices: np.ndarray, deletion_mask: SparseDeletionMask | DenseDeletionMask
) -> np.ndarray:
    """Map deleted-run starts to insertion boundaries after deletion.

    The implemented branch expects a dense boolean deletion mask. The sparse
    branch is unimplemented; callers must convert sparse masks first.
    """
    sparse = deletion_mask.ndim == 2
    if sparse:
        pass
    else:
        # Number of deleted (True) elements that appear *before* each run start
        # subtract 1 for the start itself
        prior_deleted = np.cumsum(deletion_mask)[indices] - 1
        return indices - prior_deleted


def indices_pre_deletion(
    deletion_mask: SparseDeletionMask | DenseDeletionMask
) -> np.ndarray:
    sparse = deletion_mask.ndim == 2
    if sparse:
        pass
    else:
        return np.arange(len(deletion_mask))[~deletion_mask]


def complement(seq_edit: SequenceEdit) -> SequenceEdit:
    """Take some data tokens and insert and delete positions.

    Args:
        sequence (np.ndarray): The raw data tokens.
        insertions (InsertionList): The tokens to insert at indices *post-deletion*.
        deletion_mask (DeletionMask): The tokens to delete.

    Returns:
        Tuple:
            - (np.ndarray): Tokens after applying the insertions and deletions.
            - (InsertionList): The insertions needed to invert the output string back
                to the input string.
            - (DeletionMask): The deletions needed to invert the output string back
                to the input string.
    """
    sequence = seq_edit.sequence

    # Sparse form of the deletion mask.
    if len(seq_edit.deletions):
        starts, lengths = np.array(seq_edit.deletions, dtype=np.int32).T
    else:
        starts, lengths = [], []

    # Dense form of the deletion mask.
    deletion_mask = DeletionMask.to_dense(seq_edit.deletions, full_length=len(sequence))

    # Our convention is to apply deletions before insertions.
    # The insertion index space is the canonical one, defined *post-deletion*
    reverse_insertions = [
        (int(post_ix), sequence[pre_ix:pre_ix+size])
        for post_ix, pre_ix, size in zip(
            post_deletion_index(starts, deletion_mask), starts, lengths
        )
    ]
    post_deletion_sequence = sequence[~deletion_mask]
    inserted_sequence = list(post_deletion_sequence)

    reverse_deletion_mask = [False] * len(inserted_sequence)
    for i, insert in reversed(seq_edit.insertions):
        for token in reversed(insert):
            inserted_sequence.insert(i, token)
            reverse_deletion_mask.insert(i, True)

    return SequenceEdit(
        sequence=np.array(inserted_sequence),
        deletions=DeletionMask.to_sparse(np.array(reverse_deletion_mask, dtype=bool)),
        insertions=reverse_insertions,
    )


def resolve(seq_edit: SequenceEdit) -> Sequence:
    return complement(seq_edit=seq_edit).sequence


class TokensOperations:
    def __init__(
        self,
        tokeniser,
        marginal_token_probs: np.ndarray | None = None,
        marginal_tokens: np.ndarray | None = None,
    ):
        self.tokeniser = tokeniser
        num_standard_tokens = tokeniser.max_token_value - len(tokeniser._special_tokens)
        max_standard = min(tokeniser._special_tokens.values()) - 1
        if max_standard != num_standard_tokens:
            print(
                "Warning. The tokeniser has not got standard tokens before special "
                "tokens, this means inefficient uniform token sampling is required."
            )
            special_tokens = set(tokeniser._special_tokens.values())
            tokens = np.array(
                list(set(range(tokeniser.max_token_value + 1) - special_tokens))
            )
            sample_uniform_random_tokens = lambda length: np.random.choice(
                tokens,
                size=length,
                replace=True,
            )
        else:
            sample_uniform_random_tokens = lambda length: np.random.randint(
                0, num_standard_tokens, length
            )
        if marginal_token_probs is None:
            sample_marginal_tokens = sample_uniform_random_tokens
        else:
            if marginal_tokens is None:
                tokens = num_standard_tokens
            else:
                tokens = marginal_tokens
            sample_marginal_tokens = lambda length: np.random.choice(
                tokens,
                p=marginal_token_probs,
                size=length,
                replace=True,
            )
        self.sample_marginal_tokens = sample_marginal_tokens
        self.sample_uniform_random_tokens = sample_uniform_random_tokens
        self.deletion_open_token = tokeniser._special_tokens[DEL_OPEN]
        self.deletion_start_token = tokeniser._special_tokens[DEL_START]
        self.deletion_end_token = tokeniser._special_tokens[DEL_END]
        self.insertion_open_token = tokeniser._special_tokens[INS_OPEN]
        self.insertion_loc_token = tokeniser._special_tokens[INS_LOC]
        self.edit_start_token = tokeniser._special_tokens[EDIT_START]
        self.edit_end_token = tokeniser._special_tokens[EDIT_END]
        self.indel_bos_token = tokeniser._special_tokens[INDL_BOS]
        self.eos_index = None  # No separate EOS token before EDIT_START.

        self.special_tokens = np.array([
            self.deletion_open_token,
            self.deletion_start_token,
            self.deletion_end_token,
            self.insertion_open_token,
            self.insertion_loc_token,
            self.edit_start_token,
            self.edit_end_token,
            self.indel_bos_token,
        ])

        self.deletion_cycle = np.array([
            self.deletion_open_token,
            self.deletion_start_token,
            self.deletion_end_token,
        ])

    def deserialise(self, serialised_data: SerialisedData) -> SequenceEdit:
        """Deserialise into the sequence and the insertions and deletions to be applied.

        Args:
            serialised_data (SerialisedData): Tokens and positions.

        Raises:
            ValueError: If there are multiple edit regions in the sequence.
            ValueError: If the edit start is after the edit end.

        Returns:
            SequenceEdit: Sequence with the insertions and deletions to be applied.
        """
        sequence = serialised_data.tokens
        positions = serialised_data.positions
        start, = np.where(np.array(sequence) == self.edit_start_token)
        if len(start) == 0:
            print("No edits found.")
            return SequenceEdit(sequence, [], [])
        if len(start) > 1:
            raise ValueError(f"Too many edit start tokens, positions: {start}")
        end, = np.where(np.array(sequence) == self.edit_end_token)
        if len(end) != 1:
            raise ValueError(f"Invalid number of edit end tokens, positions: {end}")
        # Unpack
        [start] = start
        [end] = end
        if end <= start:
            raise ValueError(f"end <= start: {end} <= {start}")

        standard_sequence = sequence[:start]
        edits = sequence[start+1:end]
        if len(sequence) == len(positions):
            edit_positions = positions[start+1:end]
        else:
            if len(edits) != positions:
                raise ValueError(
                    "Incompatible positions size: match edit tokens when != sequence\n"
                    f"{len(edits)=}  {len(sequence)}  {len(positions)=}"
                )
            edit_positions = positions

        # Split into deletion edits and insertion edits.
        _del_mask = (edits[:, None] == self.deletion_cycle[None, :]).any(axis=-1)
        del_edits = edits[_del_mask]
        del_pos = edit_positions[_del_mask]
        ins_edits = edits[~_del_mask]
        ins_pos = edit_positions[~_del_mask]

        def parse_deletion(tokens, positions):
            valid = (self.deletion_cycle == tokens).all()
            start = positions[1]
            end = positions[2]
            return valid, start, end

        # Parse all the deletions.
        deletions = []
        i = 0
        for _ in range((len(del_edits) // len(self.deletion_cycle)) + 1):
            slc = slice(i, i + len(self.deletion_cycle))
            if len(del_edits[slc]) < len(self.deletion_cycle):
                break
            valid, start, end = parse_deletion(del_edits[slc], del_pos[slc])
            if valid:
                deletions.append((start, end - start + 1))
                i += len(self.deletion_cycle)
            else:
                remaining = len(del_edits) - i
                for _ in range(remaining):
                    i += 1
                    if del_edits[i] == self.deletion_open_token:
                        break

        post_deletion = post_deletion_index(
            np.arange(len(standard_sequence)),
            deletion_mask=DeletionMask.to_dense(
                deletions, full_length=len(standard_sequence)
            ),
        )

        # Parse the insertions, guarded by the special tokens ins-open, ins-loc.
        def parse_ins(tokens, positions):
            # This first token should be by definition ins-loc
            assert tokens[0] == self.insertion_loc_token
            pos = positions[0]  # Pre-deletion.
            pos = post_deletion[pos]  # Post-deletion.
            insert = tokens[1:]
            return pos, insert

        # (1) Partition the sequence by `insertion_open_token`
        ins_ix, = np.where(ins_edits == self.insertion_open_token)
        # (2) Parse each insertion segment.
        insertions = []
        for i,j in zip(ins_ix, np.concatenate([ins_ix[1:],np.array([len(ins_edits)])])):
            insertions.append(parse_ins(ins_edits[i+1:j], ins_pos[i+1:j]))

        # Remove indel-bos
        ibos_mask = standard_sequence == self.indel_bos_token
        standard_sequence = standard_sequence[~ibos_mask]
        deletions = np.array([(i - 1, size) for i, size in deletions], dtype=np.int32)
        insertions = [(i - 1, ins) for i, ins in insertions]
        if ibos_mask.sum() != 1:
            raise ValueError(
                f"There should be exactly one indel-bos token, found: {ibos_mask.sum()}"
            )

        return SequenceEdit(
            sequence=standard_sequence,
            deletions=deletions if len(deletions) else np.zeros((0, 2), dtype=np.int32),
            insertions=insertions,
        )

    def serialise(
        self, sequence_edit: SequenceEdit, indel_bos_pos: int = 0
    ) -> SerialisedData:
        """Append source-ordered edit blocks to the marked source sequence.

        ``sequence_edit`` uses source-relative deletion spans and post-deletion
        insertion indices. ``indel_bos_pos`` selects the source index at which
        the boundary marker is inserted; edits before that index are rejected.
        The default is zero. Returned positions contain concrete addresses or
        stream indices, never null values. See the module docstring for the
        serialized grammar and address conventions.
        """
        if any(i < indel_bos_pos for i, _ in sequence_edit.deletions):
            raise ValueError(
                f"Incompatible deletion index found for {indel_bos_pos=}\n"
                f"{sequence_edit.deletions}"
            )
        if any(i < indel_bos_pos for i, _ in sequence_edit.insertions):
            raise ValueError(
                f"Incompatible insertion index found for {indel_bos_pos=}\n"
                f"{sequence_edit.insertions}"
            )
        sequence = np.concatenate([
            sequence_edit.sequence[:indel_bos_pos],
            np.array([self.indel_bos_token]),
            sequence_edit.sequence[indel_bos_pos:],
        ])
        dense_dels = DeletionMask.to_dense(sequence_edit.deletions, len(sequence_edit.sequence))
        deletion_mask = np.concatenate(
            [dense_dels[:indel_bos_pos], np.array([False]), dense_dels[indel_bos_pos:]]
        )
        insertions = sequence_edit.insertions
        pre_del_indices = indices_pre_deletion(deletion_mask)

        #
        eos = self.eos_index is not None
        extra = 1 + int(eos)

        # Organise deletions.
        _deletions = []
        for position, length in sequence_edit.deletions:
            pos = np.array([-1, position + 1, position + length])
            _deletions.append((position, self.deletion_cycle, pos))

        # Organise insertions.
        _insertions = []
        head = [self.insertion_open_token, self.insertion_loc_token]
        for _i, toks in insertions:
            i = pre_del_indices[_i]
            pos = np.array([-1, i] + [i] * len(toks))
            _insertions.append((i - 1, np.concatenate([head, toks]), pos))

        # Order insertions / deletions chronologically.
        edits = _deletions + _insertions
        edits.sort(key=lambda x: x[0])

        # Serialise the edits.
        seq = [
            sequence,
            np.array([self.eos_index] if eos else [], dtype=sequence.dtype),
            np.array([self.edit_start_token], dtype=sequence.dtype),
        ]
        pos = [np.arange(len(sequence) + extra, dtype=np.int32)]  # Next position pointer
        for _, tok, pt in edits:
            seq.append(tok.astype(dtype=sequence.dtype))
            pos.append(pt.astype(dtype=np.int32))
        seq.append(np.array([self.edit_end_token], dtype=sequence.dtype))
        pos.append(np.array([-1], dtype=np.int32))

        positions = np.concatenate(pos)
        positions[positions == -1] = np.arange(len(positions))[positions == -1]
        return SerialisedData(tokens=np.concatenate(seq), positions=positions)

    def resolve(self, serialised_data: SerialisedData) -> Sequence:
        return resolve(self.deserialise(serialised_data))

    def prepare_training_example(self, serialised: SerialisedData, pos_buffer: int):
        tok = serialised.tokens
        tix = serialised.positions

        # Tokens whose pointer (tix) we want to predict one step *before* they appear.
        position_tokens = np.array([
            self.deletion_start_token,
            self.deletion_end_token,
            self.insertion_loc_token,
        ])

        # Pointer-token positions in the sequence (these are the positions whose tix we want as targets)
        ptr_mask = np.isin(tok, position_tokens)
        ptr_indices, = np.where(ptr_mask)  # positions of INS_LOC / DEL_START / DEL_END
        overflow = len(ptr_indices) > pos_buffer

        # Query positions: one step earlier (these are the positions whose activations we gather)
        query_indices = ptr_indices - 1

        # Safety: drop any that would go negative
        ok = query_indices >= 0
        ptr_indices = ptr_indices[ok]
        query_indices = query_indices[ok]

        G = min(len(query_indices), pos_buffer)

        # Gather indices tell the model which hidden states to use as queries (within x for input_tokens=tok[:-1])
        gather_indices = np.zeros(pos_buffer, dtype=query_indices.dtype)
        gather_indices[:G] = query_indices[:G]

        # Valid flags for the gathered slots (so padding slots are ignored)
        gather_valid = np.arange(pos_buffer) < len(query_indices)

        # Targets: pointer values for the *pointer tokens themselves*
        # Fill unused buffer slots with -1 so CE(ignore_index=-1) cleanly ignores them.
        gathered_pos_target = np.full(pos_buffer, -1, dtype=tix.dtype)
        gathered_pos_target[:G] = tix[ptr_indices[:G]]

        # Mask token targets affected by deleted spans and the boundary marker.
        token_loss_weights = np.ones(len(tok), dtype=bool)

        for ds, de in zip(
            tix[tok == self.deletion_start_token],
            tix[tok == self.deletion_end_token],
        ):
            # +2 so that we don't train on deleted sections - or the token after it.
            token_loss_weights[ds:de + 2] = False

        # Also, don't train on the token after an insertion point.
        for il in tix[tok == self.insertion_loc_token]:
            if il + 1 < len(token_loss_weights):
                token_loss_weights[il + 1] = False

        # Pointer/control tokens are structurally forced by the edit format; the
        # position head carries their source-coordinate supervision.
        token_loss_weights[ptr_indices] = False

        token_loss_weights[tok == self.indel_bos_token] = False

        return TrainingData(
            input_tokens=tok[:-1],
            input_positions=tix[:-1],
            target=TrainingTarget(
                tokens=tok[1:],
                gathered_pos_target=gathered_pos_target,
                token_loss_weights=token_loss_weights[1:],  # align with tokens=tok[1:]
                gather_indices=gather_indices,
                gather_valid=gather_valid,
                overflow=overflow,
            ),
        )



def get_specials(specials: dict | None = None, max_token_value: int = -1):
    new_tokens = [
        DEL_OPEN,
        DEL_START,
        DEL_END,
        INS_OPEN,
        INS_LOC,
        EDIT_START,
        EDIT_END,
        INDL_BOS
    ]
    # create an extended special token dict
    if specials is None:
        specials = {}
    else:
        specials = dict(specials)
    for i, t in enumerate(new_tokens, start=1):
        specials[t] = max_token_value + i
    return specials


def get_tokeniser():
    import os
    import pickle

    from nanochat.common import get_base_dir

    # load tokenizer
    with open(os.path.join(get_base_dir(), "tokenizer.pkl"), "rb") as f:
        tokeniser = pickle.load(f)

    # build a new encoding with same core BPE + regex pattern, but new specials
    return tiktoken.Encoding(
        name=tokeniser.name + "_extended",
        pat_str=tokeniser._pat_str,
        mergeable_ranks=tokeniser._mergeable_ranks,
        special_tokens=get_specials(tokeniser._special_tokens, tokeniser.max_token_value),
    )


########################################################################################
####                                   CORRUPTION                                   ####
########################################################################################

@dataclass
class ChunkReservoir:
    """
    Classic reservoir sampling, but for fixed-length token chunks.

    - Stores up to `max_chunks` chunks.
    - Each offered chunk is assumed to be exactly `chunk_size` tokens.
    - After seeing N chunks total, the reservoir contents are an (approx) uniform sample
      of size `max_chunks` from those N chunks.

    If chunk_size is fixed, sampling a random chunk ~ uniform over chunks ~ approx uniform over tokens.
    """
    max_chunks: int
    chunk_size: int
    rng: np.random.Generator

    chunks: list[list[int]] = None
    seen_chunks: int = 0  # number of chunks offered so far

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.max_chunks <= 0:
            raise ValueError("max_chunks must be > 0")
        if self.chunks is None:
            self.chunks = []

    def offer(self, chunk: list[int]) -> None:
        """Offer one chunk to the reservoir."""
        if len(chunk) != self.chunk_size:
            raise ValueError(f"Expected chunk length {self.chunk_size}, got {len(chunk)}")

        self.seen_chunks += 1
        if len(self.chunks) < self.max_chunks:
            self.chunks.append(chunk)
            return

        # Reservoir sampling: replace an existing element with probability K / N
        # Choose j uniformly in [0, N-1]; if j < K, replace chunks[j].
        j = int(self.rng.integers(0, self.seen_chunks))
        if j < self.max_chunks:
            self.chunks[j] = chunk

    def __len__(self) -> int:
        return len(self.chunks)

    def sample_window(self, window_len: int) -> list[int]:
        """
        Sample a contiguous window of length `window_len` from a random chunk.

        Requires: chunk_size >= window_len and reservoir non-empty.
        """
        if len(self.chunks) == 0:
            raise RuntimeError("Reservoir is empty; cannot sample.")
        if window_len > self.chunk_size:
            raise ValueError(f"window_len {window_len} > chunk_size {self.chunk_size}")

        idx = int(self.rng.integers(0, len(self.chunks)))
        chunk = self.chunks[idx]
        max_start = self.chunk_size - window_len
        start = int(self.rng.integers(0, max_start + 1))  # inclusive
        return chunk[start:start + window_len]


def sample_placeholder_corruptions(
    token_budget: int,
    p_start_del: float,
    p_end_del: float,
    p_start_ins: float,
    p_end_ins: float,
    rng: Optional[RNG] = None,
):
    """Compute insertion and deletion placeholders in one go.

    Also track the total reverse serialised tokens.
    """
    _deletion_extra_tokens = 3  # Deletion start, location start and location end.
    _insertion_extra_tokens = 2  # Insertion start and location.
    _edit_tokens = 2  # Edit open and close.

    if rng is None:
        rng = np.random.default_rng()

    # Raw index is in the index space of the original sequence.
    # Deletions use raw index, insertions are done in a *post-deletion* index space.
    # Token count is for the number of serialised tokens to revert the changes.
    raw_index = 0
    token_count = _edit_tokens  # Assume an edit takes place.
    corrupt_del = []
    corrupt_ins = []
    # Calculate the distance to the next insertion and deletion.
    space_to_next_del = rng.geometric(p_start_del)
    space_to_next_ins = rng.geometric(p_start_ins) - 1  # We can insert after the first token.
    del_size = rng.geometric(p_end_del)
    ins_size = rng.geometric(p_end_ins)
    indel_count = 0
    while True:
        # ...
        indel_count += 1
        del_next = space_to_next_del < space_to_next_ins

        if del_next:
            # Push the index forward to the next deletion, record deletion.
            # Remember we are counting the reverse of a deletion which is an insertion.
            token_count += space_to_next_del + _insertion_extra_tokens + del_size
            raw_index += space_to_next_del
            corrupt_del.append((raw_index, del_size, token_count, indel_count))
            # print(f"del op:\n\t{token_count=}\n\t{space_to_next_del=}"
                #   f"\n\t{_insertion_extra_tokens=}\n\t{del_size=}\n\t{raw_index=}")
            raw_index += del_size  # Push the raw index to the end of the deletion

            # Move the insertion closer.
            space_to_next_ins -= space_to_next_del

            # Re-compute the next deletion and size.
            space_to_next_del = rng.geometric(p_start_del)
            del_size = rng.geometric(p_end_del)
            # print(f"\t{space_to_next_del=}\n\t{del_size=}\n\t{space_to_next_ins=}\n")
        else:
            # Push the index forward to the next insertion.
            raw_index += space_to_next_ins
            token_count += space_to_next_ins + _deletion_extra_tokens + ins_size
            corrupt_ins.append((raw_index, ins_size, token_count, indel_count))
            # print(f"ins op:\n\t{token_count=}\n\t{space_to_next_ins=}"
                #   f"\n\t{_deletion_extra_tokens=}\n\t{ins_size=}\n\t{raw_index=}")

            # Move the deletion closer.
            space_to_next_del -= space_to_next_ins

            # Important! We cannot delete this token, because it is used to insert after
            space_to_next_del = max(1, space_to_next_del)

            # Re-compute the next deletion and size.
            space_to_next_ins = rng.geometric(p_start_ins)
            ins_size = rng.geometric(p_end_ins)
            # print(f"\t{space_to_next_del=}\n\t{ins_size=}\n\t{space_to_next_ins=}\n")

        if token_budget <= token_count:
            break

    if len(corrupt_del):
        last_del_tc = corrupt_del[-1][2]
        if last_del_tc > token_budget:
            corrupt_del = corrupt_del[:-1]
    if len(corrupt_ins):
        last_ins_tc = corrupt_ins[-1][2]
        if last_ins_tc > token_budget:
            corrupt_ins = corrupt_ins[:-1]
    return corrupt_del, corrupt_ins


def sample_insert(
    length: int,
    current_sequence: list[int | str],
    sample_uniform_random_tokens: Callable[int, list[int | str]],
    sample_marginal_tokens: Callable[int, list[int | str]],
    reservoir,
    insertion_type_probabilities: np.ndarray,
    rng: RNG,
) -> np.ndarray:
    """Sample a sequence to insert.

    Args:
        length (int): Number of tokens to sample.
        current_sequence (list[int  |  str]): Current token sequence.
        sample_uniform_random_tokens (Callable[int, list[int  |  str]]): Sample uniform.
        sample_marginal_tokens (Callable[int, list[int  |  str]]): Sample marginal.
        data_buffer (list[list[int  |  str]]): Buffer of tokens.
        insertion_type_probabilities (np.ndarray): Insertion type probabilities:
            * Insert from the buffer.
            * Re-insert from the current sequence.
            * Sample uniform random tokens.
            * Sample random tokens from a marginal distribution.
        rng (RNG): Random number generator.

    Returns:
        np.ndarray: Sequence to insert.
    """
    branch_probs = np.array(insertion_type_probabilities)
    branch_probs[0] *= (len(reservoir) > 0)  # Only take from data buffer if we can
    branch = rng.choice(4, p=branch_probs / branch_probs.sum())
    match branch:
        case 0:
            insert_raw = reservoir.sample_window(length)
        case 1:
            i = rng.choice(len(current_sequence) - length)
            insert_raw = current_sequence[i:i+length]
        case 2:
            insert_raw = sample_uniform_random_tokens(length)
        case 3:
            insert_raw = sample_marginal_tokens(length)

    # if len(insert_raw) != length:
    #     print(f"Invalid insert! {len(insert_raw)=} != {length}")
    #     breakpoint()

    return insert_raw


def fill_insertion_placeholders(
    placeholders: list[tuple[int, int]],
    current_sequence: list[int | str],
    sample_uniform_random_tokens: Callable[int, list[int | str]],
    sample_marginal_tokens: Callable[int, list[int | str]],
    reservoir,
    insertion_type_probabilities: np.ndarray,
    rng: RNG,
):
    insertions = []
    for i, length in placeholders:
        ins = sample_insert(
            length=length,
            current_sequence=current_sequence,
            sample_uniform_random_tokens=sample_uniform_random_tokens,
            sample_marginal_tokens=sample_marginal_tokens,
            reservoir=reservoir,
            insertion_type_probabilities=insertion_type_probabilities,
            rng=rng,
        )
        insertions.append((i, ins))
    return insertions


def advance_indel_stream(
    no_del_mean: float,
    del_mean_len: float,
    no_ins_mean: float,
    ins_mean_len: float,
    token_budget: int,
    reservoir,
    current_sequence_buffer: Sequence,
    insertion_type_probabilities: np.ndarray,
    sample_uniform_random_tokens: callable,
    sample_marginal_tokens: callable,
    rng: RNG,
    min_budget: int = 10,
):
    """Sample insertions and deletions within a serialisation token budget.

    * Randomly choose between:
        * Full size indels over the full sequence -> sample a protection over some of
          the initial context (where this is randomly choosen).
        * A smaller size set of indels say two with half the context window.
    * Sample the indel placeholders.
    * Sample the insertions.
    * Prepare the training example.
    * If there are more than one - concatenate it together, accumulating the positions
      etc.
    """
    if token_budget < min_budget:
        tokens_needed = token_budget
        insertions = []
        deletions = []

    p_start_del = 1 / no_del_mean
    p_end_del = 1 / del_mean_len
    p_start_ins = 1 / no_ins_mean
    p_end_ins = 1 / ins_mean_len
    corruption_dels, placeholder_ins = sample_placeholder_corruptions(
        token_budget=token_budget,  # -1 for bos, +1 for x[1:], y[:-1]
        p_start_del=p_start_del,
        p_end_del=p_end_del,
        p_start_ins=p_start_ins,
        p_end_ins=p_end_ins,
        rng=rng,
    )

    if len(placeholder_ins) or len(corruption_dels):
        # TODO: Find a way to have new streams of tokens entering the buffer.
        insertions = fill_insertion_placeholders(
            placeholders=[x[:2] for x in placeholder_ins],
            current_sequence=current_sequence_buffer,
            sample_uniform_random_tokens=sample_uniform_random_tokens,
            sample_marginal_tokens=sample_marginal_tokens,
            reservoir=reservoir,
            insertion_type_probabilities=insertion_type_probabilities,
            rng=rng,
        )
        check_sizes = all(len(k[1]) == v[1] for k, v in zip(insertions, placeholder_ins))

        # Calculate the number of tokens needed from the stream.
        last_ins = placeholder_ins[-1][-1] if len(placeholder_ins) else -1
        last_del = corruption_dels[-1][-1] if len(corruption_dels) else -1
        insertion_last = last_ins > last_del
        if insertion_last:
            last_raw_token, _, serial_count, _ = placeholder_ins[-1]
        else:
            raw_token_start, size, serial_count, _ = corruption_dels[-1]
            last_raw_token = raw_token_start + size

        excess = token_budget - serial_count  # - 1
        tokens_needed = excess + last_raw_token

        # Push the indices of the indel and get more raw tokens at the start.
        # push = 0 #excess - 1
        push = rng.choice(excess) if excess > 0 else 0
        deletions = [(i + push, x) for i, x, _, _ in corruption_dels]
        # ins = [(i + push, x) for i, x, _, _ in placeholder_ins]
        ins = [(i + push, x) for i, x in insertions][:len(placeholder_ins)]

        # Put the insertions into *post-deletion* index space
        deletion_mask = DeletionMask.to_dense(deletions, full_length=tokens_needed)
        ixs = np.array([i for i, _ in ins], dtype=np.int32)
        ixs = np.maximum(post_deletion_index(ixs - 1, deletion_mask) if len(deletion_mask) else ixs, 0)
        multiple = (np.unique(ixs, return_counts=True)[1] > 1).any()
        assert not multiple
        # _ins = [(i+1, x) for i, (_, x) in zip(ixs, ins)]
        _ins = [(i, x) for i, (_, x) in zip(ixs, ins)]

        # ins = [(i, tokens2[:s]) for i, s in _ins]
        # assert all(a.shape[0] == s for (_, a), (_, s) in zip(ins, _ins))
        insertions = _ins
    else:
        tokens_needed = token_budget
        insertions = []
        deletions = []
        # print("NO INDELS")

    return insertions, deletions, tokens_needed


########################################################################################
####                                    TRAINING                                    ####
########################################################################################

def gamma_shaped_partition(
    n: int,
    min_size: int,
    shape: float = 3.0,
    target_mean_size: float = 128.0,
    g: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """
    Partition an integer total `n` into a list of group sizes, each at least `min_size`,
    such that the *excess* group sizes (over `min_size`) are shaped by a Gamma distribution.

    Conceptually:
        Let m_i be group sizes.
        Define excess sizes y_i = m_i - min_size >= 0.

        Before conditioning on the total sum constraint, we impose:
            y_i ~ Gamma(shape=shape, scale=scale_scaled)

        where the scale is chosen so that:
            E[m_i] = min_size + shape * scale_scaled ≈ target_mean_size.

        The final partition is obtained by:
        1. Choosing a number of groups g (explicitly, or implicitly via n / target_mean_size).
        2. Sampling i.i.d. Gamma weights for each group.
        3. Normalizing these weights to proportions.
        4. Allocating the remaining mass (n - g * min_size) proportionally,
           with integer rounding and exact sum correction.

    Notes:
    - Conditioning on sum(m_i) = n slightly distorts the marginal Gamma distribution,
      especially for small g, but the overall shape is preserved.
    - Smaller `shape` values produce heavier-tailed (more spread-out) size distributions.
    - This construction is closely related to a Dirichlet–Gamma model over group sizes.

    Args:
        n : int
            Total size to be partitioned (sum of all groups).
        min_size : int
            Minimum allowed size for each group (m_i >= min_size).
        shape : float, default=3.0
            Shape parameter of the Gamma prior over excess sizes.
            Smaller values increase variance / tail-heaviness.
        target_mean_size : float, default=128.0
            Desired average group size before conditioning on sum=n.
            Must be strictly greater than min_size.
        g : int or None, default=None
            Number of groups. If None, chosen as round(n / target_mean_size),
            clipped to [1, n // min_size].
        rng : np.random.Generator or None, default=None
            NumPy random number generator. If None, uses np.random.default_rng().

    Returns:
        List[int]
            A list of group sizes summing to n, each >= min_size.
            The order of groups is randomized.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n < min_size:
        raise ValueError("n must be >= min_size")

    if target_mean_size <= min_size:
        raise ValueError("target_mean_size must be > min_size")

    # Maximum feasible number of groups
    gmax = n // min_size

    # Choose number of groups if not provided
    if g is None:
        g = int(round(n / target_mean_size))
        g = max(1, min(g, gmax))
    else:
        if not (1 <= g <= gmax):
            raise ValueError(f"g must be in [1, {gmax}]")

    # Scale so that E[m_i] ≈ target_mean_size
    # E[m_i] = min_size + shape * scale_scaled
    scale_scaled = (target_mean_size - min_size) / shape

    # Sample Gamma weights (excess-size "preferences")
    weights = rng.gamma(shape=shape, scale=scale_scaled, size=g)

    # Allocate baseline min_size to each group
    remaining = n - g * min_size
    if remaining < 0:
        # Safety fallback (should not happen if g <= gmax)
        g = gmax
        remaining = n - g * min_size
        weights = rng.gamma(shape=shape, scale=scale_scaled, size=g)

    # Normalize weights to proportions
    w = weights / weights.sum()

    # Integer allocation of remaining mass
    extras = np.floor(w * remaining).astype(int)
    diff = remaining - extras.sum()

    # Distribute leftover units by largest fractional parts
    frac = (w * remaining) - extras
    order = np.argsort(frac)[::-1]
    extras[order[:diff]] += 1

    sizes = extras + min_size
    rng.shuffle(sizes)

    return sizes.tolist()


def cat(xs: list[TrainingData]) -> TrainingData:
    """Concatenate training examples.

    xs = [a, b, c]
    input_tokens = a.input + [a.target[-1]] + b.input + [b.target[-1]] + c.input
    output_tokens = a.target + [b.input[0]] + b.target + [c.input[0]] + c.target
    (or just input_tokens + [c.target[-1]] then do the x[:-1] / x[1:] trick)


    """
    lengths = np.array([x.input_tokens.shape[0] + 1 for x in xs])
    total_size = lengths.sum() - 1
    offsets = np.cumsum(np.concatenate([np.zeros(1, dtype=np.int32), lengths[:-1]]))

    size = xs[0].target.gather_valid.shape[0]  # pos_buffer

    gi_list = []
    tg_list = []
    for x, off in zip(xs, offsets):
        valid = x.target.gather_valid.astype(bool)
        gi_list.append(x.target.gather_indices[valid] + off)
        tg_list.append(x.target.gathered_pos_target[valid] + off)

    gi = np.concatenate(gi_list) if len(gi_list) else np.zeros((0,), dtype=np.int64)
    tg = np.concatenate(tg_list) if len(tg_list) else np.zeros((0,), dtype=np.int64)

    # number of real gathered events we have
    G = min(len(gi), size)

    # allocate fixed buffers
    gi_buf = np.zeros(size, dtype=gi.dtype)
    tg_buf = np.full(size, -1, dtype=tg.dtype)
    gv_buf = np.zeros(size, dtype=bool)

    # fill real part
    gi_buf[:G] = gi[:G]
    tg_buf[:G] = tg[:G]
    gv_buf[:G] = True

    # optional: fill remaining indices with any in-range numbers (won't matter since gv is False)
    if G < size:
        filler = np.arange(total_size - (size - G), total_size, dtype=gi.dtype)
        gi_buf[G:] = filler  # harmless

    all_tokens = []
    all_token_loss_weights = []
    all_input_positions = []
    one = np.ones(1, dtype=xs[0].target.token_loss_weights.dtype)
    for x, off in zip(xs, offsets):
        all_tokens += [x.input_tokens, x.target.tokens[-1:]]
        all_token_loss_weights += [x.target.token_loss_weights, one]
        all_input_positions += [np.array([off-1], dtype=x.input_positions.dtype), x.input_positions + off]
    all_tokens = np.concatenate(all_tokens)
    all_token_loss_weights = np.concatenate(all_token_loss_weights)[:-1]
    all_input_positions = np.concatenate(all_input_positions[1:])

    return TrainingData(
        input_tokens=all_tokens[:-1],
        input_positions=all_input_positions,
        target=TrainingTarget(
            tokens=all_tokens[1:],
            gathered_pos_target=tg_buf,
            token_loss_weights=all_token_loss_weights,
            gather_indices=gi_buf,
            gather_valid=gv_buf,
            overflow=len(gi) > size,
        ),
    )


def remove_any(insertions, deletions, keep_edit_when_identity):
    identity = len(insertions) == 0 and len(deletions) == 0
    if identity and keep_edit_when_identity:
        return NUM_EDIT_TOKENS
    else:
        return 0


def process_training_example(
    token_ops: TokensOperations,
    seq: Sequence,
    insertions: list[tuple[int, Sequence]],
    deletions: SparseDeletionMask,
    pos_buffer_size: int,
    drop_edit_when_identity: bool,
    indel_bos_index: int = 0,
) -> tuple[int, TrainingData]:
    # Empty case:
    #  - Drop the edit tokens
    #  - Keep the edit tokens
    # Otherwise
    identity = len(insertions) == 0 and len(deletions) == 0
    # if len(seq) != next_index - current_index:
    #     raise ValueError(f"{len(seq)=} != {next_index=} - {current_index=}")
    corruption = SequenceEdit(sequence=seq, insertions=insertions, deletions=deletions)
    if identity:
        seq_edit = corruption
    else:
        seq_edit = complement(corruption)
    serialised: SerialisedData = token_ops.serialise(seq_edit, indel_bos_pos=indel_bos_index)
    if identity and drop_edit_when_identity:
        serialised: SerialisedData = SerialisedData(
            tokens=serialised.tokens[:-NUM_EDIT_TOKENS],
            positions=serialised.positions[:-NUM_EDIT_TOKENS],
        )

    return token_ops.prepare_training_example(
        serialised=serialised, pos_buffer=pos_buffer_size
    )


def default_protection_function(rng: RNG, size: int) -> int:
    return rng.choice(size)


def default_indel_halflife_fn(size: int):
    scale = size / 1024
    no_del_mean = max(1, scale * 128)
    del_mean_len = max(1, scale * 8)
    no_ins_mean = max(1, scale * 128)
    ins_mean_len = max(1, scale * 8)
    return no_del_mean, del_mean_len, no_ins_mean, ins_mean_len


class InsertTypeProbs(NamedTuple):
    """
    * Insert from the buffer.
    * Re-insert from the current sequence.
    * Sample uniform random tokens.
    * Sample random tokens from a marginal distribution.
    """
    buffer: float
    myself: float
    uniform: float
    marginal: float


class CorruptionStreamParams(NamedTuple):
    no_del_mean: float
    del_mean_len: float
    no_ins_mean: float
    ins_mean_len: float
    no_corruption_prob: float
    partition_prob: float
    protection_function: callable
    indel_halflife_fn: callable
    partition_min_size: int
    partition_gamma_shape: float
    partition_slice_mean_size: float
    insertion_type_probabilities: InsertTypeProbs

    @classmethod
    def default(cls):
        return cls(
            no_del_mean=128.0,
            del_mean_len=8.0,
            no_ins_mean=128.0,
            ins_mean_len=8.0,
            no_corruption_prob=0.5,
            partition_prob=0.1,
            protection_function=default_protection_function,
            indel_halflife_fn=default_indel_halflife_fn,
            partition_min_size=48,
            partition_gamma_shape=2,
            partition_slice_mean_size=360.0,
            insertion_type_probabilities=InsertTypeProbs(
                buffer=0.6,
                myself=0.1,
                uniform=0.2,
                marginal=0.1,
            ),
        )


def get_training_data_from_stream(
    cfg: CorruptionStreamParams,
    token_buffer: list,
    reservoir,
    token_ops: TokensOperations,
    token_budget: int = 1024,
    pos_buffer_size: int = 32,
    perplexity_masks: Callable | None = None,
    rng: RNG | None = None,
) -> TrainingData:
    # We can actually shift some more context in at the start if we place the edits
    # at the end.
    # TODO: Think about whether to always shift the edits to the end. If we always
    # do this, then maybe the model will learn to never open edit regions until it
    # is particularly far into the the sequence...
    # Either way - at the scope of this function it should still be controlled by
    # budget.
    # At inference time - we will apply edits immediately after the edit close token
    # But! If we were to change the sequence length we train on- we don't want the
    # context length of the sequence to necessarily increase before these edits...
    # Fundamentally - this means we have to train on a varied sequence length
    # distribution... maybe we forgo the one-edit-region max rule... and instead
    # just process concatenated serialised sequence edits. But this may require
    # careful thinking about the position i/o.
    # --> Possible solution: do a mixture - some proportion of the time place at the
    # end, other times do smaller edit sequences cat'ed together.
    if rng is None:
        rng = np.random.default_rng(seed=0)

    branch_probs = np.array([cfg.no_corruption_prob, cfg.partition_prob, 0])
    branch_probs[-1] = 1 - branch_probs.sum()
    if not np.isclose(branch_probs.sum(), 1.0, atol=1e-5):
        raise ValueError(
            f"Invalid probabilities: {cfg.no_corruption_prob=}, {cfg.partition_prob=}"
        )
    branch_probs /= branch_probs.sum()

    # Body
    branch = rng.choice(len(branch_probs), p=branch_probs)
    drop_edit_when_identity = rng.choice(2) == 1
    # print((branch, drop_edit_when_identity))
    if branch == 0:
        insertions = []
        deletions = []
        tokens_needed = token_budget - remove_any(
            insertions=insertions,
            deletions=deletions,
            keep_edit_when_identity=not drop_edit_when_identity,
        )
        indel_bos_index = rng.choice(tokens_needed)
        training_example = process_training_example(
            token_ops=token_ops,
            seq=np.array(token_buffer[:tokens_needed]),
            insertions=insertions,
            deletions=deletions,
            pos_buffer_size=pos_buffer_size,
            drop_edit_when_identity=drop_edit_when_identity,
            indel_bos_index=indel_bos_index,
        )
    else:

        if branch == 1:
            partition = gamma_shaped_partition(
                n=token_budget,
                min_size=cfg.partition_min_size,
                shape=cfg.partition_gamma_shape,
                target_mean_size=cfg.partition_slice_mean_size,
                rng=rng,
            )
            # When we concatenate two examples we need the connecting token pair.
            # Adjust the budget accordingly.
            choice = rng.choice(len(partition))
            partition = [p if i == choice else p - 1 for i, p in enumerate(partition)]
        else:
            partition = [token_budget]

        tokens_needed = 0
        tes = []
        for size in partition:
            if cfg.protection_function is not None:
                protect = cfg.protection_function(rng, size)
            else:
                protect = 0

            if cfg.indel_halflife_fn is not None:
                nd, dm, ni, im = cfg.indel_halflife_fn(size)
            else:
                nd, dm, ni, im = (
                    cfg.no_del_mean, cfg.del_mean_len, cfg.no_ins_mean, cfg.ins_mean_len
                )

            insertions, deletions, _tokens_needed = advance_indel_stream(
                no_del_mean=nd,
                del_mean_len=dm,
                no_ins_mean=ni,
                ins_mean_len=im,
                token_budget=size-protect,
                reservoir=reservoir,
                current_sequence_buffer=np.array(token_buffer[:size]),
                insertion_type_probabilities=np.array(cfg.insertion_type_probabilities),
                sample_uniform_random_tokens=token_ops.sample_uniform_random_tokens,
                sample_marginal_tokens=token_ops.sample_marginal_tokens,
                rng=rng,
            )
            insertions = [(i + protect, x) for i, x in insertions]
            deletions = [(i + protect, x) for i, x in deletions]
            _tokens_needed += protect
            _tokens_needed -= remove_any(
                insertions=insertions,
                deletions=deletions,
                keep_edit_when_identity=not drop_edit_when_identity,
            )
            indel_bos_index = rng.choice(protect) if protect > 0 else 0
            training_example = process_training_example(
                token_ops=token_ops,
                seq=np.array(token_buffer[tokens_needed:tokens_needed+_tokens_needed]),
                insertions=insertions,
                deletions=deletions,
                pos_buffer_size=pos_buffer_size,
                drop_edit_when_identity=drop_edit_when_identity,
                indel_bos_index=indel_bos_index,
            )
            tes.append(training_example)
            if size != training_example.input_tokens.shape[0]:
                from nanochat.causal_serialisation import print_training_examples
                print_training_examples([training_example], token_ops.tokeniser)
                breakpoint()
            drop_edit_when_identity = rng.choice(2) == 1
            tokens_needed += _tokens_needed

        # cn = sum(te.input_tokens.shape[0] for te in tes)
        # if cn != 1024:
        #     from nanochat.causal_serialisation import print_training_examples
        #     print_training_examples([tes[0]], token_ops.tokeniser)
        #     breakpoint()

        if len(tes) > 1:
            training_example = cat(tes)

    if perplexity_masks is not None:
        training_example = perplexity_masks(training_example)

    # The final token can be used as the start of the next example.
    advance = tokens_needed - 1
    return advance, training_example


########################################################################################
####                                   GENERATION                                   ####
########################################################################################


class ResolutionState(Enum):
    VALID_EDIT   = "VALID_EDIT"
    PARTIAL_EDIT = "PARTIAL_EDIT"
    NO_EDIT      = "NO_EDIT"
    INVALID_EDIT = "INVALID_EDIT"


@dataclass
class _RegionParse:
    state: ResolutionState
    crop_end: int
    has_any_op: bool
    ends_with_edit_end: bool
    force_append_edit_end: bool = False


def resolve_model_output(
    sequence: Sequence,
    pointers: Positions,
    token_operations: TokensOperations,
    add_edit_close_if_missing: bool = False,
    crop_edits: int | None = None,
) -> tuple[Sequence, ResolutionState, bool]:
    """Make edits if present, mark to re-process if edits are applied.

    **NOTE** In case this has some bugs I haven't found yet - I kept the previous
    version above which shouldn't have bugs because I have trained with it for 10's of
    trillions of tokens.

    Args:
        sequence (Sequence): _description_
        positions (Positions): _description_

    Returns:
        Sequence: Processed sequence.
        ResolutionState: Report the state of the sequence.
        bool: Whether the sequence is resolved.
    """
    seq = np.asarray(sequence)
    ptr = np.asarray(pointers)
    T = len(seq)

    ES = token_operations.edit_start_token
    EE = token_operations.edit_end_token
    DO = token_operations.deletion_open_token
    DS = token_operations.deletion_start_token
    DE = token_operations.deletion_end_token
    IO = token_operations.insertion_open_token
    IL = token_operations.insertion_loc_token

    edit_specials = {ES, EE, DO, DS, DE, IO, IL}

    def is_normal(tok: int) -> bool:
        return tok not in edit_specials

    def contains_any_edit_special(arr) -> bool:
        # fast-ish: check membership by equality against each special
        # (could be optimized later)
        for s in edit_specials:
            if np.any(arr == s):
                return True
        return False

    def parse_del(i: int, end_excl: int):
        """
        Expect: DO DS DE
        Returns: (ok, new_i, complete_op_end)
          - ok=False means invalid (not partial)
          - if partial, ok is None
        """
        # Need 3 tokens
        if i + 3 > end_excl:
            return None, i, i  # partial, crop to start of this op
        if int(seq[i]) != DO or int(seq[i+1]) != DS or int(seq[i+2]) != DE:
            return False, i, i
        # Optional pointer sanity checks can go here if you want.
        return True, i + 3, i + 3

    def parse_ins(i: int, end_excl: int):
        """
        Expect: IO IL (>=1 normal token)
        Consume normals until next special or end.
        Returns (ok, new_i, complete_op_end).
        """
        if i + 2 > end_excl:
            return None, i, i
        if int(seq[i]) != IO or int(seq[i+1]) != IL:
            return False, i, i

        j = i + 2
        if j >= end_excl:
            return None, i, i
        if not is_normal(int(seq[j])):
            # must start content immediately
            return False, i, i

        # consume >=1 normal
        while j < end_excl and is_normal(int(seq[j])):
            j += 1

        return True, j, j

    def parse_edit_region(start_ix: int, end_excl: int, require_edit_end: bool):
        """
        Parses tokens in [start_ix, end_excl).
        If require_edit_end=True, region must contain EE to be VALID.
        If require_edit_end=False, reaching end_excl without EE is allowed (caller may auto-close).
        Crops off a final incomplete op if present.
        """
        assert int(seq[start_ix]) == ES
        if crop_edits is not None and crop_edits <= 0:
            return _RegionParse(
                state=ResolutionState.VALID_EDIT,
                crop_end=start_ix + 1,   # keep only ES
                has_any_op=False,
                ends_with_edit_end=False,
                force_append_edit_end=True,
            )
        i = start_ix + 1
        has_any_op = False
        n_ops = 0
        while i < end_excl:
            tok = int(seq[i])

            if tok == EE:
                return _RegionParse(
                    state=ResolutionState.VALID_EDIT,
                    crop_end=i + 1,
                    has_any_op=has_any_op,
                    ends_with_edit_end=True,
                )

            if tok == DO:
                ok, new_i, op_end = parse_del(i, end_excl)
                if ok is True:
                    has_any_op = True
                    n_ops += 1

                    if crop_edits is not None and n_ops >= crop_edits:
                        return _RegionParse(
                            state=ResolutionState.VALID_EDIT,
                            crop_end=new_i,
                            has_any_op=True,
                            ends_with_edit_end=False,
                            force_append_edit_end=True,
                        )

                    i = new_i
                    continue

                if ok is None:
                    # partial op at end -> crop before it
                    return _RegionParse(
                        state=ResolutionState.PARTIAL_EDIT,
                        crop_end=i,              # crop away the incomplete op
                        has_any_op=has_any_op,
                        ends_with_edit_end=False,
                    )
                return _RegionParse(ResolutionState.INVALID_EDIT, crop_end=i, has_any_op=has_any_op, ends_with_edit_end=False)

            if tok == IO:
                ok, new_i, op_end = parse_ins(i, end_excl)
                if ok is True:
                    has_any_op = True
                    n_ops += 1

                    # --- NEW: stop after crop_edits ops ---
                    if crop_edits is not None and n_ops >= crop_edits:
                        return _RegionParse(
                            state=ResolutionState.VALID_EDIT,
                            crop_end=new_i,
                            has_any_op=True,
                            ends_with_edit_end=False,
                            force_append_edit_end=True,
                        )
                    i = new_i
                    continue
                if ok is None:
                    return _RegionParse(
                        state=ResolutionState.PARTIAL_EDIT,
                        crop_end=i,
                        has_any_op=has_any_op,
                        ends_with_edit_end=False,
                    )
                return _RegionParse(ResolutionState.INVALID_EDIT, crop_end=i, has_any_op=has_any_op, ends_with_edit_end=False)

            # Any other token inside edit region is invalid by your grammar
            return _RegionParse(ResolutionState.INVALID_EDIT, crop_end=i, has_any_op=has_any_op, ends_with_edit_end=False)

        # Hit end_excl without EE
        if require_edit_end:
            # But we may have cropped off trailing partial ops already; here it just ended cleanly.
            return _RegionParse(ResolutionState.PARTIAL_EDIT, crop_end=end_excl, has_any_op=has_any_op, ends_with_edit_end=False)
        else:
            # Caller may append EE if desired
            return _RegionParse(ResolutionState.VALID_EDIT, crop_end=end_excl, has_any_op=has_any_op, ends_with_edit_end=False)

    # -------------------------
    # Main resolution policy
    # -------------------------

    # Case 1: first EDIT_END exists -> crop to it and require exactly one EDIT_START in that crop.
    end_ixs = np.where(seq == EE)[0]
    if end_ixs.size:
        end = int(end_ixs[0])
        crop_end = end + 1
        seq_c = seq[:crop_end]
        ptr_c = ptr[:crop_end]

        start_ixs = np.where(seq_c == ES)[0]
        # Your stricter rule: exactly one start before first end
        if start_ixs.size != 1:
            return seq_c, ResolutionState.INVALID_EDIT, True
        start = int(start_ixs[0])

        # Also enforce: no edit specials before start (optional, as you suggested)
        if contains_any_edit_special(seq_c[:start]):
            return seq_c, ResolutionState.INVALID_EDIT, True

        region = parse_edit_region(start, crop_end, require_edit_end=True)
        # region.crop_end might be < crop_end only if something went wrong; but we already have EE.
        seq_c = seq_c[:region.crop_end]
        ptr_c = ptr_c[:region.crop_end]

        # NEW: if we cropped after N ops, force-close the edit region
        if region.force_append_edit_end:
            ee_pos = len(seq_c)  # EE will be placed here
            seq_c = np.concatenate([seq_c, np.array([EE], dtype=seq_c.dtype)])
            ptr_c = np.concatenate([ptr_c, np.array([ee_pos], dtype=ptr_c.dtype)])
            region = _RegionParse(
                state=ResolutionState.VALID_EDIT,
                crop_end=len(seq_c),
                has_any_op=region.has_any_op,
                ends_with_edit_end=True,
                force_append_edit_end=False,
            )

        if region.state == ResolutionState.VALID_EDIT and region.has_any_op:
            serialised = SerialisedData(seq_c, ptr_c)
            # serialised.print(tokeniser=token_operations.tokeniser)
            seq_edit = token_operations.deserialise(serialised)
            resolved = resolve(seq_edit)
            return resolved, ResolutionState.VALID_EDIT, False

        # If it was "valid" but had zero ops, treat as NO_EDIT (your call)
        if region.state == ResolutionState.VALID_EDIT and not region.has_any_op:
            return seq_c, ResolutionState.NO_EDIT, True

        return seq_c, region.state, True

    # Case 2: no EDIT_END
    start_ixs = np.where(seq == ES)[0]
    if start_ixs.size == 0:
        # You asked: NO_EDIT only if *no* other edit specials exist
        if contains_any_edit_special(seq):
            return seq, ResolutionState.INVALID_EDIT, True
        return seq, ResolutionState.NO_EDIT, True

    # You asked: if >1 EDIT_START and no EDIT_END -> invalid
    if start_ixs.size != 1:
        return seq, ResolutionState.INVALID_EDIT, True

    start = int(start_ixs[0])

    # Optional: no edit specials before start
    if contains_any_edit_special(seq[:start]):
        return seq, ResolutionState.INVALID_EDIT, True

    # Validate region from start to end; we allow missing end so we can auto-close
    region = parse_edit_region(start, T, require_edit_end=False)

    # Crop off trailing partial op if parse requested it
    seq_c = seq[:region.crop_end]
    ptr_c = ptr[:region.crop_end]

    # NEW: if we cropped after N ops, force-close the edit region
    if getattr(region, "force_append_edit_end", False):
        ee_pos = len(seq_c)  # position EE will take
        seq_c = np.concatenate([seq_c, np.array([EE], dtype=seq_c.dtype)])
        ptr_c = np.concatenate([ptr_c, np.array([ee_pos], dtype=ptr_c.dtype)])
        region = _RegionParse(
            state=ResolutionState.VALID_EDIT,
            crop_end=len(seq_c),
            has_any_op=region.has_any_op,
            ends_with_edit_end=True,
            force_append_edit_end=False,
        )

    if not add_edit_close_if_missing:
        # If there was anything after ES that isn't a complete region, treat as PARTIAL/INVALID accordingly.
        # If region.state==VALID_EDIT here means "structure OK up to end", but missing EE: still PARTIAL externally.
        if region.state == ResolutionState.INVALID_EDIT:
            return seq_c, ResolutionState.INVALID_EDIT, True
        return seq_c, ResolutionState.PARTIAL_EDIT, True

    # add_edit_close_if_missing=True
    # If after cropping we have at least one op, we can append EE and resolve.
    if region.state != ResolutionState.INVALID_EDIT and region.has_any_op:
        # append EE
        ee_pos = len(seq_c)
        seq2 = np.concatenate([seq_c, np.array([EE], dtype=seq_c.dtype)])
        ptr2 = np.concatenate([ptr_c, np.array([ee_pos], dtype=ptr_c.dtype)])
        serialised = SerialisedData(seq2, ptr2)
        serialised.print(tokeniser=token_operations.tokeniser)
        seq_edit = token_operations.deserialise(serialised)
        resolved = resolve(seq_edit)
        # raw output was partial, but we applied an edit
        return resolved, ResolutionState.PARTIAL_EDIT, False

    # Otherwise it's partial (no complete ops) or invalid
    if region.state == ResolutionState.INVALID_EDIT:
        return seq_c, ResolutionState.INVALID_EDIT, True
    return seq_c, ResolutionState.PARTIAL_EDIT, True


########################################################################################
####                                      TEST                                      ####
########################################################################################

def same(a: SequenceEdit, b: SequenceEdit) -> bool:
    ad = np.array(a.deletions)
    bd = np.array(b.deletions)
    d = (ad == bd).all() if ad.size > 0 and bd.size > 0 else True
    s = [len(a.insertions) == len(b.insertions)]
    for (ai, ains), (bi, bins) in zip(a.insertions, b.insertions):
        s.append(ai == bi)
        s.append((ains == bins).all())
    seq = (a.sequence == b.sequence).all()
    return {"seq": seq, "del": d, "ins": all(s)}


def test_indel_complement(to, num_tries=10):
    p = len(tokens) // 4
    for _ in range(num_tries):
        sequence_and_corruption = corrupt(
            config=to.config,
            sequence=tokens,
            sample_uniform_random_tokens=to.sample_uniform_random_tokens,
            sample_marginal_tokens=to.sample_marginal_tokens,
            token_source=buffer,
            protect_up_to=p,
        )

        corrupted_with_fix = complement(sequence_and_corruption)
        recovered = complement(corrupted_with_fix)

        out = same(sequence_and_corruption, recovered)
        assert all(out.values()), f"{out}"


def test_serialisation(to, num_tries=1000):
    p = len(tokens) // 4
    for tr in range(num_tries):
        sequence_and_corruption = corrupt(
            config=to.config,
            sequence=tokens,
            sample_uniform_random_tokens=to.sample_uniform_random_tokens,
            sample_marginal_tokens=to.sample_marginal_tokens,
            token_source=buffer,
            protect_up_to=p,
        )
        serialised = to.serialise(sequence_and_corruption)
        deserialised = to.deserialise(serialised)
        # * Has an extra bos
        # * Indices / sizes in the deletion are +1 and not array.
        # * insertion index is wrong.

        out = same(sequence_and_corruption, deserialised)
        if not all(out.values()):
            print(f"Passed {tr}, but failed on this one: {out}")
            ibreakpoint()
    print(f"Successfully ran {num_tries} serialisation<->deserialisation tests!")


def test_rng(tokeniser, num_checks: int = 100):
    tops = TokensOperations(tokeniser=tokeniser)
    t = lambda x: np.squeeze(tops.tokeniser.encode_batch(list(x)))
    data = t("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    token_source = [t("abcdefghijklmnopqrstuvwxyz")]
    checks = []
    for _ in range(2):
        check = []
        for i in range(num_checks):
            check.append(
                tops.deserialise(
                    tops.corrupt(
                        data, token_source=token_source, rng=np.random.default_rng(i)
                    )
                )
            )
        checks.append(check)
    passed = [all(same(checks[0][i], checks[1][i]).values()) for i in range(num_checks)]
    assert all(passed), f"Not all tests passed: {100*np.mean(passed):.1f}%"
    print(f"{num_checks} rng tests passed.")


def run_tests():
    test_serialisation(to, num_tries=1000)
    test_indel_complement(to, num_tries=1000)
    test_rng(tokeniser, num_checks=1000)


def check_real_data(config, tokeniser):
    from nanochat.dataset import parquets_iter_batched

    # grab one row group
    texts = next(parquets_iter_batched("train"))

    print(f"{'doc':>3} | {'chars':>6} | {'tokens':>6} | {'chars/token':>12}")
    print("-" * 40)

    for i, text in enumerate(texts[:10]):  # first 10 docs
        n_chars = len(text)
        n_tokens = len(tokeniser.encode(text))
        ratio = n_chars / n_tokens if n_tokens > 0 else 0
        print(f"{i:>3} | {n_chars:>6} | {n_tokens:>6} | {ratio:>12.3f}")


    # snippet = text[:text.find("\n")]
    snippet = text
    tokens = np.array(tokeniser.encode(snippet))
    # print(snippet)

    def source(total):
        for _ in range(total):
            x = texts[np.random.randint(4)]
            yield tokeniser.encode(x)

    st = list(set(tokens))
    to = TokensOperations(tokeniser=tokeniser)
    buffer = list(source(total=2))[1:]
    serialised_training_example = to.corrupt(tokens, buffer)
    out = to.resolve(serialised_training_example)



    all_ixs = iter(range(1000))


    # for i in range(4):
    print("\n"*3 + "-"*80 + "\n"*3)
    i = next(all_ixs)
    tokens = np.array(tokeniser.encode(texts[i]))

    serialised_training_example = to.corrupt(
        tokens, buffer, info=False, protect_up_to=800
    )

    # pr_ste = to.tokeniser.decode(serialised_training_example)
    # recovered = to.tokeniser.decode(out[1:])
    # print(snippet == recovered)

    serialised_training_example.print(to.tokeniser)


def test_flatten_check_sizes(
    no_del_mean: float = 7,
    del_mean_len: float = 7,
    no_ins_mean: float = 7,
    ins_mean_len: float = 7,
    token_budget: int = 1024,
    seed=0,
    verbose=False,
) -> None:
    seed=1; verbose=True; no_del_mean: float = 20; del_mean_len: float = 7; no_ins_mean: float = 20; ins_mean_len: float = 7; token_budget: int = 100
    p_start_del = 1 / no_del_mean
    p_end_del = 1 / del_mean_len
    p_start_ins = 1 / no_ins_mean
    p_end_ins = 1 / ins_mean_len
    rng = np.random.default_rng(seed=seed)
    _corruption_dels, _placeholder_ins = sample_placeholder_corruptions(
        token_budget=token_budget - 1,  # -1 for bos
        p_start_del=p_start_del,
        p_end_del=p_end_del,
        p_start_ins=p_start_ins,
        p_end_ins=p_end_ins,
        rng=rng,
    )

    from nanochat.dataset import parquets_iter_batched
    texts = next(parquets_iter_batched("train"))
    tokens1 = np.array(tokeniser.encode(texts[0]))
    tokens2 = np.array(tokeniser.encode(texts[1]))

    # Done: I think my BOS is different to the one expected ... I need a bos to insert
    # after... I think there's delimits documents.
    current_index = 0
    # TODO: Find a way to have new streams of tokens entering the buffer.
    insertions = fill_insertion_placeholders(
        placeholders=[x[:2] for x in _placeholder_ins],
        current_sequence=tokens1,
        sample_uniform_random_tokens=tops.sample_uniform_random_tokens,
        sample_marginal_tokens=tops.sample_marginal_tokens,
        data_buffer=[tokens2],
        reinsert_self_slice_weight=0.06,
        buffer_insert_weight=0.9,
        marginal_token_weight=0.02,
        rng=rng,
    )
    if len(_placeholder_ins) or len(_corruption_dels):

        # Temp
        ix = max(
            _placeholder_ins[-1][-1] if len(_placeholder_ins) else -1,
            _corruption_dels[-1][-1] if len(_corruption_dels) else -1
        )
        for i in range(1, 1 + ix):
            placeholder_ins = [(a, b, c, d) for a, b, c, d in _placeholder_ins if d <= i]
            corruption_dels = [(a, b, c, d) for a, b, c, d in _corruption_dels if d <= i]


            # Calculate the number of tokens needed from the stream.
            last_ins = placeholder_ins[-1][-1] if len(placeholder_ins) else -1
            last_del = corruption_dels[-1][-1] if len(corruption_dels) else -1
            insertion_last = last_ins > last_del
            if insertion_last:
                last_raw_token, _, serial_count, _ = placeholder_ins[-1]
            else:
                raw_token_start, size, serial_count, _ = corruption_dels[-1]
                last_raw_token = raw_token_start + size

            excess = token_budget - serial_count - 1
            tokens_needed = excess + last_raw_token
            # We can actually shift some more context in at the start if we place the edits
            # at the end.
            # TODO: Think about whether to always shift the edits to the end. If we always
            # do this, then maybe the model will learn to never open edit regions until it
            # is particularly far into the the sequence...
            # Either way - at the scope of this function it should still be controlled by
            # budget.
            # At inference time - we will apply edits immediately after the edit close token
            # But! If we were to change the sequence length we train on- we don't want the
            # context length of the sequence to necessarily increase before these edits...
            # Fundamentally - this means we have to train on a varied sequence length
            # distribution... maybe we forgo the one-edit-region max rule... and instead
            # just process concatenated serialised sequence edits. But this may require
            # careful thinking about the position i/o.
            # --> Possible solution: do a mixture - some proportion of the time place at the
            # end, other times do smaller edit sequences cat'ed together.

            # Push the indices of the indel and get more raw tokens at the start.
            push = 0 #excess - 1
            # push = rng.choice(excess)
            dels = [(i + push, x) for i, x, _, _ in corruption_dels]
            # ins = [(i + push, x) for i, x, _, _ in placeholder_ins]
            ins = [(i + push, x) for i, x in insertions][:len(placeholder_ins)]

            # Put the insertions into *post-deletion* index space
            deletion_mask = DeletionMask.to_dense(dels, full_length=tokens_needed)
            ixs = np.array([i for i, _ in ins], dtype=np.int32)

            multiple = (np.unique(ixs, return_counts=True)[1] > 1).any()
            assert not multiple
            # _ins = [(i+1, x) for i, (_, x) in zip(ixs, ins)]
            _ins = [(i, x) for i, (_, x) in zip(ixs, ins)]

            # ins = [(i, tokens2[:s]) for i, s in _ins]
            # assert all(a.shape[0] == s for (_, a), (_, s) in zip(ins, _ins))
            ins = _ins

            corruption = SequenceEdit(
                sequence=tokens1[current_index:current_index+tokens_needed],
                insertions=ins,
                deletions=dels,
            )
            sm = same(complement(complement(corruption)), corruption)
            assert all(sm.values()), f"{sm}"
            ser = tops.serialise(complement(corruption))
            if verbose:
                ser.print(tops.tokeniser)
                # tops.serialise(corruption).print(tops.tokeniser)
                print(f"Serialised size: {ser.tokens.shape[0]} ({token_budget=})")
                print(f"{ins=}")
                print(f"{dels=}")
                print("-"*80)
            assert ser.tokens.shape[0] == token_budget, f"n={ser.tokens.shape[0]}"
    else:
        corruption = SequenceEdit(
            sequence=data[current_index:current_index+token_budget-1],
            insertions=[],
            deletions=[],
        )
        print("NO INDELS")

    # print(dels)
    # print(ins)


def main():
    tokeniser = get_tokeniser()

    tops = TokensOperations(tokeniser=tokeniser)

    from collections import deque

    from nanochat.dataset import parquets_iter_batched

    texts = next(parquets_iter_batched("train"))
    tokens1 = np.array(tokeniser.encode(texts[0]))
    tokens2 = np.array(tokeniser.encode(texts[1]))
    # cfg = CorruptionStreamParams.default()
    cfg = CorruptionStreamParams(
        no_del_mean=128.0,
        del_mean_len=8.0,
        no_ins_mean=128.0,
        ins_mean_len=8.0,
        no_corruption_prob=0.5,
        partition_prob=0.1,
        protection_function=default_protection_function,
        indel_halflife_fn=default_indel_halflife_fn,
        partition_min_size=48,
        partition_gamma_shape=2,
        partition_slice_mean_size=360.0,
        insertion_type_probabilities=InsertTypeProbs(
            buffer=0.6,
            myself=0.1,
            uniform=0.2,
            marginal=0.1,
        ),
    )
    rng = np.random.default_rng(seed=0)
    reservoir = ChunkReservoir(
        max_chunks=8192,
        chunk_size=1024,
        rng=rng,
    )
    ts = list(np.tile(tokens1, 100))
    das = []
    while True:
        a, da = get_training_data_from_stream(
            cfg=cfg,
            token_buffer=ts,  # Ensure this is at least B*T+1
            reservoir=reservoir,  # Adapt to deal with an empty buffer.
            token_ops=tops,
            token_budget=1024,
            pos_buffer_size=32,
            rng=rng,
        )
        ts = ts[a:]
        das.append(da)
        if len(das) >= 5:
            break


    print_training_examples(das, tops.tokeniser, max_rows=1025)


if __name__ == "__main__":
    main()
    # ----------------------------------------------------------------------------------
    # TODO:
    # [1] - Ensure that the edits aren't cropped out.
    # [1] - Buffer.
    # [1] - Merge updates. >> This will break datloader integration, KV cache
    # [1] - Serial Dataloader integration.
    # [0] - Mini problem.
    #   [0] - Data dist. (...)
    #   [0] - Check the NN code. (add training example, check cache)
    #   [0] - Generation code. (...)
    # [0] - BOS Dataloader integration.
    # [0] - Adapting to mid-training / SFT.
    # ----------------------------------------------------------------------------------
    # - Probs:
    #   - Check del / insert probs.
    #   - No corruption guarding probs.
    # - Also - give the model a mask to show which tokens it can edit (i.e. early block
    #     might be fixed) --> Maybe - just mask out logits at inference time?
