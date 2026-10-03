"""Causal Edit Serialization: edit scripts, token streams and training targets.

Source-coordinate edit scripts
------------------------------
``SequenceEdit`` pairs a source token sequence with an ordered list of edits:

* ``Insert(pos, content)`` inserts at boundary ``0 <= pos <= len(sequence)``.
* ``Delete(start, end)`` removes the half-open span ``[start, end)``.
* ``Replace(start, end, content)`` replaces that span with a payload.
* ``Copy(start, end, dest)`` inserts a copy of the source span at ``dest``.
* ``Move(start, end, dest)`` removes the source span and reinserts it at ``dest``.

All coordinates refer to the original source, not to an incrementally edited
output. Edits must satisfy the ordering and overlap checks in ``_validate_edits``;
move operations are lowered to copy/delete operations for those checks and
execution. A move destination cannot lie strictly inside its source span.

Serialized programs
-------------------
``TokensOperations.serialise`` inserts ``INDL_BOS`` into the source (at position
zero by default), appends ``EDIT_START``, then the edit blocks and ``EDIT_END``.
The default, role-specific token grammar is:

    insert:  ins-open, ins-loc, payload...
    delete:  del-open, del-start, del-end
    replace: replace-open, replace-start, replace-end, payload..., replace-eos
    copy:    copy-open, copy-start, copy-end, copy-dest
    move:    move-open, move-start, move-end, move-dest

Insertion payloads end implicitly at the next edit opener or ``EDIT_END``.
Replacement payloads have an explicit ``replace-eos`` terminator. There is no
insertion-EOS token in this implementation. The shared-pointer ablation reuses
one pointer token ID for the location roles; the surrounding grammar still
identifies each role.

Serialized addresses and dual RoPE
----------------------------------
``SerialisedData`` has parallel ``tokens`` and ``positions`` arrays. Its addresses
refer to token indices in the serialized source, including the inserted
``INDL_BOS`` marker. This differs from the source-coordinate edit API:

* Span-start pointers address the first token of the span. Span-end pointers
  address the last token, so the serialized endpoint is inclusive.
* An insertion/copy/move destination addresses the token immediately before the
  destination boundary. ``INDL_BOS`` is the anchor for boundary zero.
* Insertion payload tokens inherit their destination anchor. Replacement
  payload tokens inherit their span-start anchor.
* Source tokens and other control tokens use their own serialized indices.

For example, deleting ``x`` and inserting ``d e`` after ``c`` gives the output
``a b c d e f g``. With ``IB = INDL_BOS``, ``ES = EDIT_START``, ``EE = EDIT_END``,
and abbreviated delete/insert token names, the serialized stream is:

    tokens:     IB a b x c f g ES DO DS DE IO IL d  e  EE
    index i:     0 1 2 3 4 5 6  7  8  9 10 11 12 13 14 15
    address p:   0 1 2 3 4 5 6  7  8  3  3 11  4  4  4 15

The decoder in ``gpt.py`` combines stream index ``i`` and address index ``p``
through dual RoPE. These addresses do not define a sparse attention mask:
attention remains causal over the serialized history. ``engine_extended.py``
implements the grammar and pointer constraints during generation.

Training alignment
------------------
``prepare_training_example`` shifts the token stream for next-token prediction.
For each pointer token it gathers the preceding hidden-state index as the
localization query and uses that pointer's address as the target. Pointer token
identities receive no vocabulary loss; their addresses retain localization
supervision. Vocabulary masks also exclude ``INDL_BOS``, deleted/replaced source
spans and their following token, and the token following an insertion anchor.
These are loss masks, not attention masks. The returned fixed-size pointer
buffer reports overflow when a program has more pointers than it can hold.

The legacy ``causal_serialisation.py`` is still used for shared corruption and
reservoir utilities and for the older insertion/deletion-only loader/decoder.
This module implements the five-operation CES format used by the released models.
"""
from typing import NamedTuple, Union

import numpy as np
import pandas as pd
import tiktoken

Index = int
Length = int
Sequence = list[int | str] | np.ndarray
DenseDeletionMask = np.ndarray[bool]
Positions = np.ndarray
RNG = Union[np.random.Generator, np.random.RandomState]


DEL_OPEN = "<|del-open|>"
DEL_START = "<|del-start|>"
DEL_END = "<|del-end|>"
INS_OPEN = "<|ins-open|>"
INS_LOC = "<|ins-loc|>"
REPLACE_OPEN = "<|replace-open|>"
REPLACE_START = "<|replace-start|>"
REPLACE_END = "<|replace-end|>"
REPLACE_EOS = "<|replace-eos|>"
COPY_OPEN = "<|copy-open|>"
COPY_START = "<|copy-start|>"
COPY_END = "<|copy-end|>"
COPY_DEST = "<|copy-dest|>"
MOVE_OPEN = "<|move-open|>"
MOVE_START = "<|move-start|>"
MOVE_END = "<|move-end|>"
MOVE_DEST = "<|move-dest|>"
EDIT_START = "<|edit-start|>"
EDIT_END = "<|edit-end|>"
INDL_BOS = "<|indel-bos|>"

NUM_EDIT_TOKENS = 2

POSTION_TOKENS = [
    DEL_START,
    DEL_END,
    INS_LOC,
    REPLACE_START,
    REPLACE_END,
    COPY_START,
    COPY_END,
    COPY_DEST,
    MOVE_START,
    MOVE_END,
    MOVE_DEST,
]

# CES pointer-token ablations. ``role_specific`` is the released/original CES
# representation. ``shared`` reuses the existing INS_LOC vocabulary row for
# every pointer-bearing grammar position, keeping the tokenizer vocabulary and
# model parameter count fixed so that only the serialization changes.
CES_POINTER_TOKEN_MODES = ("role_specific", "shared")
SHARED_POINTER_TOKEN = INS_LOC


def validate_ces_pointer_token_mode(mode: str) -> str:
    if mode not in CES_POINTER_TOKEN_MODES:
        raise ValueError(
            f"Unknown CES pointer-token mode {mode!r}; expected one of "
            f"{CES_POINTER_TOKEN_MODES}."
        )
    return mode


INDEL_SPECIALS = [
    DEL_OPEN,
    DEL_START,
    DEL_END,
    INS_OPEN,
    INS_LOC,
    REPLACE_OPEN,
    REPLACE_START,
    REPLACE_END,
    REPLACE_EOS,
    COPY_OPEN,
    COPY_START,
    COPY_END,
    COPY_DEST,
    MOVE_OPEN,
    MOVE_START,
    MOVE_END,
    MOVE_DEST,
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
    REPLACE_OPEN: "magenta",
    REPLACE_START: "bold magenta",
    REPLACE_END: "magenta",
    REPLACE_EOS: "magenta",
    COPY_OPEN: "cyan",
    COPY_START: "bold cyan",
    COPY_END: "cyan",
    COPY_DEST: "cyan bold",
    MOVE_OPEN: "bright_blue",
    MOVE_START: "bold bright_blue",
    MOVE_END: "bright_blue",
    MOVE_DEST: "bright_blue bold",
    EDIT_START: "yellow",
    EDIT_END: "yellow",
    INDL_BOS: "blue",
}

########################################################################################
####                                      TYPE                                      ####
########################################################################################

class Insert(NamedTuple):
    """Insert ``content`` at source boundary ``pos``."""
    pos: Index
    content: Sequence


class Delete(NamedTuple):
    """Delete the half-open source span ``[start, end)``."""
    start: Index
    end: Index


class Replace(NamedTuple):
    """Replace the half-open source span ``[start, end)`` with ``content``."""
    start: Index
    end: Index
    content: Sequence


class Copy(NamedTuple):
    """Copy source span ``[start, end)`` and insert it at source boundary ``dest``."""
    start: Index
    end: Index
    dest: Index


class Move(NamedTuple):
    """Move source span ``[start, end)`` so it is reinserted at source boundary ``dest``."""
    start: Index
    end: Index
    dest: Index


Edit = Insert | Delete | Replace | Copy | Move


class SequenceEdit(NamedTuple):
    """A base sequence plus an ordered left-to-right edit script."""
    sequence: Sequence
    edits: list[Edit]


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


        toks = np.array(tokeniser.decode_batch([[t] for t in self.tokens]))
        colours = _serial_debug_colours(toks=toks, positions=self.positions)

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


def _serial_debug_colours(toks: np.ndarray, positions: np.ndarray) -> list[str | None]:
    """Colour source spans and edit payloads for debug printing."""
    colours: list[str | None] = [None] * len(toks)
    source_pos_colours: dict[int, str] = {}

    if EDIT_START not in toks:
        return [STYLE.get(t, None) for t in toks]

    edit_start_ix = int(np.where(toks == EDIT_START)[0][0])
    edit_end_matches, = np.where(toks == EDIT_END)
    edit_end_ix = int(edit_end_matches[0]) if len(edit_end_matches) else len(toks)

    for ds, de in zip(
        positions[toks == DEL_START],
        positions[toks == DEL_END],
    ):
        for pos in range(int(ds), int(de) + 1):
            source_pos_colours[pos] = STYLE[DEL_START]

    for rs, re in zip(
        positions[toks == REPLACE_START],
        positions[toks == REPLACE_END],
    ):
        for pos in range(int(rs), int(re) + 1):
            source_pos_colours[pos] = STYLE[REPLACE_START]

    for cs, ce in zip(
        positions[toks == COPY_START],
        positions[toks == COPY_END],
    ):
        for pos in range(int(cs), int(ce) + 1):
            source_pos_colours.setdefault(pos, STYLE[COPY_START])

    for cd in positions[toks == COPY_DEST]:
        source_pos_colours.setdefault(int(cd), STYLE[COPY_DEST])

    for ms, me in zip(
        positions[toks == MOVE_START],
        positions[toks == MOVE_END],
    ):
        for pos in range(int(ms), int(me) + 1):
            source_pos_colours.setdefault(pos, STYLE[MOVE_START])

    for md in positions[toks == MOVE_DEST]:
        source_pos_colours.setdefault(int(md), STYLE[MOVE_DEST])

    edit_toks = toks[edit_start_ix + 1:edit_end_ix]
    edit_pos = positions[edit_start_ix + 1:edit_end_ix]
    ins_open_rel, = np.where(edit_toks == INS_OPEN)
    ins_palette = _alternating_green_colormap(len(ins_open_rel))

    rel = 0
    ins_ix = 0
    while rel < len(edit_toks):
        token = edit_toks[rel]
        abs_ix = edit_start_ix + 1 + rel
        if token == INS_OPEN and rel + 1 < len(edit_toks) and edit_toks[rel + 1] == INS_LOC:
            colour = ins_palette[ins_ix]
            ins_ix += 1
            anchor = int(edit_pos[rel + 1])
            source_pos_colours.setdefault(anchor, colour)
            rel += 2
            while rel < len(edit_toks) and edit_toks[rel] not in (
                DEL_OPEN, INS_OPEN, REPLACE_OPEN, COPY_OPEN, MOVE_OPEN
            ):
                colours[edit_start_ix + 1 + rel] = colour
                rel += 1
            continue
        if token == REPLACE_OPEN:
            rel += 3
            while rel < len(edit_toks) and edit_toks[rel] != REPLACE_EOS:
                colours[edit_start_ix + 1 + rel] = STYLE[REPLACE_START]
                rel += 1
            if rel < len(edit_toks):
                colours[edit_start_ix + 1 + rel] = STYLE[REPLACE_EOS]
                rel += 1
            continue
        if token == COPY_OPEN:
            if rel + 3 < len(edit_toks):
                colours[edit_start_ix + 1 + rel + 1] = STYLE[COPY_START]
                colours[edit_start_ix + 1 + rel + 2] = STYLE[COPY_END]
                colours[edit_start_ix + 1 + rel + 3] = STYLE[COPY_DEST]
            rel += 4
            continue
        if token == MOVE_OPEN:
            if rel + 3 < len(edit_toks):
                colours[edit_start_ix + 1 + rel + 1] = STYLE[MOVE_START]
                colours[edit_start_ix + 1 + rel + 2] = STYLE[MOVE_END]
                colours[edit_start_ix + 1 + rel + 3] = STYLE[MOVE_DEST]
            rel += 4
            continue
        colours[abs_ix] = STYLE.get(token, None)
        rel += 1

    for i, (tok, pos) in enumerate(zip(toks, positions)):
        if colours[i] is None:
            colours[i] = source_pos_colours.get(int(pos), STYLE.get(tok, None))
    return colours


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
    colours = np.array(_serial_debug_colours(tok, pos), dtype=object)
    sdf["in-tok"] = colours
    sdf["in-pos"] = colours
    sdf.targ = sdf.targ.apply(lambda x: None)
    sdf["mask"] = sdf["mask"].apply(lambda x: None if x else "#8B50EB")
    sdf["ref"] = sdf["ref"].apply(lambda x: "#EB50B7" if len(x) else None)
    return sdf


def print_training_examples(training_examples, tokeniser, max_rows=1025):
    dfs = [da.df(tokeniser) for da in training_examples]
    style_dfs = [colour_df(df) for df in dfs]
    print_df_grid_rich(dfs, style_dfs, max_rows=max_rows)


########################################################################################
####                                   OPERATIONS                                   ####
########################################################################################


def _edit_key(edit: Edit) -> tuple[int, int]:
    """Sort inserts before span edits when they share the same left boundary."""
    if isinstance(edit, Insert):
        return (edit.pos, 0)
    if isinstance(edit, Copy):
        return (edit.dest, 0)
    if isinstance(edit, Move):
        return (min(edit.start, edit.dest), 0)
    return (edit.start, 1)


def _validate_edits(sequence: Sequence, edits: list[Edit]) -> None:
    """Validate that edits form a well-ordered source-coordinate edit script."""
    n = len(sequence)
    last_boundary = 0
    last_span_end = 0
    last_key = (-1, -1)
    for edit in edits:
        key = _edit_key(edit)
        if key < last_key:
            raise ValueError("Edits must be sorted left-to-right in source coordinates.")
        last_key = key
        if isinstance(edit, Insert):
            if not (0 <= edit.pos <= n):
                raise ValueError(f"Invalid insertion boundary: {edit.pos}")
            if last_span_end > edit.pos > last_boundary:
                raise ValueError(f"Insertion falls inside a removed span: {edit.pos}")
            last_boundary = edit.pos
            continue
        if isinstance(edit, Copy):
            if not (0 <= edit.start < edit.end <= n):
                raise ValueError(f"Invalid copy source span: {(edit.start, edit.end)}")
            if not (0 <= edit.dest <= n):
                raise ValueError(f"Invalid copy destination boundary: {edit.dest}")
            if last_span_end > edit.dest > last_boundary:
                raise ValueError(f"Copy destination falls inside a removed span: {edit.dest}")
            last_boundary = edit.dest
            continue
        if isinstance(edit, Move):
            if not (0 <= edit.start < edit.end <= n):
                raise ValueError(f"Invalid move source span: {(edit.start, edit.end)}")
            if not (0 <= edit.dest <= n):
                raise ValueError(f"Invalid move destination boundary: {edit.dest}")
            if edit.start < edit.dest < edit.end:
                raise ValueError(
                    f"Move destination falls strictly inside the moved span: {edit.dest}"
                )
            last_boundary = min(edit.start, edit.dest)
            if edit.start < last_span_end:
                raise ValueError(f"Overlapping span edits at {(edit.start, edit.end)}")
            last_span_end = edit.end
            continue
        if not (0 <= edit.start < edit.end <= n):
            raise ValueError(f"Invalid span edit: {(edit.start, edit.end)}")
        if edit.start < last_span_end:
            raise ValueError(f"Overlapping span edits at {(edit.start, edit.end)}")
        last_boundary = edit.start
        last_span_end = edit.end


def _validate_move_edits(sequence: Sequence, edits: list[Edit]) -> None:
    """Validate move-specific constraints before moves are lowered."""
    n = len(sequence)
    for edit in edits:
        if not isinstance(edit, Move):
            continue
        if not (0 <= edit.start < edit.end <= n):
            raise ValueError(f"Invalid move source span: {(edit.start, edit.end)}")
        if not (0 <= edit.dest <= n):
            raise ValueError(f"Invalid move destination boundary: {edit.dest}")
        if edit.start < edit.dest < edit.end:
            raise ValueError(
                f"Move destination falls strictly inside the moved span: {edit.dest}"
            )


def _lower_moves(seq_edit: SequenceEdit) -> SequenceEdit:
    """Expand move edits into copy+delete edits for execution."""
    lowered: list[Edit] = []
    for edit in seq_edit.edits:
        if isinstance(edit, Move):
            lowered.append(Copy(edit.start, edit.end, edit.dest))
            lowered.append(Delete(edit.start, edit.end))
        else:
            lowered.append(edit)
    lowered.sort(key=_edit_key)
    return SequenceEdit(sequence=np.array(seq_edit.sequence), edits=lowered)


def _lower_moves_with_origins(seq_edit: SequenceEdit) -> list[tuple[Edit, int | None]]:
    """Expand move edits while retaining enough origin information for inverse moves."""
    lowered: list[tuple[Edit, int | None]] = []
    for origin_ix, edit in enumerate(seq_edit.edits):
        if isinstance(edit, Move):
            lowered.append((Copy(edit.start, edit.end, edit.dest), origin_ix))
            lowered.append((Delete(edit.start, edit.end), origin_ix))
        else:
            lowered.append((edit, None))
    lowered.sort(key=lambda item: _edit_key(item[0]))
    return lowered


def _apply_edit_script(
    seq_edit: SequenceEdit,
    *,
    preserve_inverse_moves: bool = False,
) -> tuple[np.ndarray, list[Edit]]:
    """Apply an edit script and build the inverse script in output coordinates."""
    sequence = np.array(seq_edit.sequence)
    _validate_move_edits(sequence, list(seq_edit.edits))
    lowered = _lower_moves_with_origins(seq_edit)
    edits = [edit for edit, _origin in lowered]
    _validate_edits(sequence, edits)

    result = []
    inverse_edits: list[Edit | None] = []
    move_inverse_parts: dict[int, dict[str, int]] = {}
    src_cursor = 0
    out_cursor = 0

    for edit, move_origin in lowered:
        if isinstance(edit, Insert):
            untouched = sequence[src_cursor:edit.pos]
            result.extend(untouched)
            out_cursor += len(untouched)
            payload = np.array(edit.content)
            if len(payload):
                result.extend(payload)
                inverse_edits.append(Delete(out_cursor, out_cursor + len(payload)))
                out_cursor += len(payload)
            src_cursor = edit.pos
            continue
        if isinstance(edit, Copy):
            untouched = sequence[src_cursor:edit.dest]
            result.extend(untouched)
            out_cursor += len(untouched)
            payload = np.array(sequence[edit.start:edit.end])
            if len(payload):
                result.extend(payload)
                if preserve_inverse_moves and move_origin is not None:
                    part = move_inverse_parts.setdefault(move_origin, {})
                    part["copy_start"] = out_cursor
                    part["copy_end"] = out_cursor + len(payload)
                else:
                    inverse_edits.append(Delete(out_cursor, out_cursor + len(payload)))
                out_cursor += len(payload)
            src_cursor = edit.dest
            continue

        untouched = sequence[src_cursor:edit.start]
        result.extend(untouched)
        out_cursor += len(untouched)
        removed = np.array(sequence[edit.start:edit.end])
        if isinstance(edit, Delete):
            if len(removed):
                if preserve_inverse_moves and move_origin is not None:
                    part = move_inverse_parts.setdefault(move_origin, {})
                    part["delete_dest"] = out_cursor
                    part["inverse_index"] = len(inverse_edits)
                    inverse_edits.append(None)
                else:
                    inverse_edits.append(Insert(out_cursor, removed))
        else:
            payload = np.array(edit.content)
            result.extend(payload)
            inverse_edits.append(Replace(out_cursor, out_cursor + len(payload), removed))
            out_cursor += len(payload)
        src_cursor = edit.end

    tail = sequence[src_cursor:]
    result.extend(tail)
    if preserve_inverse_moves:
        for part in move_inverse_parts.values():
            if {"copy_start", "copy_end", "delete_dest"} <= set(part):
                inverse_edits[part["inverse_index"]] = Move(
                    part["copy_start"], part["copy_end"], part["delete_dest"]
                )
    return np.array(result), [edit for edit in inverse_edits if edit is not None]


def complement(seq_edit: SequenceEdit) -> SequenceEdit:
    """Return the exact inverse edit script for ``seq_edit``."""
    resolved, inverse_edits = _apply_edit_script(seq_edit, preserve_inverse_moves=True)
    return SequenceEdit(sequence=resolved, edits=inverse_edits)


def resolve(seq_edit: SequenceEdit) -> Sequence:
    """Apply ``seq_edit`` and return the edited sequence."""
    resolved, _ = _apply_edit_script(seq_edit)
    return resolved


class TokensOperations:
    def __init__(
        self,
        tokeniser,
        marginal_token_probs: np.ndarray | None = None,
        marginal_tokens: np.ndarray | None = None,
        pointer_token_mode: str = "role_specific",
    ):
        self.tokeniser = tokeniser
        self.pointer_token_mode = validate_ces_pointer_token_mode(pointer_token_mode)
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
        self.replace_open_token = tokeniser._special_tokens[REPLACE_OPEN]
        self.replace_start_token = tokeniser._special_tokens[REPLACE_START]
        self.replace_end_token = tokeniser._special_tokens[REPLACE_END]
        self.replace_eos_token = tokeniser._special_tokens[REPLACE_EOS]
        self.copy_open_token = tokeniser._special_tokens[COPY_OPEN]
        self.copy_start_token = tokeniser._special_tokens[COPY_START]
        self.copy_end_token = tokeniser._special_tokens[COPY_END]
        self.copy_dest_token = tokeniser._special_tokens[COPY_DEST]
        self.move_open_token = tokeniser._special_tokens[MOVE_OPEN]
        self.move_start_token = tokeniser._special_tokens[MOVE_START]
        self.move_end_token = tokeniser._special_tokens[MOVE_END]
        self.move_dest_token = tokeniser._special_tokens[MOVE_DEST]
        if self.pointer_token_mode == "shared":
            shared_pointer_token = tokeniser._special_tokens[SHARED_POINTER_TOKEN]
            self.deletion_start_token = shared_pointer_token
            self.deletion_end_token = shared_pointer_token
            self.insertion_loc_token = shared_pointer_token
            self.replace_start_token = shared_pointer_token
            self.replace_end_token = shared_pointer_token
            self.copy_start_token = shared_pointer_token
            self.copy_end_token = shared_pointer_token
            self.copy_dest_token = shared_pointer_token
            self.move_start_token = shared_pointer_token
            self.move_end_token = shared_pointer_token
            self.move_dest_token = shared_pointer_token
        self.edit_start_token = tokeniser._special_tokens[EDIT_START]
        self.edit_end_token = tokeniser._special_tokens[EDIT_END]
        self.indel_bos_token = tokeniser._special_tokens[INDL_BOS]
        self.eos_index = None  # TODO: Remove support for this. # tokeniser._special_tokens.get(EOS, None)  # Not always necessary

        self.special_tokens = np.array([
            self.deletion_open_token,
            self.deletion_start_token,
            self.deletion_end_token,
            self.insertion_open_token,
            self.insertion_loc_token,
            self.replace_open_token,
            self.replace_start_token,
            self.replace_end_token,
            self.replace_eos_token,
            self.copy_open_token,
            self.copy_start_token,
            self.copy_end_token,
            self.copy_dest_token,
            self.move_open_token,
            self.move_start_token,
            self.move_end_token,
            self.move_dest_token,
            self.edit_start_token,
            self.edit_end_token,
            self.indel_bos_token,
        ])

        self.deletion_cycle = np.array([
            self.deletion_open_token,
            self.deletion_start_token,
            self.deletion_end_token,
        ])
        self.edit_opener_tokens = np.array([
            self.deletion_open_token,
            self.insertion_open_token,
            self.replace_open_token,
            self.copy_open_token,
            self.move_open_token,
        ])

    def _pointer_indices_by_role(self, tokens: np.ndarray) -> dict[str, np.ndarray]:
        """Locate pointer tokens by grammatical role rather than token identity.

        Token equality is sufficient for the original CES vocabulary, but is
        ambiguous when every pointer role shares one token ID. The edit opener
        and finite serialization grammar make every role contextual and fully
        recoverable.
        """
        roles: dict[str, list[int]] = {
            "deletion_start": [],
            "deletion_end": [],
            "insertion_loc": [],
            "replace_start": [],
            "replace_end": [],
            "copy_start": [],
            "copy_end": [],
            "copy_dest": [],
            "move_start": [],
            "move_end": [],
            "move_dest": [],
        }
        edit_starts = np.flatnonzero(tokens == self.edit_start_token)
        if len(edit_starts) == 0:
            return {name: np.array([], dtype=np.int64) for name in roles}
        edit_ends = np.flatnonzero(tokens == self.edit_end_token)

        def scan_program(start: int, end: int) -> None:
            i = start + 1

            def require_pointer(index: int, expected: int, role: str) -> None:
                if index >= end or int(tokens[index]) != expected:
                    raise ValueError(
                        f"Malformed CES program: expected {role} at token index {index}."
                    )
                roles[role].append(index)

            while i < end:
                token = int(tokens[i])
                if token == self.deletion_open_token:
                    require_pointer(i + 1, self.deletion_start_token, "deletion_start")
                    require_pointer(i + 2, self.deletion_end_token, "deletion_end")
                    i += 3
                elif token == self.insertion_open_token:
                    require_pointer(i + 1, self.insertion_loc_token, "insertion_loc")
                    i += 2
                    while i < end and int(tokens[i]) not in self.edit_opener_tokens:
                        i += 1
                elif token == self.replace_open_token:
                    require_pointer(i + 1, self.replace_start_token, "replace_start")
                    require_pointer(i + 2, self.replace_end_token, "replace_end")
                    i += 3
                    while i < end and int(tokens[i]) != self.replace_eos_token:
                        i += 1
                    if i >= end:
                        raise ValueError(
                            "Malformed CES program: replacement has no replace-eos token."
                        )
                    i += 1
                elif token == self.copy_open_token:
                    require_pointer(i + 1, self.copy_start_token, "copy_start")
                    require_pointer(i + 2, self.copy_end_token, "copy_end")
                    require_pointer(i + 3, self.copy_dest_token, "copy_dest")
                    i += 4
                elif token == self.move_open_token:
                    require_pointer(i + 1, self.move_start_token, "move_start")
                    require_pointer(i + 2, self.move_end_token, "move_end")
                    require_pointer(i + 3, self.move_dest_token, "move_dest")
                    i += 4
                else:
                    raise ValueError(
                        f"Malformed CES program: unexpected token {token} at edit index {i}."
                    )

        for start in edit_starts:
            later_ends = edit_ends[edit_ends > start]
            if len(later_ends) == 0:
                raise ValueError(
                    f"Malformed CES program: edit-start at {int(start)} has no edit-end."
                )
            scan_program(int(start), int(later_ends[0]))

        return {
            name: np.asarray(indices, dtype=np.int64)
            for name, indices in roles.items()
        }

    @staticmethod
    def _source_to_serial_token_ix(source_ix: int, indel_bos_pos: int) -> int:
        """Map a source token index to the serial token index after inserting indel-bos."""
        return source_ix if source_ix < indel_bos_pos else source_ix + 1

    @staticmethod
    def _serial_token_ix_to_source(serial_ix: int, indel_bos_pos: int) -> int:
        """Invert ``_source_to_serial_token_ix`` for real source tokens."""
        if serial_ix == indel_bos_pos:
            raise ValueError("Cannot map indel-bos token to a source token index.")
        return serial_ix if serial_ix < indel_bos_pos else serial_ix - 1

    def _insert_pos_to_anchor(self, pos: int, source_length: int, indel_bos_pos: int) -> int:
        """Map a source boundary to the serial anchor token used by ``INS_LOC``."""
        if not (0 <= pos <= source_length):
            raise ValueError(f"Invalid insertion boundary: {pos}")
        if pos == 0:
            return indel_bos_pos
        return self._source_to_serial_token_ix(pos - 1, indel_bos_pos)

    def _anchor_to_insert_pos(self, anchor: int, indel_bos_pos: int) -> int:
        """Recover a source insertion boundary from a serial ``INS_LOC`` anchor."""
        if anchor == indel_bos_pos:
            return 0
        return self._serial_token_ix_to_source(anchor, indel_bos_pos) + 1

    def deserialise(self, serialised_data: SerialisedData) -> SequenceEdit:
        """Deserialise into an ordered source-coordinate edit script."""
        sequence = serialised_data.tokens
        positions = serialised_data.positions
        start, = np.where(np.array(sequence) == self.edit_start_token)
        if len(start) == 0:
            print("No edits found.")
            return SequenceEdit(np.array(sequence), [])
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
        ibos_mask = standard_sequence == self.indel_bos_token
        if ibos_mask.sum() != 1:
            raise ValueError(
                f"There should be exactly one indel-bos token, found: {ibos_mask.sum()}"
            )
        indel_bos_pos = int(np.where(ibos_mask)[0][0])
        source_sequence = standard_sequence[~ibos_mask]

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

        parsed_edits: list[Edit] = []
        i = 0
        while i < len(edits):
            token = edits[i]
            if token == self.deletion_open_token:
                seg = edits[i:i+3]
                seg_pos = edit_positions[i:i+3]
                if len(seg) == 3 and np.array_equal(seg, self.deletion_cycle):
                    del_start = self._serial_token_ix_to_source(int(seg_pos[1]), indel_bos_pos)
                    del_end = self._serial_token_ix_to_source(int(seg_pos[2]), indel_bos_pos) + 1
                    parsed_edits.append(Delete(del_start, del_end))
                    i += 3
                    continue
            elif token == self.insertion_open_token:
                j = i + 1
                while j < len(edits) and edits[j] not in self.edit_opener_tokens:
                    j += 1
                if j > i + 1 and edits[i + 1] == self.insertion_loc_token:
                    pos = self._anchor_to_insert_pos(int(edit_positions[i + 1]), indel_bos_pos)
                    parsed_edits.append(Insert(pos, np.array(edits[i + 2:j])))
                    i = j
                    continue
            elif token == self.replace_open_token:
                eos_rel, = np.where(edits[i:] == self.replace_eos_token)
                if len(eos_rel):
                    j = i + int(eos_rel[0])
                    if j >= i + 3 and edits[i + 1] == self.replace_start_token and edits[i + 2] == self.replace_end_token:
                        rep_start = self._serial_token_ix_to_source(int(edit_positions[i + 1]), indel_bos_pos)
                        rep_end = self._serial_token_ix_to_source(int(edit_positions[i + 2]), indel_bos_pos) + 1
                        parsed_edits.append(
                            Replace(rep_start, rep_end, np.array(edits[i + 3:j]))
                        )
                        i = j + 1
                        continue
            elif token == self.copy_open_token:
                seg = edits[i:i+4]
                seg_pos = edit_positions[i:i+4]
                if (
                    len(seg) == 4
                    and seg[1] == self.copy_start_token
                    and seg[2] == self.copy_end_token
                    and seg[3] == self.copy_dest_token
                ):
                    copy_start = self._serial_token_ix_to_source(int(seg_pos[1]), indel_bos_pos)
                    copy_end = self._serial_token_ix_to_source(int(seg_pos[2]), indel_bos_pos) + 1
                    copy_dest = self._anchor_to_insert_pos(int(seg_pos[3]), indel_bos_pos)
                    parsed_edits.append(Copy(copy_start, copy_end, copy_dest))
                    i += 4
                    continue
            elif token == self.move_open_token:
                seg = edits[i:i+4]
                seg_pos = edit_positions[i:i+4]
                if (
                    len(seg) == 4
                    and seg[1] == self.move_start_token
                    and seg[2] == self.move_end_token
                    and seg[3] == self.move_dest_token
                ):
                    move_start = self._serial_token_ix_to_source(int(seg_pos[1]), indel_bos_pos)
                    move_end = self._serial_token_ix_to_source(int(seg_pos[2]), indel_bos_pos) + 1
                    move_dest = self._anchor_to_insert_pos(int(seg_pos[3]), indel_bos_pos)
                    parsed_edits.append(Move(move_start, move_end, move_dest))
                    i += 4
                    continue
            i += 1

        _validate_move_edits(source_sequence, parsed_edits)
        _validate_edits(
            source_sequence,
            _lower_moves(SequenceEdit(source_sequence, parsed_edits)).edits,
        )
        return SequenceEdit(sequence=np.array(source_sequence), edits=parsed_edits)

    def serialise(
        self, sequence_edit: SequenceEdit, indel_bos_pos: int = 0
    ) -> SerialisedData:
        """Serialise an ordered source-coordinate edit script.

        The token-level format keeps the causal edit blocks:
        * delete: ``[del-open, del-start, del-end]``
        * insert: ``[ins-open, ins-loc, ...payload]``
        * replace: ``[replace-open, replace-start, replace-end, ...payload, replace-eos]``
        * copy: ``[copy-open, copy-start, copy-end, copy-dest]``
        * move: ``[move-open, move-start, move-end, move-dest]``

        The attached positions remain in serial token coordinates so the model can
        point back into the visible source portion of the sequence.
        """
        sequence = np.array(sequence_edit.sequence)
        edits = list(sequence_edit.edits)
        _validate_move_edits(sequence, edits)
        _validate_edits(sequence, _lower_moves(SequenceEdit(sequence, edits)).edits)
        sequence = np.concatenate([
            sequence[:indel_bos_pos],
            np.array([self.indel_bos_token]),
            sequence[indel_bos_pos:],
        ])

        #
        eos = self.eos_index is not None
        extra = 1 + int(eos)

        serialised_edits = []
        replace_head = [self.replace_open_token, self.replace_start_token, self.replace_end_token]
        insert_head = [self.insertion_open_token, self.insertion_loc_token]
        for ordinal, edit in enumerate(edits):
            if isinstance(edit, Delete):
                pos = np.array([
                    -1,
                    self._source_to_serial_token_ix(edit.start, indel_bos_pos),
                    self._source_to_serial_token_ix(edit.end - 1, indel_bos_pos),
                ])
                serialised_edits.append((ordinal, np.array(self.deletion_cycle), pos))
            elif isinstance(edit, Insert):
                anchor = self._insert_pos_to_anchor(edit.pos, len(sequence_edit.sequence), indel_bos_pos)
                toks = np.array(edit.content)
                pos = np.array([-1, anchor] + [anchor] * len(toks))
                serialised_edits.append((ordinal, np.concatenate([insert_head, toks]), pos))
            elif isinstance(edit, Copy):
                source_start = self._source_to_serial_token_ix(edit.start, indel_bos_pos)
                source_end = self._source_to_serial_token_ix(edit.end - 1, indel_bos_pos)
                dest_anchor = self._insert_pos_to_anchor(
                    edit.dest, len(sequence_edit.sequence), indel_bos_pos
                )
                tok = np.array([
                    self.copy_open_token,
                    self.copy_start_token,
                    self.copy_end_token,
                    self.copy_dest_token,
                ])
                pos = np.array([-1, source_start, source_end, dest_anchor])
                serialised_edits.append((ordinal, tok, pos))
            elif isinstance(edit, Move):
                source_start = self._source_to_serial_token_ix(edit.start, indel_bos_pos)
                source_end = self._source_to_serial_token_ix(edit.end - 1, indel_bos_pos)
                dest_anchor = self._insert_pos_to_anchor(
                    edit.dest, len(sequence_edit.sequence), indel_bos_pos
                )
                tok = np.array([
                    self.move_open_token,
                    self.move_start_token,
                    self.move_end_token,
                    self.move_dest_token,
                ])
                pos = np.array([-1, source_start, source_end, dest_anchor])
                serialised_edits.append((ordinal, tok, pos))
            else:
                anchor = self._source_to_serial_token_ix(edit.start, indel_bos_pos)
                end_pos = self._source_to_serial_token_ix(edit.end - 1, indel_bos_pos)
                toks = np.array(edit.content)
                pos = np.array([-1, anchor, end_pos] + [anchor] * len(toks) + [-1])
                tok = np.concatenate([replace_head, toks, np.array([self.replace_eos_token])])
                serialised_edits.append((ordinal, tok, pos))

        # Serialise the edits.
        seq = [
            sequence,
            np.array([self.eos_index] if eos else [], dtype=sequence.dtype),
            np.array([self.edit_start_token], dtype=sequence.dtype),
        ]
        pos = [np.arange(len(sequence) + extra, dtype=np.int32)]  # Next position pointer
        for _, tok, pt in serialised_edits:
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
        pointer_roles = self._pointer_indices_by_role(tok)

        # Pointer-token positions in the sequence (these are the positions whose
        # tix we want as targets). Sort to preserve causal sequence order.
        ptr_indices = np.sort(np.concatenate(list(pointer_roles.values())))
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

        # Vocabulary loss masks are applied before next-token shifting.
        token_loss_weights = np.ones(len(tok), dtype=bool)

        for ds, de in zip(
            tix[pointer_roles["deletion_start"]],
            tix[pointer_roles["deletion_end"]],
        ):
            # +2 so that we don't train on deleted sections - or the token after it.
            token_loss_weights[ds:de + 2] = False

        for rs, re in zip(
            tix[pointer_roles["replace_start"]],
            tix[pointer_roles["replace_end"]],
        ):
            token_loss_weights[rs:re + 2] = False

        # Also, don't train on the token after an insertion point.
        for il in tix[pointer_roles["insertion_loc"]]:
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
        REPLACE_OPEN,
        REPLACE_START,
        REPLACE_END,
        REPLACE_EOS,
        COPY_OPEN,
        COPY_START,
        COPY_END,
        COPY_DEST,
        MOVE_OPEN,
        MOVE_START,
        MOVE_END,
        MOVE_DEST,
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
