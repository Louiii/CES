"""Generate and persist source-coordinate edit scripts with gpt-5-nano.

The model emits a compact edit DSL anchored against the original source text.
This module parses those lines into ``SequenceEdit`` objects, validates them
locally, and can write the results to JSONL or parquet for later statistics and
training-example generation.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.json as pajson
import pyarrow.parquet as pq
import requests

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanochat.causal_serialisation_extended import (
    Delete,
    Insert,
    Move,
    Replace,
    SequenceEdit,
    _edit_key,
    complement,
    resolve,
)


DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_REASONING_EFFORT = "minimal"
DEFAULT_MAX_OUTPUT_TOKENS = 600
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

PROMPT_VERSION = "v3_region_pointer_pre_post"
GEC_PROMPT_VERSION = f"{PROMPT_VERSION}_gec_exact_v3"
EDIT_OP_NAMES = ("insert", "delete", "replace", "move")
TASK_CORRUPT = "corrupt"
TASK_IMPROVE = "improve"
TASK_CHOICES = (TASK_CORRUPT, TASK_IMPROVE)
REGIME_ATOMIC_LOCAL = "atomic_local"
REGIME_COEDIT_LIKE = "coedit_like"
REGIME_GEC_POSTTRAIN = "gec_posttrain"
REGIME_LONG_DOC_REPAIR = "long_doc_repair"
REGIME_MOVE_ENRICHMENT = "move_enrichment"
REGIME_CHOICES = (
    REGIME_ATOMIC_LOCAL,
    REGIME_COEDIT_LIKE,
    REGIME_GEC_POSTTRAIN,
    REGIME_LONG_DOC_REPAIR,
    REGIME_MOVE_ENRICHMENT,
)

OPERATION_FORMATS = {
    "delete": 'Del(region=RegionInclusive(start_search_phrase="...", start_exact="...", end_search_phrase="...", end_exact="..."))',
    "replace": 'Replace(region=RegionInclusive(start_search_phrase="...", start_exact="...", end_search_phrase="...", end_exact="..."), text="...")',
    "insert": 'Insert(pointer=Pointer(search_phrase="...", pre="...", post="..."), text="...")',
    "move": 'Move(region=RegionInclusive(start_search_phrase="...", start_exact="...", end_search_phrase="...", end_exact="..."), pointer=Pointer(search_phrase="...", pre="...", post="..."))',
}

OP_NAME_TO_ALLOWED = {
    "Del": "delete",
    "Replace": "replace",
    "Insert": "insert",
    "Move": "move",
}


def build_output_format_doc() -> str:
    lines = ["Output only operation lines. One operation per line. Use exactly one of these forms:"]
    lines.extend(["", *[OPERATION_FORMATS[op] for op in EDIT_OP_NAMES]])
    return "\n".join(lines)


def build_gec_output_format_doc() -> str:
    """A local-span DSL that avoids overlapping two-ended region anchors."""

    return "\n".join([
        "Output only one operation line. Use exactly one of these forms:",
        "",
        'DelExact(search_phrase="...", exact="...")',
        'ReplaceExact(search_phrase="...", exact="...", text="...")',
        OPERATION_FORMATS["insert"],
    ])

RULES_DOC = """- All search phrases and anchors must refer to the original source text, not to text created by earlier edits.
- Keep search phrases short and local. Prefer a short reliable exact source phrase that uniquely identifies the location; usually 4 to 16 words is enough.
- `RegionInclusive(...)` is an inclusive source span. `start_search_phrase` and `end_search_phrase` must be copied exactly from the source text.
- In `RegionInclusive(...)`, `start_exact` must occur inside `start_search_phrase` and marks the inclusive start of the span. `end_exact` must occur inside `end_search_phrase` and marks the inclusive end of the span.
- `Pointer(...)` identifies a boundary in the original source text. Always provide both `pre` and `post`.
- In `Pointer(...)`, `search_phrase` must be copied exactly from the source text. The boundary is immediately between `pre` and `post`, and the source text inside `search_phrase` must contain the exact substring `pre + post`.
- Example pointer: if the source text contains `chicken, seafood`, then `Pointer(search_phrase="chicken, seafood", pre="chicken, ", post="seafood")` points to the boundary between `chicken, ` and `seafood`.
- For `Move`, the destination pointer must be outside the selected source region. Do not place the pointer inside the text selected by the region.
- For `Del`, `Replace`, and `Move`, choose `start_exact` near the beginning of the intended span and `end_exact` near the end. Avoid vague anchors like single punctuation marks or generic one-word phrases unless they are truly unique and necessary.
- `text` is the only field that may contain new text not present in the source.
- Output only valid operation lines, with no numbering and no extra commentary."""

GEC_RULES_DOC = """- `search_phrase` must be copied exactly from the source and should usually contain 4 to 16 words of local context.
- For `DelExact` and `ReplaceExact`, `exact` must occur inside `search_phrase` and must contain only the source token or short phrase being changed.
- `text` is the replacement or insertion text and is the only field that may contain new text.
- `Pointer(...)` identifies a source boundary. Its `search_phrase` must contain the exact substring `pre + post`; the insertion occurs between `pre` and `post`.
- Do not output RegionInclusive, Del, Replace, Move, explanations, numbering, markdown, or the full rewritten text."""


@dataclass
class DroppedOperation:
    stage: str
    reason: str
    raw_line: str


@dataclass
class OperationSpec:
    op_name: str
    kwargs: dict[str, Any]
    raw_line: str


@dataclass
class UsageSummary:
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None


@dataclass
class EditExample:
    row_index: int
    task: str
    regime: str
    source_text: str
    corrupted_text: str
    sequence_edit: SequenceEdit
    inverse_sequence_edit: SequenceEdit
    operations: list[str]
    dropped_operations: list[DroppedOperation]
    raw_model_text: str
    raw_response: dict[str, Any]
    usage: UsageSummary


ANSI_RESET = "\033[0m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_BRIGHT_BLUE = "\033[94m"
ANSI_BOLD = "\033[1m"
ANSI_ITALIC = "\033[3m"


def read_documents(
    parquet_path: str,
    text_column: str = "text",
    *,
    row_limit: int | None = 1,
    row_offset: int = 0,
) -> Iterator[tuple[int, str]]:
    """Yield ``(row_index, text)`` pairs from a parquet text column."""
    pf = pq.ParquetFile(parquet_path)
    if pf.num_row_groups == 0:
        raise ValueError(f"Parquet file has no row groups: {parquet_path}")

    emitted = 0
    global_row = 0
    for rg_ix in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_ix, columns=[text_column])
        if text_column not in rg.column_names:
            raise ValueError(
                f"Column {text_column!r} not found in parquet file. "
                f"Available columns: {rg.column_names}"
            )
        texts = rg.column(text_column).to_pylist()
        for text in texts:
            if global_row < row_offset:
                global_row += 1
                continue
            if row_limit is not None and emitted >= row_limit:
                return
            if not isinstance(text, str):
                raise ValueError(f"Document {global_row} is not a string: {type(text)!r}")
            yield global_row, text
            emitted += 1
            global_row += 1


def build_prompt(
    source_text: str,
    *,
    task: str = TASK_CORRUPT,
    regime: str = REGIME_ATOMIC_LOCAL,
) -> str:
    """Build the edit-generation prompt for gpt-5-nano."""
    output_format_doc = (
        build_gec_output_format_doc()
        if task == TASK_CORRUPT and regime == REGIME_GEC_POSTTRAIN
        else build_output_format_doc()
    )

    if task == TASK_CORRUPT and regime == REGIME_GEC_POSTTRAIN:
        task_block = """Your task is to introduce a very small number of realistic grammatical errors into otherwise clean text.

Requirements:
- Preserve the original meaning, factual content, language, tone, and nearly all wording.
- Make exactly one error in most examples; make two only when both are short and independent.
- Prefer errors seen in natural writing: article or determiner choice, prepositions, subject-verb agreement, tense, word form, singular/plural, missing or extra function words, duplicated short words, punctuation, or a short awkward local construction.
- Do not introduce factual drift, contradictions, new claims, broad stylistic rewrites, or reordered paragraphs.
- Do not use Move operations."""
    elif task == TASK_CORRUPT:
        task_block = """Your task is to describe a small set of meaningful corruptions to the text.

Requirements:
- Keep the same language and general topic.
- Preserve most of the original text.
- The implied edited result should still be coherent, but it should be slightly worse than the original.
- Prefer realistic mistakes: omissions, duplicated phrases, local contradictions, awkward wording, misplaced clauses, factual drift, repeated fragments, missing connectives, slightly wrong substitutions, and reordered spans."""
    elif task == TASK_IMPROVE:
        task_block = """Your task is to describe a small set of meaningful improvements to the text.

Requirements:
- Keep the same language and general topic.
- Preserve most of the original text.
- The implied edited result should be slightly better than the original.
- Prefer realistic edits: deleting redundancies, replacing awkward wording, moving misplaced clauses, restoring missing connectives, correcting local contradictions, and reusing existing source spans when that is natural."""
    else:
        raise ValueError(f"Unsupported task: {task}")

    if regime not in REGIME_CHOICES:
        raise ValueError(f"Unsupported regime: {regime}")

    move_enrichment_text = """Regime: move_enrichment
- Intentionally include plausible move operations when they fit naturally.
- Good examples are misplaced fragments or locally reordered spans.
- Keep these edits plausible, not bizarre."""

    regime_block = {
        REGIME_ATOMIC_LOCAL: """Regime: atomic_local
- Prefer several short, independent, local edits over one broad rewrite.
- Target omissions, duplicated phrases, short replacements, or local insertions.
- Keep each edit tightly bounded to a contiguous span.""",
        REGIME_COEDIT_LIKE: """Regime: coedit_like
- Prefer realistic revision-style edits: clarification, grammar repair, wording swaps, and local restructuring.
- Bias toward replacements and deletions that improve or degrade clarity without rewriting the whole passage.
- Keep changes locally scoped and independently meaningful.""",
        REGIME_GEC_POSTTRAIN: """Regime: gec_posttrain
- Produce one localized grammatical corruption whenever possible.
- A second corruption is allowed only when it is short, realistic, and independent.
- Prefer Replace, Insert, or Del; never use Move.
- Keep every changed span short and preserve the intended meaning.""",
        REGIME_LONG_DOC_REPAIR: """Regime: long_doc_repair
- Spread edits across the document when natural instead of clustering only at the start.
- Still keep each edit bounded and contiguous.
- Prefer multiple local issues over one global rewrite.""",
        REGIME_MOVE_ENRICHMENT: move_enrichment_text,
    }[regime]

    if regime == REGIME_GEC_POSTTRAIN:
        constraints = """Constraints:
- Output exactly one or two short contiguous-span edits.
- At least 80% of examples should require exactly one edit.
- Prefer a single-token or short-phrase ReplaceExact, Insert, or DelExact.
- Never output Move.
- For ReplaceExact or DelExact, `exact` must contain only the token or short
  phrase that actually changes. Never include unchanged surrounding clauses.
- For example, to corrupt `she walks` to `she walk`, ReplaceExact only `walks`
  with `walk`; do not replace the surrounding sentence.
- Keep a replaced or deleted source span below 12 words.
- The corrupted result must remain understandable but grammatically or locally stylistically worse.
- Do not merely replace a word with an equally valid synonym.
- Keep localisation concise and source-anchored.
- Do not output the full rewritten text.
- Do not output explanations outside the requested operation lines."""
    else:
        constraints = """Constraints:
- Make between 2 and 5 substantial contiguous-span edits.
- At least one edit should be larger than a trivial synonym or punctuation tweak.
- Prefer edits that correspond clearly to insert, delete, replace, or move.
- Avoid outputs where the only changes are punctuation, formatting, or tiny synonym swaps.
- Keep localisation concise: use short reliable source phrases that still identify the intended span or destination.
- Do not output the full rewritten text.
- Do not output explanations outside the requested operation lines."""

    return f"""You are given a piece of real text.

{task_block}

{regime_block}

{constraints}

Output format:
{output_format_doc}

Rules for the operation fields:
{GEC_RULES_DOC if task == TASK_CORRUPT and regime == REGIME_GEC_POSTTRAIN else RULES_DOC}

SOURCE TEXT:
```text
{source_text}
```

Reminder of required output format:
{output_format_doc}

Reminder of anchor rules:
{RULES_DOC}

Pointer example:
If the source text contains `chicken, seafood`, then
`Pointer(search_phrase="chicken, seafood", pre="chicken, ", post="seafood")`
points to the boundary between `chicken, ` and `seafood`.
"""


def extract_text_from_response(response_json: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI Responses API payload."""
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    outputs = response_json.get("output", [])
    parts: list[str] = []
    for item in outputs:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text", "")
                if text:
                    parts.append(text)
    joined = "".join(parts).strip()
    if joined:
        return joined

    try:
        return response_json["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError):
        pass

    raise ValueError(
        "Could not extract assistant text from OpenAI response:\n"
        f"{json.dumps(response_json, indent=2)[:4000]}"
    )


def extract_usage(response_json: dict[str, Any]) -> UsageSummary:
    usage = response_json.get("usage", {})
    output_details = usage.get("output_tokens_details", {})
    return UsageSummary(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        reasoning_tokens=output_details.get("reasoning_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


def parse_operation_lines(raw_text: str) -> tuple[list[OperationSpec], list[DroppedOperation]]:
    """Parse the compact edit DSL and retain structured drop reasons."""
    specs: list[OperationSpec] = []
    dropped: list[DroppedOperation] = []

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            node = ast.parse(line, mode="eval").body
        except SyntaxError:
            dropped.append(DroppedOperation("parse", "invalid_python_syntax", raw_line))
            continue
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            dropped.append(DroppedOperation("parse", "not_a_function_call", raw_line))
            continue
        op_name = node.func.id
        if op_name not in {
            "Del", "Replace", "Insert", "Move", "DelExact", "ReplaceExact"
        }:
            dropped.append(DroppedOperation("parse", "unknown_operation", raw_line))
            continue
        if node.args:
            dropped.append(DroppedOperation("parse", "positional_arguments_not_allowed", raw_line))
            continue
        try:
            kwargs = {kw.arg: _parse_dsl_value(kw.value) for kw in node.keywords if kw.arg is not None}
        except ValueError as exc:
            dropped.append(DroppedOperation("parse", str(exc), raw_line))
            continue
        if len(kwargs) != len(node.keywords):
            dropped.append(DroppedOperation("parse", "keyword_names_required", raw_line))
            continue
        specs.append(OperationSpec(op_name=op_name, kwargs=kwargs, raw_line=raw_line))

    return specs, dropped


def _parse_dsl_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.args:
            raise ValueError("nested_positional_arguments_not_allowed")
        func_name = node.func.id
        if func_name not in {"RegionInclusive", "Pointer"}:
            raise ValueError("unknown_nested_structure")
        kwargs: dict[str, str] = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError("keyword_names_required")
            if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, str):
                raise ValueError("nested_arguments_must_be_string_literals")
            kwargs[kw.arg] = kw.value.value
        return {"kind": func_name, "fields": kwargs}
    raise ValueError("arguments_must_be_strings_or_supported_nested_calls")


def _find_all_occurrences(text: str, needle: str) -> list[int]:
    out = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            break
        out.append(idx)
        start = idx + 1
    return out


def _locate_phrase_exact(
    text: str,
    phrase: str,
    exact: str,
    *,
    min_start: int = 0,
) -> tuple[int, int]:
    phrase_hits = [idx for idx in _find_all_occurrences(text, phrase) if idx >= min_start]
    if not phrase_hits:
        raise ValueError(f"Could not find search phrase: {phrase!r}")
    rel = phrase.find(exact)
    if rel == -1:
        raise ValueError(f"Exact anchor {exact!r} not found inside search phrase {phrase!r}")
    abs_start = phrase_hits[0] + rel
    return abs_start, abs_start + len(exact)


def _require_nested_kind(value: Any, expected_kind: str, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict) or value.get("kind") != expected_kind or not isinstance(value.get("fields"), dict):
        raise ValueError(f"{field_name}_must_be_{expected_kind}")
    return value["fields"]


def _validate_region_fields(fields: dict[str, str]) -> list[str]:
    required = {"start_search_phrase", "start_exact", "end_search_phrase", "end_exact"}
    return sorted(required - set(fields))


def _validate_pointer_fields(fields: dict[str, str]) -> list[str]:
    required = {"search_phrase", "pre", "post"}
    return sorted(required - set(fields))


def _locate_region(source_text: str, region: dict[str, str]) -> tuple[int, int]:
    missing = _validate_region_fields(region)
    if missing:
        raise ValueError(f"missing_region_keys:{','.join(missing)}")
    start, _ = _locate_phrase_exact(source_text, region["start_search_phrase"], region["start_exact"])
    _, end = _locate_phrase_exact(
        source_text,
        region["end_search_phrase"],
        region["end_exact"],
        min_start=start,
    )
    if end < start:
        raise ValueError("region_end_before_start")
    return start, end


def _locate_pointer(source_text: str, pointer: dict[str, str]) -> int:
    missing = _validate_pointer_fields(pointer)
    if missing:
        raise ValueError(f"missing_pointer_keys:{','.join(missing)}")
    search_phrase = pointer["search_phrase"]
    pre = pointer["pre"]
    post = pointer["post"]
    if pre == "" and post == "":
        raise ValueError("pointer_requires_nonempty_pre_or_post")
    phrase_hits = _find_all_occurrences(source_text, search_phrase)
    if not phrase_hits:
        raise ValueError(f"Could not find search phrase: {search_phrase!r}")
    boundary_rel = search_phrase.find(pre + post)
    if boundary_rel == -1:
        raise ValueError(f"Pointer pre+post not found inside search phrase: {pre!r}+{post!r}")
    return phrase_hits[0] + boundary_rel + len(pre)


def _canonicalize_edit(source_text: str, edit: Any) -> Any | None:
    if isinstance(edit, Replace):
        source_span = source_text[edit.start:edit.end]
        replacement = "".join(np.array(edit.content).tolist())

        prefix = 0
        max_prefix = min(len(source_span), len(replacement))
        while prefix < max_prefix and source_span[prefix] == replacement[prefix]:
            prefix += 1

        source_span = source_span[prefix:]
        replacement = replacement[prefix:]
        start = edit.start + prefix
        end = edit.end

        suffix = 0
        max_suffix = min(len(source_span), len(replacement))
        while suffix < max_suffix and source_span[-(suffix + 1)] == replacement[-(suffix + 1)]:
            suffix += 1

        if suffix:
            source_span = source_span[:-suffix]
            replacement = replacement[:-suffix]
            end -= suffix

        if source_span == replacement:
            return None
        if source_span == "":
            if replacement == "":
                return None
            return Insert(start, np.array(list(replacement)))
        if replacement == "":
            return Delete(start, end)
        return Replace(start, end, np.array(list(replacement)))

    if isinstance(edit, Insert):
        content = "".join(np.array(edit.content).tolist())
        if content == "":
            return None
        return edit

    return edit


def _build_sequence_edit(
    source_text: str,
    specs: list[OperationSpec],
) -> tuple[SequenceEdit, list[DroppedOperation]]:
    parsed_edits: list[tuple[Any, str]] = []
    dropped: list[DroppedOperation] = []
    for spec in specs:
        try:
            if spec.op_name == "DelExact":
                search_phrase = spec.kwargs.get("search_phrase")
                exact = spec.kwargs.get("exact")
                if not isinstance(search_phrase, str) or not isinstance(exact, str):
                    raise ValueError("missing_required_keys:search_phrase,exact")
                start, end = _locate_phrase_exact(source_text, search_phrase, exact)
                canonical = _canonicalize_edit(source_text, Delete(start, end))
                if canonical is not None:
                    parsed_edits.append((canonical, spec.raw_line))
            elif spec.op_name == "ReplaceExact":
                search_phrase = spec.kwargs.get("search_phrase")
                exact = spec.kwargs.get("exact")
                text = spec.kwargs.get("text")
                if (
                    not isinstance(search_phrase, str)
                    or not isinstance(exact, str)
                    or not isinstance(text, str)
                ):
                    raise ValueError("missing_required_keys:search_phrase,exact,text")
                start, end = _locate_phrase_exact(source_text, search_phrase, exact)
                canonical = _canonicalize_edit(
                    source_text, Replace(start, end, np.array(list(text)))
                )
                if canonical is not None:
                    parsed_edits.append((canonical, spec.raw_line))
            elif spec.op_name == "Del":
                region = _require_nested_kind(spec.kwargs.get("region"), "RegionInclusive", "region")
                start, end = _locate_region(source_text, region)
                canonical = _canonicalize_edit(source_text, Delete(start, end))
                if canonical is not None:
                    parsed_edits.append((canonical, spec.raw_line))
            elif spec.op_name == "Replace":
                region = _require_nested_kind(spec.kwargs.get("region"), "RegionInclusive", "region")
                text = spec.kwargs.get("text")
                if not isinstance(text, str):
                    raise ValueError("missing_required_keys:text")
                start, end = _locate_region(source_text, region)
                canonical = _canonicalize_edit(source_text, Replace(start, end, np.array(list(text))))
                if canonical is not None:
                    parsed_edits.append((canonical, spec.raw_line))
            elif spec.op_name == "Insert":
                pointer = _require_nested_kind(spec.kwargs.get("pointer"), "Pointer", "pointer")
                insert_text = spec.kwargs.get("text")
                if not isinstance(insert_text, str):
                    raise ValueError("missing_required_keys:text")
                pos = _locate_pointer(source_text, pointer)
                canonical = _canonicalize_edit(source_text, Insert(pos, np.array(list(insert_text))))
                if canonical is not None:
                    parsed_edits.append((canonical, spec.raw_line))
            elif spec.op_name == "Move":
                region = _require_nested_kind(spec.kwargs.get("region"), "RegionInclusive", "region")
                pointer = _require_nested_kind(spec.kwargs.get("pointer"), "Pointer", "pointer")
                start, end = _locate_region(source_text, region)
                dest = _locate_pointer(source_text, pointer)
                canonical = _canonicalize_edit(source_text, Move(start, end, dest))
                if canonical is not None:
                    parsed_edits.append((canonical, spec.raw_line))
        except Exception as exc:
            reason = str(exc) if isinstance(exc, ValueError) and str(exc) else exc.__class__.__name__
            dropped.append(DroppedOperation("localise", reason, spec.raw_line))

    sequence = np.array(list(source_text))
    parsed_edits.sort(key=lambda item: _edit_key(item[0]))

    kept: list[Any] = []
    for edit, raw_line in parsed_edits:
        candidate = SequenceEdit(sequence=sequence, edits=kept + [edit])
        try:
            resolve(candidate)
        except Exception as exc:
            dropped.append(DroppedOperation("validate", exc.__class__.__name__, raw_line))
            continue
        kept.append(edit)

    return SequenceEdit(sequence=sequence, edits=kept), dropped


def edit_to_dict(edit: Any) -> dict[str, Any]:
    if isinstance(edit, Insert):
        return {"type": "insert", "pos": int(edit.pos), "content": "".join(np.array(edit.content).tolist())}
    if isinstance(edit, Delete):
        return {"type": "delete", "start": int(edit.start), "end": int(edit.end)}
    if isinstance(edit, Replace):
        return {
            "type": "replace",
            "start": int(edit.start),
            "end": int(edit.end),
            "content": "".join(np.array(edit.content).tolist()),
        }
    if isinstance(edit, Move):
        return {"type": "move", "start": int(edit.start), "end": int(edit.end), "dest": int(edit.dest)}
    raise TypeError(f"Unsupported edit type: {type(edit)!r}")


def dict_to_edit(data: dict[str, Any]) -> Any:
    edit_type = data["type"]
    if edit_type == "insert":
        return Insert(int(data["pos"]), np.array(list(data["content"])))
    if edit_type == "delete":
        return Delete(int(data["start"]), int(data["end"]))
    if edit_type == "replace":
        return Replace(int(data["start"]), int(data["end"]), np.array(list(data["content"])))
    if edit_type == "move":
        return Move(int(data["start"]), int(data["end"]), int(data["dest"]))
    raise ValueError(f"Unsupported edit type: {edit_type}")


def sequence_edit_to_json(sequence_edit: SequenceEdit) -> str:
    edits = [edit_to_dict(edit) for edit in sequence_edit.edits]
    return json.dumps(edits, ensure_ascii=False)


def sequence_edit_from_json(source_text: str, payload: str) -> SequenceEdit:
    edits = [dict_to_edit(item) for item in json.loads(payload)]
    return SequenceEdit(sequence=np.array(list(source_text)), edits=edits)


def edit_span_size(edit: Any) -> int:
    if isinstance(edit, Insert):
        return len(edit.content)
    if isinstance(edit, (Delete, Replace, Move)):
        return int(edit.end) - int(edit.start)
    raise TypeError(f"Unsupported edit type: {type(edit)!r}")


def edit_location_features(edit: Any, source_len: int) -> dict[str, float | int | None]:
    denom = max(source_len, 1)
    if isinstance(edit, Insert):
        return {
            "start": None,
            "end": None,
            "pos": int(edit.pos),
            "dest": None,
            "relative_start": None,
            "relative_end": None,
            "relative_pos": float(edit.pos / denom),
            "relative_dest": None,
        }
    if isinstance(edit, (Delete, Replace)):
        return {
            "start": int(edit.start),
            "end": int(edit.end),
            "pos": None,
            "dest": None,
            "relative_start": float(edit.start / denom),
            "relative_end": float(edit.end / denom),
            "relative_pos": None,
            "relative_dest": None,
        }
    if isinstance(edit, Move):
        return {
            "start": int(edit.start),
            "end": int(edit.end),
            "pos": None,
            "dest": int(edit.dest),
            "relative_start": float(edit.start / denom),
            "relative_end": float(edit.end / denom),
            "relative_pos": None,
            "relative_dest": float(edit.dest / denom),
        }
    raise TypeError(f"Unsupported edit type: {type(edit)!r}")


def edit_counts(sequence_edit: SequenceEdit) -> dict[str, int]:
    counts = {"insert": 0, "delete": 0, "replace": 0, "move": 0}
    for edit in sequence_edit.edits:
        counts[edit_to_dict(edit)["type"]] += 1
    return counts


def _style_wrap(text: str, *styles: str) -> str:
    if not text:
        return text
    return "".join(styles) + text + ANSI_RESET


def render_operation_diff(example: EditExample) -> str:
    """Render one annotated source document with local edit markup only."""
    text = example.source_text
    inserts_by_pos: dict[int, list[str]] = {}
    span_starts: dict[int, Any] = {}

    def add_insert(pos: int, rendered: str) -> None:
        inserts_by_pos.setdefault(pos, []).append(rendered)

    for edit in sorted(example.sequence_edit.edits, key=_edit_key):
        if isinstance(edit, Insert):
            add_insert(edit.pos, _style_wrap("".join(np.array(edit.content).tolist()), ANSI_GREEN, ANSI_BOLD))
        elif isinstance(edit, Replace):
            span_starts[edit.start] = edit
        elif isinstance(edit, Delete):
            span_starts[edit.start] = edit
        elif isinstance(edit, Move):
            span_starts[edit.start] = edit
            moved = text[edit.start:edit.end]
            add_insert(edit.dest, _style_wrap(moved, ANSI_BRIGHT_BLUE, ANSI_BOLD))

    out: list[str] = []
    i = 0
    n = len(text)
    while i <= n:
        if i in inserts_by_pos:
            out.extend(inserts_by_pos[i])
        if i == n:
            break
        edit = span_starts.get(i)
        if edit is None:
            out.append(text[i])
            i += 1
            continue
        if isinstance(edit, Delete):
            out.append(_style_wrap(text[edit.start:edit.end], ANSI_RED))
            i = edit.end
        elif isinstance(edit, Replace):
            out.append(_style_wrap(text[edit.start:edit.end], ANSI_MAGENTA, ANSI_ITALIC))
            replacement = "".join(np.array(edit.content).tolist())
            out.append(_style_wrap(replacement, ANSI_MAGENTA, ANSI_BOLD))
            i = edit.end
        elif isinstance(edit, Move):
            out.append(_style_wrap(text[edit.start:edit.end], ANSI_BRIGHT_BLUE, ANSI_ITALIC))
            i = edit.end
        else:
            out.append(text[i])
            i += 1

    return "".join(out)


def colour_operation_line(line: str) -> str:
    """Colour one raw operation line by operation family."""
    if line.startswith("Del("):
        return _style_wrap(line, ANSI_RED)
    if line.startswith("Insert("):
        return _style_wrap(line, ANSI_GREEN, ANSI_BOLD)
    if line.startswith("Replace("):
        return _style_wrap(line, ANSI_MAGENTA)
    if line.startswith("Move("):
        return _style_wrap(line, ANSI_BRIGHT_BLUE)
    return line


def call_gpt5_nano(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    """Call the OpenAI Responses API."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in the environment before running."
        )

    payload: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You produce compact, source-anchored edit ces.workflows. "
                            "Always follow the requested output format exactly."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ],
        "max_output_tokens": max_output_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}

    verify: bool | str = True
    env_verify = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
    if env_verify:
        verify = env_verify
    else:
        certifi_path = None
        try:
            import certifi

            certifi_path = certifi.where()
        except Exception:
            certifi_path = None
        if certifi_path and os.path.exists(certifi_path):
            verify = certifi_path
        elif os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
            verify = "/etc/ssl/certs/ca-certificates.crt"

    response = requests.post(
        OPENAI_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
        verify=verify,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:8000]
        raise requests.HTTPError(
            f"{exc}\nResponse body:\n{body}",
            response=response,
        ) from exc
    return response.json()


def build_edit_example_from_text(
    source_text: str,
    *,
    row_index: int = 0,
    task: str = TASK_CORRUPT,
    regime: str = REGIME_ATOMIC_LOCAL,
    model: str = DEFAULT_MODEL,
    temperature: float | None = None,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> EditExample:
    """Query the model for one document and parse the result."""
    prompt = build_prompt(source_text, task=task, regime=regime)
    raw_response = call_gpt5_nano(
        prompt,
        model=model,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )
    raw_model_text = extract_text_from_response(raw_response)
    specs, dropped_parse = parse_operation_lines(raw_model_text)
    sequence_edit, dropped_localisation = _build_sequence_edit(source_text, specs)
    corrupted_text = "".join(resolve(sequence_edit).tolist())
    inverse_sequence_edit = complement(sequence_edit)
    return EditExample(
        row_index=row_index,
        task=task,
        regime=regime,
        source_text=source_text,
        corrupted_text=corrupted_text,
        sequence_edit=sequence_edit,
        inverse_sequence_edit=inverse_sequence_edit,
        operations=[line.strip() for line in raw_model_text.splitlines() if line.strip()],
        dropped_operations=dropped_parse + dropped_localisation,
        raw_model_text=raw_model_text,
        raw_response=raw_response,
        usage=extract_usage(raw_response),
    )


def build_edit_examples(
    parquet_path: str,
    *,
    model: str = DEFAULT_MODEL,
    text_column: str = "text",
    row_limit: int | None = 1,
    row_offset: int = 0,
    task: str = TASK_CORRUPT,
    regime: str = REGIME_ATOMIC_LOCAL,
    temperature: float | None = None,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    continue_on_error: bool = False,
) -> list[EditExample]:
    examples = []
    for row_index, source_text in read_documents(
        parquet_path,
        text_column=text_column,
        row_limit=row_limit,
        row_offset=row_offset,
    ):
        try:
            examples.append(
                build_edit_example_from_text(
                    source_text,
                    row_index=row_index,
                    task=task,
                    regime=regime,
                    model=model,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                )
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            print(f"ROW {row_index} FAILED: {exc}", file=sys.stderr)
    return examples


def _temperature_label(temperature: float | None) -> str:
    return "default" if temperature is None else f"{temperature:g}"


def _infer_source_shard_id(source_parquet_path: str | None) -> str | None:
    if not source_parquet_path:
        return None
    stem = Path(source_parquet_path).stem
    return stem if stem.startswith("shard_") else None


def example_to_record(
    example: EditExample,
    *,
    model: str,
    temperature: float | None = None,
    source_dataset: str | None = None,
    source_parquet_path: str | None = None,
    source_shard_id: str | None = None,
) -> dict[str, Any]:
    forward_counts = edit_counts(example.sequence_edit)
    inverse_counts = edit_counts(example.inverse_sequence_edit)
    source_len = len(example.source_text)
    corrupted_len = len(example.corrupted_text)
    span_sizes = [edit_span_size(edit) for edit in example.sequence_edit.edits]
    source_shard_id = source_shard_id or _infer_source_shard_id(source_parquet_path)
    return {
        "row_index": example.row_index,
        "task": example.task,
        "regime": example.regime,
        "model": model,
        "prompt_version": (
            GEC_PROMPT_VERSION
            if example.regime == REGIME_GEC_POSTTRAIN
            else PROMPT_VERSION
        ),
        "temperature": temperature,
        "temperature_label": _temperature_label(temperature),
        "source_dataset": source_dataset,
        "source_parquet_path": source_parquet_path,
        "source_shard_id": source_shard_id,
        "source_text": example.source_text,
        "corrupted_text": example.corrupted_text,
        "source_char_len": source_len,
        "corrupted_char_len": corrupted_len,
        "char_delta": corrupted_len - source_len,
        "operations": example.operations,
        "num_operations_model": len(example.operations),
        "num_edits_kept": len(example.sequence_edit.edits),
        "num_dropped_operations": len(example.dropped_operations),
        "has_valid_edit_script": bool(example.sequence_edit.edits),
        "kept_fraction": (
            len(example.sequence_edit.edits) / len(example.operations)
            if example.operations
            else 0.0
        ),
        "edit_span_sum": int(sum(span_sizes)),
        "edit_span_max": int(max(span_sizes)) if span_sizes else 0,
        "forward_edits_json": sequence_edit_to_json(example.sequence_edit),
        "inverse_edits_json": sequence_edit_to_json(example.inverse_sequence_edit),
        "forward_edit_counts_json": json.dumps(forward_counts),
        "inverse_edit_counts_json": json.dumps(inverse_counts),
        "dropped_operations_json": json.dumps([asdict(item) for item in example.dropped_operations]),
        "raw_model_text": example.raw_model_text,
        "raw_response_json": json.dumps(example.raw_response),
        "usage_input_tokens": example.usage.input_tokens,
        "usage_output_tokens": example.usage.output_tokens,
        "usage_reasoning_tokens": example.usage.reasoning_tokens,
        "usage_total_tokens": example.usage.total_tokens,
    }


def write_examples_jsonl(
    path: str,
    examples: Iterable[EditExample],
    *,
    model: str,
    temperature: float | None = None,
    source_dataset: str | None = None,
    source_parquet_path: str | None = None,
    source_shard_id: str | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(
                json.dumps(
                    example_to_record(
                        example,
                        model=model,
                        temperature=temperature,
                        source_dataset=source_dataset,
                        source_parquet_path=source_parquet_path,
                        source_shard_id=source_shard_id,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_examples_parquet(
    path: str,
    examples: Iterable[EditExample],
    *,
    model: str,
    temperature: float | None = None,
    source_dataset: str | None = None,
    source_parquet_path: str | None = None,
    source_shard_id: str | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        example_to_record(
            example,
            model=model,
            temperature=temperature,
            source_dataset=source_dataset,
            source_parquet_path=source_parquet_path,
            source_shard_id=source_shard_id,
        )
        for example in examples
    ]
    if not records:
        raise ValueError("No examples to write.")
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path)


def load_records(path: str) -> list[dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        table = pajson.read_json(path)
    elif suffix == ".parquet":
        table = pq.read_table(path)
    else:
        raise ValueError(f"Unsupported input format: {path}")
    return table.to_pylist()


def print_example(example: EditExample, *, show_raw_response: bool, show_texts: bool) -> None:
    print(f"ROW INDEX: {example.row_index}")
    print("OPERATIONS:")
    for op in example.operations:
        print(colour_operation_line(op))
    if example.dropped_operations:
        print("\nDROPPED OPERATIONS:")
        for item in example.dropped_operations:
            label = f"[{item.stage}:{item.reason}] {item.raw_line}"
            print(_style_wrap(label, ANSI_YELLOW))
    print("\n" + "=" * 80 + "\n")

    if show_texts:
        print("SOURCE:\n")
        print(example.source_text)
        print("\n" + "=" * 80 + "\n")
        print("CORRUPTED:\n")
        print(example.corrupted_text)
        print("\n" + "=" * 80 + "\n")

    print(f"{ANSI_BOLD}DIFF:{ANSI_RESET}\n")
    print(render_operation_diff(example))

    if show_raw_response:
        print("\n" + "=" * 80 + "\n")
        print("RAW MODEL TEXT:\n")
        print(example.raw_model_text)
        print("\n" + "=" * 80 + "\n")
        print("RAW RESPONSE JSON:\n")
        print(json.dumps(example.raw_response, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read parquet documents and ask gpt-5-nano to emit edit ces.workflows."
    )
    parser.add_argument("parquet_path", help="Path to the parquet file to read.")
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of the text column in the parquet file. Default: text",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to call. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--task",
        choices=TASK_CHOICES,
        default=TASK_CORRUPT,
        help="Generate corruption edits or improvement edits.",
    )
    parser.add_argument(
        "--regime",
        choices=REGIME_CHOICES,
        default=REGIME_ATOMIC_LOCAL,
        help="Prompt regime label to use for generation.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=1,
        help="Maximum number of rows to process. Default: 1",
    )
    parser.add_argument(
        "--row-offset",
        type=int,
        default=0,
        help="Start from this global row index. Default: 0",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature for the Responses API.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        help=f"Responses API reasoning effort. Default: {DEFAULT_REASONING_EFFORT}",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"Responses API max output tokens. Default: {DEFAULT_MAX_OUTPUT_TOKENS}",
    )
    parser.add_argument(
        "--output-jsonl",
        help="Optional path to write batch records as JSONL.",
    )
    parser.add_argument(
        "--output-parquet",
        help="Optional path to write batch records as parquet.",
    )
    parser.add_argument(
        "--show-raw-response",
        action="store_true",
        help="Print the raw JSON response as well.",
    )
    parser.add_argument(
        "--show-texts",
        action="store_true",
        help="Print the full source and corrupted texts before the diff.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip rows that fail instead of aborting the full run.",
    )
    args = parser.parse_args()

    examples = build_edit_examples(
        args.parquet_path,
        model=args.model,
        text_column=args.text_column,
        row_limit=args.row_limit,
        row_offset=args.row_offset,
        task=args.task,
        regime=args.regime,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        continue_on_error=args.continue_on_error,
    )
    if not examples:
        raise ValueError("No examples were generated.")

    if args.output_jsonl:
        write_examples_jsonl(
            args.output_jsonl,
            examples,
            model=args.model,
            temperature=args.temperature,
            source_parquet_path=args.parquet_path,
        )
    if args.output_parquet:
        write_examples_parquet(
            args.output_parquet,
            examples,
            model=args.model,
            temperature=args.temperature,
            source_parquet_path=args.parquet_path,
        )

    for example in examples:
        print_example(
            example,
            show_raw_response=args.show_raw_response,
            show_texts=args.show_texts,
        )
        if len(examples) > 1:
            print("\n" + "#" * 80 + "\n")


if __name__ == "__main__":
    main()
