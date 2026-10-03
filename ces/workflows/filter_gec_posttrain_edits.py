"""Filter generated GEC corruptions to a conservative local-edit envelope."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein


def rejection_reason(
    row: dict[str, Any],
    *,
    max_edits: int,
    max_span_chars: int,
    max_distance_chars: int,
    max_distance_ratio: float,
) -> str | None:
    edits = int(row.get("num_edits_kept") or 0)
    if edits < 1 or edits > max_edits:
        return "edit_count"
    counts = json.loads(row.get("forward_edit_counts_json") or "{}")
    if int(counts.get("move", 0)):
        return "move"
    if int(row.get("edit_span_max") or 0) > max_span_chars:
        return "span_too_large"
    source = str(row.get("source_text") or "")
    corrupted = str(row.get("corrupted_text") or "")
    if not source or not corrupted or source == corrupted:
        return "empty_or_identity"
    distance = int(Levenshtein.distance(source, corrupted))
    if distance > max_distance_chars:
        return "distance_too_large"
    if distance / max(len(source), 1) > max_distance_ratio:
        return "distance_ratio_too_large"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-edits",
        type=int,
        default=2,
        help="Maximum number of retained local edits per example.",
    )
    parser.add_argument("--max-span-chars", type=int, default=64)
    parser.add_argument("--max-distance-chars", type=int, default=64)
    parser.add_argument("--max-distance-ratio", type=float, default=0.15)
    parser.add_argument(
        "--approval-jsonl",
        type=Path,
        default=None,
        help="Optional corruption-audit judgments; retain only safe_for_training rows.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    inputs = sorted(args.input_dir.glob("batch_*.parquet"))
    if not inputs:
        raise FileNotFoundError(f"No batch_*.parquet files under {args.input_dir}")
    existing = sorted(args.output_dir.glob("batch_*.parquet"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Filtered batches already exist under {args.output_dir}; pass --overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    approved: set[str] | None = None
    if args.approval_jsonl is not None:
        judgments = [
            json.loads(line)
            for line in args.approval_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        approved = {
            str(row["doc_key"])
            for row in judgments
            if bool(row.get("safe_for_training"))
        }

    reasons: Counter[str] = Counter()
    kept_total = 0
    rows_total = 0
    for path in inputs:
        rows = pq.read_table(path).to_pylist()
        kept = []
        for row in rows:
            rows_total += 1
            key = f"{row.get('source_shard_id')}:{int(row.get('row_index') or 0)}"
            reason = (
                "judge_not_approved"
                if approved is not None and key not in approved
                else rejection_reason(
                    row,
                    max_edits=args.max_edits,
                    max_span_chars=args.max_span_chars,
                    max_distance_chars=args.max_distance_chars,
                    max_distance_ratio=args.max_distance_ratio,
                )
            )
            if reason is None:
                kept.append(row)
            else:
                reasons[reason] += 1
        if kept:
            pq.write_table(pa.Table.from_pylist(kept), args.output_dir / path.name)
            kept_total += len(kept)

    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "rows": rows_total,
        "kept": kept_total,
        "kept_fraction": kept_total / max(rows_total, 1),
        "rejected": dict(sorted(reasons.items())),
        "thresholds": {
            "max_edits": args.max_edits,
            "max_span_chars": args.max_span_chars,
            "max_distance_chars": args.max_distance_chars,
            "max_distance_ratio": args.max_distance_ratio,
        },
        "approval_jsonl": (
            str(args.approval_jsonl.resolve())
            if args.approval_jsonl is not None
            else None
        ),
    }
    (args.output_dir / "filter_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
