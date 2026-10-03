"""Score saved final-text records with ERRANT F0.5 and risk/coverage metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from nanochat.gec_metrics import (
    errant_runtime_metadata,
    load_errant_annotator,
    risk_coverage_curve,
    score_errant_records,
    summarise_scored_examples,
)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload, dict) and isinstance(payload.get("examples"), list):
        return payload["examples"]
    raise ValueError(f"Could not find records in {path}")


def _serialise_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            {"tp": value.tp, "fp": value.fp, "fn": value.fn}
            if key == "_errant_counts"
            else value
        )
        for key, value in row.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--prediction-field", default="prediction_text")
    parser.add_argument("--confidence-field", default=None)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--output-records", type=Path, default=None)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    args = parser.parse_args()
    if args.beta <= 0:
        parser.error("--beta must be positive")

    records = _read_records(args.input)
    annotator = load_errant_annotator()
    scored = score_errant_records(
        records,
        annotator=annotator,
        prediction_field=args.prediction_field,
    )
    summary = {
        "input": str(args.input.resolve()),
        "prediction_field": args.prediction_field,
        "runtime": errant_runtime_metadata(annotator),
        "overall": summarise_scored_examples(
            scored,
            beta=args.beta,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
        "risk_coverage": (
            risk_coverage_curve(
                scored,
                confidence_field=args.confidence_field,
                beta=args.beta,
            )
            if args.confidence_field
            else None
        ),
        "confidence_field": args.confidence_field,
        "direction_definition": (
            "Token-level Levenshtein distance to the single CoEdit reference: "
            "improves if prediction is closer than source, worse if farther, same otherwise."
        ),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_records:
        args.output_records.parent.mkdir(parents=True, exist_ok=True)
        with args.output_records.open("w", encoding="utf-8") as handle:
            for row in scored:
                handle.write(json.dumps(_serialise_row(row), ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
