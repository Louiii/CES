"""ERRANT-based final-text metrics and target-blind risk/coverage utilities."""

from __future__ import annotations

import importlib.metadata
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import errant
import numpy as np
from errant.commands.compare_m2 import compareEdits
from rapidfuzz.distance import Levenshtein


@dataclass(frozen=True)
class ErrantCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "ErrantCounts") -> "ErrantCounts":
        return ErrantCounts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def _normalise_text(text: str) -> str:
    return " ".join(text.split())


def _fscore(counts: ErrantCounts, beta: float = 0.5) -> dict[str, float | int]:
    precision = counts.tp / (counts.tp + counts.fp) if counts.tp + counts.fp else 1.0
    recall = counts.tp / (counts.tp + counts.fn) if counts.tp + counts.fn else 1.0
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    fscore = (1.0 + beta_sq) * precision * recall / denom if denom else 0.0
    return {
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
        "precision": precision,
        "recall": recall,
        f"f{beta:g}": fscore,
    }


def load_errant_annotator(language: str = "en"):
    return errant.load(language)


def errant_runtime_metadata(annotator) -> dict[str, str]:
    return {
        "errant_version": importlib.metadata.version("errant"),
        "spacy_version": importlib.metadata.version("spacy"),
        "spacy_pipeline": str(annotator.nlp.meta.get("name", "unknown")),
        "spacy_pipeline_version": str(annotator.nlp.meta.get("version", "unknown")),
        "language": str(annotator.lang),
        "tokenization": "spaCy raw-text tokenization via annotator.parse(tokenise=True)",
        "matching": "ERRANT span-based correction without category matching",
    }


def _edit_dict(edits) -> dict[tuple[int, int, str], list[str]]:
    output: dict[tuple[int, int, str], list[str]] = {}
    for edit in edits:
        key = (int(edit.o_start), int(edit.o_end), str(edit.c_str))
        output.setdefault(key, []).append(str(edit.type))
    return output


def _token_text(doc) -> list[str]:
    return [token.text for token in doc]


def score_errant_example(
    *,
    annotator,
    source_text: str,
    target_text: str,
    prediction_text: str | None,
) -> dict[str, Any]:
    source_text = _normalise_text(source_text)
    target_text = _normalise_text(target_text)
    prediction_text = _normalise_text(prediction_text or source_text)
    source = annotator.parse(source_text, tokenise=True)
    target = annotator.parse(target_text, tokenise=True)
    prediction = annotator.parse(prediction_text, tokenise=True)
    reference_edits = annotator.annotate(source, target)
    hypothesis_edits = annotator.annotate(source, prediction)
    tp, fp, fn, categories = compareEdits(
        _edit_dict(hypothesis_edits),
        _edit_dict(reference_edits),
    )
    source_target_distance = int(Levenshtein.distance(_token_text(source), _token_text(target)))
    prediction_target_distance = int(
        Levenshtein.distance(_token_text(prediction), _token_text(target))
    )
    distance_delta = source_target_distance - prediction_target_distance
    direction = "improves" if distance_delta > 0 else "worse" if distance_delta < 0 else "same"
    return {
        "counts": ErrantCounts(tp=tp, fp=fp, fn=fn),
        "hypothesis_edits": len(hypothesis_edits),
        "reference_edits": len(reference_edits),
        "changed": prediction_text != source_text,
        "source_target_token_distance": source_target_distance,
        "prediction_target_token_distance": prediction_target_distance,
        "target_distance_delta": distance_delta,
        "target_distance_direction": direction,
        "categories": categories,
        "prediction_text": prediction_text,
    }


def _bootstrap_fscore(
    row_counts: np.ndarray,
    *,
    beta: float,
    samples: int,
    seed: int,
) -> list[float] | None:
    if not len(row_counts) or samples <= 0:
        return None
    rng = np.random.default_rng(seed)
    values = []
    remaining = samples
    while remaining:
        chunk = min(500, remaining)
        indices = rng.integers(0, len(row_counts), size=(chunk, len(row_counts)))
        totals = row_counts[indices].sum(axis=1)
        tp, fp, fn = totals[:, 0], totals[:, 1], totals[:, 2]
        precision = np.divide(tp, tp + fp, out=np.ones_like(tp, dtype=float), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.ones_like(tp, dtype=float), where=(tp + fn) != 0)
        beta_sq = beta * beta
        denom = beta_sq * precision + recall
        score = np.divide(
            (1 + beta_sq) * precision * recall,
            denom,
            out=np.zeros_like(precision),
            where=denom != 0,
        )
        values.append(score)
        remaining -= chunk
    distribution = np.concatenate(values)
    return [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))]


def _counts_for_row(row: dict[str, Any]) -> ErrantCounts:
    return row["counts"] if "counts" in row else row["_errant_counts"]


def summarise_scored_examples(
    rows: list[dict[str, Any]],
    *,
    beta: float = 0.5,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 20260727,
) -> dict[str, Any]:
    totals = ErrantCounts()
    directions = Counter()
    categories: dict[str, list[int]] = {}
    for row in rows:
        totals += _counts_for_row(row)
        directions[row["target_distance_direction"]] += 1
        for category, counts in row["categories"].items():
            current = categories.setdefault(category, [0, 0, 0])
            for index, value in enumerate(counts):
                current[index] += int(value)
    summary = _fscore(totals, beta)
    row_counts = np.asarray(
        [
            [counts.tp, counts.fp, counts.fn]
            for row in rows
            for counts in [_counts_for_row(row)]
        ],
        dtype=np.int64,
    )
    summary.update(
        {
            "examples": len(rows),
            "changed_examples": sum(bool(row["changed"]) for row in rows),
            "coverage": (
                sum(bool(row["changed"]) for row in rows) / len(rows) if rows else None
            ),
            "improves": directions["improves"],
            "same": directions["same"],
            "worse": directions["worse"],
            "improvement_rate": directions["improves"] / len(rows) if rows else None,
            "harm_rate": directions["worse"] / len(rows) if rows else None,
            "mean_target_distance_delta": (
                float(np.mean([row["target_distance_delta"] for row in rows]))
                if rows
                else None
            ),
            "bootstrap_samples": bootstrap_samples,
            f"f{beta:g}_ci95": _bootstrap_fscore(
                row_counts,
                beta=beta,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            ),
            "categories": dict(sorted(categories.items())),
        }
    )
    return summary


def score_errant_records(
    records: Iterable[dict[str, Any]],
    *,
    annotator,
    prediction_field: str,
) -> list[dict[str, Any]]:
    output = []
    for record in records:
        source = str(record["source_text"])
        target = str(record["target_text"])
        prediction = record.get(prediction_field)
        scored = score_errant_example(
            annotator=annotator,
            source_text=source,
            target_text=target,
            prediction_text=prediction if isinstance(prediction, str) else source,
        )
        output.append({**record, **{key: value for key, value in scored.items() if key != "counts"}, "_errant_counts": scored["counts"]})
    return output


def risk_coverage_curve(
    scored_rows: list[dict[str, Any]],
    *,
    confidence_field: str,
    beta: float = 0.5,
) -> list[dict[str, Any]]:
    """Return all tie-safe target-blind confidence operating points."""

    base_fn = sum(int(row["reference_edits"]) for row in scored_rows)
    candidates = []
    for row in scored_rows:
        confidence = row.get(confidence_field)
        if not row["changed"] or not isinstance(confidence, (int, float)):
            continue
        confidence = float(confidence)
        if not math.isfinite(confidence):
            continue
        candidates.append((confidence, row))
    candidates.sort(key=lambda item: item[0], reverse=True)

    points = []
    totals = ErrantCounts(tp=0, fp=0, fn=base_fn)
    directions = Counter({"same": len(scored_rows)})

    def append_point(threshold: float | None, accepted: int) -> None:
        score = _fscore(totals, beta)
        points.append(
            {
                "threshold": threshold,
                "accepted": accepted,
                "examples": len(scored_rows),
                "coverage": accepted / len(scored_rows) if scored_rows else None,
                **score,
                "improves": directions["improves"],
                "same": directions["same"],
                "worse": directions["worse"],
                "improvement_rate": (
                    directions["improves"] / len(scored_rows) if scored_rows else None
                ),
                "harm_rate": directions["worse"] / len(scored_rows) if scored_rows else None,
            }
        )

    append_point(None, 0)
    accepted = 0
    index = 0
    while index < len(candidates):
        threshold = candidates[index][0]
        while index < len(candidates) and candidates[index][0] == threshold:
            row = candidates[index][1]
            counts: ErrantCounts = row["_errant_counts"]
            totals = ErrantCounts(
                tp=totals.tp + counts.tp,
                fp=totals.fp + counts.fp,
                fn=totals.fn - int(row["reference_edits"]) + counts.fn,
            )
            directions["same"] -= 1
            directions[row["target_distance_direction"]] += 1
            accepted += 1
            index += 1
        append_point(threshold, accepted)
    return points
