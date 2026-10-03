"""Attach frozen-model likelihoods to previously generated edit records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanochat.checkpoint_manager import build_model, load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from ces.workflows.controlled_edit_eval import (
    _read_jsonl,
    _write_jsonl,
    attach_likelihood_scores,
    summarise_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--scorer-model-tag", required=True)
    parser.add_argument(
        "--scorer-checkpoint-dir",
        type=Path,
        default=None,
        help="Direct checkpoint directory; overrides --scorer-model-tag lookup",
    )
    parser.add_argument("--scorer-step", type=int, default=None)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    parser.add_argument("--device-type", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_jsonl)
    summary_output = Path(args.summary_json) if args.summary_json else output.with_suffix(".summary.json")
    if not args.overwrite and (output.exists() or summary_output.exists()):
        raise FileExistsError("Output exists; use --overwrite or choose new output paths")
    records = _read_jsonl(Path(args.input_jsonl))
    if not records:
        raise ValueError("No evaluation records found")

    device_type = autodetect_device_type() if not args.device_type else args.device_type
    _ddp, rank, _local_rank, world_size, device = compute_init(device_type)
    if rank != 0 or world_size != 1:
        raise RuntimeError("Frozen scoring expects a single process")
    if args.scorer_checkpoint_dir is not None:
        if args.scorer_step is None:
            raise ValueError("Direct scorer checkpoint requires --scorer-step")
        scorer, tokenizer, meta = build_model(
            str(args.scorer_checkpoint_dir.resolve()),
            args.scorer_step,
            device,
            phase="eval",
        )
        scorer_name = str(args.scorer_checkpoint_dir.resolve())
    else:
        scorer, tokenizer, meta = load_model(
            "base",
            device,
            phase="eval",
            model_tag=args.scorer_model_tag,
            step=args.scorer_step,
        )
        scorer_name = args.scorer_model_tag
    vocab_size = tokenizer.get_vocab_size()
    max_token = max(
        token
        for record in records
        for key in ("source_tokens", "target_tokens", "resolved_tokens")
        for token in record[key]
    )
    if max_token >= vocab_size:
        raise ValueError(f"Evaluation token id {max_token} exceeds scorer vocabulary {vocab_size}")

    attach_likelihood_scores(records, scorer=scorer, device=device, batch_size=args.score_batch_size)
    summary = {
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "records": len(records),
        "scorer_model_tag": scorer_name,
        "scorer_step": int(meta["step"]),
        "scorer_tokenizer_vocab_size": vocab_size,
        "uncertainty": {
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "groups": summarise_records(
            records,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
    }
    _write_jsonl(output, records)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_jsonl": str(output),
        "summary_json": str(summary_output),
        "scorer": f"{scorer_name}@{meta['step']}",
        "overall": summary["groups"]["overall"],
    }, indent=2))
    compute_cleanup()


if __name__ == "__main__":
    main()
