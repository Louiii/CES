#!/usr/bin/env python
"""Evaluate iterative Levenshtein checkpoints on resolved final text."""

from __future__ import annotations

import argparse
import difflib
import json
import os
from dataclasses import asdict

import torch

from nanochat.checkpoint_manager import load_checkpoint
from nanochat.levenshtein import LevenshteinConfig, LevenshteinTransformer
from nanochat.tokenizer import get_levenshtein_tokeniser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument(
        "--manifest",
        default="data/evals/controlled_edit_val_v1/manifest.jsonl",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    state, _optimizer, metadata = load_checkpoint(
        args.checkpoint_dir, args.step, device, load_optimizer=False
    )
    if metadata.get("model_type") != "iterative_levenshtein_style":
        raise ValueError("Checkpoint is not an iterative Levenshtein model")
    config = LevenshteinConfig(**metadata["model_config"])
    with torch.device("meta"):
        model = LevenshteinTransformer(config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(
        {key.removeprefix("_orig_mod."): value for key, value in state.items()},
        strict=True,
        assign=True,
    )
    model.eval()
    tokenizer = get_levenshtein_tokeniser()
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("Checkpoint/tokenizer vocabulary mismatch")

    os.makedirs(args.output_dir, exist_ok=True)
    records_path = os.path.join(args.output_dir, "records.jsonl")
    control_ids = {config.pad_id, config.mask_id, config.bos_id, config.eos_id}
    records = []
    with open(args.manifest, encoding="utf-8") as source_file:
        for index, line in enumerate(source_file):
            if args.limit >= 0 and index >= args.limit:
                break
            row = json.loads(line)
            predicted_tokens, decode_stats = model.iterative_decode(
                row["source_tokens"], max_length=config.sequence_len
            )
            leaked_controls = sum(token in control_ids for token in predicted_tokens)
            predicted_tokens = [token for token in predicted_tokens if token not in control_ids]
            target_tokens = list(map(int, row["target_tokens"]))
            source_tokens = list(map(int, row["source_tokens"]))
            target_ratio = difflib.SequenceMatcher(
                a=predicted_tokens, b=target_tokens, autojunk=False
            ).ratio()
            source_ratio = difflib.SequenceMatcher(
                a=predicted_tokens, b=source_tokens, autojunk=False
            ).ratio()
            record = {
                "benchmark_id": row["benchmark_id"],
                "stratum": row["stratum"],
                "source_tokens": source_tokens,
                "target_tokens": target_tokens,
                "predicted_tokens": predicted_tokens,
                "predicted_text": tokenizer.decode(predicted_tokens),
                "exact_match": predicted_tokens == target_tokens,
                "source_exact": predicted_tokens == source_tokens,
                "target_sequence_ratio": target_ratio,
                "source_sequence_ratio": source_ratio,
                "direction_delta": target_ratio - source_ratio,
                "leaked_control_tokens": leaked_controls,
                **decode_stats,
            }
            records.append(record)

    with open(records_path, "w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    denominator = max(len(records), 1)
    summary = {
        "model_type": "iterative_levenshtein_style",
        "checkpoint_step": args.step,
        "model_config": asdict(config),
        "examples": len(records),
        "exact_match_rate": sum(record["exact_match"] for record in records) / denominator,
        "source_exact_rate": sum(record["source_exact"] for record in records) / denominator,
        "mean_target_sequence_ratio": sum(record["target_sequence_ratio"] for record in records) / denominator,
        "mean_direction_delta": sum(record["direction_delta"] for record in records) / denominator,
        "mean_rounds": sum(record["rounds"] for record in records) / denominator,
        "mean_encoder_passes": sum(record["encoder_passes"] for record in records) / denominator,
        "control_leak_rate": sum(record["leaked_control_tokens"] > 0 for record in records) / denominator,
        "records_path": records_path,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
