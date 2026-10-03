"""Convert paired corruption rows into target-blind clean identity prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--preserve-controlled-fields",
        action="store_true",
        help="Emit a no-op controlled benchmark when token/program fields are present",
    )
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = []
    for index, row in enumerate(rows):
        clean = str(row["target_text"])
        identity = {
            "benchmark_id": f"identity:{row.get('benchmark_id', index)}",
            "source_text": clean,
            "target_text": clean,
            "split": "training",
            "role": "rl_training_identity",
            "source_split": "train",
        }
        if args.preserve_controlled_fields:
            target_tokens = [int(token) for token in row["target_tokens"]]
            indel_bos = int(row["prompt_tokens"][0])
            edit_start = int(row["prompt_tokens"][-1])
            edit_end = int(row["gold_program_tokens"][-1])
            prompt_tokens = [indel_bos, *target_tokens, edit_start]
            identity.update({
                "split": "validation",
                "role": "identity_validation",
                "source_split": "validation",
                "stratum": "identity_k0",
                "repair_op_mode": "identity",
                "repair_num_ops": 0,
                "density_regime": "identity",
                "source_shard_id": row.get("source_shard_id", "identity"),
                "source_row_index": int(row.get("source_row_index", index)),
                "source_token_offset": int(row.get("source_token_offset", 0)),
                "construction_seed": int(row.get("construction_seed", 0)),
                "source_num_tokens": len(target_tokens),
                "target_num_tokens": len(target_tokens),
                "source_tokens": target_tokens,
                "target_tokens": target_tokens,
                "source_token_sha256": row.get("target_token_sha256"),
                "target_token_sha256": row.get("target_token_sha256"),
                "gold_edits": [],
                "gold_op_counts": {},
                "prompt_tokens": prompt_tokens,
                "prompt_positions": list(range(len(prompt_tokens))),
                "gold_program_tokens": [edit_end],
                "gold_program_positions": [len(prompt_tokens)],
            })
        output.append(identity)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"input_rows": len(rows), "output_rows": len(output), "output": str(args.output_jsonl)}))


if __name__ == "__main__":
    main()
