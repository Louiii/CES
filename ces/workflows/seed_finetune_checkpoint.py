"""Create a step-zero checkpoint view for fresh-optimizer fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nanochat.common import get_base_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-tag", required=True)
    parser.add_argument("--source-step", required=True, type=int)
    parser.add_argument("--output-model-tag", required=True)
    args = parser.parse_args()

    checkpoint_root = Path(get_base_dir()) / "base_checkpoints"
    source = checkpoint_root / args.source_model_tag
    output = checkpoint_root / args.output_model_tag
    source_model = source / f"model_{args.source_step:06d}.pt"
    source_meta = source / f"meta_{args.source_step:06d}.json"
    output_model = output / "model_000000.pt"
    output_meta = output / "meta_000000.json"
    if not source_model.is_file() or not source_meta.is_file():
        raise FileNotFoundError(f"Missing source checkpoint {source_model}")
    output.mkdir(parents=True, exist_ok=True)
    if output_model.exists() or output_meta.exists():
        raise FileExistsError(f"Fine-tune seed already exists in {output}")

    os.link(source_model, output_model)
    metadata = json.loads(source_meta.read_text(encoding="utf-8"))
    metadata["step"] = 0
    metadata["val_bpb"] = None
    metadata["initialization"] = {
        "kind": "checkpoint_finetune",
        "source_model_tag": args.source_model_tag,
        "source_step": args.source_step,
    }
    metadata["loop_state"] = {
        "min_val_bpb": float("inf"),
        "smooth_train_loss": 0.0,
        "total_training_time": 0.0,
        "raw_tokens_seen": 0,
        "clean_examples_seen": 0,
        "edit_mass_seen": 0,
        "valid_pointer_targets_seen": 0,
    }
    output_meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "source_model": str(source_model),
                "output_model": str(output_model),
                "hard_linked": output_model.stat().st_ino == source_model.stat().st_ino,
                "output_meta": str(output_meta),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
