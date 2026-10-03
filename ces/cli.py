"""Command-line entry point for CES artifacts and reproduction workflows.

The CLI keeps heavyweight dependencies inside the commands that need them, so
listing commands or inspecting a training recipe does not import PyTorch.
"""

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


WORKFLOWS = {
    "prepare": {
        "checkpoint": "seed_finetune_checkpoint",
        "controlled": "build_controlled_edit_benchmark",
        "gec": "build_coedit_gec_benchmark",
        "transfer": "build_coedit_gec_transfer",
        "corpus": "build_manifest_ces_training_corpus",
        "identity": "build_identity_edit_manifest",
    },
    "train": {
        "pretrain": "base_train",
        "levenshtein": "train_iterative_levenshtein",
        "sft": "ces_sft",
        "balanced-sft": "ces_balanced_sft",
        "rl": "ces_rl",
    },
    "evaluate": {
        "base": "base_eval",
        "controlled": "controlled_edit_eval",
        "gec": "method_gec_eval",
        "ces-gec": "human_edit_eval",
        "text": "text_edit_baseline_eval",
        "levenshtein": "eval_iterative_levenshtein",
        "oracle": "oracle_ces_eval",
        "oracle-gec": "oracle_gec_edit_likelihood",
        "tail": "post_edit_tail_likelihood",
        "latency": "benchmark_optimized_inference",
    },
    "report": {
        "controlled": "score_controlled_edit_records",
        "gec": "score_gec_records",
        "tail": "analyze_post_edit_tail_likelihood",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Causal Edit Serialization: artifacts, editing, and reproducible workflows"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    for command_name, workflows in WORKFLOWS.items():
        workflow_parser = commands.add_parser(command_name)
        workflow_parser.add_argument("workflow", choices=workflows)
        workflow_parser.add_argument("--config", type=Path)
        workflow_parser.add_argument("--dry-run", action="store_true")

    download_parser = commands.add_parser("download")
    download_parser.add_argument("--destination", type=Path, required=True)
    download_parser.add_argument(
        "--revision", required=True, help="Immutable Hub commit or release tag"
    )
    download_parser.add_argument(
        "--include",
        action="append",
        required=True,
        help="Repeat for selected file patterns; no automatic full download",
    )

    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("artifacts", type=Path)
    verify_parser.add_argument("--include", default="*")

    edit_parser = commands.add_parser("edit")
    edit_parser.add_argument("--checkpoint", type=Path, required=True)
    edit_parser.add_argument("--tokenizer", type=Path, required=True)
    edit_parser.add_argument("--text", required=True)
    edit_parser.add_argument("--device", default="cpu")
    edit_parser.add_argument("--max-tokens", type=int, default=128)
    edit_parser.add_argument("--seed", type=int, default=42)
    return parser


def run_workflow(
    args: argparse.Namespace,
    extra_arguments: list[str],
    parser: argparse.ArgumentParser,
) -> None:
    """Expand a saved recipe and launch its underlying Python module."""
    config = json.loads(args.config.read_text()) if args.config else {}
    config_arguments = config.get("arguments", [])
    if not isinstance(config_arguments, list) or not all(
        isinstance(value, str) for value in config_arguments
    ):
        parser.error("config.arguments must be a list of strings")

    environment = os.environ.copy()
    environment.update(
        {
            name: os.path.expandvars(str(value))
            for name, value in config.get("environment", {}).items()
        }
    )
    module_name = WORKFLOWS[args.command][args.workflow]
    command = [
        sys.executable,
        "-m",
        f"ces.workflows.{module_name}",
        *[os.path.expandvars(value) for value in config_arguments],
        *extra_arguments,
    ]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "environment": config.get("environment", {}),
                    "prepare": config.get("prepare", []),
                },
                indent=2,
            )
        )
        return

    if any("$" in value for value in command):
        parser.error(
            "Set all environment variables required by the config before running"
        )

    # Fine-tuning recipes may first create a step-zero view of pretrained weights.
    for preparation_arguments in config.get("prepare", []):
        preparation_command = [
            sys.executable,
            "-m",
            "ces.workflows.seed_finetune_checkpoint",
            *[os.path.expandvars(value) for value in preparation_arguments],
        ]
        if any("$" in value for value in preparation_command):
            parser.error("Set config preparation environment variables")
        subprocess.run(preparation_command, env=environment, check=True)

    raise SystemExit(subprocess.call(command, env=environment))


def verify_artifacts(artifacts: Path, include: str = "*") -> int:
    """Check matching manifest entries without loading whole files into memory."""
    artifact_root = artifacts.resolve()
    verified_files = 0
    manifest_lines = (artifact_root / "SHA256SUMS").read_text().splitlines()

    for line in manifest_lines:
        expected_digest, relative_path = line.split("  ", 1)
        if not fnmatch.fnmatch(relative_path, include):
            continue

        artifact_path = (artifact_root / relative_path).resolve()
        if not artifact_path.is_relative_to(artifact_root):
            raise ValueError("Unsafe manifest path")

        digest = hashlib.sha256()
        with artifact_path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(block)

        if digest.hexdigest() != expected_digest:
            raise ValueError(f"Checksum mismatch: {relative_path}")
        verified_files += 1

    if verified_files == 0:
        raise ValueError("No matching checksum entries")
    return verified_files


def main() -> None:
    parser = build_parser()
    args, extra_arguments = parser.parse_known_args()
    if extra_arguments[:1] == ["--"]:
        extra_arguments = extra_arguments[1:]

    if args.command in WORKFLOWS:
        run_workflow(args, extra_arguments, parser)
        return
    if extra_arguments:
        parser.error("unrecognised arguments: " + " ".join(extra_arguments))

    if args.command == "download":
        from huggingface_hub import snapshot_download

        snapshot_download(
            "lorob/CES",
            repo_type="dataset",
            revision=args.revision,
            local_dir=args.destination,
            allow_patterns=args.include,
        )
    elif args.command == "verify":
        count = verify_artifacts(args.artifacts, args.include)
        print(json.dumps({"verified_files": count}))
    elif args.command == "edit":
        from ces.inference import edit

        result = edit(
            checkpoint=args.checkpoint,
            tokenizer_path=args.tokenizer,
            text=args.text,
            device=args.device,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
