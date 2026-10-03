"""Check the public CLI's recipe execution and selective verification contracts."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from ces.cli import build_parser, run_workflow, verify_artifacts
from ces.inference import tokenizer_runtime


def test_dry_run_expands_paths_without_launching(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "recipe.json"
    config_path.write_text(
        json.dumps({"arguments": ["--model-tag", "${CES_RUNS}/demo"]})
    )
    monkeypatch.setenv("CES_RUNS", str(tmp_path))
    monkeypatch.setattr(
        "ces.cli.subprocess.call",
        lambda *args, **kwargs: pytest.fail("Dry run started a process"),
    )
    parser = build_parser()
    args = parser.parse_args(
        ["train", "pretrain", "--config", str(config_path), "--dry-run"]
    )

    run_workflow(args, ["--num-iterations", "1"], parser)

    preview = json.loads(capsys.readouterr().out)
    assert preview["command"][1:3] == ["-m", "ces.workflows.base_train"]
    assert preview["command"][3:] == [
        "--model-tag",
        str(tmp_path / "demo"),
        "--num-iterations",
        "1",
    ]


def test_workflow_prepares_checkpoint_before_training(tmp_path, monkeypatch):
    config_path = tmp_path / "recipe.json"
    config_path.write_text(
        json.dumps(
            {
                "arguments": ["--num-iterations", "1"],
                "environment": {"NANOCHAT_BASE_DIR": str(tmp_path)},
                "prepare": [["--source-step", "100"]],
            }
        )
    )
    calls = []

    def prepare(command, *, env, check):
        calls.append(("prepare", command, env))
        assert check is True

    def train(command, *, env):
        calls.append(("train", command, env))
        return 7

    monkeypatch.setattr("ces.cli.subprocess.run", prepare)
    monkeypatch.setattr("ces.cli.subprocess.call", train)
    parser = build_parser()
    args = parser.parse_args(["train", "pretrain", "--config", str(config_path)])
    with pytest.raises(SystemExit) as result:
        run_workflow(args, [], parser)

    assert result.value.code == 7
    assert [call[0] for call in calls] == ["prepare", "train"]
    assert calls[0][1][2] == "ces.workflows.seed_finetune_checkpoint"
    assert calls[1][1][2] == "ces.workflows.base_train"
    assert calls[1][2]["NANOCHAT_BASE_DIR"] == str(tmp_path)


def test_verification_checks_only_selected_files_and_detects_changes(tmp_path):
    artifact = tmp_path / "tokenizer.bin"
    artifact.write_bytes(b"frozen tokenizer")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(
        f"{digest}  tokenizer.bin\n{'0' * 64}  not-downloaded.pt\n"
    )
    assert verify_artifacts(tmp_path, "*.bin") == 1

    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_artifacts(tmp_path, "*.bin")


def test_verification_rejects_manifest_path_outside_artifacts(tmp_path):
    (tmp_path / "SHA256SUMS").write_text(f"{'0' * 64}  ../outside.bin\n")
    with pytest.raises(ValueError, match="Unsafe manifest path"):
        verify_artifacts(tmp_path)


@pytest.mark.parametrize("previous", [None, "existing-runtime"])
def test_tokenizer_runtime_restores_environment_after_error(
    tmp_path, monkeypatch, previous
):
    if previous is None:
        monkeypatch.delenv("NANOCHAT_BASE_DIR", raising=False)
    else:
        monkeypatch.setenv("NANOCHAT_BASE_DIR", previous)

    with pytest.raises(RuntimeError, match="example error"):
        with tokenizer_runtime(tmp_path):
            runtime = Path(os.environ["NANOCHAT_BASE_DIR"])
            assert (runtime / "tokenizer").resolve() == tmp_path
            raise RuntimeError("example error")

    assert os.environ.get("NANOCHAT_BASE_DIR") == previous
    assert not runtime.exists()
