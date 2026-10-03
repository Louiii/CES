import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanochat.checkpoint_manager import delete_checkpoint


def _touch(path: Path) -> None:
    path.write_bytes(b"checkpoint")


def test_delete_checkpoint_removes_only_this_ranks_files(tmp_path):
    step = 123
    model = tmp_path / "model_000123.pt"
    meta = tmp_path / "meta_000123.json"
    optim0 = tmp_path / "optim_000123_rank0.pt"
    optim1 = tmp_path / "optim_000123_rank1.pt"
    for path in (model, meta, optim0, optim1):
        _touch(path)

    delete_checkpoint(tmp_path, step, rank=1)
    assert not optim1.exists()
    assert model.exists()
    assert meta.exists()
    assert optim0.exists()

    delete_checkpoint(tmp_path, step, rank=0)
    assert not model.exists()
    assert not meta.exists()
    assert not optim0.exists()


def test_delete_checkpoint_tolerates_missing_files(tmp_path):
    delete_checkpoint(tmp_path, 999, rank=0)
