"""Exact tokenizer contract for reproducible CES training runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from nanochat.common import get_base_dir


CANONICAL_TOKENIZER_NAME = "ces32k_v1"
CANONICAL_BASE_VOCAB_SIZE = 32748
CANONICAL_EXTENDED_VOCAB_SIZE = 32768
CANONICAL_TOKENIZER_SHA256 = "9ad2b2dbad95b16713c2e91cc6bbacc6de66fc29bdd2ba8d296ac1d6b2fc997c"
CANONICAL_TOKEN_BYTES_SHA256 = "03385f49e054a7facbd1205710ed7c0fcc307a35b0b07ee00147ce17ad65e2a9"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_canonical_tokenizer(*, tokenizer: Any | None = None) -> dict[str, Any]:
    """Fail unless the runtime contains the exact frozen CES 32K tokenizer."""

    tokenizer_dir = Path(get_base_dir()) / "tokenizer"
    tokenizer_path = tokenizer_dir / "tokenizer.pkl"
    token_bytes_path = tokenizer_dir / "token_bytes.pt"
    for path in (tokenizer_path, token_bytes_path):
        if not path.is_file():
            raise RuntimeError(
                f"Canonical CES tokenizer file is missing: {path}. Set NANOCHAT_BASE_DIR to a runtime directory with the released tokenizer/ files."
            )
    tokenizer_sha = _sha256(tokenizer_path)
    token_bytes_sha = _sha256(token_bytes_path)
    if tokenizer_sha != CANONICAL_TOKENIZER_SHA256:
        raise RuntimeError(
            f"Refusing noncanonical tokenizer at {tokenizer_path}: {tokenizer_sha}; "
            f"expected {CANONICAL_TOKENIZER_SHA256} ({CANONICAL_TOKENIZER_NAME})."
        )
    if token_bytes_sha != CANONICAL_TOKEN_BYTES_SHA256:
        raise RuntimeError(
            f"Refusing noncanonical token-byte table at {token_bytes_path}: {token_bytes_sha}; "
            f"expected {CANONICAL_TOKEN_BYTES_SHA256} ({CANONICAL_TOKENIZER_NAME})."
        )
    if tokenizer is not None and int(tokenizer.get_vocab_size()) != CANONICAL_EXTENDED_VOCAB_SIZE:
        raise RuntimeError(
            f"Refusing extended tokenizer vocabulary {tokenizer.get_vocab_size()}; "
            f"expected {CANONICAL_EXTENDED_VOCAB_SIZE}."
        )
    return {
        "name": CANONICAL_TOKENIZER_NAME,
        "base_vocab_size": CANONICAL_BASE_VOCAB_SIZE,
        "extended_vocab_size": CANONICAL_EXTENDED_VOCAB_SIZE,
        "tokenizer_sha256": tokenizer_sha,
        "token_bytes_sha256": token_bytes_sha,
    }
