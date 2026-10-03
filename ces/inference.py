"""Load a released CES checkpoint and generate a greedy edit program.

This is the small editing example used by the CLI. It executes the generated
program but does not apply the paper's selective confirmation filters.
"""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Iterator


@contextmanager
def tokenizer_runtime(tokenizer_path: str | Path) -> Iterator[None]:
    """Expose downloaded tokenizer files through the legacy loader's layout.

    The loader expects ``NANOCHAT_BASE_DIR/tokenizer``. A temporary symlink
    supplies that layout without copying assets. Because the environment is
    process-wide, this helper is intended for the single-request CLI, not for
    concurrent requests using different tokenizers in the same process.
    """
    with tempfile.TemporaryDirectory(prefix="ces-runtime-") as runtime:
        tokenizer_directory = Path(tokenizer_path).resolve()
        Path(runtime, "tokenizer").symlink_to(
            tokenizer_directory, target_is_directory=True
        )
        previous_runtime = os.environ.get("NANOCHAT_BASE_DIR")
        os.environ["NANOCHAT_BASE_DIR"] = runtime
        try:
            yield
        finally:
            if previous_runtime is None:
                os.environ.pop("NANOCHAT_BASE_DIR", None)
            else:
                os.environ["NANOCHAT_BASE_DIR"] = previous_runtime


def edit(
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    text: str,
    device: str = "cpu",
    max_tokens: int = 128,
    seed: int = 42,
) -> dict:
    """Return the edited text, execution status and generated token/address program."""
    # Import the model implementation only after its tokenizer layout is available.
    with tokenizer_runtime(tokenizer_path):
        import numpy as np
        import torch

        from nanochat.causal_serialisation_extended import (
            SequenceEdit,
            SerialisedData,
            TokensOperations,
        )
        from nanochat.checkpoint_manager import build_model, find_last_step
        from nanochat.engine_extended import Engine

        checkpoint_directory = str(checkpoint)
        checkpoint_step = find_last_step(checkpoint_directory)
        model, tokenizer, metadata = build_model(
            checkpoint_directory,
            checkpoint_step,
            torch.device(device),
            "eval",
        )
        if not model.config.use_position_head:
            raise ValueError(
                "edit expects a CES checkpoint; use evaluate for baseline methods"
            )

        pointer_mode = metadata.get("user_config", {}).get(
            "ces_pointer_token_mode", "role_specific"
        )
        token_operations = TokensOperations(
            tokenizer.enc, pointer_token_mode=pointer_mode
        )

        # Serialise an empty program, then retain the source and edit-start marker.
        # This uses the same source/address convention as training and evaluation.
        source_tokens = [tokenizer.get_bos_token_id(), *tokenizer.encode(text)]
        source_example = SequenceEdit(sequence=np.array(source_tokens), edits=[])
        serialised = token_operations.serialise(source_example, indel_bos_pos=0)
        edit_start_index = int(
            np.flatnonzero(serialised.tokens == token_operations.edit_start_token)[0]
        )
        prompt_length = edit_start_index + 1
        generation_budget = min(max_tokens, model.config.sequence_len - prompt_length)
        if generation_budget <= 0:
            raise ValueError("Source exceeds checkpoint context length")

        engine = Engine(model, tokenizer, pointer_token_mode=pointer_mode)
        with torch.inference_mode():
            samples, _, pointers = engine.generate_batch(
                serialised.tokens[:prompt_length].tolist(),
                tix=serialised.positions[:prompt_length].tolist(),
                max_tokens=generation_budget,
                temperature=0.0,
                pointer_temperature=0.0,
                seed=seed,
                stop_after_edit_end=True,
            )

        generated_program = {
            "tokens": samples[0][prompt_length:],
            "pointers": pointers[0][prompt_length:],
        }
        try:
            resolved_tokens = token_operations.resolve(
                SerialisedData(
                    tokens=np.array(samples[0]),
                    positions=np.array(pointers[0]),
                )
            ).tolist()
            output_tokens = [
                token
                for token in resolved_tokens
                if token != tokenizer.get_bos_token_id()
            ]
            return {
                "text": tokenizer.decode(output_tokens),
                "status": "valid",
                **generated_program,
            }
        except ValueError as error:
            return {
                "text": None,
                "status": "invalid_program",
                "error": str(error),
                **generated_program,
            }
