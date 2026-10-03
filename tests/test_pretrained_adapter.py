from types import SimpleNamespace

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.pretrained_adapter import adapt_pretrained_state, new_token_ids


class TinyTokenizer:
    def __init__(self, pieces, specials):
        self.enc = SimpleNamespace(
            _mergeable_ranks={piece: index for index, piece in enumerate(pieces)},
            _special_tokens={
                token: len(pieces) + index for index, token in enumerate(specials)
            },
        )

    def get_vocab_size(self):
        return len(self.enc._mergeable_ranks) + len(self.enc._special_tokens)


def build_model(*, vocab_size, append, position_head):
    config = GPTConfig(
        sequence_len=16,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        use_dual_rope=append,
        dual_rope_style="append" if append else "split",
        use_position_head=position_head,
        window_pattern="L",
    )
    model = GPT(config)
    model.init_weights()
    model.eval()
    return model


def test_standard_to_appended_import_is_function_preserving_for_shared_logits():
    torch.manual_seed(7)
    source_tokenizer = TinyTokenizer([b"a", b"b", b"c"], ["<|bos|>"])
    target_tokenizer = TinyTokenizer(
        [b"a", b"b", b"c"],
        ["<|bos|>", "<|edit-start|>", "<|edit-end|>"],
    )
    source = build_model(
        vocab_size=source_tokenizer.get_vocab_size(), append=False, position_head=False
    )
    target = build_model(
        vocab_size=target_tokenizer.get_vocab_size(), append=True, position_head=True
    )

    report = adapt_pretrained_state(
        target_model=target,
        target_tokenizer=target_tokenizer,
        source_state=source.state_dict(),
        source_tokenizer=source_tokenizer,
        source_metadata={"step": 12, "model_config": vars(source.config), "tokenizer": "base"},
        seed=123,
    )

    ids = torch.tensor([[3, 0, 1, 2, 0]], dtype=torch.long)
    with torch.no_grad():
        source_logits = source(ids)
        target_logits = target(ids)
    torch.testing.assert_close(
        target_logits[..., : source_tokenizer.get_vocab_size()],
        source_logits,
        atol=2e-5,
        rtol=2e-5,
    )
    assert report["shared_token_count"] == 4
    assert report["new_token_count"] == 2
    assert new_token_ids(source_tokenizer, target_tokenizer) == [4, 5]
    for block in target.transformer.h:
        torch.testing.assert_close(block.attn.c_q_pointer.weight, block.attn.c_q.weight)
        torch.testing.assert_close(block.attn.c_k_pointer.weight, block.attn.c_k.weight)
        assert block.attn.pointer_attn_gate.item() == 0.0


def test_new_rows_are_deterministic_finite_and_nonidentical():
    source_tokenizer = TinyTokenizer([b"a", b"b"], ["<|bos|>"])
    target_tokenizer = TinyTokenizer(
        [b"a", b"b"], ["<|bos|>", "<|edit-start|>", "<|edit-end|>"]
    )
    torch.manual_seed(1)
    source = build_model(
        vocab_size=source_tokenizer.get_vocab_size(), append=False, position_head=False
    )
    targets = []
    for model_seed in (2, 999):
        torch.manual_seed(model_seed)
        target = build_model(
            vocab_size=target_tokenizer.get_vocab_size(), append=True, position_head=True
        )
        adapt_pretrained_state(
            target_model=target,
            target_tokenizer=target_tokenizer,
            source_state=source.state_dict(),
            source_tokenizer=source_tokenizer,
            source_metadata={"step": 1, "model_config": vars(source.config), "tokenizer": "base"},
            seed=456,
        )
        targets.append(target)

    rows = [3, 4]
    for name in ("transformer.wte.weight", "lm_head.weight", "value_embeds.1.weight"):
        first = targets[0].state_dict()[name][rows]
        second = targets[1].state_dict()[name][rows]
        assert torch.isfinite(first).all()
        assert not torch.equal(first[0], first[1])
        torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_unexpected_nonvocabulary_mismatch_fails_loudly():
    source_tokenizer = TinyTokenizer([b"a"], ["<|bos|>"])
    target_tokenizer = TinyTokenizer([b"a"], ["<|bos|>", "<|edit-start|>"])
    source = build_model(vocab_size=2, append=False, position_head=False)
    target = build_model(vocab_size=3, append=True, position_head=True)
    state = dict(source.state_dict())
    state["transformer.h.0.mlp.c_fc.weight"] = state[
        "transformer.h.0.mlp.c_fc.weight"
    ][:-1]
    try:
        adapt_pretrained_state(
            target_model=target,
            target_tokenizer=target_tokenizer,
            source_state=state,
            source_tokenizer=source_tokenizer,
            source_metadata={},
        )
    except ValueError as error:
        assert "shape mismatch" in str(error)
    else:
        raise AssertionError("Expected incompatible trunk shape to be rejected")


def test_optimizer_roles_are_complete_for_standard_and_appended_models():
    standard = build_model(vocab_size=8, append=False, position_head=False)
    appended = build_model(vocab_size=10, append=True, position_head=True)

    def optimizer_for(model):
        return model.setup_optimizer(
            unembedding_lr=0.004,
            embedding_lr=0.3,
            scalar_lr=0.5,
            matrix_lr=0.05,
            weight_decay=0.1,
        )

    standard_optimizer = optimizer_for(standard)
    appended_optimizer = optimizer_for(appended)

    assert all(group["params"] for group in standard_optimizer.param_groups)
    assert all(group["params"] for group in appended_optimizer.param_groups)
    assert {group["adapt_role"] for group in standard_optimizer.param_groups} == {
        "backbone",
        "vocab",
    }
    assert {group["adapt_role"] for group in appended_optimizer.param_groups} == {
        "backbone",
        "new",
        "vocab",
    }
