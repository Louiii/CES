from types import SimpleNamespace

import torch

from nanochat.predicted_fim import generate_predicted_fim


class FakeTokenizer:
    def __init__(self):
        self.specials = {"<|assistant_end|>": 2}
        self.next_special = 200

    def get_bos_token_id(self):
        return 1

    def encode_special(self, text):
        if text not in self.specials:
            self.specials[text] = self.next_special
            self.next_special += 1
        return self.specials[text]

    def encode(self, text):
        return [20 + ord(char) % 50 for char in text]


class FakeLocatorModel:
    def __init__(self):
        self.config = SimpleNamespace(
            sequence_len=256,
            n_kv_head=1,
            n_embd=8,
            n_head=1,
            n_layer=1,
            use_dual_rope=False,
        )
        self.position_head = object()
        self.calls = []

    def get_device(self):
        return torch.device("cpu")

    def get_kv_cache_head_dim(self):
        return 8

    def forward(
        self,
        ids,
        *,
        kv_cache,
        pos_k_cache,
        position_queries=None,
        return_pos_logits=False,
        **_kwargs,
    ):
        batch, length = ids.shape
        self.calls.append(ids[0].tolist())
        kv_cache.advance(length)
        pos_k_cache.advance(length)
        logits = torch.full((batch, length, 300), -10.0)
        logits[..., 2] = 10.0
        if position_queries is not None:
            position_logits = torch.full((batch, 2, length), -100.0)
            position_logits[:, 0, 1] = 10.0
            position_logits[:, 1, 2] = 10.0
        else:
            position_logits = torch.zeros((batch, 1, pos_k_cache.get_pos()))
        assert return_pos_logits
        return logits, position_logits


def test_predicted_fim_reuses_source_cache_and_applies_predicted_span():
    tokenizer = FakeTokenizer()
    model = FakeLocatorModel()
    source = [1, 10, 11, 12, 13]

    result = generate_predicted_fim(
        model,
        tokenizer,
        source,
        max_new_tokens=8,
        temperature=0.0,
    )

    assert result.status == "valid"
    assert result.terminated is True
    assert (result.predicted_start, result.predicted_end) == (2, 3)
    assert result.generated_completion_tokens == []
    assert result.resolved_tokens == [1, 10, 12, 13]
    assert len(model.calls) == 2
    assert model.calls[0][:len(source)] == source
    assert model.calls[1] == result.appended_prompt_tokens
