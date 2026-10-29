"""BERT encoder behaviour with mocked tokenizer/model (no weights downloaded)."""

import pytest

from app.lazy_model import ModelState
from query_processing.bert_encoder import BertQueryEncoder
from query_processing.understanding import QueryUnderstanding

torch = pytest.importorskip("torch")


class FakeTokenizer:
    def __init__(self):
        self.calls = 0

    def __call__(self, text, add_special_tokens=True, **_):
        self.calls += 1
        ids = [1000 + i for i, _ in enumerate(text.split())]
        return {"input_ids": self.build_inputs_with_special_tokens(ids) if add_special_tokens else ids}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [101] + list(ids) + [102]

    def prepare_for_model(self, ids, return_tensors=None, **_):
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}

    def convert_ids_to_tokens(self, ids):
        return ["[CLS]" if i == 101 else "[SEP]" if i == 102 else f"tok{i}" for i in ids]


class FakeBert:
    def __call__(self, input_ids, attention_mask):
        class Out:
            last_hidden_state = torch.ones(1, input_ids.shape[1], 8)
        return Out()


def test_loader_is_lazy_and_called_once():
    calls = []

    def loader():
        calls.append(1)
        return FakeTokenizer(), FakeBert()

    enc = BertQueryEncoder("fake", loader=loader)
    assert calls == [] and enc.model.state == ModelState.NOT_LOADED
    enc.encode("one two")
    enc.encode("three")
    assert calls == [1]
    assert enc.model.state == ModelState.READY


def test_encode_returns_normalized_pooled_embedding():
    tokenizer = FakeTokenizer()
    enc = BertQueryEncoder("fake", loader=lambda: (tokenizer, FakeBert()))
    out = enc.encode("alpha beta gamma")
    assert out.wordpieces[0] == "[CLS]" and out.wordpieces[-1] == "[SEP]"
    assert out.token_count == 5
    assert out.embedding_dim == 8
    assert sum(v * v for v in out.embedding) == pytest.approx(1.0, rel=1e-4)
    assert out.truncated is False
    assert tokenizer.calls == 1


def test_encode_flags_truncation():
    enc = BertQueryEncoder("fake", max_tokens=4, loader=lambda: (FakeTokenizer(), FakeBert()))
    out = enc.encode("a b c d e f g")
    assert out.token_count == 4
    assert out.truncated is True


def test_unavailable_model_degrades_gracefully():
    def broken():
        raise OSError("weights not found")

    enc = BertQueryEncoder("missing-model", loader=broken)
    assert enc.encode("anything") is None
    status = enc.model.status()
    assert status["state"] == "unavailable"
    assert "weights not found" in status["error"]

    analysis = QueryUnderstanding(enc).analyze("what is term alpha")
    assert analysis.bert_available is False
    assert analysis.summary()["bert"]["available"] is False
    assert analysis.pre.keywords == ["term", "alpha"]
