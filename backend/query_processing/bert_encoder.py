"""BERT query understanding.

What BERT does here: it converts the question into WordPiece tokens and runs them
through a pretrained transformer encoder, producing one contextual vector per
token. We mean-pool those vectors into a single fixed-size question embedding
that a knowledge-base adapter can use for semantic (dense) search.

What BERT does NOT do here: it holds no medical facts and cannot verify answers.
It only represents the meaning of the question.
"""

import logging
from dataclasses import dataclass, field

from app.lazy_model import LazyModel

logger = logging.getLogger(__name__)


@dataclass
class EncodedQuery:
    wordpieces: list[str] = field(default_factory=list)
    token_count: int = 0
    truncated: bool = False
    embedding: list[float] | None = None

    @property
    def embedding_dim(self) -> int:
        return len(self.embedding) if self.embedding else 0


def _load_hf_encoder(model_name: str, offline: bool):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=offline)
    model = AutoModel.from_pretrained(model_name, local_files_only=offline)
    model.eval()
    return tokenizer, model


class BertQueryEncoder:
    def __init__(self, model_name: str, max_tokens: int = 128, offline: bool = False, loader=None):
        self.max_tokens = max_tokens
        self._model = LazyModel(
            "query-encoder",
            model_name,
            loader or (lambda: _load_hf_encoder(model_name, offline)),
        )

    @property
    def model(self) -> LazyModel:
        return self._model

    def encode(self, text: str) -> EncodedQuery | None:
        """Return tokens + pooled embedding, or None if the model is unavailable."""
        loaded = self._model.get()
        if loaded is None:
            return None
        tokenizer, model = loaded

        import torch

        content_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        special_count = tokenizer.num_special_tokens_to_add(pair=False)
        content_limit = max(0, self.max_tokens - special_count)
        truncated = len(content_ids) > content_limit
        ids = tokenizer.build_inputs_with_special_tokens(content_ids[:content_limit])
        batch = tokenizer.prepare_for_model(
            ids,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state  # (1, seq, dim)
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)

        ids = batch["input_ids"][0].tolist()
        return EncodedQuery(
            wordpieces=tokenizer.convert_ids_to_tokens(ids),
            token_count=len(ids),
            truncated=truncated,
            embedding=[round(v, 6) for v in pooled[0].tolist()],
        )
