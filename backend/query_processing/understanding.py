"""Combines rule-based preprocessing with BERT encoding into one QueryAnalysis."""

from dataclasses import dataclass

from query_processing.bert_encoder import BertQueryEncoder, EncodedQuery
from query_processing.preprocess import PreprocessedQuery, preprocess


@dataclass
class QueryAnalysis:
    pre: PreprocessedQuery
    encoded: EncodedQuery | None

    @property
    def bert_available(self) -> bool:
        return self.encoded is not None and self.encoded.embedding is not None

    def summary(self) -> dict:
        return {
            "normalized_text": self.pre.normalized_text,
            "keywords": self.pre.keywords,
            "question_type": self.pre.question_type,
            "safety_flags": self.pre.safety_flags,
            "bert": {
                "available": self.bert_available,
                "token_count": self.encoded.token_count if self.encoded else 0,
                "wordpieces": self.encoded.wordpieces[:64] if self.encoded else [],
                "truncated": self.encoded.truncated if self.encoded else False,
                "embedding_dim": self.encoded.embedding_dim if self.encoded else 0,
            },
        }


class QueryUnderstanding:
    def __init__(self, encoder: BertQueryEncoder):
        self.encoder = encoder

    def analyze(self, question: str) -> QueryAnalysis:
        pre = preprocess(question)
        # BERT sees the cleaned text with original casing (cased models benefit).
        return QueryAnalysis(pre=pre, encoded=self.encoder.encode(pre.clean_text))
