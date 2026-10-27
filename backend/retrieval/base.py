"""Retrieval contract between MedQuery and an external medical knowledge base.

MedQuery ships WITHOUT a knowledge base. To connect one, implement `Retriever`
(see `retrieval/adapters/custom_adapter_template.py`) and point the
`MEDQUERY_KB_ADAPTER` setting at your class.

Input  -> `RetrievalQuery`  (the processed question, keywords, optional BERT embedding)
Output -> `RetrievalResult` (a status plus zero or more `Passage` objects)

Rules every adapter must follow:
  * Return only passages that genuinely come from your knowledge base.
  * Use `RetrievalStatus.NO_RESULTS` (with an empty list) when nothing matched.
  * Raise an exception or return `RetrievalStatus.ERROR` on failure; never
    fabricate passages or sources.
  * `relevance_score` should be "higher is more relevant"; normalising it to
    0..1 lets `MEDQUERY_KB_MIN_RELEVANCE` filter meaningfully.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RetrievalStatus(str, Enum):
    OK = "OK"
    NO_RESULTS = "NO_RESULTS"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class RetrievalQuery(BaseModel):
    original_text: str
    normalized_text: str
    keywords: list[str] = Field(default_factory=list)
    question_type: str = "general"
    # Mean-pooled BERT embedding of the question; None if the encoder is unavailable.
    embedding: list[float] | None = None
    top_k: int = 5


class Passage(BaseModel):
    text: str
    source_id: str
    source_title: str | None = None
    source_url: str | None = None
    relevance_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    status: RetrievalStatus
    passages: list[Passage] = Field(default_factory=list)
    message: str | None = None
    adapter: str = "unknown"


class Retriever(ABC):
    """Base class for knowledge-base adapters."""

    #: Human-readable adapter name, reported by /api/health and in responses.
    name: str = "retriever"

    def __init__(self, connection: str = "", **options: Any) -> None:
        self.connection = connection
        self.options = options

    @property
    def is_configured(self) -> bool:
        return True

    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Return passages relevant to `query`. Called from a worker thread."""

    def close(self) -> None:
        """Release connections/resources. Optional."""
