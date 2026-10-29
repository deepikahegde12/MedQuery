"""Response models for a processed query."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from retrieval.base import Passage, RetrievalStatus


class AnswerStatus(str, Enum):
    ANSWERED = "ANSWERED"                  # answer extracted from evidence with adequate confidence
    LOW_CONFIDENCE = "LOW_CONFIDENCE"      # answer extracted, but the model is unsure
    EVIDENCE_ONLY = "EVIDENCE_ONLY"        # evidence found; no answer synthesised (mode or model unavailable)
    NO_ANSWER_IN_EVIDENCE = "NO_ANSWER_IN_EVIDENCE"  # evidence found, but it does not answer the question
    NO_EVIDENCE = "NO_EVIDENCE"            # knowledge base returned nothing relevant
    KB_NOT_CONFIGURED = "KB_NOT_CONFIGURED"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"  # timeout or adapter error
    EMERGENCY_GUIDANCE = "EMERGENCY_GUIDANCE"


class Answer(BaseModel):
    text: str
    confidence: float
    method: str
    evidence_index: int
    verified: bool = False  # MedQuery never verifies medical correctness


class Evidence(Passage):
    used_for_answer: bool = False
    highlight_start: int | None = None
    highlight_end: int | None = None


class SourceRef(BaseModel):
    source_id: str
    title: str | None = None
    url: str | None = None


class RetrievalInfo(BaseModel):
    status: RetrievalStatus
    adapter: str
    message: str | None = None
    passages_returned: int = 0


class Notice(BaseModel):
    level: str  # "info" | "warning" | "danger"
    code: str
    message: str


class QueryResult(BaseModel):
    id: str
    question: str
    status: AnswerStatus
    answer: Answer | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    retrieval: RetrievalInfo
    analysis: dict[str, Any] = Field(default_factory=dict)
    notices: list[Notice] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    created_at: datetime
    disclaimer: str
