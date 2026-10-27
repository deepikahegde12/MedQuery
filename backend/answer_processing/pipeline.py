"""End-to-end question pipeline:

    question -> preprocessing + BERT encoding -> knowledge retrieval
             -> evidence gate -> answer generation -> QueryResult

The evidence gate is the key safety property: if retrieval did not return
usable passages, no answer is produced at all.
"""

import logging
import time
import uuid
from datetime import datetime, timezone

from answer_processing.generators import AnswerGenerator
from answer_processing.models import (
    Answer,
    AnswerStatus,
    Evidence,
    Notice,
    QueryResult,
    RetrievalInfo,
    SourceRef,
)
from query_processing.understanding import QueryUnderstanding
from retrieval.base import RetrievalQuery, RetrievalStatus
from retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "MedQuery is an educational prototype. It does not provide diagnosis, treatment, "
    "or prescriptions, and its output is not verified medical advice. Consult a "
    "qualified healthcare professional for medical decisions."
)

_RETRIEVAL_TO_ANSWER_STATUS = {
    RetrievalStatus.NOT_CONFIGURED: AnswerStatus.KB_NOT_CONFIGURED,
    RetrievalStatus.NO_RESULTS: AnswerStatus.NO_EVIDENCE,
    RetrievalStatus.TIMEOUT: AnswerStatus.RETRIEVAL_FAILED,
    RetrievalStatus.ERROR: AnswerStatus.RETRIEVAL_FAILED,
}

_STATUS_NOTICES = {
    AnswerStatus.KB_NOT_CONFIGURED: Notice(
        level="warning", code="kb_not_configured",
        message="Evidence retrieval is unavailable because no medical knowledge base is connected. "
                "No answer has been generated.",
    ),
    AnswerStatus.NO_EVIDENCE: Notice(
        level="warning", code="no_evidence",
        message="No relevant evidence was found in the knowledge base, so no answer is given.",
    ),
    AnswerStatus.RETRIEVAL_FAILED: Notice(
        level="warning", code="retrieval_failed",
        message="The knowledge base could not be reached. No answer has been generated. Please retry.",
    ),
    AnswerStatus.LOW_CONFIDENCE: Notice(
        level="warning", code="low_confidence",
        message="The extracted answer has low confidence. Read the supporting evidence carefully.",
    ),
    AnswerStatus.NO_ANSWER_IN_EVIDENCE: Notice(
        level="info", code="no_answer_in_evidence",
        message="Related passages were found, but none appears to directly answer the question.",
    ),
    AnswerStatus.EVIDENCE_ONLY: Notice(
        level="info", code="evidence_only",
        message="Showing the most relevant passages; no answer was extracted.",
    ),
}

_SAFETY_NOTICES = {
    "possible_emergency": Notice(
        level="danger", code="possible_emergency",
        message="If this is a medical emergency, contact your local emergency number immediately. "
                "MedQuery cannot help in urgent situations.",
    ),
    "personal_medical_advice": Notice(
        level="info", code="personal_medical_advice",
        message="MedQuery cannot assess individual cases or recommend doses. "
                "Please ask a doctor or pharmacist about your specific situation.",
    ),
}


class AnswerPipeline:
    def __init__(
        self,
        understanding: QueryUnderstanding,
        retrieval: RetrievalService,
        generator: AnswerGenerator,
        top_k: int = 5,
        min_confidence: float = 0.15,
    ):
        self.understanding = understanding
        self.retrieval = retrieval
        self.generator = generator
        self.top_k = top_k
        self.min_confidence = min_confidence

    def run(self, question: str) -> QueryResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        analysis = self.understanding.analyze(question)
        timings["understanding"] = _ms(t0)

        t1 = time.perf_counter()
        retrieval = self.retrieval.retrieve(
            RetrievalQuery(
                original_text=analysis.pre.clean_text,
                normalized_text=analysis.pre.normalized_text,
                keywords=analysis.pre.keywords,
                question_type=analysis.pre.question_type,
                embedding=analysis.encoded.embedding if analysis.encoded else None,
                top_k=self.top_k,
            )
        )
        timings["retrieval"] = _ms(t1)

        evidence = [Evidence(**p.model_dump()) for p in retrieval.passages]
        answer: Answer | None = None

        # --- Evidence gate -------------------------------------------------
        if retrieval.status != RetrievalStatus.OK or not evidence:
            status = _RETRIEVAL_TO_ANSWER_STATUS.get(retrieval.status, AnswerStatus.NO_EVIDENCE)
            evidence = []
        else:
            t2 = time.perf_counter()
            generated = None
            if self.generator.available:
                generated = self.generator.generate(analysis.pre.clean_text, retrieval.passages)
                if generated is None:
                    status = AnswerStatus.NO_ANSWER_IN_EVIDENCE
                else:
                    status = (
                        AnswerStatus.ANSWERED
                        if generated.confidence >= self.min_confidence
                        else AnswerStatus.LOW_CONFIDENCE
                    )
            else:
                status = AnswerStatus.EVIDENCE_ONLY
            if self.generator.name == "evidence_only":
                status = AnswerStatus.EVIDENCE_ONLY
            timings["answer"] = _ms(t2)

            if generated is not None and 0 <= generated.passage_index < len(evidence):
                ev = evidence[generated.passage_index]
                ev.used_for_answer = True
                ev.highlight_start, ev.highlight_end = generated.start, generated.end
                answer = Answer(
                    text=generated.text,
                    confidence=round(generated.confidence, 4),
                    method=generated.method,
                    evidence_index=generated.passage_index,
                )

        notices = [_SAFETY_NOTICES[f] for f in analysis.pre.safety_flags if f in _SAFETY_NOTICES]
        if status in _STATUS_NOTICES:
            notices.append(_STATUS_NOTICES[status])
        if not analysis.bert_available:
            notices.append(Notice(
                level="info", code="bert_unavailable",
                message="The BERT query encoder is unavailable; keyword-based query processing was used.",
            ))
        if status == AnswerStatus.EVIDENCE_ONLY and self.generator.name != "evidence_only":
            notices.append(Notice(
                level="info", code="qa_model_unavailable",
                message="The answer-extraction model is unavailable, so only evidence is shown.",
            ))

        timings["total"] = _ms(t0)
        logger.info("Query processed: status=%s retrieval=%s total=%.0fms",
                    status.value, retrieval.status.value, timings["total"])

        return QueryResult(
            id=uuid.uuid4().hex,
            question=analysis.pre.clean_text,
            status=status,
            answer=answer,
            evidence=evidence,
            sources=_dedupe_sources(evidence),
            retrieval=RetrievalInfo(
                status=retrieval.status,
                adapter=retrieval.adapter,
                message=retrieval.message,
                passages_returned=len(retrieval.passages),
            ),
            analysis=analysis.summary(),
            notices=notices,
            timings_ms=timings,
            created_at=datetime.now(timezone.utc),
            disclaimer=DISCLAIMER,
        )


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _dedupe_sources(evidence: list[Evidence]) -> list[SourceRef]:
    seen: dict[str, SourceRef] = {}
    for ev in evidence:
        seen.setdefault(ev.source_id, SourceRef(source_id=ev.source_id, title=ev.source_title, url=ev.source_url))
    return list(seen.values())
