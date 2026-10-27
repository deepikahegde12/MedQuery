"""Answer generators: turn retrieved passages into an answer.

Generators only ever work from the passages they are given. They are never
called when retrieval produced no evidence.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.lazy_model import LazyModel
from retrieval.base import Passage

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    text: str
    confidence: float
    passage_index: int        # which passage the answer came from
    start: int | None = None  # character offsets inside that passage
    end: int | None = None
    method: str = "extractive"


class AnswerGenerator(ABC):
    name: str = "generator"

    @property
    def available(self) -> bool:
        return True

    @abstractmethod
    def generate(self, question: str, passages: list[Passage]) -> GeneratedAnswer | None:
        """Return an answer grounded in `passages`, or None if none is supported."""

    def status(self) -> dict:
        return {"mode": self.name}


class EvidenceOnlyGenerator(AnswerGenerator):
    """Does not synthesise an answer; the UI shows the ranked passages instead."""

    name = "evidence_only"

    def generate(self, question: str, passages: list[Passage]) -> GeneratedAnswer | None:
        return None


def _load_hf_qa(model_name: str, offline: bool):
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=offline)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, local_files_only=offline)
    return pipeline("question-answering", model=model, tokenizer=tokenizer, device=-1)


class ExtractiveQAGenerator(AnswerGenerator):
    """BERT reading comprehension (SQuAD-style): selects an exact span from a passage.

    Because the answer is copied verbatim from the evidence, it cannot introduce
    claims that are not in the knowledge base.
    """

    name = "extractive"

    def __init__(self, model_name: str, max_answer_tokens: int = 64, offline: bool = False, loader=None):
        self.max_answer_tokens = max_answer_tokens
        self.model = LazyModel("qa", model_name, loader or (lambda: _load_hf_qa(model_name, offline)))

    @property
    def available(self) -> bool:
        return self.model.get() is not None

    def generate(self, question: str, passages: list[Passage]) -> GeneratedAnswer | None:
        qa = self.model.get()
        if qa is None:
            return None
        best: GeneratedAnswer | None = None
        for idx, passage in enumerate(passages):
            try:
                out = qa(
                    question=question,
                    context=passage.text,
                    top_k=1,
                    max_answer_len=self.max_answer_tokens,
                    handle_impossible_answer=True,
                )
            except Exception:  # noqa: BLE001
                logger.exception("QA model failed on passage %s", passage.source_id)
                continue
            if isinstance(out, list):
                out = out[0] if out else None
            if not out or not str(out.get("answer", "")).strip():
                continue  # model judged the passage does not contain an answer
            score = float(out.get("score", 0.0))
            if best is None or score > best.confidence:
                best = GeneratedAnswer(
                    text=str(out["answer"]).strip(),
                    confidence=score,
                    passage_index=idx,
                    start=out.get("start"),
                    end=out.get("end"),
                )
        return best

    def status(self) -> dict:
        return {"mode": self.name, **self.model.status()}
