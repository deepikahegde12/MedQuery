"""Answer generators: turn retrieved passages into an answer.

Generators only ever work from the passages they are given. They are never
called when retrieval produced no evidence.
"""

import logging
import math
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.lazy_model import LazyModel
from query_processing.preprocess import STOPWORDS, normalize
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

    def supports_question(self, question: str) -> bool:
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

    The span is copied from a passage and verified against that passage, but
    selection can still be wrong and must not be treated as medical validation.
    """

    name = "extractive"
    disagreement_margin = 0.05

    def __init__(self, model_name: str, max_answer_tokens: int = 64, offline: bool = False, loader=None):
        self.max_answer_tokens = max_answer_tokens
        self.model = LazyModel("qa", model_name, loader or (lambda: _load_hf_qa(model_name, offline)))
        self._inference_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.model.get() is not None

    def generate(self, question: str, passages: list[Passage]) -> GeneratedAnswer | None:
        if not self.supports_question(question):
            return None
        qa = self.model.get()
        if qa is None:
            return None
        candidates: list[GeneratedAnswer] = []
        intents = _question_intents(question)
        intent = intents[0] if intents else None
        matching = [idx for idx, passage in enumerate(passages) if intent and _heading_matches(passage, intent)]
        has_headings = any(p.metadata.get("section_heading") for p in passages)
        if intent and has_headings and not matching:
            return None
        selected_indices = matching or list(range(len(passages)))
        seen_answers: set[tuple[str, str]] = set()
        for idx in selected_indices:
            passage = passages[idx]
            try:
                with self._inference_lock:
                    out = qa(question=question, context=passage.text, top_k=1,
                             max_answer_len=self.max_answer_tokens,
                             handle_impossible_answer=True)
            except Exception:  # noqa: BLE001
                logger.exception("QA model failed on passage %s", passage.source_id)
                continue
            if isinstance(out, list):
                out = out[0] if out else None
            if not out or not str(out.get("answer", "")).strip():
                continue  # model judged the passage does not contain an answer
            try:
                score = float(out.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                continue
            answer_text = str(out["answer"]).strip()
            if not answer_text or len(answer_text.split()) > self.max_answer_tokens:
                continue
            start, end = out.get("start"), out.get("end")
            context = passage.text
            if (type(start) is not int or type(end) is not int or start < 0 or end <= start
                    or end > len(context) or context[start:end] != answer_text):
                occurrences = list(re.finditer(re.escape(answer_text), context))
                if len(occurrences) != 1:
                    continue
                start, end = occurrences[0].span()
            key = (passage.source_id, _normalise_answer(answer_text))
            if key in seen_answers:
                continue
            seen_answers.add(key)
            candidates.append(GeneratedAnswer(
                text=answer_text,
                confidence=score,
                passage_index=idx,
                start=start,
                end=end,
            ))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (
            bool(intent and _heading_matches(passages[item.passage_index], intent)),
            item.confidence,
        ), reverse=True)
        best = candidates[0]
        for other in candidates[1:]:
            if abs(best.confidence - other.confidence) > self.disagreement_margin:
                continue
            if passages[best.passage_index].source_id == passages[other.passage_index].source_id:
                continue
            if _answers_explicitly_conflict(best, other, passages):
                # Different spans are often complementary. Abstain only when their
                # supporting statements directly conflict under a narrow heuristic.
                logger.info("Abstaining on near-tied extractive answers from multiple sources")
                return None
        return best

    def supports_question(self, question: str) -> bool:
        """This response schema can safely provide only one non-comparison span."""
        intents = _question_intents(question)
        return len(intents) <= 1 and not {"comparison", "multi_part"}.intersection(intents)

    def status(self) -> dict:
        return {"mode": self.name, **self.model.status()}


def _normalise_answer(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", value.casefold())).strip()


_INTENT_PATTERNS = {
    # "Warning signs" is its own intent; treating the word "signs" as
    # symptoms too would incorrectly make a single warning-sign query compound.
    "symptoms": re.compile(r"\b(symptoms?|(?<!warning )signs?|feel like|presentation)\b"),
    "risk_factors": re.compile(r"\b(risk factors?)\b"),
    "causes": re.compile(r"\b(causes?|caused|etiology|triggers?)\b"),
    "prevention": re.compile(r"\b(prevent\w*|avoid\w*|reduce the risk|lower the risk|vaccinat\w*)\b"),
    "diagnosis": re.compile(r"\b(diagnos\w*|tests?|detect\w*|screen\w*)\b"),
    "medication_safety": re.compile(r"\b(side effects?|interactions?|dos(?:e|age)|labels?|safe storage)\b"),
    "treatment": re.compile(r"\b(treat\w*|therapy|cure|manage\w*|medications?|medicines?|drugs?)\b"),
    "complications": re.compile(r"\b(complications?|long[- ]term effects?)\b"),
    "warning": re.compile(r"\b(warning signs?|emergency|urgent help|when should\b|when to seek)\b"),
}

_INTENT_HEADINGS = {
    "symptoms": ("symptom", "sign"),
    "risk_factors": ("risk factor", "risk and check"),
    "causes": ("cause", "trigger", "why"),
    "prevention": ("prevent", "avoid", "risk reduction", "vaccin", "hand hygiene"),
    "diagnosis": ("diagnos", "test", "screen", "assessment", "check"),
    "medication_safety": ("side effect", "interaction", "label", "storage", "medicine"),
    "treatment": ("treat", "manage", "therapy", "medicine", "medication"),
    "complications": ("complication", "long-term effect", "risk"),
    "warning": ("warning", "emergency", "urgent", "when should", "when to seek", "when to contact"),
    "definition": ("what is", "what are", "meaning", "definition", "overview"),
}


def _question_intents(question: str) -> list[str]:
    text = normalize(question)
    if re.search(r"\b(difference between|compare|comparison|versus|vs\.?|distinguish(?:es|ed)?)\b", text):
        return ["comparison"]
    if re.search(r"\b(?:and|or|also)\s+(?:what|which|who|when|where|why|how|can|does|is|are|should)\b", text):
        return ["multi_part"]
    intents = [name for name, pattern in _INTENT_PATTERNS.items() if pattern.search(text)]
    if "symptoms" in intents and "causes" in intents:
        starts_with_symptom_question = re.match(r"^(what|which)\s+(?:are\s+)?(?:common\s+)?symptoms?\b", text)
        starts_with_cause_question = re.match(r"^(what|why)\s+(?:are\s+the\s+)?causes?\b", text)
        conjunction_between = bool(re.search(r"\b(and|or|also)\b", text))
        if starts_with_symptom_question and not conjunction_between:
            intents.remove("causes")
        elif starts_with_cause_question and not conjunction_between:
            intents.remove("symptoms")
    if re.match(r"^(define|meaning of|what does .+ mean)\b", text):
        intents.append("definition")
    elif not intents and re.match(r"^(what is|what are)\b", text):
        intents.append("definition")
    elif re.match(r"^what is\b", text) and re.search(r"\b(and|also)\b", text):
        intents.append("definition")
    return list(dict.fromkeys(intents))


def _heading_matches(passage: Passage, intent: str) -> bool:
    heading = str(passage.metadata.get("section_heading") or "").casefold()
    if not heading:
        heading = passage.text.splitlines()[0].lstrip("# ").casefold()
    return any(term in heading for term in _INTENT_HEADINGS.get(intent, ()))


def _answers_explicitly_conflict(left: GeneratedAnswer, right: GeneratedAnswer,
                                 passages: list[Passage]) -> bool:
    left_sentence = _sentence_for_span(passages[left.passage_index].text, left.start, left.end)
    right_sentence = _sentence_for_span(passages[right.passage_index].text, right.start, right.end)
    left_tokens = {_normalise_conflict_token(token) for token in re.findall(r"[a-z0-9]+", left_sentence.casefold())}
    right_tokens = {_normalise_conflict_token(token) for token in re.findall(r"[a-z0-9]+", right_sentence.casefold())}
    negation = {"no", "not", "never", "without", "neither", "cannot", "can't"}
    left_polarity = bool(left_tokens & negation)
    right_polarity = bool(right_tokens & negation)
    shared = (left_tokens & right_tokens) - STOPWORDS - negation
    left_context = left_tokens - STOPWORDS - negation
    right_context = right_tokens - STOPWORDS - negation
    union = left_context | right_context
    overlap = len(shared) / len(union) if union else 0.0
    if overlap < 0.65:
        return False
    if left_polarity != right_polarity:
        return True
    left_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", left.text)
    right_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", right.text)
    return len(left_numbers) == len(right_numbers) == 1 and left_numbers[0] != right_numbers[0]


def _normalise_conflict_token(token: str) -> str:
    """Treat a simple plural or third-person verb ending as its base token."""
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _sentence_for_span(text: str, start: int | None, end: int | None) -> str:
    if start is None or end is None:
        return text
    left = max((match.end() for match in re.finditer(r"[.!?;\n]", text[:start])), default=0)
    right_match = re.search(r"[.!?;\n]", text[end:])
    right = end + right_match.start() if right_match else len(text)
    return text[left:right]
