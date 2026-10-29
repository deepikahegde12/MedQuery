from answer_processing.generators import EvidenceOnlyGenerator, ExtractiveQAGenerator, GeneratedAnswer
from answer_processing.models import AnswerStatus
from tests.doubles import FakeEncoder, FakeGenerator, StaticRetriever, make_passages, make_pipeline, ok_retriever
from retrieval.base import RetrievalResult, RetrievalStatus


def codes(result):
    return [n.code for n in result.notices]


# ------------------------------------------------------------ evidence available
def test_answer_when_evidence_available():
    gen = FakeGenerator(GeneratedAnswer(text="ALPHA1", confidence=0.8, passage_index=1, start=40, end=46))
    result = make_pipeline(ok_retriever(2), gen).run("what is the answer token")
    assert result.status == AnswerStatus.ANSWERED
    assert result.answer.text == "ALPHA1"
    assert result.answer.evidence_index == 1
    assert result.answer.verified is False
    assert [e.used_for_answer for e in result.evidence] == [False, True]
    assert (result.evidence[1].highlight_start, result.evidence[1].highlight_end) == (40, 46)
    assert [s.source_id for s in result.sources] == ["test-doc-0", "test-doc-1"]
    assert result.retrieval.status == RetrievalStatus.OK
    assert result.disclaimer


def test_low_confidence_answer_is_flagged():
    gen = FakeGenerator(GeneratedAnswer(text="ALPHA0", confidence=0.05, passage_index=0))
    result = make_pipeline(ok_retriever(), gen, min_confidence=0.2).run("what is the answer token")
    assert result.status == AnswerStatus.LOW_CONFIDENCE
    assert "low_confidence" in codes(result)


def test_evidence_without_answer():
    result = make_pipeline(ok_retriever(), FakeGenerator(None)).run("unrelated question here")
    assert result.status == AnswerStatus.NO_ANSWER_IN_EVIDENCE
    assert result.answer is None
    assert len(result.evidence) == 2


def test_evidence_only_mode():
    result = make_pipeline(ok_retriever(), EvidenceOnlyGenerator()).run("what is the answer token")
    assert result.status == AnswerStatus.EVIDENCE_ONLY
    assert result.answer is None and len(result.evidence) == 2
    assert "qa_model_unavailable" not in codes(result)


def test_unavailable_qa_model_falls_back_to_evidence_only():
    gen = FakeGenerator(available=False)
    result = make_pipeline(ok_retriever(), gen).run("what is the answer token")
    assert result.status == AnswerStatus.EVIDENCE_ONLY
    assert gen.calls == 0
    assert "qa_model_unavailable" in codes(result)


def test_retrieval_query_carries_bert_embedding_and_keywords():
    retriever = ok_retriever()
    make_pipeline(retriever).run("What's the answer token?")
    q = retriever.queries[0]
    assert q.embedding == [0.1, 0.2, 0.3]
    assert q.keywords == ["answer", "token"]
    assert q.normalized_text == "what is the answer token?"


# ------------------------------------------------------------ evidence unavailable
def test_not_configured_produces_no_answer():
    gen = FakeGenerator(GeneratedAnswer(text="SHOULD NOT APPEAR", confidence=1.0, passage_index=0))
    result = make_pipeline(None, gen).run("what is term x")
    assert result.status == AnswerStatus.KB_NOT_CONFIGURED
    assert result.answer is None and result.evidence == [] and result.sources == []
    assert gen.calls == 0
    assert "kb_not_configured" in codes(result)


def test_empty_retrieval_produces_no_answer():
    gen = FakeGenerator(GeneratedAnswer(text="SHOULD NOT APPEAR", confidence=1.0, passage_index=0))
    retriever = StaticRetriever(RetrievalResult(status=RetrievalStatus.NO_RESULTS))
    result = make_pipeline(retriever, gen).run("what is term x")
    assert result.status == AnswerStatus.NO_EVIDENCE
    assert result.answer is None and gen.calls == 0


def test_retrieval_error_produces_no_answer():
    gen = FakeGenerator(GeneratedAnswer(text="SHOULD NOT APPEAR", confidence=1.0, passage_index=0))
    result = make_pipeline(StaticRetriever(exc=ConnectionError()), gen).run("what is term x")
    assert result.status == AnswerStatus.RETRIEVAL_FAILED
    assert result.answer is None and gen.calls == 0


def test_safety_and_bert_notices():
    result = make_pipeline(retriever=ok_retriever(), encoder=FakeEncoder(available=False)).run(
        "Should I take this medication?",
    )
    assert {"personal_medical_advice", "bert_unavailable"} <= set(codes(result))


def test_personal_medical_advice_shows_evidence_without_extracted_answer():
    gen = FakeGenerator(GeneratedAnswer(text="ALPHA0", confidence=0.99, passage_index=0))
    result = make_pipeline(ok_retriever(), gen).run("Can I take ibuprofen with warfarin?")
    assert result.status == AnswerStatus.EVIDENCE_ONLY
    assert result.answer is None
    assert result.evidence
    assert gen.calls == 0
    assert "personal_medical_advice" in codes(result)


# ------------------------------------------------------------ extractive generator
class FakeQA:
    def __init__(self, outputs):
        self.outputs = outputs
        self.contexts = []

    def __call__(self, question=None, context=None, **kw):
        self.contexts.append(context)
        return self._next()

    def _next(self):
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_extractive_picks_highest_scoring_non_empty_span():
    qa = FakeQA([
        {"answer": "", "score": 0.99, "start": 0, "end": 0},       # "no answer" for passage 0
        {"answer": "ALPHA1", "score": 0.4, "start": 40, "end": 46},
        RuntimeError("boom"),                                         # passage 2 fails -> skipped
        {"answer": "ALPHA3", "score": 0.7, "start": 40, "end": 46},
    ])
    gen = ExtractiveQAGenerator("fake", loader=lambda: qa)
    ans = gen.generate("q", make_passages(4))
    assert (ans.text, ans.passage_index, ans.confidence) == ("ALPHA3", 3, 0.7)


def test_extractive_returns_none_when_no_span():
    gen = ExtractiveQAGenerator("fake", loader=lambda: FakeQA([{"answer": "", "score": 0.9}]))
    assert gen.generate("q", make_passages(1)) is None


def test_extractive_prefers_matching_symptom_heading_and_keeps_span_attribution():
    from retrieval.base import Passage

    passages = [
        Passage(text="Influenza is a respiratory infection.", source_id="general",
                metadata={"section_heading": "What is the flu?"}),
        Passage(text="Symptoms include fever and cough.", source_id="symptoms",
                metadata={"section_heading": "What are the symptoms of the flu?"}),
    ]
    qa = FakeQA([
        {"answer": "fever and cough", "score": 0.52, "start": 17, "end": 32},
    ])
    answer = ExtractiveQAGenerator("fake", loader=lambda: qa).generate(
        "What are common symptoms of influenza?", passages,
    )
    assert answer is not None
    assert answer.text == "fever and cough"
    assert answer.passage_index == 1
    assert passages[answer.passage_index].text[answer.start:answer.end] == answer.text
    assert qa.contexts == [passages[1].text]


def test_extractive_rejects_answer_not_found_in_associated_passage():
    qa = FakeQA([{"answer": "invented text", "score": 0.99, "start": 0, "end": 5}])
    assert ExtractiveQAGenerator("fake", loader=lambda: qa).generate("q", make_passages(1)) is None


def test_qa_model_is_loaded_once_across_generations():
    loads = []
    passage = make_passages(1)[0]
    qa = FakeQA([
        {"answer": "ALPHA0", "score": 0.8, "start": 50, "end": 56},
        {"answer": "ALPHA0", "score": 0.8, "start": 50, "end": 56},
    ])

    def load():
        loads.append(True)
        return qa

    generator = ExtractiveQAGenerator("fake", loader=load)
    assert generator.generate("q", [passage]).text == "ALPHA0"
    assert generator.generate("q", [passage]).text == "ALPHA0"
    assert len(loads) == 1


def test_extractive_unavailable_model():
    def broken():
        raise ImportError("transformers missing")

    gen = ExtractiveQAGenerator("fake", loader=broken)
    assert gen.available is False
    assert gen.generate("q", make_passages(1)) is None
    assert gen.status()["state"] == "unavailable"


def test_extractive_abstains_on_multi_intent_question_before_qa():
    qa = FakeQA([])
    loads = []
    gen = ExtractiveQAGenerator("fake", loader=lambda: (loads.append(True) or qa))
    assert gen.generate("What are symptoms and how is it diagnosed?", make_passages(1)) is None
    assert qa.contexts == []
    assert loads == []


def test_warning_signs_is_not_misclassified_as_a_multi_intent_question():
    from answer_processing.generators import _question_intents

    assert _question_intents("What general warning signs need emergency assessment?") == ["warning"]


def test_second_interrogative_clause_and_distinction_abstain():
    from answer_processing.generators import _question_intents

    assert _question_intents("What symptoms occur, and how is it diagnosed?") == ["multi_part"]
    assert _question_intents("How does the summary distinguish two conditions?") == ["comparison"]


def test_complementary_answers_from_different_sources_are_not_a_conflict():
    from retrieval.base import Passage

    passages = [
        Passage(text="Symptoms include fever and chills.", source_id="one"),
        Passage(text="Symptoms include cough and fatigue.", source_id="two"),
    ]
    qa = FakeQA([
        {"answer": "fever and chills", "score": 0.8, "start": 18, "end": 34},
        {"answer": "cough and fatigue", "score": 0.79, "start": 18, "end": 35},
    ])
    result = ExtractiveQAGenerator("fake", loader=lambda: qa).generate("What symptoms?", passages)
    assert result is not None
    assert result.passage_index == 0


def test_extractive_abstains_on_explicit_near_duplicate_polarity_conflict():
    from retrieval.base import Passage

    passages = [
        Passage(text="The condition causes fever.", source_id="one"),
        Passage(text="The condition does not cause fever.", source_id="two"),
    ]
    qa = FakeQA([
        {"answer": "fever", "score": 0.8, "start": 23, "end": 28},
        {"answer": "fever", "score": 0.79, "start": 32, "end": 37},
    ])
    assert ExtractiveQAGenerator("fake", loader=lambda: qa).generate("What causes fever?", passages) is None
