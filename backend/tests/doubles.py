"""Shared test doubles.

Everything here is TEST-ONLY. The passages are neutral placeholder strings, not
medical content, and none of these classes are reachable from the running app.
"""

from fastapi.testclient import TestClient

from answer_processing.generators import AnswerGenerator, GeneratedAnswer
from answer_processing.pipeline import AnswerPipeline
from app.config import Settings
from app.container import Services
from app.lazy_model import LazyModel
from app.main import create_app
from app.storage import QueryStore
from query_processing.bert_encoder import EncodedQuery
from query_processing.understanding import QueryUnderstanding
from retrieval.base import Passage, RetrievalQuery, RetrievalResult, RetrievalStatus, Retriever
from retrieval.not_configured import NotConfiguredRetriever
from retrieval.service import RetrievalService


def make_passages(n: int = 2) -> list[Passage]:
    return [
        Passage(
            text=f"Placeholder passage number {i}. The answer token is ALPHA{i}.",
            source_id=f"test-doc-{i}",
            source_title=f"Test Document {i}",
            source_url=f"https://example.test/doc/{i}",
            relevance_score=1.0 - i * 0.1,
        )
        for i in range(n)
    ]


class FakeEncoder:
    """Stands in for BertQueryEncoder without loading weights."""

    def __init__(self, available: bool = True):
        self.available = available
        self.model = LazyModel("query-encoder", "fake-bert", lambda: object())
        self.calls: list[str] = []

    def encode(self, text: str) -> EncodedQuery | None:
        self.calls.append(text)
        if not self.available:
            return None
        return EncodedQuery(wordpieces=["[CLS]", "test", "[SEP]"], token_count=3, embedding=[0.1, 0.2, 0.3])


class StaticRetriever(Retriever):
    name = "static_test"

    def __init__(self, result: RetrievalResult | None = None, exc: Exception | None = None):
        super().__init__()
        self.result = result
        self.exc = exc
        self.queries: list[RetrievalQuery] = []

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        self.queries.append(query)
        if self.exc:
            raise self.exc
        return self.result.model_copy(deep=True)


def ok_retriever(n: int = 2) -> StaticRetriever:
    return StaticRetriever(RetrievalResult(status=RetrievalStatus.OK, passages=make_passages(n)))


class FakeGenerator(AnswerGenerator):
    name = "fake"

    def __init__(self, answer: GeneratedAnswer | None = None, available: bool = True):
        self.answer = answer
        self._available = available
        self.calls = 0

    @property
    def available(self) -> bool:
        return self._available

    def generate(self, question, passages):
        self.calls += 1
        return self.answer


def make_pipeline(retriever=None, generator=None, encoder=None, **kw) -> AnswerPipeline:
    return AnswerPipeline(
        QueryUnderstanding(encoder or FakeEncoder()),
        RetrievalService(retriever or NotConfiguredRetriever(), timeout_seconds=kw.pop("timeout", 2.0),
                         min_relevance=kw.pop("min_relevance", 0.0)),
        generator or FakeGenerator(),
        **kw,
    )


def make_client(retriever=None, generator=None, encoder=None) -> TestClient:
    settings = Settings(database_path=":memory:", environment="test", _env_file=None)
    encoder = encoder or FakeEncoder()
    generator = generator or FakeGenerator()
    retrieval = RetrievalService(retriever or NotConfiguredRetriever(), timeout_seconds=2.0)
    pipeline = AnswerPipeline(QueryUnderstanding(encoder), retrieval, generator)
    services = Services(settings, encoder, retrieval, generator, pipeline, QueryStore(":memory:"))
    return TestClient(create_app(settings, services))

