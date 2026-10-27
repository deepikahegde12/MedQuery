import time

from tests.doubles import StaticRetriever, make_passages
from retrieval.base import Passage, RetrievalQuery, RetrievalResult, RetrievalStatus, Retriever
from retrieval.not_configured import NotConfiguredRetriever
from retrieval.service import RetrievalService, load_retriever

QUERY = RetrievalQuery(original_text="q", normalized_text="q", top_k=5)


def test_not_configured_returns_explicit_status_and_no_passages():
    r = NotConfiguredRetriever()
    assert r.is_configured is False
    result = r.retrieve(QUERY)
    assert result.status == RetrievalStatus.NOT_CONFIGURED
    assert result.passages == []
    assert "knowledge base" in result.message.lower()


def test_load_retriever_empty_path_is_not_configured():
    assert isinstance(load_retriever(""), NotConfiguredRetriever)


def test_load_retriever_invalid_paths_fall_back_safely():
    for path in ["no_colon", "missing.module:Cls", "retrieval.base:RetrievalQuery"]:
        r = load_retriever(path)
        assert isinstance(r, NotConfiguredRetriever)
        assert r.retrieve(QUERY).status == RetrievalStatus.NOT_CONFIGURED


def test_load_retriever_loads_a_retriever_subclass():
    r = load_retriever("retrieval.adapters.custom_adapter_template:MyKnowledgeBaseAdapter", "conn-string")
    assert r.name == "my_knowledge_base"
    assert r.connection == "conn-string"


def test_unimplemented_template_adapter_reports_error_not_evidence():
    svc = RetrievalService(load_retriever("retrieval.adapters.custom_adapter_template:MyKnowledgeBaseAdapter"))
    result = svc.retrieve(QUERY)
    assert result.status == RetrievalStatus.ERROR
    assert result.passages == []


def test_service_returns_timeout():
    class Slow(Retriever):
        name = "slow"

        def retrieve(self, query):
            time.sleep(1.0)
            return RetrievalResult(status=RetrievalStatus.OK, passages=make_passages())

    result = RetrievalService(Slow(), timeout_seconds=0.1).retrieve(QUERY)
    assert result.status == RetrievalStatus.TIMEOUT
    assert result.passages == []


def test_service_handles_adapter_exception():
    result = RetrievalService(StaticRetriever(exc=RuntimeError("db down"))).retrieve(QUERY)
    assert result.status == RetrievalStatus.ERROR
    assert result.adapter == "static_test"


def test_service_rejects_malformed_result():
    class Bad(Retriever):
        def retrieve(self, query):
            return {"passages": []}

    assert RetrievalService(Bad()).retrieve(QUERY).status == RetrievalStatus.ERROR


def test_empty_ok_result_becomes_no_results():
    svc = RetrievalService(StaticRetriever(RetrievalResult(status=RetrievalStatus.OK, passages=[])))
    assert svc.retrieve(QUERY).status == RetrievalStatus.NO_RESULTS


def test_min_relevance_filters_sorts_and_limits():
    passages = [
        Passage(text="low", source_id="a", relevance_score=0.1),
        Passage(text="high", source_id="b", relevance_score=0.9),
        Passage(text="mid", source_id="c", relevance_score=0.5),
        Passage(text="   ", source_id="d", relevance_score=1.0),
    ]
    svc = RetrievalService(StaticRetriever(RetrievalResult(status=RetrievalStatus.OK, passages=passages)),
                           min_relevance=0.3)
    result = svc.retrieve(QUERY.model_copy(update={"top_k": 1}))
    assert [p.source_id for p in result.passages] == ["b"]


def test_all_filtered_becomes_no_results():
    svc = RetrievalService(
        StaticRetriever(RetrievalResult(status=RetrievalStatus.OK,
                                        passages=[Passage(text="x", source_id="a", relevance_score=0.01)])),
        min_relevance=0.5,
    )
    assert svc.retrieve(QUERY).status == RetrievalStatus.NO_RESULTS


def test_non_ok_status_never_carries_passages():
    svc = RetrievalService(StaticRetriever(RetrievalResult(status=RetrievalStatus.ERROR, passages=make_passages())))
    result = svc.retrieve(QUERY)
    assert result.status == RetrievalStatus.ERROR and result.passages == []
