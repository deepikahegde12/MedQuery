import json
import sys
import types

import numpy as np
import pytest

from retrieval.adapters.faiss_adapter import FaissRetriever
from retrieval.base import RetrievalQuery, RetrievalStatus
from retrieval.ingest import corpus_fingerprint


def _write_index_files(path):
    path.mkdir()
    chunks = [{
        "document_id": "doc-a", "chunk_id": "doc-a-00000", "text": "Evidence text.",
        "title": "Verified title", "organization": "Health agency",
        "source_url": "https://health.example/source", "topic": "topic",
        "section_heading": "Section", "publication_date": None,
        "ingested_at": "2025-01-01T00:00:00+00:00", "content_hash": "hash",
    }]
    manifest = {"embedding_model": "fake-model", "dimension": 2, "chunk_count": 1,
                "corpus_fingerprint": corpus_fingerprint(chunks)}
    (path / "index_manifest.json").write_text(json.dumps(manifest))
    (path / "index.faiss").write_bytes(b"fake")
    (path / "chunks.json").write_text(json.dumps(chunks))


def test_missing_index_is_explicitly_unconfigured(tmp_path):
    retriever = FaissRetriever(str(tmp_path / "missing"))
    assert retriever.is_configured is False
    assert retriever.retrieve(RetrievalQuery(original_text="q", normalized_text="q")).status == RetrievalStatus.NOT_CONFIGURED


def test_retrieval_uses_index_model_and_preserves_source(monkeypatch, tmp_path):
    _write_index_files(tmp_path / "kb")
    index = types.SimpleNamespace(
        ntotal=1, d=2,
        search=lambda query, k: (np.array([[0.8]], dtype="float32"), np.array([[0]], dtype="int64")),
    )
    monkeypatch.setitem(sys.modules, "faiss", types.SimpleNamespace(read_index=lambda path: index))
    calls = []
    encode_calls = []

    class Encoder:
        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            encode_calls.append(texts)
            return np.array([[1.0, 0.0]], dtype="float32")

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Encoder))
    retriever = FaissRetriever(str(tmp_path / "kb"))
    lexical_index = retriever._lexical_docs
    query = RetrievalQuery(original_text="question", normalized_text="question", top_k=3)
    result = retriever.retrieve(query)
    assert result.status == RetrievalStatus.OK
    assert result.passages[0].source_title == "Verified title"
    assert result.passages[0].source_url == "https://health.example/source"
    assert result.passages[0].metadata["chunk_id"] == "doc-a-00000"
    assert result.passages[0].relevance_score == pytest.approx(0.9)
    assert result.passages[0].metadata["dense_rank"] == 1
    assert result.passages[0].metadata["rank_fusion_score"] == 1 / 61
    assert result.passages[0].metadata["lexical_rank"] is None
    assert retriever.retrieve(query).status == RetrievalStatus.OK
    assert retriever._lexical_docs is lexical_index
    assert calls == [("fake-model", {"revision": None, "local_files_only": True})]
    assert encode_calls == [["question"], ["question"]]


def test_lexical_rank_can_promote_exact_section_match(monkeypatch, tmp_path):
    path = tmp_path / "hybrid"
    path.mkdir()
    chunks = [
        {"document_id": "general", "chunk_id": "general-1", "text": "Influenza is a respiratory infection.",
         "title": "Influenza", "organization": "Health agency", "source_url": "https://example.test/general",
         "section_heading": "What is the flu?", "content_hash": "one"},
        {"document_id": "symptoms", "chunk_id": "symptoms-1", "text": "Symptoms include fever and cough.",
         "title": "Influenza", "organization": "Health agency", "source_url": "https://example.test/symptoms",
         "section_heading": "What are the symptoms?", "content_hash": "two"},
    ]
    manifest = {"embedding_model": "fake-model", "dimension": 2, "chunk_count": 2,
                "corpus_fingerprint": corpus_fingerprint(chunks)}
    (path / "index_manifest.json").write_text(json.dumps(manifest))
    (path / "index.faiss").write_bytes(b"fake")
    (path / "chunks.json").write_text(json.dumps(chunks))
    index = types.SimpleNamespace(
        ntotal=2, d=2,
        search=lambda query, k: (np.array([[0.8, 0.7]], dtype="float32"), np.array([[0, 1]], dtype="int64")),
    )
    monkeypatch.setitem(sys.modules, "faiss", types.SimpleNamespace(read_index=lambda path: index))

    class Encoder:
        def __init__(self, name, **kwargs):
            pass

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0]], dtype="float32")

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Encoder))
    result = FaissRetriever(str(path)).retrieve(RetrievalQuery(
        original_text="What are common symptoms?", normalized_text="what are common symptoms?",
        query_variants=["what are common symptoms?"], top_k=1,
    ))
    assert result.passages[0].source_id == "symptoms"
    assert result.passages[0].metadata["lexical_rank"] == 1


def test_lexical_initialization_failure_keeps_dense_fallback(monkeypatch, tmp_path):
    _write_index_files(tmp_path / "kb")
    index = types.SimpleNamespace(
        ntotal=1, d=2,
        search=lambda query, k: (np.array([[0.8]], dtype="float32"), np.array([[0]], dtype="int64")),
    )
    monkeypatch.setitem(sys.modules, "faiss", types.SimpleNamespace(read_index=lambda path: index))

    def fail_lexical_initialization(self):
        raise MemoryError("test")

    monkeypatch.setattr(FaissRetriever, "_build_lexical_index", fail_lexical_initialization)

    class Encoder:
        def __init__(self, name, **kwargs):
            pass

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0]], dtype="float32")

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Encoder))
    retriever = FaissRetriever(str(tmp_path / "kb"))
    assert retriever.is_configured is True
    assert "MemoryError" in retriever.lexical_error
    result = retriever.retrieve(RetrievalQuery(original_text="question", normalized_text="question"))
    assert result.status == RetrievalStatus.OK
    assert result.passages[0].source_id == "doc-a"
