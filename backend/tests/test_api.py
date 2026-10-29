import pytest
from fastapi.testclient import TestClient

from answer_processing.generators import GeneratedAnswer
from tests.doubles import FakeGenerator, make_client, ok_retriever


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    kb = body["data"]["components"]["knowledge_base"]
    assert kb == {
        "configured": False,
        "adapter": "not_configured",
        "status": "NOT_CONFIGURED",
        "message": "No medical knowledge base is connected. Set MEDQUERY_KB_ADAPTER to enable evidence retrieval.",
        "embedding_model_loaded": None,
    }


def test_query_without_knowledge_base(client):
    r = client.post("/api/query", json={"question": "  What is term X?  "})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["status"] == "KB_NOT_CONFIGURED"
    assert data["answer"] is None and data["evidence"] == [] and data["sources"] == []
    assert data["retrieval"]["status"] == "NOT_CONFIGURED"
    assert data["question"] == "What is term X?"
    assert data["analysis"]["question_type"] == "definition"


def test_emergency_query_skips_retrieval_and_answer_generation():
    from tests.doubles import FakeEncoder, FakeGenerator
    from retrieval.base import RetrievalStatus

    retriever = ok_retriever()
    generator = FakeGenerator()
    encoder = FakeEncoder()
    with make_client(retriever=retriever, generator=generator, encoder=encoder) as c:
        data = c.post("/api/query", json={"question": "I have chest pain right now"}).json()["data"]
    assert data["status"] == "EMERGENCY_GUIDANCE"
    assert data["answer"] is None and data["evidence"] == []
    assert data["retrieval"]["status"] == RetrievalStatus.SKIPPED_SAFETY.value
    assert retriever.queries == []
    assert encoder.calls == []
    assert generator.calls == 0
    assert {"understanding", "retrieval", "answer", "total"} <= set(data["timings_ms"])
    assert data["timings_ms"]["answer"] == 0.0
    assert any(n["code"] == "possible_emergency" for n in data["notices"])


def test_query_with_evidence_and_persistence():
    gen = FakeGenerator(GeneratedAnswer(text="ALPHA0", confidence=0.9, passage_index=0, start=40, end=46))
    with make_client(retriever=ok_retriever(), generator=gen) as c:
        data = c.post("/api/query", json={"question": "what is the answer token"}).json()["data"]
        assert data["status"] == "ANSWERED"
        assert data["answer"]["text"] == "ALPHA0"
        assert data["sources"][0]["url"] == "https://example.test/doc/0"
        assert data["evidence"][0]["source_title"] == "Test Document 0"

        fetched = c.get(f"/api/query/{data['id']}").json()["data"]
        assert fetched["id"] == data["id"] and fetched["answer"]["text"] == "ALPHA0"

        hist = c.get("/api/history").json()["data"]
        assert hist[0]["id"] == data["id"] and hist[0]["status"] == "ANSWERED"

        assert c.delete("/api/history").json()["data"]["deleted"] == 1
        assert c.get("/api/history").json()["data"] == []


def test_personal_medication_query_keeps_evidence_but_skips_answer_extraction():
    gen = FakeGenerator(GeneratedAnswer(text="ALPHA0", confidence=0.99, passage_index=0))
    with make_client(retriever=ok_retriever(), generator=gen) as c:
        data = c.post("/api/query", json={"question": "Can I take ibuprofen with warfarin?"}).json()["data"]
    assert data["answer"] is None
    assert data["status"] == "EVIDENCE_ONLY"
    assert data["evidence"] and data["sources"]
    assert gen.calls == 0
    assert any(notice["code"] == "personal_medical_advice" for notice in data["notices"])


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "  a "},
        {"question": "x" * 501},
        {"question": "12345 ?? !!"},
        {"question": 42},
        {"question": None},
    ],
)
def test_query_validation_errors(client, payload):
    r = client.post("/api/query", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert body["success"] is False
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["message"]


def test_invalid_json_body(client):
    r = client.post("/api/query", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert r.json()["success"] is False


def test_request_too_large(client):
    r = client.post("/api/query", json={"question": "x" * 10_000})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


def test_get_unknown_and_malformed_query_id(client):
    assert client.get("/api/query/" + "a" * 32).status_code == 404
    r = client.get("/api/query/bad-id!")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_id"


def test_unknown_api_route_returns_json_404(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    assert r.json()["success"] is False


def test_method_not_allowed(client):
    r = client.get("/api/query")
    assert r.status_code == 405
    assert r.json()["success"] is False


def test_internal_error_is_wrapped():
    class Exploding(FakeGenerator):
        def generate(self, question, passages):
            raise RuntimeError("secret internals")

    with make_client(retriever=ok_retriever(), generator=Exploding()) as c,             TestClient(c.app, raise_server_exceptions=False) as quiet:
        r = quiet.post("/api/query", json={"question": "what is the answer token"})
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["code"] == "internal_error"
    assert "secret" not in body["error"]["message"]


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "MedQuery" in r.text
    assert client.get("/js/app.js").status_code == 200
