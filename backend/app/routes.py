import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.container import Services
from app.schemas import QueryRequest, error, ok

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _services(request: Request) -> Services:
    return request.app.state.services


@router.get("/health")
def health(request: Request):
    s = _services(request)
    return ok({
        "status": "ok",
        "app": s.settings.app_name,
        "components": {
            "query_encoder": s.encoder.model.status(),
            "knowledge_base": {
                "configured": s.retrieval.is_configured,
                "adapter": s.retrieval.retriever.name,
                "status": "CONFIGURED" if s.retrieval.is_configured else "NOT_CONFIGURED",
                "message": s.retrieval.unavailable_reason,
                "embedding_model_loaded": s.retrieval.embedding_model_loaded,
            },
            "answer_generator": s.generator.status(),
        },
        "limits": {
            "min_question_chars": s.settings.min_question_chars,
            "max_question_chars": s.settings.max_question_chars,
        },
    })


@router.post("/query")
def create_query(body: QueryRequest, request: Request):
    s = _services(request)
    result = s.pipeline.run(body.question)
    try:
        s.store.save(result)
    except Exception:  # noqa: BLE001 - history is best-effort; the answer still returns
        logger.exception("Failed to persist query %s", result.id)
    return ok(result.model_dump(mode="json"))


@router.get("/query/{query_id}")
def get_query(query_id: str, request: Request):
    if not query_id.isalnum() or len(query_id) > 64:
        return JSONResponse(status_code=400, content=error("invalid_id", "Malformed query id."))
    result = _services(request).store.get(query_id)
    if result is None:
        return JSONResponse(status_code=404, content=error("not_found", "No query with that id."))
    return ok(result.model_dump(mode="json"))


@router.get("/history")
def history(request: Request):
    s = _services(request)
    return ok(s.store.history(s.settings.history_limit))


@router.delete("/history")
def clear_history(request: Request):
    return ok({"deleted": _services(request).store.clear()})
