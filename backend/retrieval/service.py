"""Loads the configured knowledge-base adapter and calls it safely."""

import importlib
import inspect
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from retrieval.base import RetrievalQuery, RetrievalResult, RetrievalStatus, Retriever
from retrieval.not_configured import NotConfiguredRetriever

logger = logging.getLogger(__name__)


def load_retriever(adapter_path: str, connection: str = "", **options) -> Retriever:
    """Instantiate the adapter named by "package.module:ClassName".

    Falls back to NotConfiguredRetriever (with an explanatory reason) when the
    path is empty or invalid, so the app still starts.
    """
    if not adapter_path.strip():
        return NotConfiguredRetriever()

    module_name, sep, class_name = adapter_path.partition(":")
    if not sep or not module_name or not class_name:
        logger.error("Invalid MEDQUERY_KB_ADAPTER %r (expected 'module:ClassName')", adapter_path)
        return NotConfiguredRetriever(reason=f"Invalid knowledge-base adapter path '{adapter_path}'.")

    try:
        cls = getattr(importlib.import_module(module_name), class_name)
        if not (isinstance(cls, type) and issubclass(cls, Retriever)):
            raise TypeError(f"{adapter_path} is not a Retriever subclass")
        signature = inspect.signature(cls)
        accepts_extra = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        supported_options = options if accepts_extra else {
            key: value for key, value in options.items() if key in signature.parameters
        }
        retriever = cls(connection=connection, **supported_options)
    except Exception as exc:  # noqa: BLE001 - adapter code is user-supplied
        logger.exception("Failed to load knowledge-base adapter %s", adapter_path)
        return NotConfiguredRetriever(reason=f"Knowledge-base adapter failed to load: {exc}")

    logger.info("Knowledge-base adapter loaded: %s", retriever.name)
    return retriever


class RetrievalService:
    """Wraps an adapter with a timeout, error handling and relevance filtering."""

    def __init__(self, retriever: Retriever, timeout_seconds: float = 10.0,
                 min_relevance: float = 0.0, default_top_k: int = 5):
        self.retriever = retriever
        self.timeout_seconds = timeout_seconds
        self.min_relevance = min_relevance
        self.default_top_k = default_top_k
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kb")

    @property
    def is_configured(self) -> bool:
        return self.retriever.is_configured

    @property
    def unavailable_reason(self) -> str | None:
        """Explain index or lazy embedding-model problems, when the adapter knows."""
        model_error = getattr(self.retriever, "model_error", None)
        if model_error:
            return model_error
        if self.is_configured:
            return None
        return (getattr(self.retriever, "load_error", None)
                or getattr(self.retriever, "reason", None)
                or "No persisted knowledge index is available.")

    @property
    def embedding_model_loaded(self) -> bool | None:
        """Expose lazy model state for adapters that provide it."""
        encoder = getattr(self.retriever, "encoder", None)
        return None if not hasattr(self.retriever, "encoder") else encoder is not None

    @property
    def requires_query_embedding(self) -> bool:
        return bool(getattr(self.retriever, "requires_query_embedding", True))

    def retrieve_question(self, question: str, top_k: int | None = None) -> RetrievalResult:
        """Convenience entry point for callers without precomputed query analysis.

        The main API continues to pass its richer BERT/query-analysis object via
        ``retrieve``. This helper does not substitute BERT vectors for the adapter's
        sentence embeddings.
        """
        normalized = question.strip().lower()
        limit = self.default_top_k if top_k is None else top_k
        if not normalized or limit <= 0:
            return RetrievalResult(status=RetrievalStatus.NO_RESULTS, adapter=self.retriever.name)
        return self.retrieve(RetrievalQuery(
            original_text=question.strip(), normalized_text=normalized, top_k=limit,
        ))

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        name = getattr(self.retriever, "name", type(self.retriever).__name__)
        future = self._executor.submit(self.retriever.retrieve, query)
        try:
            result = future.result(timeout=self.timeout_seconds)
        except FutureTimeout:
            future.cancel()
            logger.warning("Knowledge-base adapter %s timed out after %.1fs", name, self.timeout_seconds)
            return RetrievalResult(
                status=RetrievalStatus.TIMEOUT,
                message=f"The knowledge base did not respond within {self.timeout_seconds:g} seconds.",
                adapter=name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Knowledge-base adapter %s raised an error", name)
            return RetrievalResult(
                status=RetrievalStatus.ERROR,
                message=f"The knowledge base returned an error ({type(exc).__name__}).",
                adapter=name,
            )

        if not isinstance(result, RetrievalResult):
            logger.error("Adapter %s returned %r instead of RetrievalResult", name, type(result))
            return RetrievalResult(
                status=RetrievalStatus.ERROR,
                message="The knowledge base returned a malformed response.",
                adapter=name,
            )

        result.adapter = name
        if result.status == RetrievalStatus.OK:
            kept = [
                p for p in result.passages
                if p.text.strip() and (
                    (p.relevance_score is not None and p.relevance_score >= self.min_relevance)
                    or (p.relevance_score is None and self.min_relevance <= 0)
                )
            ]
            kept.sort(key=lambda p: p.relevance_score if p.relevance_score is not None else float("-inf"), reverse=True)
            kept.sort(key=lambda p: float(p.metadata.get("rank_fusion_score") or 0.0), reverse=True)
            result.passages = _deduplicate_passages(kept)[: query.top_k]
            if not result.passages:
                result.status = RetrievalStatus.NO_RESULTS
                result.message = result.message or "No sufficiently relevant evidence was found."
        else:
            # Non-OK statuses must never carry evidence.
            result.passages = []
        return result

    def close(self) -> None:
        try:
            self.retriever.close()
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)


def _deduplicate_passages(passages):
    """Drop contained overlap chunks while retaining the most useful context."""
    unique = []
    seen: dict[tuple[str, str], list[tuple[str, int]]] = {}
    exact: set[tuple[str, str, str]] = set()
    for passage in passages:
        normalized = re.sub(r"\s+", " ", passage.text).strip().casefold()
        section = str(passage.metadata.get("section_heading") or "").casefold()
        exact_key = (passage.source_id, section, normalized)
        if exact_key in exact:
            continue
        exact.add(exact_key)
        if not section:
            unique.append(passage)
            continue
        key = (passage.source_id, section)
        previous = seen.setdefault(key, [])
        # Overlap chunks often repeat a whole short paragraph. Keep the highest
        # ranked copy when equally informative; if one contains another, retain
        # the longer context and its own source attribution.
        replaced = False
        for position, (prior, unique_index) in enumerate(previous):
            if normalized == prior or normalized in prior:
                replaced = True
                break
            if prior in normalized:
                unique[unique_index] = passage
                previous[position] = (normalized, unique_index)
                replaced = True
                break
        if replaced:
            continue
        previous.append((normalized, len(unique)))
        unique.append(passage)
    return unique
