"""Loads the configured knowledge-base adapter and calls it safely."""

import importlib
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from retrieval.base import RetrievalQuery, RetrievalResult, RetrievalStatus, Retriever
from retrieval.not_configured import NotConfiguredRetriever

logger = logging.getLogger(__name__)


def load_retriever(adapter_path: str, connection: str = "") -> Retriever:
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
        retriever = cls(connection=connection)
    except Exception as exc:  # noqa: BLE001 - adapter code is user-supplied
        logger.exception("Failed to load knowledge-base adapter %s", adapter_path)
        return NotConfiguredRetriever(reason=f"Knowledge-base adapter failed to load: {exc}")

    logger.info("Knowledge-base adapter loaded: %s", retriever.name)
    return retriever


class RetrievalService:
    """Wraps an adapter with a timeout, error handling and relevance filtering."""

    def __init__(self, retriever: Retriever, timeout_seconds: float = 10.0, min_relevance: float = 0.0):
        self.retriever = retriever
        self.timeout_seconds = timeout_seconds
        self.min_relevance = min_relevance
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kb")

    @property
    def is_configured(self) -> bool:
        return self.retriever.is_configured

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
                if p.text.strip() and (p.relevance_score is None or p.relevance_score >= self.min_relevance)
            ]
            kept.sort(key=lambda p: p.relevance_score if p.relevance_score is not None else float("-inf"), reverse=True)
            result.passages = kept[: query.top_k]
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
