from retrieval.base import RetrievalQuery, RetrievalResult, RetrievalStatus, Retriever


class NotConfiguredRetriever(Retriever):
    """Default adapter used until a real knowledge base is connected.

    It never returns passages; it only reports that retrieval is unavailable.
    """

    name = "not_configured"

    def __init__(self, connection: str = "", reason: str | None = None, **options) -> None:
        super().__init__(connection, **options)
        self.reason = reason or (
            "No medical knowledge base is connected. Set MEDQUERY_KB_ADAPTER to "
            "enable evidence retrieval."
        )

    @property
    def is_configured(self) -> bool:
        return False

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        return RetrievalResult(
            status=RetrievalStatus.NOT_CONFIGURED,
            passages=[],
            message=self.reason,
            adapter=self.name,
        )
