"""TEMPLATE — connect your own medical knowledge base here.

This file is intentionally empty of any medical content. Copy it (or edit it in
place), implement `retrieve`, then set in `.env`:

    MEDQUERY_KB_ADAPTER=retrieval.adapters.custom_adapter_template:MyKnowledgeBaseAdapter
    MEDQUERY_KB_CONNECTION=<whatever your adapter needs: a path, DB URL, index name...>

Until `retrieve` is implemented, selecting this adapter makes every query
return status ERROR (it raises NotImplementedError) — it never fakes evidence.
"""

from retrieval.base import Passage, RetrievalQuery, RetrievalResult, RetrievalStatus, Retriever


class MyKnowledgeBaseAdapter(Retriever):
    name = "my_knowledge_base"

    def __init__(self, connection: str = "", **options) -> None:
        super().__init__(connection, **options)
        # Open your index / database / client here using `self.connection`.
        # Example: self.client = SomeClient(self.connection)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        # Use any of: query.normalized_text, query.keywords, query.question_type,
        # query.embedding (BERT vector, may be None), query.top_k.
        #
        # hits = self.client.search(...)
        # passages = [
        #     Passage(
        #         text=hit.text,
        #         source_id=hit.id,
        #         source_title=hit.title,
        #         source_url=hit.url,
        #         relevance_score=hit.score,
        #         metadata={...},
        #     )
        #     for hit in hits
        # ]
        # status = RetrievalStatus.OK if passages else RetrievalStatus.NO_RESULTS
        # return RetrievalResult(status=status, passages=passages, adapter=self.name)
        raise NotImplementedError("MyKnowledgeBaseAdapter.retrieve is not implemented yet")

    def close(self) -> None:
        # Close your client/connection here.
        pass


__all__ = ["MyKnowledgeBaseAdapter", "Passage", "RetrievalResult", "RetrievalStatus"]
