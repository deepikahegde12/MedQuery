"""FAISS cosine-search adapter for the local index produced by retrieval.ingest."""

import json
import logging
import math
from pathlib import Path
import re
import threading

from retrieval.base import Passage, RetrievalQuery, RetrievalResult, RetrievalStatus, Retriever
from retrieval.ingest import corpus_fingerprint
from query_processing.preprocess import STOPWORDS

logger = logging.getLogger(__name__)


class FaissRetriever(Retriever):
    name = "faiss"
    requires_query_embedding = False

    def __init__(self, connection: str = "", allow_model_download: bool = False,
                 hybrid_enabled: bool = True, **options) -> None:
        super().__init__(connection, **options)
        self.allow_model_download = allow_model_download
        self.hybrid_enabled = hybrid_enabled
        self.path = Path(connection) if connection else Path(__file__).resolve().parents[2] / "data" / "medical_kb"
        self.index = None
        self.chunks: list[dict] = []
        self.encoder = None
        self.model_name: str | None = None
        self.model_revision: str | None = None
        self.dimension: int | None = None
        self.load_error: str | None = None
        self.model_error: str | None = None
        self._encoder_lock = threading.Lock()
        self._lexical_docs: list[list[str]] = []
        self._lexical_doc_freq: dict[str, int] = {}
        self._lexical_avg_length = 0.0
        self.lexical_error: str | None = None
        manifest_path = self.path / "index_manifest.json"
        index_path = self.path / "index.faiss"
        chunks_path = self.path / "chunks.json"
        if not (manifest_path.is_file() and index_path.is_file() and chunks_path.is_file()):
            self.load_error = f"FAISS index is incomplete in {self.path}; build it with retrieval.build_index."
            return
        try:
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError("FAISS is unavailable; install backend/requirements-ml.txt") from exc
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.model_name = manifest["embedding_model"]
            self.model_revision = manifest.get("embedding_model_revision")
            self.dimension = int(manifest["dimension"])
            if self.dimension <= 0:
                raise ValueError("FAISS manifest dimension must be positive")
            self.index = faiss.read_index(str(index_path))
            self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            if self.index.ntotal == 0:
                raise ValueError("FAISS index contains no vectors")
            if self.index.ntotal != len(self.chunks) or self.index.ntotal != manifest["chunk_count"]:
                raise ValueError("FAISS index and chunk metadata counts differ")
            if self.index.d != self.dimension:
                raise ValueError("FAISS index dimension does not match its manifest")
            if corpus_fingerprint(self.chunks) != manifest.get("corpus_fingerprint"):
                raise ValueError("FAISS index and chunk metadata have different corpus fingerprints")
            if any(not all(row.get(key) for key in ("document_id", "chunk_id", "text", "title", "source_url"))
                   for row in self.chunks):
                raise ValueError("FAISS chunk metadata is missing required source fields")
            if hybrid_enabled:
                try:
                    self._build_lexical_index()
                except Exception as exc:  # noqa: BLE001 - retain dense-only operation
                    self.lexical_error = f"Lexical index could not be initialized ({type(exc).__name__}): {exc}"
                    logger.warning("%s; continuing with FAISS-only retrieval", self.lexical_error)
        except Exception as exc:
            logger.exception("Failed to load FAISS knowledge base from %s", self.path)
            self.index = None
            self.chunks = []
            self.load_error = f"FAISS knowledge base could not be loaded ({type(exc).__name__}): {exc}"

    @property
    def is_configured(self) -> bool:
        return self.index is not None and bool(self.chunks)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        if not self.is_configured:
            return RetrievalResult(status=RetrievalStatus.NOT_CONFIGURED,
                                   message=self.load_error or "No valid FAISS knowledge index is available.", adapter=self.name)
        if not query.normalized_text.strip() or query.top_k <= 0:
            return RetrievalResult(status=RetrievalStatus.NO_RESULTS, adapter=self.name)
        try:
            encoder = self._get_encoder()
            vector = encoder.encode([query.original_text], convert_to_numpy=True, normalize_embeddings=True)
            import numpy as np
            vector = np.asarray(vector, dtype="float32")
            if vector.shape != (1, self.dimension) or not np.isfinite(vector).all():
                raise ValueError("query encoder returned an invalid vector for the persisted FAISS dimension")
            scores, positions = self.index.search(vector, len(self.chunks))
            dense_order = [int(pos) for pos in positions[0].tolist() if pos >= 0]
            try:
                lexical_order = self._lexical_search(query) if self._lexical_docs else []
            except Exception as exc:  # noqa: BLE001 - dense search remains usable
                self.lexical_error = f"Lexical search failed ({type(exc).__name__}): {exc}"
                logger.warning("%s; continuing with FAISS-only retrieval", self.lexical_error)
                lexical_order = []
            dense_rank = {position: rank for rank, position in enumerate(dense_order, start=1)}
            lexical_rank = {position: rank for rank, position in enumerate(lexical_order, start=1)}
            score_by_position = {int(pos): float(score) for score, pos in zip(scores[0].tolist(), positions[0].tolist()) if pos >= 0}
            positions_by_fusion = sorted(
                dense_rank,
                key=lambda pos: -(
                    1.0 / (60 + dense_rank[pos])
                    + (1.0 / (60 + lexical_rank[pos]) if pos in lexical_rank else 0.0)
                ),
            )
            candidate_depth = min(len(self.chunks), max(query.top_k * 4, query.top_k))
            passages = []
            for position in positions_by_fusion[:candidate_depth]:
                if position < 0:
                    continue
                score = score_by_position[position]
                row = self.chunks[position]
                rrf_score = 1.0 / (60 + dense_rank[position])
                if position in lexical_rank:
                    rrf_score += 1.0 / (60 + lexical_rank[position])
                passages.append(Passage(
                    text=row["text"], source_id=row["document_id"], source_title=row["title"],
                    source_url=row["source_url"], relevance_score=max(0.0, min(1.0, (float(score) + 1.0) / 2.0)),
                    metadata={**{key: row.get(key) for key in (
                        "organization", "topic", "section_heading", "publication_date", "updated_date",
                        "retrieved_at", "content_provenance", "license", "usage_notes", "ingested_at",
                        "document_content_hash", "content_hash", "source_path", "source_metadata")}, "chunk_id": row["chunk_id"],
                        "document_id": row["document_id"], "similarity": float(score),
                        "dense_rank": dense_rank[position], "lexical_rank": lexical_rank.get(position),
                        "rank_fusion_score": rrf_score},
                ))
            return RetrievalResult(status=RetrievalStatus.OK if passages else RetrievalStatus.NO_RESULTS,
                                   passages=passages, adapter=self.name)
        except Exception as exc:
            logger.exception("FAISS retrieval failed")
            hint = " Check the cached embedding model or enable MEDQUERY_KB_ALLOW_MODEL_DOWNLOAD." if isinstance(exc, OSError) else ""
            return RetrievalResult(status=RetrievalStatus.ERROR,
                                   message=f"Semantic retrieval failed ({type(exc).__name__}): {exc}.{hint}", adapter=self.name)

    def _get_encoder(self):
        if self.encoder is not None:
            return self.encoder
        with self._encoder_lock:
            if self.encoder is not None:
                return self.encoder
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                self.model_error = "sentence-transformers is unavailable; install backend/requirements-ml.txt"
                raise RuntimeError(self.model_error) from exc
            try:
                encoder = SentenceTransformer(
                    self.model_name,
                    revision=self.model_revision,
                    local_files_only=not self.allow_model_download,
                )
            except Exception as exc:
                mode = ("API downloads are disabled; cache this model or set "
                        "MEDQUERY_KB_ALLOW_MODEL_DOWNLOAD=true" if not self.allow_model_download
                        else "check model name, revision, and network access")
                self.model_error = f"Could not load query embedding model {self.model_name!r}; {mode}: {exc}"
                raise RuntimeError(self.model_error) from exc
            actual_dimension = encoder.get_sentence_embedding_dimension()
            if actual_dimension is not None and int(actual_dimension) != self.dimension:
                self.model_error = "query embedding model dimension differs from the persisted FAISS index"
                raise ValueError(self.model_error)
            self.encoder = encoder
            self.model_error = None
            return encoder

    def _build_lexical_index(self) -> None:
        self._lexical_docs = [
            _tokens(" ".join((str(row.get("title") or ""), str(row.get("section_heading") or ""), row["text"])))
            for row in self.chunks
        ]
        doc_freq: dict[str, int] = {}
        for terms in self._lexical_docs:
            for term in set(terms):
                doc_freq[term] = doc_freq.get(term, 0) + 1
        self._lexical_doc_freq = doc_freq
        self._lexical_avg_length = sum(map(len, self._lexical_docs)) / max(1, len(self._lexical_docs))

    def _lexical_search(self, query: RetrievalQuery) -> list[int]:
        terms = list(dict.fromkeys(
            term for term in _tokens(" ".join(query.query_variants or [query.normalized_text]))
            if term not in STOPWORDS
        ))
        if not terms:
            return []
        n = len(self._lexical_docs)
        scores: list[tuple[float, int]] = []
        for position, document in enumerate(self._lexical_docs):
            frequencies: dict[str, int] = {}
            for token in document:
                frequencies[token] = frequencies.get(token, 0) + 1
            score = 0.0
            for term in terms:
                df = self._lexical_doc_freq.get(term, 0)
                tf = frequencies.get(term, 0)
                if not tf or not df:
                    continue
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = tf + 1.5 * (1 - 0.75 + 0.75 * len(document) / max(self._lexical_avg_length, 1e-9))
                score += idf * tf * (1.5 + 1) / denom
            if score > 0:
                scores.append((score, position))
        scores.sort(key=lambda pair: (-pair[0], pair[1]))
        return [position for _, position in scores]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.casefold())


__all__ = ["FaissRetriever"]
