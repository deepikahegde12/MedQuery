"""Builds the application's long-lived services from Settings."""

from dataclasses import dataclass

from answer_processing.generators import AnswerGenerator, EvidenceOnlyGenerator, ExtractiveQAGenerator
from answer_processing.pipeline import AnswerPipeline
from app.config import Settings
from app.storage import QueryStore
from query_processing.bert_encoder import BertQueryEncoder
from query_processing.understanding import QueryUnderstanding
from retrieval.service import RetrievalService, load_retriever


@dataclass
class Services:
    settings: Settings
    encoder: BertQueryEncoder
    retrieval: RetrievalService
    generator: AnswerGenerator
    pipeline: AnswerPipeline
    store: QueryStore

    def close(self) -> None:
        self.retrieval.close()
        self.store.close()


def build_services(settings: Settings) -> Services:
    encoder = BertQueryEncoder(settings.query_model_name, settings.query_max_tokens, settings.hf_offline)
    retrieval = RetrievalService(
        load_retriever(settings.kb_adapter, settings.kb_connection),
        timeout_seconds=settings.kb_timeout_seconds,
        min_relevance=settings.kb_min_relevance,
    )
    if settings.answer_mode == "extractive":
        generator: AnswerGenerator = ExtractiveQAGenerator(
            settings.qa_model_name, settings.qa_max_answer_tokens, settings.hf_offline
        )
    else:
        generator = EvidenceOnlyGenerator()

    pipeline = AnswerPipeline(
        QueryUnderstanding(encoder), retrieval, generator,
        top_k=settings.kb_top_k, min_confidence=settings.qa_min_confidence,
    )
    return Services(settings, encoder, retrieval, generator, pipeline, QueryStore(settings.database_path))
