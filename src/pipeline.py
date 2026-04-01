"""
End-to-End RAG Pipeline. Orchestrates all six layers.
FIX #1: ingest_text() now correctly calls ingest_bytes().
FIX #2: flush_bm25() for batch ingestion efficiency.
FIX #9: Uses embedder.dimension as source of truth for Qdrant collection.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from config.settings import Settings, LLMBackend
from prompts.clinical_rag import SYSTEM_PROMPT
from src.models import (
    AuditRecord,
    Citation,
    DocumentType,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievalResult,
)
from src.ingestion.loader import DocumentLoader
from src.chunking.chunker import ChunkerConfig, get_chunker
from src.embedding.embedder import LocalEmbedder
from src.vectorstore.qdrant_store import QdrantStore
from src.vectorstore.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.inference.base import LLMClientBase
from src.inference.vllm_client import VLLMClient
from src.inference.ollama_client import OllamaClient
from src.filtering.phi_scanner import PHIOutputScanner
from src.filtering.hallucination_checker import HallucinationChecker
from src.audit.logger import AuditLogger

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._init_components()

    def _init_components(self) -> None:
        s = self.settings

        # Layer 1
        self.loader = DocumentLoader(deidentify=s.enable_phi_detection, phi_threshold=s.phi_detection_threshold)

        # Layer 2
        self.chunker = get_chunker(s.chunk_strategy.value, ChunkerConfig(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap))

        # Layer 3 — FIX #9: Use embedder.dimension as source of truth
        self.embedder = LocalEmbedder(model_name=s.embedding_model, device=s.embedding_device, batch_size=s.embedding_batch_size)
        actual_dim = self.embedder.dimension  # Forces model load + reads real dimension
        self.dense_store = QdrantStore(host=s.qdrant_host, port=s.qdrant_port, collection_name=s.qdrant_collection, dimension=actual_dim)
        self.sparse_store = BM25Store()

        # Layer 4
        self.reranker = CrossEncoderReranker(model_name=s.reranker_model, device=s.embedding_device)
        self.retriever = HybridRetriever(embedder=self.embedder, dense_store=self.dense_store, sparse_store=self.sparse_store, reranker=self.reranker, rrf_k=s.retrieval_rrf_k, rerank_top_n=s.retrieval_rerank_top_n)

        # Layer 5
        self.llm = self._create_llm_client()

        # Layer 6
        self.phi_scanner = PHIOutputScanner(threshold=s.phi_detection_threshold)
        self.hallucination_checker = HallucinationChecker()

        # Audit
        self.audit = AuditLogger(log_dir=s.audit_log_dir, enabled=s.enable_audit_logging)

    def _create_llm_client(self) -> LLMClientBase:
        s = self.settings
        if s.llm_backend == LLMBackend.VLLM:
            return VLLMClient(host=s.vllm_host, model=s.llm_model)
        return OllamaClient(host=s.ollama_host, model=s.ollama_model)

    # === Ingestion ===

    def ingest_file(self, path: str | Path, document_type: DocumentType = DocumentType.OTHER, acl_tags: list[str] | None = None, source: str = "", title: str = "") -> IngestResponse:
        doc = self.loader.load_file(path, document_type=document_type, acl_tags=acl_tags, source=source, title=title)
        chunks = self.chunker.chunk(doc.text, doc.metadata, doc.sections)
        if chunks:
            embeddings = self.embedder.embed_chunks_to_list([c.text for c in chunks])
            self.dense_store.upsert(chunks, embeddings)
            self.sparse_store.add(chunks)
        return IngestResponse(document_id=doc.metadata.document_id, chunks_created=len(chunks))

    def ingest_text(self, text: str, title: str = "", document_type: DocumentType = DocumentType.OTHER, acl_tags: list[str] | None = None, source: str = "") -> IngestResponse:
        """FIX #1: Actually ingests the text instead of reading /dev/null."""
        return self.ingest_bytes(
            content=text.encode("utf-8"),
            filename="inline_text.md",
            document_type=document_type,
            acl_tags=acl_tags,
            source=source,
            title=title,
        )

    def ingest_bytes(self, content: bytes, filename: str, document_type: DocumentType = DocumentType.OTHER, acl_tags: list[str] | None = None, source: str = "", title: str = "") -> IngestResponse:
        doc = self.loader.load_bytes(content, filename, document_type=document_type, acl_tags=acl_tags, source=source, title=title)
        chunks = self.chunker.chunk(doc.text, doc.metadata, doc.sections)
        if chunks:
            embeddings = self.embedder.embed_chunks_to_list([c.text for c in chunks])
            self.dense_store.upsert(chunks, embeddings)
            self.sparse_store.add(chunks)
        return IngestResponse(document_id=doc.metadata.document_id, chunks_created=len(chunks))

    def flush_bm25(self) -> None:
        """FIX #2: Explicit BM25 rebuild after batch ingestion."""
        self.sparse_store.rebuild()

    # === Query ===

    def query(self, request: QueryRequest) -> QueryResponse:
        start = time.time()

        # FIX Bug 7: Validate and coerce acl_tags from untyped filters dict
        raw_acl = request.filters.get("acl_tags")
        if isinstance(raw_acl, str):
            acl_tags: list[str] | None = [raw_acl]
        elif isinstance(raw_acl, list):
            acl_tags = [str(t) for t in raw_acl]
        else:
            acl_tags = None

        retrieval = self.retriever.retrieve(query=request.query, user_id=request.user_id, top_k=request.max_results, acl_tags=acl_tags)

        context = self._build_context(retrieval)
        prompt = SYSTEM_PROMPT.format(context=context, query=request.query)
        try:
            answer = self.llm.generate(prompt, max_tokens=2048, temperature=0.1)
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            answer = "I was unable to generate a response due to a system error. Please try again."

        phi_detections = self.phi_scanner.scan(answer) if self.settings.enable_phi_detection else []
        if phi_detections:
            logger.warning("PHI detected in output — redacting")
            answer = self.phi_scanner.redact(answer)

        # FIX Bug 4: Phantom citations are now logged and acted upon
        phantom = self.hallucination_checker.check_citations(answer, retrieval.passages)
        if phantom:
            logger.warning("Phantom citations in response: %s — flagging for review", phantom)
        confidence = self.hallucination_checker.estimate_confidence(retrieval.passages, answer)
        is_abstention = self.hallucination_checker.has_abstention_markers(answer)

        citations = [Citation(chunk_id=p.chunk.chunk_id, document_title=p.chunk.metadata.title, section=p.chunk.section_path, relevance_score=p.reranker_score) for p in retrieval.passages[:10]]
        elapsed = (time.time() - start) * 1000

        self.audit.log(AuditRecord(user_id=request.user_id, query=request.query, retrieved_chunk_ids=[p.chunk.chunk_id for p in retrieval.passages], model_id=str(self.settings.llm_backend.value), confidence=confidence, is_abstention=is_abstention, phi_detected=len(phi_detections) > 0, phantom_citations_found=len(phantom) > 0, latency_ms=elapsed))

        sources = [{"chunk_id": p.chunk.chunk_id, "title": p.chunk.metadata.title, "source": p.chunk.metadata.source, "section": p.chunk.section_path, "text_preview": p.chunk.text[:200], "reranker_score": round(p.reranker_score, 4)} for p in retrieval.passages[:10]] if request.include_sources else []

        return QueryResponse(answer=answer, citations=citations, confidence=confidence, is_abstention=is_abstention, sources=sources, latency_ms=round(elapsed, 1))

    def _build_context(self, retrieval: RetrievalResult) -> str:
        parts = []
        for p in retrieval.passages[:15]:
            parts.append(f"[DOC_ID: {p.chunk.chunk_id}]\nTitle: {p.chunk.metadata.title}\nSource: {p.chunk.metadata.source}\nSection: {p.chunk.section_path}\nContent: {p.chunk.text}\n---")
        return "\n\n".join(parts) if parts else "No relevant passages were retrieved."
