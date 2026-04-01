"""Layer 4 — Hybrid Retrieval: BM25 + Dense ANN + Cross-Encoder + RRF. ACL at retrieval (C2)."""

from __future__ import annotations

import logging
import time

from src.models import Chunk, RetrievedPassage, RetrievalResult
from src.embedding.embedder import LocalEmbedder
from src.vectorstore.qdrant_store import QdrantStore
from src.vectorstore.bm25_store import BM25Store
from src.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, embedder: LocalEmbedder, dense_store: QdrantStore, sparse_store: BM25Store, reranker: CrossEncoderReranker, rrf_k: int = 60, rerank_top_n: int = 50):
        self.embedder = embedder
        self.dense_store = dense_store
        self.sparse_store = sparse_store
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.rerank_top_n = rerank_top_n

    def retrieve(self, query: str, user_id: str, top_k: int = 20, acl_tags: list[str] | None = None) -> RetrievalResult:
        start = time.time()
        query_vec = self.embedder.embed_query(query).tolist()
        dense_results = self.dense_store.search(query_vec, top_k=self.rerank_top_n, acl_tags=acl_tags)
        sparse_results = self.sparse_store.search(query, top_k=self.rerank_top_n, acl_tags=acl_tags)
        fused = self._rrf_fuse(dense_results, sparse_results)

        result_passages = []
        if fused:
            passages_text = [chunk.text for chunk, _ in fused[:self.rerank_top_n]]
            reranked = self.reranker.rerank(query, passages_text, top_k=top_k)
            for orig_idx, rerank_score in reranked:
                chunk, rrf_score = fused[orig_idx]
                dense_score = next((s for c, s in dense_results if c.chunk_id == chunk.chunk_id), 0.0)
                sparse_score = next((s for c, s in sparse_results if c.chunk_id == chunk.chunk_id), 0.0)
                result_passages.append(RetrievedPassage(chunk=chunk, dense_score=dense_score, sparse_score=sparse_score, rrf_score=rrf_score, reranker_score=rerank_score))

        elapsed = (time.time() - start) * 1000
        logger.info("Hybrid retrieval: %d results in %.1fms", len(result_passages), elapsed)
        return RetrievalResult(query=query, user_id=user_id, passages=result_passages, retrieval_time_ms=elapsed)

    def _rrf_fuse(self, dense_results: list[tuple[Chunk, float]], sparse_results: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
        scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}
        for rank, (chunk, _) in enumerate(dense_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_map[chunk.chunk_id] = chunk
        for rank, (chunk, _) in enumerate(sparse_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_map[chunk.chunk_id] = chunk
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_map[cid], score) for cid, score in ranked]
