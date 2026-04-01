"""
Layer 3 — Qdrant Vector Store (self-hosted, Apache 2.0)
FIX #4: delete_by_document now uses FilterSelector (correct Qdrant API).
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.models import Chunk, DocumentMetadata, DocumentType
from src.vectorstore.base import VectorStoreBase

logger = logging.getLogger(__name__)


class QdrantStore(VectorStoreBase):
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "clinical_rag", dimension: int = 1024):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.dimension = dimension
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s' (dim=%d)", self.collection_name, self.dimension)

    def _chunk_to_payload(self, chunk: Chunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id, "document_id": chunk.document_id,
            "text": chunk.text, "section_path": chunk.section_path,
            "parent_chunk_id": chunk.parent_chunk_id or "",
            "chunk_index": chunk.chunk_index, "text_hash": chunk.text_hash,
            "title": chunk.metadata.title, "source": chunk.metadata.source,
            "document_type": chunk.metadata.document_type.value,
            "acl_tags": chunk.metadata.acl_tags, "mesh_terms": chunk.metadata.mesh_terms,
            "ingestion_timestamp": chunk.metadata.ingestion_timestamp.isoformat(),
        }

    def _payload_to_chunk(self, payload: dict[str, Any]) -> Chunk:
        return Chunk(
            chunk_id=payload["chunk_id"], document_id=payload["document_id"],
            text=payload["text"], section_path=payload.get("section_path", ""),
            parent_chunk_id=payload.get("parent_chunk_id") or None,
            chunk_index=payload.get("chunk_index", 0), text_hash=payload.get("text_hash", ""),
            metadata=DocumentMetadata(
                document_id=payload["document_id"], title=payload.get("title", ""),
                source=payload.get("source", ""),
                document_type=DocumentType(payload.get("document_type", "other")),
                acl_tags=payload.get("acl_tags", []), mesh_terms=payload.get("mesh_terms", []),
            ),
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        points = [PointStruct(id=c.chunk_id, vector=emb, payload=self._chunk_to_payload(c)) for c, emb in zip(chunks, embeddings)]
        for i in range(0, len(points), 100):
            self.client.upsert(collection_name=self.collection_name, points=points[i:i+100])
        logger.info("Upserted %d chunks into Qdrant", len(chunks))
        return len(chunks)

    def search(self, query_embedding: list[float], top_k: int = 20, acl_tags: list[str] | None = None) -> list[tuple[Chunk, float]]:
        query_filter = None
        if acl_tags:
            query_filter = Filter(must=[FieldCondition(key="acl_tags", match=MatchAny(any=acl_tags))])
        results = self.client.search(collection_name=self.collection_name, query_vector=query_embedding, query_filter=query_filter, limit=top_k, with_payload=True)
        return [(self._payload_to_chunk(r.payload), r.score) for r in results if r.payload]  # type: ignore

    def delete_by_document(self, document_id: str) -> int:
        """FIX #4: Use FilterSelector wrapping the Filter (correct Qdrant API)."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
            ),
        )
        logger.info("Deleted chunks for document %s", document_id)
        return 0

    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0
