"""Abstract vector store interface — enables swappable backends."""

from __future__ import annotations
from abc import ABC, abstractmethod
from src.models import Chunk


class VectorStoreBase(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int: ...
    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 20, acl_tags: list[str] | None = None) -> list[tuple[Chunk, float]]: ...
    @abstractmethod
    def delete_by_document(self, document_id: str) -> int: ...
    @abstractmethod
    def count(self) -> int: ...
