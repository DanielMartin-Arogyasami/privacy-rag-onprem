"""
Layer 3 — BM25 Sparse Index
FIX #2: Deferred rebuild — add() marks dirty, rebuild() does the actual work.
FIX #11: ACL filtering is post-scoring (documented as known limitation).
"""

from __future__ import annotations

import logging
import re

from rank_bm25 import BM25Okapi

from src.models import Chunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Store:
    """In-memory BM25 index with deferred rebuild for batch efficiency."""

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._acl_map: dict[str, list[str]] = {}
        self._dirty: bool = False  # FIX #2: Deferred rebuild flag

    def index(self, chunks: list[Chunk]) -> int:
        """Build BM25 index from scratch (replaces all data)."""
        self._chunks = list(chunks)
        self._corpus = [_tokenize(c.text) for c in chunks]
        self._acl_map = {c.chunk_id: c.metadata.acl_tags for c in chunks}
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None
        self._dirty = False
        logger.info("BM25 index built: %d documents", len(chunks))
        return len(chunks)

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks — marks index as dirty. Call rebuild() when ready."""
        self._chunks.extend(chunks)
        self._corpus.extend([_tokenize(c.text) for c in chunks])
        for c in chunks:
            self._acl_map[c.chunk_id] = c.metadata.acl_tags
        self._dirty = True

    def rebuild(self) -> None:
        """Rebuild BM25 index if dirty. Call after batch ingestion."""
        if not self._dirty:
            return
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
        self._dirty = False
        logger.info("BM25 index rebuilt: %d documents", len(self._chunks))

    def search(self, query: str, top_k: int = 20, acl_tags: list[str] | None = None) -> list[tuple[Chunk, float]]:
        """
        Search with BM25 scores.
        NOTE (FIX #11): ACL filtering is post-scoring. For corpora where most chunks
        are restricted, consider separate BM25 indices per ACL scope.
        """
        # Auto-rebuild if dirty
        if self._dirty:
            self.rebuild()
        if not self._bm25 or not self._chunks:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        results: list[tuple[int, float]] = []
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            if acl_tags:
                chunk_acl = self._acl_map.get(self._chunks[i].chunk_id, [])
                if not any(tag in chunk_acl for tag in acl_tags):
                    continue
            results.append((i, float(score)))
        results.sort(key=lambda x: x[1], reverse=True)
        return [(self._chunks[i], s) for i, s in results[:top_k]]

    @property
    def size(self) -> int:
        return len(self._chunks)
