"""Layer 3 — Local Embedding Generation. No external API calls (C3)."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cpu", batch_size: int = 64):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._dimension: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s on %s...", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Loaded — dimension=%d", self._dimension)

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._dimension  # type: ignore

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        return np.asarray(self._model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 100), dtype=np.float32)  # type: ignore

    def embed_query(self, query: str) -> np.ndarray:
        self._ensure_loaded()
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
        return np.asarray(self._model.encode(query, normalize_embeddings=True), dtype=np.float32)  # type: ignore

    def embed_chunks_to_list(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_texts(texts).tolist()
