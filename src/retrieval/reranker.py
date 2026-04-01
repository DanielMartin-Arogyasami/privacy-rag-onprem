"""Layer 4 — Cross-Encoder Reranker. Runs locally (C3)."""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker %s on %s...", self.model_name, self.device)
        self._model = CrossEncoder(self.model_name, device=self.device)
        logger.info("Reranker loaded")

    def rerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[tuple[int, float]]:
        self._ensure_loaded()
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        scores = self._model.predict(pairs)  # type: ignore
        indexed = sorted(enumerate(float(s) for s in scores), key=lambda x: x[1], reverse=True)
        return indexed[:top_k] if top_k else indexed
