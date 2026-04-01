"""Layer 6 — Hallucination & Phantom Citation Checker. Enforces C1."""

from __future__ import annotations
import logging
import re
from src.models import RetrievedPassage

logger = logging.getLogger(__name__)


class HallucinationChecker:
    CITATION_PATTERN = re.compile(r"\[Source:\s*([^\],]+)(?:,\s*([^\]]*))?\]", re.IGNORECASE)

    def check_citations(self, response_text: str, retrieved_passages: list[RetrievedPassage]) -> list[str]:
        cited_ids = {m.group(1).strip() for m in self.CITATION_PATTERN.finditer(response_text)}
        valid_refs = {p.chunk.chunk_id for p in retrieved_passages} | {p.chunk.document_id for p in retrieved_passages} | {p.chunk.metadata.title for p in retrieved_passages}
        phantom = [cid for cid in cited_ids if cid not in valid_refs]
        if phantom:
            logger.warning("Phantom citations detected: %s", phantom)
        return phantom

    def has_abstention_markers(self, response_text: str) -> bool:
        phrases = ["i do not have sufficient evidence", "the retrieved documents do not contain", "i cannot find information", "insufficient evidence", "no relevant passages"]
        lower = response_text.lower()
        return any(p in lower for p in phrases)

    def estimate_confidence(self, retrieved_passages: list[RetrievedPassage], response_text: str) -> float:
        if not retrieved_passages:
            return 0.0
        top_score = max(p.reranker_score for p in retrieved_passages)
        num_citations = len(self.CITATION_PATTERN.findall(response_text))
        sentences = len(re.split(r"[.!?]+", response_text))
        citation_density = min(num_citations / max(sentences, 1), 1.0)
        passage_factor = min(len(retrieved_passages) / 5.0, 1.0)
        confidence = 0.4 * min(top_score, 1.0) + 0.3 * citation_density + 0.3 * passage_factor
        return round(min(max(confidence, 0.0), 1.0), 3)
