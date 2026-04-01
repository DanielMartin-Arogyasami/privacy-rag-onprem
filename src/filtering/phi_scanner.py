"""Layer 6 — PHI Output Scanner (Defense in Depth, final layer)."""

from __future__ import annotations
import logging
from src.models import PHIDetection
from src.ingestion.deidentifier import PHIDeidentifier

logger = logging.getLogger(__name__)


class PHIOutputScanner:
    def __init__(self, threshold: float = 0.7):
        self._phi = PHIDeidentifier(threshold=threshold)

    def scan(self, text: str) -> list[PHIDetection]:
        return [PHIDetection(entity_type=d["entity_type"], start=d["start"], end=d["end"], score=d["score"]) for d in self._phi.detect(text)]

    def is_clean(self, text: str) -> bool:
        return len(self.scan(text)) == 0

    def redact(self, text: str) -> str:
        return self._phi.deidentify(text).cleaned_text
