"""
Layer 1 — PHI De-identification (Defense in Depth, Layer 1)
Uses Microsoft Presidio for local PHI detection and redaction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DeidentificationResult:
    original_text: str
    cleaned_text: str
    detections: list[dict] = field(default_factory=list)
    entities_found: int = 0


class PHIDeidentifier:
    """Presidio-based local PHI detection and redaction."""

    def __init__(self, threshold: float = 0.7, language: str = "en"):
        self.threshold = threshold
        self.language = language
        self.entities = [
            "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN",
            "MEDICAL_LICENSE", "IP_ADDRESS", "DATE_TIME", "LOCATION",
        ]
        self._analyzer = None
        self._anonymizer = None

    def _ensure_loaded(self) -> None:
        if self._analyzer is not None:
            return
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        logger.info("Presidio PHI engine loaded")

    def detect(self, text: str) -> list[dict]:
        self._ensure_loaded()
        results = self._analyzer.analyze(  # type: ignore
            text=text, entities=self.entities,
            language=self.language, score_threshold=self.threshold,
        )
        return [
            {"entity_type": r.entity_type, "start": r.start, "end": r.end, "score": r.score}
            for r in results
        ]

    def deidentify(self, text: str) -> DeidentificationResult:
        self._ensure_loaded()
        results = self._analyzer.analyze(  # type: ignore
            text=text, entities=self.entities,
            language=self.language, score_threshold=self.threshold,
        )
        if not results:
            return DeidentificationResult(text, text, [], 0)
        anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)  # type: ignore
        detections = [{"entity_type": r.entity_type, "start": r.start, "end": r.end, "score": r.score} for r in results]
        return DeidentificationResult(text, anonymized.text, detections, len(results))

    def is_clean(self, text: str) -> bool:
        return len(self.detect(text)) == 0
