"""Layer 1 — Document Loader: parse → de-identify → attach metadata."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from src.models import DocumentMetadata, DocumentType
from src.ingestion.parser import DocumentParser, ParsedDocument
from src.ingestion.deidentifier import PHIDeidentifier

logger = logging.getLogger(__name__)


@dataclass
class LoadedDocument:
    parsed: ParsedDocument
    metadata: DocumentMetadata
    content_hash: str

    @property
    def text(self) -> str:
        return self.parsed.text

    @property
    def sections(self) -> list[dict]:
        return self.parsed.sections


class DocumentLoader:
    def __init__(self, deidentify: bool = True, phi_threshold: float = 0.7):
        self.parser = DocumentParser()
        self.deidentify = deidentify
        self._phi = PHIDeidentifier(threshold=phi_threshold) if deidentify else None

    def load_bytes(
        self, content: bytes, filename: str,
        document_type: DocumentType = DocumentType.OTHER,
        acl_tags: list[str] | None = None, source: str = "", title: str = "",
    ) -> LoadedDocument:
        parsed = self.parser.parse(content, filename)
        logger.info("Parsed %s: %d chars, %d sections", filename, len(parsed.text), len(parsed.sections))

        if self.deidentify and self._phi:
            result = self._phi.deidentify(parsed.text)
            if result.entities_found > 0:
                logger.warning("Redacted %d PHI entities in %s", result.entities_found, filename)
                parsed.text = result.cleaned_text
                for sec in parsed.sections:
                    sec["text"] = self._phi.deidentify(sec.get("text", "")).cleaned_text

        metadata = DocumentMetadata(
            title=title or parsed.metadata.get("title", Path(filename).stem),
            source=source, document_type=document_type, acl_tags=acl_tags or [],
        )
        return LoadedDocument(parsed, metadata, hashlib.sha256(content).hexdigest())

    def load_file(self, path: str | Path, **kwargs) -> LoadedDocument:
        path = Path(path)
        return self.load_bytes(path.read_bytes(), path.name, **kwargs)
