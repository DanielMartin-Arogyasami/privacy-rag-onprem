"""
Layer 2 — Structure-Aware Chunking
FIX #16: get_chunker raises ValueError (not KeyError) for unknown strategies.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import xxhash

from src.models import Chunk, DocumentMetadata

logger = logging.getLogger(__name__)


@dataclass
class ChunkerConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100

    def __post_init__(self) -> None:
        # FIX Bug 5: Prevent infinite loop in FixedSizeChunker when step=0
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        if self.min_chunk_size < 0:
            raise ValueError(f"min_chunk_size must be >= 0, got {self.min_chunk_size}")


def _make_chunk(
    text: str, document_id: str, metadata: DocumentMetadata, index: int,
    section_path: str = "", char_start: int = 0, char_end: int = 0,
    parent_chunk_id: str | None = None,
) -> Chunk:
    return Chunk(
        document_id=document_id, text=text.strip(), section_path=section_path,
        parent_chunk_id=parent_chunk_id, chunk_index=index,
        char_start=char_start, char_end=char_end, metadata=metadata,
        text_hash=xxhash.xxh64(text.encode()).hexdigest(),
    )


class FixedSizeChunker:
    def __init__(self, config: ChunkerConfig | None = None):
        self.config = config or ChunkerConfig()

    def chunk(self, text: str, metadata: DocumentMetadata, sections: list[dict] | None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.config.chunk_size - self.config.chunk_overlap
        pos, idx = 0, 0
        while pos < len(text):
            end = min(pos + self.config.chunk_size, len(text))
            chunk_text = text[pos:end]
            if len(chunk_text.strip()) < self.config.min_chunk_size and chunks:
                chunks[-1].text += " " + chunk_text.strip()
                break
            chunks.append(_make_chunk(chunk_text, metadata.document_id, metadata, idx, char_start=pos, char_end=end))
            pos += step
            idx += 1
        logger.info("FixedSizeChunker: %d chunks from %d chars", len(chunks), len(text))
        return chunks


class StructureAwareChunker:
    def __init__(self, config: ChunkerConfig | None = None):
        self.config = config or ChunkerConfig()

    def chunk(self, text: str, metadata: DocumentMetadata, sections: list[dict] | None = None) -> list[Chunk]:
        if not sections:
            return FixedSizeChunker(self.config).chunk(text, metadata)
        chunks: list[Chunk] = []
        doc_id = metadata.document_id
        idx = 0
        for section in sections:
            title = section.get("title", "")
            sec_text = section.get("text", "").strip()
            if not sec_text:
                continue
            if len(sec_text) <= self.config.chunk_size:
                full_text = f"{title}\n\n{sec_text}" if title else sec_text
                chunks.append(_make_chunk(full_text, doc_id, metadata, idx, section_path=title))
                idx += 1
            else:
                parent = _make_chunk(f"{title}\n\n{sec_text[:200]}...", doc_id, metadata, idx, section_path=title)
                chunks.append(parent)
                idx += 1
                paragraphs = re.split(r"\n\s*\n", sec_text)
                buffer = ""
                for para in paragraphs:
                    if len(buffer) + len(para) > self.config.chunk_size:
                        if buffer.strip():
                            chunks.append(_make_chunk(buffer.strip(), doc_id, metadata, idx, section_path=title, parent_chunk_id=parent.chunk_id))
                            idx += 1
                        buffer = para
                    else:
                        buffer += "\n\n" + para if buffer else para
                if buffer.strip() and len(buffer.strip()) >= self.config.min_chunk_size:
                    chunks.append(_make_chunk(buffer.strip(), doc_id, metadata, idx, section_path=title, parent_chunk_id=parent.chunk_id))
                    idx += 1
        logger.info("StructureAwareChunker: %d chunks from %d sections", len(chunks), len(sections))
        return chunks


class SemanticChunker:
    STAT_PATTERNS = [r"p\s*[<>=]\s*\d", r"CI\s*[\[\(]", r"HR\s*[=:]", r"\d+\.?\d*\s*%", r"n\s*=\s*\d+"]

    def __init__(self, config: ChunkerConfig | None = None):
        self.config = config or ChunkerConfig()

    def chunk(self, text: str, metadata: DocumentMetadata, sections: list[dict] | None = None) -> list[Chunk]:
        base = StructureAwareChunker(self.config).chunk(text, metadata, sections)
        merged: list[Chunk] = []
        skip = False
        for i, c in enumerate(base):
            if skip:
                skip = False
                continue
            if i + 1 < len(base):
                nxt = base[i + 1]
                has_stats = any(re.search(p, nxt.text[:200], re.IGNORECASE) for p in self.STAT_PATTERNS)
                if has_stats and len(c.text) + len(nxt.text) <= self.config.chunk_size * 1.5:
                    merged.append(_make_chunk(c.text + "\n\n" + nxt.text, c.document_id, metadata, len(merged), section_path=c.section_path, parent_chunk_id=c.parent_chunk_id))
                    skip = True
                    continue
            merged.append(c)
        return merged


# FIX #16: Raises ValueError with helpful message instead of KeyError
_CHUNKERS = {"fixed": FixedSizeChunker, "structure_aware": StructureAwareChunker, "semantic": SemanticChunker}


def get_chunker(strategy: str, config: ChunkerConfig | None = None):
    """Factory: returns the appropriate chunker."""
    cls = _CHUNKERS.get(strategy)
    if cls is None:
        raise ValueError(f"Unknown chunking strategy: '{strategy}'. Choose from: {list(_CHUNKERS.keys())}")
    return cls(config)
