"""Tests for Layer 2 — Chunking. FIX #16: Tests ValueError not KeyError."""

import pytest
from src.chunking.chunker import FixedSizeChunker, StructureAwareChunker, SemanticChunker, ChunkerConfig, get_chunker
from src.models import DocumentMetadata


@pytest.fixture
def metadata():
    return DocumentMetadata(document_id="test-001", title="Test")


class TestFixedSizeChunker:
    def test_basic(self, metadata):
        chunks = FixedSizeChunker(ChunkerConfig(chunk_size=100, chunk_overlap=20)).chunk("A" * 250, metadata)
        assert len(chunks) >= 2

    def test_short_text(self, metadata):
        chunks = FixedSizeChunker(ChunkerConfig(chunk_size=1000)).chunk("Short.", metadata)
        assert len(chunks) == 1


class TestStructureAwareChunker:
    def test_sections(self, metadata):
        sections = [{"title": "Intro", "text": "Intro text.", "level": 1}, {"title": "Methods", "text": "Methods text.", "level": 1}]
        chunks = StructureAwareChunker(ChunkerConfig(chunk_size=500)).chunk("text", metadata, sections)
        assert len(chunks) == 2
        assert "Intro" in chunks[0].section_path

    def test_fallback(self, metadata):
        # chunk_overlap must be < chunk_size (ChunkerConfig validation)
        chunks = StructureAwareChunker(ChunkerConfig(chunk_size=100, chunk_overlap=20)).chunk("A" * 250, metadata)
        assert len(chunks) >= 2


class TestGetChunker:
    def test_valid(self):
        assert isinstance(get_chunker("fixed"), FixedSizeChunker)
        assert isinstance(get_chunker("structure_aware"), StructureAwareChunker)
        assert isinstance(get_chunker("semantic"), SemanticChunker)

    def test_invalid_raises_valueerror(self):
        # FIX #16: ValueError not KeyError
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("nonexistent")

    def test_overlap_gte_chunk_size_raises(self):
        """FIX Bug 5: Prevent infinite loop when overlap >= chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkerConfig(chunk_size=500, chunk_overlap=500)
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkerConfig(chunk_size=500, chunk_overlap=600)
