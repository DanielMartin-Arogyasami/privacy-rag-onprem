"""
Integration tests for the pipeline.
FIX #1: Tests ingest_text() actually works.
"""

import pytest
from unittest.mock import MagicMock
from config.settings import Settings, LLMBackend
from src.pipeline import RAGPipeline
from src.models import QueryRequest


@pytest.fixture
def pipeline_no_llm():
    settings = Settings(embedding_device="cpu", llm_backend=LLMBackend.OLLAMA, enable_phi_detection=False, enable_audit_logging=False)
    pipe = RAGPipeline(settings)
    pipe.llm = MagicMock()
    pipe.llm.generate.return_value = "Metformin is a first-line treatment for type 2 diabetes."
    pipe.llm.is_available.return_value = True
    return pipe


@pytest.mark.slow
class TestPipelineIngestion:
    def test_ingest_bytes(self, pipeline_no_llm):
        result = pipeline_no_llm.ingest_bytes(content=b"# Test\n\nMetformin for diabetes.", filename="test.md", source="test", title="Test")
        assert result.chunks_created > 0

    def test_ingest_text(self, pipeline_no_llm):
        """FIX #1: ingest_text must actually ingest content."""
        result = pipeline_no_llm.ingest_text(text="Metformin is used for diabetes.", title="Inline Test")
        assert result.chunks_created > 0
        assert pipeline_no_llm.dense_store.count() > 0

    def test_ingest_and_query(self, pipeline_no_llm):
        pipeline_no_llm.ingest_bytes(content=b"# Metformin\n\nMetformin treats diabetes. Side effects: nausea.", filename="t.md", source="test", title="Met")
        pipeline_no_llm.flush_bm25()
        resp = pipeline_no_llm.query(QueryRequest(query="metformin side effects?", user_id="test"))
        assert resp.answer
        assert resp.latency_ms > 0
