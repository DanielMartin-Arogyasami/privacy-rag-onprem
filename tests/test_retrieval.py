"""Tests for Layer 4 — Retrieval. FIX #2: Tests deferred rebuild."""

import pytest
from src.vectorstore.bm25_store import BM25Store
from src.filtering.hallucination_checker import HallucinationChecker
from src.models import RetrievedPassage


@pytest.fixture
def bm25_with_data(sample_chunks):
    store = BM25Store()
    store.index(sample_chunks)
    return store


class TestBM25Store:
    def test_search(self, bm25_with_data):
        results = bm25_with_data.search("metformin diabetes", top_k=3)
        assert len(results) > 0
        assert any("metformin" in c.text.lower() for c, _ in results)

    def test_empty_query(self, bm25_with_data):
        assert bm25_with_data.search("", top_k=3) == []

    def test_acl_filtering(self, sample_chunks):
        sample_chunks[0].metadata.acl_tags = ["team_a"]
        sample_chunks[1].metadata.acl_tags = ["team_b"]
        store = BM25Store()
        store.index(sample_chunks)
        for chunk, _ in store.search("metformin", top_k=5, acl_tags=["team_a"]):
            assert "team_a" in chunk.metadata.acl_tags

    def test_deferred_rebuild(self, sample_chunks):
        """FIX #2: add() defers rebuild; search auto-rebuilds."""
        store = BM25Store()
        store.add(sample_chunks)
        assert store._dirty is True
        results = store.search("metformin", top_k=3)
        assert store._dirty is False
        assert len(results) > 0


class TestHallucinationChecker:
    def test_no_phantoms(self, sample_chunks):
        checker = HallucinationChecker()
        passages = [RetrievedPassage(chunk=sample_chunks[0], reranker_score=0.8)]
        response = f"Metformin is used [Source: {sample_chunks[0].chunk_id}]"
        assert checker.check_citations(response, passages) == []

    def test_phantom(self, sample_chunks):
        checker = HallucinationChecker()
        passages = [RetrievedPassage(chunk=sample_chunks[0], reranker_score=0.8)]
        assert "fake-id" in checker.check_citations("Claim [Source: fake-id]", passages)

    def test_abstention(self):
        checker = HallucinationChecker()
        assert checker.has_abstention_markers("I do not have sufficient evidence to answer.")
        assert not checker.has_abstention_markers("Metformin treats diabetes.")

    def test_confidence(self, sample_chunks):
        checker = HallucinationChecker()
        passages = [RetrievedPassage(chunk=sample_chunks[i], reranker_score=0.8 - i * 0.1) for i in range(3)]
        c = checker.estimate_confidence(passages, "Claim [Source: x]. More [Source: y].")
        assert 0.0 <= c <= 1.0
