"""
API tests.
FIX #3: Tests auth enforcement.
FIX #7: Tests that errors don't leak internals.
FIX Bug 8: Uses FastAPI dependency_overrides instead of import-time mock patching.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.models import QueryResponse, Citation
from src.api.main import app
from src.api.deps import get_pipeline
from src.api.auth import require_auth


def _make_mock_pipeline() -> MagicMock:
    mock = MagicMock()
    mock.dense_store.count.return_value = 100
    mock.sparse_store.size = 100
    mock.llm.is_available.return_value = True
    mock.query.return_value = QueryResponse(
        answer="Metformin for T2D.",
        citations=[Citation(chunk_id="c1", document_title="Test", relevance_score=0.9)],
        confidence=0.85,
        latency_ms=150.0,
    )
    return mock


@pytest.fixture
def client():
    """FIX Bug 8: dependency_overrides are import-order-safe."""
    mock_pipeline = _make_mock_pipeline()
    # Override both the pipeline dependency and auth (disable for most tests)
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[require_auth] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def failing_client():
    """Client whose pipeline.query always raises."""
    mock_pipeline = _make_mock_pipeline()
    mock_pipeline.query.side_effect = RuntimeError("secret internal error detail")
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[require_auth] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestHealth:
    def test_ok(self, client):
        assert client.get("/api/health").status_code == 200


class TestQuery:
    def test_query(self, client):
        resp = client.post("/api/query", json={"query": "metformin?", "user_id": "test"})
        assert resp.status_code == 200
        assert "answer" in resp.json()

    def test_missing_field(self, client):
        assert client.post("/api/query", json={}).status_code == 422

    def test_query_too_long(self, client):
        """FIX #6: Input validation rejects oversized queries."""
        resp = client.post("/api/query", json={"query": "x" * 10001, "user_id": "test"})
        assert resp.status_code == 422

    def test_error_no_leak(self, failing_client):
        """FIX #7: Errors return generic message, not stack trace."""
        resp = failing_client.post("/api/query", json={"query": "test", "user_id": "t"})
        assert resp.status_code == 500
        assert "secret" not in resp.text
        assert "Internal server error" in resp.json()["detail"]


class TestRoot:
    def test_root(self, client):
        resp = client.get("/")
        assert "Privacy-Preserving" in resp.json()["service"]
