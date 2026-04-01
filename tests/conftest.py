"""Shared test fixtures."""

import pytest
from src.models import Chunk, DocumentMetadata, DocumentType


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    return DocumentMetadata(document_id="test-doc-001", title="Test Document", source="Test", document_type=DocumentType.LITERATURE, acl_tags=["public"])


@pytest.fixture
def sample_chunks(sample_metadata) -> list[Chunk]:
    texts = [
        "Metformin is a biguanide used as first-line treatment for type 2 diabetes.",
        "Common adverse effects include nausea, diarrhea, and abdominal discomfort.",
        "Lactic acidosis is a rare but serious complication with an incidence of 0.03 per 1000 patient-years.",
        "SGLT2 inhibitors are recommended as add-on therapy for patients with cardiovascular risk.",
        "Warfarin has numerous drug interactions including with NSAIDs and certain antibiotics.",
    ]
    return [Chunk(chunk_id=f"chunk-{i:03d}", document_id=sample_metadata.document_id, text=t, chunk_index=i, metadata=sample_metadata) for i, t in enumerate(texts)]
