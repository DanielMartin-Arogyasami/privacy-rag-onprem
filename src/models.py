"""
Shared Pydantic models enforcing formal constraints C1–C5 at the type level.
FIX #6: Added Field validation constraints (max_length, ge/le ranges).
FIX #13: Removed unused FilteredResponse and GeneratedResponse models.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# === Document & Chunk Models (Layers 1-3) ===

class DocumentType(str, Enum):
    CLINICAL_PROTOCOL = "clinical_protocol"
    DRUG_LABEL = "drug_label"
    ADVERSE_EVENT = "adverse_event"
    LITERATURE = "literature"
    REGULATORY = "regulatory"
    CLINICAL_TRIAL = "clinical_trial"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    OTHER = "other"


class DocumentMetadata(BaseModel):
    """Metadata for every ingested document — supports C4 auditability."""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(default="", max_length=2000)
    source: str = Field(default="", max_length=500)
    document_type: DocumentType = DocumentType.OTHER
    version: str = "1.0"
    ingestion_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    acl_tags: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    ontology_concepts: list[str] = Field(default_factory=list)
    language: str = "en"
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A single retrieval unit with full provenance chain."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    text: str
    section_path: str = ""
    parent_chunk_id: str | None = None
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    text_hash: str = ""


# === Retrieval Models (Layer 4) ===

class RetrievedPassage(BaseModel):
    """A passage returned by hybrid retrieval with all scores."""
    chunk: Chunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    reranker_score: float = 0.0


class RetrievalResult(BaseModel):
    """Full retrieval output for a query."""
    query: str
    user_id: str
    passages: list[RetrievedPassage] = Field(default_factory=list)
    retrieval_time_ms: float = 0.0


# === Generation Models (Layer 5) ===

class Citation(BaseModel):
    """Links a claim to a retrieved passage — enforces C1."""
    chunk_id: str
    document_title: str = ""
    section: str = ""
    relevance_score: float = 0.0


# === Filtering Models (Layer 6) ===

class PHIDetection(BaseModel):
    """A detected PHI entity in output."""
    entity_type: str
    start: int
    end: int
    score: float


# === Audit Models (C4) ===

class AuditRecord(BaseModel):
    """Complete audit trail for a query-response cycle."""
    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str
    query: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    response_id: str = ""
    model_id: str = ""
    confidence: float = 0.0
    is_abstention: bool = False
    phi_detected: bool = False
    phantom_citations_found: bool = False  # FIX Bug 4: Track phantom citation results
    latency_ms: float = 0.0


# === API Models ===

class QueryRequest(BaseModel):
    # FIX #6: Input validation with length limits and range constraints
    query: str = Field(min_length=1, max_length=10000)
    user_id: str = Field(default="anonymous", max_length=200)
    max_results: int = Field(default=20, ge=1, le=100)
    include_sources: bool = True
    filters: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    is_abstention: bool = False
    sources: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0


class IngestRequest(BaseModel):
    title: str = Field(default="", max_length=2000)
    source: str = Field(default="", max_length=500)
    document_type: DocumentType = DocumentType.OTHER
    acl_tags: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int
    status: str = "success"
