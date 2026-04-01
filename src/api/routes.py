"""
API routes.
FIX #3: Auth dependency on all routes.
FIX #7: Exception handler returns generic message, logs full error.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException

from src.models import DocumentType, IngestResponse, QueryRequest, QueryResponse
from src.pipeline import RAGPipeline
from src.api.deps import get_pipeline
from src.api.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["rag"], dependencies=[Depends(require_auth)])


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    """Submit a clinical query to the RAG system."""
    try:
        return pipeline.query(request)
    except Exception:
        # FIX #7: Log full error internally, return generic message to client
        logger.exception("Query failed for user=%s", request.user_id)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    title: str = Form(""),
    source: str = Form(""),
    document_type: str = Form("other"),
    acl_tags: str = Form(""),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    content = await file.read()
    filename = file.filename or "upload.txt"
    tags = [t.strip() for t in acl_tags.split(",") if t.strip()]
    try:
        doc_type = DocumentType(document_type)
    except ValueError:
        doc_type = DocumentType.OTHER
    try:
        result = pipeline.ingest_bytes(content=content, filename=filename, document_type=doc_type, acl_tags=tags, source=source, title=title)
        pipeline.flush_bm25()  # Rebuild BM25 after ingest
        return result
    except Exception:
        logger.exception("Ingest failed for file=%s", filename)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
def health(pipeline: RAGPipeline = Depends(get_pipeline)):
    return {"status": "ok", "vector_store_count": pipeline.dense_store.count(), "bm25_count": pipeline.sparse_store.size, "llm_available": pipeline.llm.is_available()}


@router.get("/stats")
def stats(pipeline: RAGPipeline = Depends(get_pipeline)):
    return {"total_chunks": pipeline.dense_store.count(), "bm25_indexed": pipeline.sparse_store.size, "embedding_model": pipeline.settings.embedding_model, "llm_backend": pipeline.settings.llm_backend.value}
