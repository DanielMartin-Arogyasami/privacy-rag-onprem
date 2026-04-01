"""Centralized configuration — all values from environment or .env file."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMBackend(str, Enum):
    VLLM = "vllm"
    OLLAMA = "ollama"


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    STRUCTURE_AWARE = "structure_aware"
    SEMANTIC = "semantic"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # LLM
    llm_backend: LLMBackend = LLMBackend.OLLAMA
    llm_model: str = "TheBloke/Llama-3.3-70B-Instruct-AWQ"
    vllm_host: str = "http://localhost:8001"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Embedding — FIX #9: removed embedding_dimension; derived from model at runtime
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 64

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "clinical_rag"

    # Retrieval
    retrieval_top_k: int = 20
    retrieval_rrf_k: int = 60
    retrieval_rerank_top_n: int = 50

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_strategy: ChunkStrategy = ChunkStrategy.STRUCTURE_AWARE

    # PHI
    enable_phi_detection: bool = True
    phi_detection_threshold: float = 0.7

    # Audit
    audit_log_dir: Path = Path("./logs/audit")
    enable_audit_logging: bool = True

    # API — FIX #3: auth settings are now enforced
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "change-me-in-production"
    enable_auth: bool = True
    rate_limit: str = "30/minute"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


@lru_cache
def get_settings() -> Settings:
    return Settings()
