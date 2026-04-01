"""FastAPI dependency injection — singleton pipeline."""

from __future__ import annotations
from functools import lru_cache
from config.settings import get_settings
from src.pipeline import RAGPipeline


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(settings=get_settings())
