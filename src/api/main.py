"""
FastAPI application — Privacy-Preserving Clinical RAG.
FIX #17: Added SlowAPI rate limiting.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from config.settings import get_settings
from src.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

settings = get_settings()

# FIX #17: Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])

app = FastAPI(
    title="Privacy-Preserving Clinical RAG",
    description="On-premises RAG for clinical data systems. All processing stays within the security perimeter.",
    version="0.2.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])

app.include_router(router)


@app.get("/")
def root():
    return {"service": "Privacy-Preserving Clinical RAG", "version": "0.2.0", "docs": "/docs"}
