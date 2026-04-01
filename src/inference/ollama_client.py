"""Layer 5 — Ollama Client. Local dev inference (C3)."""

from __future__ import annotations
import logging
import httpx
from src.inference.base import LLMClientBase

logger = logging.getLogger(__name__)


class OllamaClient(LLMClientBase):
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.1:8b"):
        self.host = host.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=180.0)

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.1) -> str:
        try:
            resp = self._client.post(f"{self.host}/api/generate", json={"model": self.model, "prompt": prompt, "options": {"temperature": temperature, "num_predict": max_tokens}, "stream": False})
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            raise

    def is_available(self) -> bool:
        try:
            return self._client.get(f"{self.host}/api/tags").status_code == 200
        except Exception:
            return False
