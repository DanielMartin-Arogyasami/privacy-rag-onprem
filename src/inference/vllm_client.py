"""Layer 5 — vLLM Client. Connects to local vLLM server (C3)."""

from __future__ import annotations
import logging
import httpx
from src.inference.base import LLMClientBase

logger = logging.getLogger(__name__)


class VLLMClient(LLMClientBase):
    def __init__(self, host: str = "http://localhost:8001", model: str = ""):
        self.host = host.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=120.0)

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.1) -> str:
        try:
            resp = self._client.post(f"{self.host}/v1/completions", json={"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "stop": ["</s>", "[/INST]"]})
            resp.raise_for_status()
            return resp.json()["choices"][0]["text"].strip()
        except Exception as e:
            logger.error("vLLM generation failed: %s", e)
            raise

    def is_available(self) -> bool:
        try:
            return self._client.get(f"{self.host}/v1/models").status_code == 200
        except Exception:
            return False
