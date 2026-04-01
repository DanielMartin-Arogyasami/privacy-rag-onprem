from __future__ import annotations
from abc import ABC, abstractmethod

class LLMClientBase(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.1) -> str: ...
    @abstractmethod
    def is_available(self) -> bool: ...
