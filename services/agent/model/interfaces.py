"""LLM client interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Invoke the LLM with messages and return the response."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def get_provider(self) -> str:
        """Return the provider name."""
        pass
