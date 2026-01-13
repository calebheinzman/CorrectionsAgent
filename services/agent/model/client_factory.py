"""LLM client factory using model registry."""
from __future__ import annotations

from typing import Optional

from mock_apis.cloud_apis import mock_model_registry

from .gemini_client import GeminiClient
from .interfaces import LLMClient


def create_client_from_registry(task: str = "agent") -> tuple[Optional[LLMClient], Optional[str]]:
    """
    Create an LLM client based on model registry configuration.
    
    Args:
        task: The task name to look up in the registry
    
    Returns:
        Tuple of (client, error_message). If successful, error_message is None.
    """
    model_info = mock_model_registry.get_current_model(task=task)
    
    if not model_info:
        return None, f"No model configured in registry for task '{task}'"
    
    provider = model_info.get("provider", "").lower()
    model_id = model_info.get("model_id", "")
    
    if not provider or not model_id:
        return None, f"Invalid model info in registry: {model_info}"
    
    if provider == "google" or provider == "gemini":
        client = GeminiClient(model_name=model_id)
        if not client.is_available():
            return client, client.get_init_error()
        return client, None
    
    return None, f"Unsupported provider '{provider}' in registry"
