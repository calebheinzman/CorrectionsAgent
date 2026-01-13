"""Client for model registry (mock)."""
from __future__ import annotations

from typing import Optional

from mock_apis.cloud_apis import mock_model_registry


class RegistryClient:
    """Client for the model registry."""

    def get_current_model(self, task: Optional[str] = None) -> Optional[dict]:
        """
        Get the current model configuration.

        Args:
            task: Optional task name to get model for

        Returns:
            Model info dict if configured, None otherwise
        """
        return mock_model_registry.get_current_model(task=task)

    def get_policy_bundle(self) -> Optional[dict]:
        """
        Get the current policy bundle.

        Returns:
            Policy bundle dict if configured, None otherwise
        """
        return mock_model_registry.get_policy_bundle()


_client_instance: Optional[RegistryClient] = None


def get_client() -> RegistryClient:
    """Get or create the singleton client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = RegistryClient()
    return _client_instance
