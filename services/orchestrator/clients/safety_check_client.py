"""Client for Safety Check service."""
from __future__ import annotations

import time
from typing import Optional

import httpx

from ..schemas import GuardrailDecision
from ..settings import get_settings


class SafetyCheckClient:
    """HTTP client for the Safety Check service."""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        settings = get_settings()
        self._base_url = base_url or settings.service_urls.safety_check
        self._timeout = timeout or settings.timeouts.safety_check_timeout

    def check(self, request_id: str, question: str, user_id: Optional[str] = None) -> GuardrailDecision:
        """
        Call the safety check service.

        Args:
            request_id: The request ID to propagate
            question: The question to check
            user_id: Optional user ID

        Returns:
            GuardrailDecision with the safety check result

        Raises:
            httpx.TimeoutException: If the request times out
            httpx.HTTPStatusError: If the service returns an error
        """
        payload = {
            "request_id": request_id,
            "question": question,
        }
        if user_id:
            payload["user_id"] = user_id

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/v1/safety/check",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            return GuardrailDecision(
                allowed=data.get("allowed"),
                reason=data.get("reason", ""),
                policy=data.get("policy"),
                categories=data.get("categories", []),
            )


_client_instance: Optional[SafetyCheckClient] = None


def get_client() -> SafetyCheckClient:
    """Get or create the singleton client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = SafetyCheckClient()
    return _client_instance


def check_safety(request_id: str, question: str, user_id: Optional[str] = None) -> GuardrailDecision:
    """Convenience function to check safety using the singleton client."""
    return get_client().check(request_id, question, user_id)
