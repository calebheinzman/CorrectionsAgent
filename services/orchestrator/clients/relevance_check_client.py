"""Client for Relevance Check service."""
from __future__ import annotations

from typing import Optional

import httpx

from ..schemas import GuardrailDecision
from ..settings import get_settings


class RelevanceCheckClient:
    """HTTP client for the Relevance Check service."""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        settings = get_settings()
        self._base_url = base_url or settings.service_urls.relevance_check
        self._timeout = timeout or settings.timeouts.relevance_check_timeout

    def check(self, request_id: str, question: str, user_id: Optional[str] = None) -> GuardrailDecision:
        """
        Call the relevance check service.

        Args:
            request_id: The request ID to propagate
            question: The question to check
            user_id: Optional user ID

        Returns:
            GuardrailDecision with the relevance check result

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
                f"{self._base_url}/v1/relevance/check",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            return GuardrailDecision(
                relevant=data.get("relevant"),
                reason=data.get("reason", ""),
                matched_domains=data.get("matched_domains", []),
                model_id=data.get("model_id"),
            )


_client_instance: Optional[RelevanceCheckClient] = None


def get_client() -> RelevanceCheckClient:
    """Get or create the singleton client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = RelevanceCheckClient()
    return _client_instance


def check_relevance(request_id: str, question: str, user_id: Optional[str] = None) -> GuardrailDecision:
    """Convenience function to check relevance using the singleton client."""
    return get_client().check(request_id, question, user_id)
