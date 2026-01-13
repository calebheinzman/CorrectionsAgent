"""Client for Agent service."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from ..schemas import AgentTrace, Citation
from ..settings import get_settings


class AgentResult:
    """Result from the agent service."""

    def __init__(
        self,
        answer: str,
        citations: List[Citation],
        trace: AgentTrace,
    ):
        self.answer = answer
        self.citations = citations
        self.trace = trace


class AgentClient:
    """HTTP client for the Agent service."""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        settings = get_settings()
        self._base_url = base_url or settings.service_urls.agent
        self._timeout = timeout or settings.timeouts.agent_timeout

    def answer(
        self,
        request_id: str,
        question: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AgentResult:
        """
        Call the agent service to answer a question.

        Args:
            request_id: The request ID to propagate
            question: The question to answer
            user_id: Optional user ID
            session_id: Optional session ID

        Returns:
            AgentResult with answer, citations, and trace

        Raises:
            httpx.TimeoutException: If the request times out
            httpx.HTTPStatusError: If the service returns an error
        """
        payload: Dict[str, Any] = {
            "request_id": request_id,
            "question": question,
        }
        if user_id:
            payload["user_id"] = user_id
        if session_id:
            payload["session_id"] = session_id

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/v1/answer",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            citations = [
                Citation(
                    source_type=c.get("source_type", ""),
                    source_id=c.get("source_id", ""),
                    excerpt=c.get("excerpt"),
                )
                for c in data.get("citations", [])
            ]

            model_info = data.get("model_info")
            trace = AgentTrace(
                tool_calls=data.get("tool_calls", []),
                model_info=model_info,
            )

            return AgentResult(
                answer=data.get("answer", ""),
                citations=citations,
                trace=trace,
            )


_client_instance: Optional[AgentClient] = None


def get_client() -> AgentClient:
    """Get or create the singleton client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = AgentClient()
    return _client_instance


def get_answer(
    request_id: str,
    question: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> AgentResult:
    """Convenience function to get an answer using the singleton client."""
    return get_client().answer(request_id, question, user_id, session_id)
