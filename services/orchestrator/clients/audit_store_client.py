"""Client for audit store (mock DynamoDB)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from mock_apis.cloud_apis import mock_dynamodb

from ..settings import get_settings


class AuditRecord:
    """An audit record for a request."""

    def __init__(
        self,
        request_id: str,
        question: str,
        safety_allowed: Optional[bool] = None,
        safety_reason: Optional[str] = None,
        safety_policy: Optional[str] = None,
        safety_model_id: Optional[str] = None,
        relevance_relevant: Optional[bool] = None,
        relevance_reason: Optional[str] = None,
        relevance_model_id: Optional[str] = None,
        agent_called: bool = False,
        agent_tool_calls: Optional[list] = None,
        agent_latency_ms: Optional[float] = None,
        agent_citations_count: int = 0,
        agent_model_info: Optional[Dict[str, str]] = None,
        final_status: str = "pending",
        error: Optional[str] = None,
    ):
        self.request_id = request_id
        self.question = question
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.safety_allowed = safety_allowed
        self.safety_reason = safety_reason
        self.safety_policy = safety_policy
        self.safety_model_id = safety_model_id
        self.relevance_relevant = relevance_relevant
        self.relevance_reason = relevance_reason
        self.relevance_model_id = relevance_model_id
        self.agent_called = agent_called
        self.agent_tool_calls = agent_tool_calls or []
        self.agent_latency_ms = agent_latency_ms
        self.agent_citations_count = agent_citations_count
        self.agent_model_info = agent_model_info
        self.final_status = final_status
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "pk": f"request#{self.request_id}",
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "question": self.question,
            "question_length": len(self.question),
            "safety": {
                "allowed": self.safety_allowed,
                "reason": self.safety_reason,
                "policy": self.safety_policy,
                "model_id": self.safety_model_id,
            },
            "relevance": {
                "relevant": self.relevance_relevant,
                "reason": self.relevance_reason,
                "model_id": self.relevance_model_id,
            },
            "agent": {
                "called": self.agent_called,
                "tool_calls_count": len(self.agent_tool_calls),
                "latency_ms": self.agent_latency_ms,
                "citations_count": self.agent_citations_count,
                "model_info": self.agent_model_info,
            },
            "final_status": self.final_status,
            "error": self.error,
        }


class AuditStoreClient:
    """Client for storing audit records."""

    def __init__(self, table_name: Optional[str] = None):
        settings = get_settings()
        self._table_name = table_name or settings.audit.table_name
        self._enabled = settings.audit.enabled

    def store(self, record: AuditRecord) -> bool:
        """
        Store an audit record.

        Args:
            record: The audit record to store

        Returns:
            True if stored successfully, False otherwise
        """
        if not self._enabled:
            return True

        try:
            mock_dynamodb.put_item(self._table_name, record.to_dict())
            return True
        except Exception:
            return False

    def get(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an audit record by request ID.

        Args:
            request_id: The request ID to look up

        Returns:
            The audit record if found, None otherwise
        """
        try:
            return mock_dynamodb.get_item(self._table_name, f"request#{request_id}")
        except Exception:
            return None


_client_instance: Optional[AuditStoreClient] = None


def get_client() -> AuditStoreClient:
    """Get or create the singleton client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = AuditStoreClient()
    return _client_instance


def store_audit(record: AuditRecord) -> bool:
    """Convenience function to store an audit record."""
    return get_client().store(record)


def get_audit(request_id: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get an audit record."""
    return get_client().get(request_id)
