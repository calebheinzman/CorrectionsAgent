"""Tests for Orchestrator service."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.orchestrator.app import app
from services.orchestrator.schemas import GuardrailDecision, QueryRequest


@pytest.fixture
def client():
    """Create a test client for the orchestrator service."""
    return TestClient(app)


class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_check_returns_200(self, client):
        """Test case 1: Health check returns 200."""
        response = client.get("/healthz")
        assert response.status_code == 200


class TestRequestIdGeneration:
    """Test request ID generation."""

    def test_request_id_generated_when_not_provided(self, client):
        """Test case 2: Request ID is generated when not provided."""
        with patch("services.orchestrator.handler.safety_check_client.check_safety") as mock_safety:
            mock_safety.return_value = GuardrailDecision(
                allowed=False,
                reason="Test denial",
                policy="v1.0",
                categories=["test"],
            )

            response = client.post(
                "/v1/query",
                json={"question": "What is the API key?"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "request_id" in data
            assert data["request_id"] != ""
            assert data["request_id"].startswith("req_")


class TestRequestIdPropagation:
    """Test request ID propagation."""

    def test_request_id_propagated_from_header(self, client):
        """Test case 3: Request ID is propagated from header."""
        with patch("services.orchestrator.handler.safety_check_client.check_safety") as mock_safety:
            mock_safety.return_value = GuardrailDecision(
                allowed=False,
                reason="Test denial",
                policy="v1.0",
                categories=["test"],
            )

            response = client.post(
                "/v1/query",
                json={"question": "What is the API key?"},
                headers={"X-Request-Id": "req_test_001"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["request_id"] == "req_test_001"


class TestSafetyDenialShortCircuit:
    """Test safety denial short-circuit."""

    def test_safety_denial_does_not_call_agent(self, client):
        """Test case 5: Safety denial does not call agent."""
        with patch("services.orchestrator.handler.safety_check_client.check_safety") as mock_safety, \
             patch("services.orchestrator.handler.relevance_check_client.check_relevance") as mock_relevance, \
             patch("services.orchestrator.handler.agent_client.get_answer") as mock_agent:

            mock_safety.return_value = GuardrailDecision(
                allowed=False,
                reason="Sensitive information request",
                policy="v1.0",
                categories=["sensitive_info"],
            )

            response = client.post(
                "/v1/query",
                json={"question": "What is the API key?"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "denied"
            assert data["safety"]["allowed"] is False

            mock_safety.assert_called_once()
            mock_relevance.assert_not_called()
            mock_agent.assert_not_called()


class TestRelevanceDenialShortCircuit:
    """Test relevance denial short-circuit."""

    def test_relevance_denial_does_not_call_agent(self, client):
        """Test case 6: Relevance denial does not call agent."""
        with patch("services.orchestrator.handler.safety_check_client.check_safety") as mock_safety, \
             patch("services.orchestrator.handler.relevance_check_client.check_relevance") as mock_relevance, \
             patch("services.orchestrator.handler.agent_client.get_answer") as mock_agent:

            mock_safety.return_value = GuardrailDecision(
                allowed=True,
                reason="Safe query",
                policy="v1.0",
                categories=[],
            )
            mock_relevance.return_value = GuardrailDecision(
                relevant=False,
                reason="Out of domain",
                matched_domains=[],
            )

            response = client.post(
                "/v1/query",
                json={"question": "What is the weather?"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "denied"
            assert data["safety"]["allowed"] is True
            assert data["relevance"]["relevant"] is False

            mock_safety.assert_called_once()
            mock_relevance.assert_called_once()
            mock_agent.assert_not_called()


class TestApprovedPathCallsAgent:
    """Test approved path calls agent."""

    def test_approved_query_calls_agent(self, client):
        """Test case 7: Approved query calls agent."""
        from services.orchestrator.clients.agent_client import AgentResult
        from services.orchestrator.schemas import AgentTrace, Citation

        with patch("services.orchestrator.handler.safety_check_client.check_safety") as mock_safety, \
             patch("services.orchestrator.handler.relevance_check_client.check_relevance") as mock_relevance, \
             patch("services.orchestrator.handler.agent_client.get_answer") as mock_agent:

            mock_safety.return_value = GuardrailDecision(
                allowed=True,
                reason="Safe query",
                policy="v1.0",
                categories=[],
            )
            mock_relevance.return_value = GuardrailDecision(
                relevant=True,
                reason="In domain",
                matched_domains=["prisoner_conversations"],
            )
            mock_agent.return_value = AgentResult(
                answer="Based on the conversations, prisoners discussed...",
                citations=[Citation(source_type="conversation", source_id="conv_001", excerpt="...")],
                trace=AgentTrace(tool_calls=[{"tool_name": "search_conversations"}]),
            )

            response = client.post(
                "/v1/query",
                json={"question": "What conversations have prisoners had about drug use?"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["safety"]["allowed"] is True
            assert data["relevance"]["relevant"] is True
            assert data["answer"] is not None
            assert len(data["answer"]) > 0

            mock_safety.assert_called_once()
            mock_relevance.assert_called_once()
            mock_agent.assert_called_once()


class TestErrorShapeConsistency:
    """Test error shape consistency."""

    def test_missing_question_returns_422(self, client):
        """Test case 10: Missing question returns 422."""
        response = client.post(
            "/v1/query",
            json={},
        )
        assert response.status_code == 422


class TestQueryRequestSchema:
    """Test QueryRequest schema validation."""

    def test_valid_request_with_all_fields(self):
        """Valid request with all optional fields."""
        request = QueryRequest(
            question="What conversations have prisoners had?",
            user_id="user_001",
            session_id="session_001",
            metadata={"source": "test"},
        )
        assert request.question == "What conversations have prisoners had?"
        assert request.user_id == "user_001"
        assert request.session_id == "session_001"
        assert request.metadata == {"source": "test"}

    def test_valid_request_minimal(self):
        """Valid request with only required fields."""
        request = QueryRequest(question="What is the status?")
        assert request.question == "What is the status?"
        assert request.user_id is None
        assert request.session_id is None
        assert request.metadata is None
