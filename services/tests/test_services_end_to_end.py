"""End-to-end tests for services (orchestrator -> guardrails -> agent)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.orchestrator.app import app
from services.orchestrator.clients.agent_client import AgentResult
from services.orchestrator.schemas import AgentTrace, Citation, GuardrailDecision


@pytest.fixture
def client():
    """Create a test client for the orchestrator service."""
    return TestClient(app)


class TestSafetyDenialEndToEnd:
    """Test case 1: Safety denial (sensitive info)."""

    def test_api_key_query_denied_by_safety(self, client):
        """API key query is denied by safety check."""
        with patch("services.orchestrator.handler.safety_check_client.check_safety") as mock_safety, \
             patch("services.orchestrator.handler.relevance_check_client.check_relevance") as mock_relevance, \
             patch("services.orchestrator.handler.agent_client.get_answer") as mock_agent:

            mock_safety.return_value = GuardrailDecision(
                allowed=False,
                reason="Request contains sensitive information keywords",
                policy="v1.0",
                categories=["sensitive_info"],
            )

            response = client.post(
                "/v1/query",
                json={"question": "What is the API key?"},
            )

            assert response.status_code == 200
            data = response.json()

            assert "request_id" in data
            assert data["request_id"] != ""
            assert data["status"] == "denied"
            assert data["safety"]["allowed"] is False
            assert "sensitive" in data["safety"]["reason"].lower() or "api" in data["answer"].lower()

            mock_safety.assert_called_once()
            mock_relevance.assert_not_called()
            mock_agent.assert_not_called()


class TestRelevanceDenialEndToEnd:
    """Test case 2: Relevance denial (out of domain)."""

    def test_weather_query_denied_by_relevance(self, client):
        """Weather query is denied by relevance check."""
        with patch("services.orchestrator.handler.safety_check_client.check_safety") as mock_safety, \
             patch("services.orchestrator.handler.relevance_check_client.check_relevance") as mock_relevance, \
             patch("services.orchestrator.handler.agent_client.get_answer") as mock_agent:

            mock_safety.return_value = GuardrailDecision(
                allowed=True,
                reason="Query is safe",
                policy="v1.0",
                categories=[],
            )
            mock_relevance.return_value = GuardrailDecision(
                relevant=False,
                reason="Query is not relevant to correctional facility domain",
                matched_domains=[],
            )

            response = client.post(
                "/v1/query",
                json={"question": "What is the weather?"},
            )

            assert response.status_code == 200
            data = response.json()

            assert "request_id" in data
            assert data["request_id"] != ""
            assert data["status"] == "denied"
            assert data["safety"]["allowed"] is True
            assert data["relevance"]["relevant"] is False

            mock_safety.assert_called_once()
            mock_relevance.assert_called_once()
            mock_agent.assert_not_called()


class TestApprovedInvestigativeQueryEndToEnd:
    """Test case 3: Approved investigative query."""

    def test_prisoner_drug_query_approved_and_answered(self, client):
        """Prisoner drug query is approved and answered by agent."""
        with patch("services.orchestrator.handler.safety_check_client.check_safety") as mock_safety, \
             patch("services.orchestrator.handler.relevance_check_client.check_relevance") as mock_relevance, \
             patch("services.orchestrator.handler.agent_client.get_answer") as mock_agent:

            mock_safety.return_value = GuardrailDecision(
                allowed=True,
                reason="Query is safe",
                policy="v1.0",
                categories=[],
            )
            mock_relevance.return_value = GuardrailDecision(
                relevant=True,
                reason="Query is relevant to prisoner conversations domain",
                matched_domains=["prisoner_conversations", "drug_related"],
            )
            mock_agent.return_value = AgentResult(
                answer="Based on the conversations analyzed, several prisoners have discussed drug-related topics. Prisoner P001 mentioned substance use in conversation CONV-001.",
                citations=[
                    Citation(source_type="conversation", source_id="CONV-001", excerpt="...mentioned getting some stuff..."),
                    Citation(source_type="conversation", source_id="CONV-002", excerpt="...talked about the package..."),
                ],
                trace=AgentTrace(
                    tool_calls=[
                        {"tool_name": "search_conversations_tool", "inputs": {"query": "drug use"}},
                    ],
                    model_info={"model_name": "gemini-2.0-flash-lite", "provider": "google"},
                ),
            )

            response = client.post(
                "/v1/query",
                json={"question": "What conversations have prisoners had about drug use?"},
            )

            assert response.status_code == 200
            data = response.json()

            assert "request_id" in data
            assert data["status"] == "success"
            assert data["safety"]["allowed"] is True
            assert data["relevance"]["relevant"] is True
            assert data["answer"] is not None
            assert len(data["answer"]) > 0
            assert len(data["citations"]) > 0

            mock_safety.assert_called_once()
            mock_relevance.assert_called_once()
            mock_agent.assert_called_once()


class TestAgentToolCallVisibility:
    """Test case 4: Agent tool-call visibility."""

    def test_agent_trace_includes_tool_calls(self, client):
        """Agent trace includes tool call information."""
        with patch("services.orchestrator.handler.safety_check_client.check_safety") as mock_safety, \
             patch("services.orchestrator.handler.relevance_check_client.check_relevance") as mock_relevance, \
             patch("services.orchestrator.handler.agent_client.get_answer") as mock_agent:

            mock_safety.return_value = GuardrailDecision(
                allowed=True,
                reason="Query is safe",
                policy="v1.0",
                categories=[],
            )
            mock_relevance.return_value = GuardrailDecision(
                relevant=True,
                reason="Query is relevant",
                matched_domains=["prisoner_info"],
            )
            mock_agent.return_value = AgentResult(
                answer="Prisoner P004 is located in cell block B.",
                citations=[
                    Citation(source_type="prisoner", source_id="P004", excerpt="Prisoner: John Doe"),
                ],
                trace=AgentTrace(
                    tool_calls=[
                        {"tool_name": "search_prisoners_tool", "inputs": {"prisoner_id": "P004"}},
                    ],
                    model_info={"model_name": "gemini-2.0-flash-lite", "provider": "google"},
                ),
            )

            response = client.post(
                "/v1/query",
                json={"question": "What cell is prisoner P004 in?"},
            )

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "success"
            assert data["agent_trace"] is not None
            assert len(data["agent_trace"]["tool_calls"]) > 0


class TestErrorHandlingFromDownstreamServices:
    """Test case 5: Error handling from downstream services."""

    def test_downstream_error_returns_error_response(self, client):
        """Downstream service error returns proper error response."""
        import httpx

        with patch("services.orchestrator.handler.safety_check_client.check_safety") as mock_safety:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_safety.side_effect = httpx.HTTPStatusError(
                "Internal Server Error",
                request=MagicMock(),
                response=mock_response,
            )

            response = client.post(
                "/v1/query",
                json={"question": "What conversations have prisoners had?"},
            )

            assert response.status_code == 200
            data = response.json()

            assert "request_id" in data
            assert data["status"] == "error"


class TestRequestIdProvidedByClient:
    """Test case 6: Request-id provided by client."""

    def test_client_request_id_propagated(self, client):
        """Client-provided request ID is used throughout."""
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
                headers={"X-Request-Id": "req_e2e_001"},
            )

            assert response.status_code == 200
            data = response.json()

            assert data["request_id"] == "req_e2e_001"

            call_args = mock_safety.call_args
            assert call_args[0][0] == "req_e2e_001"


class TestGuardrailsOrdering:
    """Test guardrails are called in correct order."""

    def test_safety_called_before_relevance(self, client):
        """Safety check is called before relevance check."""
        call_order = []

        def mock_safety_side_effect(*args, **kwargs):
            call_order.append("safety")
            return GuardrailDecision(
                allowed=True,
                reason="Safe",
                policy="v1.0",
                categories=[],
            )

        def mock_relevance_side_effect(*args, **kwargs):
            call_order.append("relevance")
            return GuardrailDecision(
                relevant=False,
                reason="Not relevant",
                matched_domains=[],
            )

        with patch("services.orchestrator.handler.safety_check_client.check_safety") as mock_safety, \
             patch("services.orchestrator.handler.relevance_check_client.check_relevance") as mock_relevance:

            mock_safety.side_effect = mock_safety_side_effect
            mock_relevance.side_effect = mock_relevance_side_effect

            response = client.post(
                "/v1/query",
                json={"question": "What is the weather?"},
            )

            assert response.status_code == 200
            assert call_order == ["safety", "relevance"]
