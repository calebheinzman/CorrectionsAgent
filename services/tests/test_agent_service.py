"""Tests for Agent service."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.agent.app import app
from services.agent.schemas import AgentRequest


@pytest.fixture
def client():
    """Create a test client for the agent service."""
    return TestClient(app)


class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_check_returns_200(self, client):
        """Test case 1: Health check returns 200."""
        response = client.get("/healthz")
        assert response.status_code == 200


class TestRequiredFields:
    """Test required field validation."""

    def test_missing_question_returns_422(self, client):
        """Test case 2: Missing question returns 422."""
        response = client.post(
            "/v1/answer",
            json={"request_id": "req_test_001"},
        )
        assert response.status_code == 422

    def test_missing_request_id_returns_422(self, client):
        """Missing request_id returns 422."""
        response = client.post(
            "/v1/answer",
            json={"question": "What conversations have prisoners had?"},
        )
        assert response.status_code == 422


class TestRequestIdEcho:
    """Test request_id echoing."""

    def test_echo_request_id(self, client):
        """Test case 3: Response includes the same request_id."""
        response = client.post(
            "/v1/answer",
            json={
                "request_id": "req_test_001",
                "question": "What conversations have prisoners had about drug use?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "req_test_001"


class TestResponseSchemaCompleteness:
    """Test response schema completeness."""

    def test_response_has_all_fields(self, client):
        """Test case 8: Response includes all documented fields."""
        response = client.post(
            "/v1/answer",
            json={
                "request_id": "req_test_schema",
                "question": "What conversations have prisoners had?",
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert "request_id" in data
        assert "answer" in data
        assert "citations" in data
        assert "tool_calls" in data

        assert isinstance(data["request_id"], str)
        assert isinstance(data["answer"], str)
        assert isinstance(data["citations"], list)
        assert isinstance(data["tool_calls"], list)


class TestAgentRequestSchema:
    """Test AgentRequest schema validation."""

    def test_valid_request_with_all_fields(self):
        """Valid request with all optional fields."""
        request = AgentRequest(
            request_id="req_001",
            question="What conversations have prisoners had?",
            user_id="user_001",
            session_id="session_001",
        )
        assert request.request_id == "req_001"
        assert request.question == "What conversations have prisoners had?"
        assert request.user_id == "user_001"
        assert request.session_id == "session_001"

    def test_valid_request_minimal(self):
        """Valid request with only required fields."""
        request = AgentRequest(
            request_id="req_002",
            question="What is the status?",
        )
        assert request.request_id == "req_002"
        assert request.user_id is None
        assert request.session_id is None
