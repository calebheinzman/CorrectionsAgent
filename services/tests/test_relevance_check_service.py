"""Tests for Relevance Check service."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.guardrails.relevance_check.app import app
from services.guardrails.relevance_check.schemas import RelevanceCheckRequest
from services.guardrails.relevance_check.service import RelevanceCheckService


@pytest.fixture
def client():
    """Create a test client for the relevance check service."""
    return TestClient(app)


@pytest.fixture
def service():
    """Create a relevance check service instance."""
    return RelevanceCheckService()


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
            "/v1/relevance/check",
            json={"request_id": "req_test_001"},
        )
        assert response.status_code == 422

    def test_missing_request_id_returns_422(self, client):
        """Missing request_id returns 422."""
        response = client.post(
            "/v1/relevance/check",
            json={"question": "What is the weather?"},
        )
        assert response.status_code == 422


class TestRequestIdEcho:
    """Test request_id echoing."""

    def test_echo_request_id(self, client):
        """Test case 3: Response includes the same request_id."""
        response = client.post(
            "/v1/relevance/check",
            json={
                "request_id": "req_test_001",
                "question": "What conversations have prisoners had?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "req_test_001"


class TestOutOfDomainQuery:
    """Test out-of-domain query detection."""

    def test_weather_query_not_relevant(self, client):
        """Test case 4: Out-of-domain query (weather)."""
        response = client.post(
            "/v1/relevance/check",
            json={
                "request_id": "req_test_002",
                "question": "What is the weather?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["relevant"] is False
        assert data["reason"] != ""

    def test_sports_query_not_relevant(self, client):
        """Sports query is not relevant."""
        response = client.post(
            "/v1/relevance/check",
            json={
                "request_id": "req_test_003",
                "question": "Who won the football game last night?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["relevant"] is False

    def test_recipe_query_not_relevant(self, client):
        """Recipe query is not relevant."""
        response = client.post(
            "/v1/relevance/check",
            json={
                "request_id": "req_test_004",
                "question": "How do I make chocolate cake?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["relevant"] is False


class TestInDomainQuery:
    """Test in-domain query detection."""

    def test_prisoner_drug_query_relevant(self, client):
        """Test case 5: In-domain query (prisoner/drug)."""
        response = client.post(
            "/v1/relevance/check",
            json={
                "request_id": "req_test_005",
                "question": "What conversations have prisoners had about drug use?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["relevant"] is True

    def test_incident_query_relevant(self, client):
        """Incident query is relevant."""
        response = client.post(
            "/v1/relevance/check",
            json={
                "request_id": "req_test_006",
                "question": "What incidents have been reported?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["relevant"] is True

    def test_prisoner_info_query_relevant(self, client):
        """Prisoner info query is relevant."""
        response = client.post(
            "/v1/relevance/check",
            json={
                "request_id": "req_test_007",
                "question": "What cell is prisoner P004 in?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["relevant"] is True


class TestDeterminism:
    """Test deterministic behavior."""

    def test_same_question_same_result(self, client):
        """Test case 6: Same question produces consistent results."""
        question = "What is the weather?"
        results = []

        for i in range(3):
            response = client.post(
                "/v1/relevance/check",
                json={
                    "request_id": f"req_test_det_{i}",
                    "question": question,
                },
            )
            assert response.status_code == 200
            results.append(response.json()["relevant"])

        assert all(r == results[0] for r in results)


class TestResponseSchemaCompleteness:
    """Test response schema completeness."""

    def test_response_has_all_fields(self, client):
        """Test case 7: Response includes all documented fields."""
        response = client.post(
            "/v1/relevance/check",
            json={
                "request_id": "req_test_schema",
                "question": "What conversations have prisoners had?",
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert "request_id" in data
        assert "relevant" in data
        assert "reason" in data

        assert isinstance(data["request_id"], str)
        assert isinstance(data["relevant"], bool)
        assert isinstance(data["reason"], str)


class TestServiceDirectly:
    """Test the service class directly."""

    def test_service_check_relevance_relevant(self, service):
        """Service identifies relevant questions."""
        request = RelevanceCheckRequest(
            request_id="req_direct_001",
            question="What prisoners are in cell block A?",
        )
        response = service.check_relevance(request)
        assert response.relevant is True

    def test_service_check_relevance_not_relevant(self, service):
        """Service identifies irrelevant questions."""
        request = RelevanceCheckRequest(
            request_id="req_direct_002",
            question="What is the capital of France?",
        )
        response = service.check_relevance(request)
        assert response.relevant is False
