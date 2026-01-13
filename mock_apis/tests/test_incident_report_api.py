"""Tests for Incident Report API."""
import pytest
from fastapi.testclient import TestClient

from mock_apis.internal_apis.app import app
from mock_apis.internal_apis import incident_report_api


@pytest.fixture(scope="module")
def client():
    """Create test client with sample data."""
    sample_incidents = [
        {
            "incident_id": "INC-017",
            "date": "2026-01-09",
            "type": "Contraband Concern",
            "severity": "high",
            "description": "Staff documented a contraband-related concern following increased tension in Housing Unit B.",
            "involved_prisoner_ids": ["P004"],
            "involved_prisoner_names": ["Casey Garcia"],
            "facility_id": "FAC-001",
            "facility_name": "North River Correctional Center",
            "location": "Housing Unit B",
            "shift": "evening",
            "outcome": "Resolved; referred for follow-up review",
            "linked_conversation_ids": ["conv_001", "conv_002"],
            "linked_report_ids": ["rpt_003"],
        },
        {
            "incident_id": "inc_006",
            "date": "2025-09-18",
            "type": "Altercation",
            "severity": "medium",
            "description": "A brief altercation was reported and resolved without further escalation.",
            "involved_prisoner_ids": ["P005"],
            "involved_prisoner_names": ["Riley Miller"],
            "facility_id": "FAC-001",
            "facility_name": "North River Correctional Center",
            "location": "Yard",
            "shift": "day",
            "outcome": "Resolved",
            "linked_conversation_ids": [],
            "linked_report_ids": [],
        },
        {
            "incident_id": "inc_007",
            "date": "2026-01-15",
            "type": "Medical Incident",
            "severity": "low",
            "description": "Minor medical incident handled by on-site staff.",
            "involved_prisoner_ids": ["P006"],
            "involved_prisoner_names": ["Morgan Davis"],
            "facility_id": "FAC-001",
            "facility_name": "North River Correctional Center",
            "location": "Medical",
            "shift": "night",
            "outcome": "Resolved",
            "linked_conversation_ids": [],
            "linked_report_ids": [],
        },
        {
            "incident_id": "inc_008",
            "date": "2026-01-20",
            "type": "Threat/Intimidation",
            "severity": "high",
            "description": "Reported threat incident requiring immediate attention.",
            "involved_prisoner_ids": ["P004"],
            "involved_prisoner_names": ["Casey Garcia"],
            "facility_id": "FAC-001",
            "facility_name": "North River Correctional Center",
            "location": "Housing Unit A",
            "shift": "evening",
            "outcome": "Under investigation",
            "linked_conversation_ids": ["conv_001"],
            "linked_report_ids": [],
        },
    ]
    incident_report_api.load_incident_reports(sample_incidents)
    return TestClient(app)


class TestIncidentReportAPI:
    """Test cases for Incident Report API."""

    def test_list_sanity(self, client):
        """Test 1: List sanity - GET /incidents?limit=5&offset=0"""
        response = client.get("/incidents?limit=5&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        for item in data["items"]:
            assert "incident_id" in item
            assert "date" in item
            assert "type" in item
            assert "severity" in item
            assert "description" in item

    def test_exact_id_lookup(self, client):
        """Test 2: Exact ID lookup - GET /incidents/INC-017"""
        response = client.get("/incidents/INC-017")
        assert response.status_code == 200
        data = response.json()
        assert data["incident_id"] == "INC-017"

    def test_filter_by_prisoner_id(self, client):
        """Test 3: Filter by prisoner_id - GET /incidents?prisoner_id=P004"""
        response = client.get("/incidents?prisoner_id=P004")
        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert "P004" in item["involved_prisoner_ids"]

    def test_severity_filter(self, client):
        """Test 4: Severity filter - GET /incidents?severity=high"""
        response = client.get("/incidents?severity=high")
        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["severity"] == "high"

    def test_date_range_filtering(self, client):
        """Test 5: Date range filtering"""
        response = client.get("/incidents?start_date=2026-01-01&end_date=2026-01-31")
        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["date"] >= "2026-01-01"
            assert item["date"] <= "2026-01-31"

    def test_invalid_date_range(self, client):
        """Test 6: Invalid date range - start after end"""
        response = client.get("/incidents?start_date=2026-02-01&end_date=2026-01-01")
        assert response.status_code == 400

    def test_similar_text_search_without_vector_store(self, client):
        """Test 7: Similar-text search (without vector store falls back to list)"""
        response = client.get("/incidents?query=contraband&top_k=5")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data

    def test_pagination_and_sorting(self, client):
        """Test 8: Pagination + sorting"""
        response = client.get("/incidents?sort_by=date&sort_order=desc&limit=3&offset=0")
        assert response.status_code == 200
        data = response.json()
        items = data["items"]
        for i in range(len(items) - 1):
            assert items[i]["date"] >= items[i + 1]["date"]
