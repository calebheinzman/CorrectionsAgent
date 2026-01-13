"""Tests for Conversation API."""
import pytest
from fastapi.testclient import TestClient

from mock_apis.internal_apis.app import app
from mock_apis.internal_apis import conversation_api


@pytest.fixture(scope="module")
def client():
    """Create test client with sample data."""
    sample_conversations = [
        {
            "conversation_id": "conv_001",
            "timestamp": "2026-01-09T16:00:00Z",
            "prisoner_ids": ["P004"],
            "prisoner_names": ["Casey Garcia"],
            "transcript": "Discussion about facility tension and staffing issues.",
            "facility_id": "FAC-001",
            "facility_name": "North River Correctional Center",
            "communication_type": "inmate_call",
            "call_duration_seconds": 410,
            "outside_contact_name": "Jamie Parker",
            "outside_contact_relation": "partner",
            "alert_categories": ["facility_security", "threats_violence"],
            "keyword_hits": ["short staffed", "fight", "tense"],
            "alert_confidence": 0.82,
            "review_status": "unreviewed",
        },
        {
            "conversation_id": "conv_002",
            "timestamp": "2026-01-09T08:30:00Z",
            "prisoner_ids": ["P004"],
            "prisoner_names": ["Casey Garcia"],
            "transcript": "Talk about contraband pressure and drug use concerns.",
            "facility_id": "FAC-001",
            "facility_name": "North River Correctional Center",
            "communication_type": "inmate_call",
            "call_duration_seconds": 520,
            "outside_contact_name": "Pat Reynolds",
            "outside_contact_relation": "parent",
            "alert_categories": ["contraband_drugs", "facility_security"],
            "keyword_hits": ["contraband", "pills", "debts"],
            "alert_confidence": 0.9,
            "review_status": "unreviewed",
        },
        {
            "conversation_id": "conv_003",
            "timestamp": "2026-01-05T13:15:00Z",
            "prisoner_ids": ["P004"],
            "prisoner_names": ["Casey Garcia"],
            "transcript": "Concerns about coercive behavior.",
            "facility_id": "FAC-001",
            "facility_name": "North River Correctional Center",
            "communication_type": "inmate_text",
            "call_duration_seconds": 25,
            "outside_contact_name": "Chris Nguyen",
            "outside_contact_relation": "sibling",
            "alert_categories": ["prea_related"],
            "keyword_hits": ["coercive", "pressure"],
            "alert_confidence": 0.72,
            "review_status": "reviewed",
        },
        {
            "conversation_id": "conv_004",
            "timestamp": "2026-01-08T21:10:00Z",
            "prisoner_ids": ["P005"],
            "prisoner_names": ["Riley Miller"],
            "transcript": "Mental health concerns and wellness check.",
            "facility_id": "FAC-001",
            "facility_name": "North River Correctional Center",
            "communication_type": "inmate_call",
            "call_duration_seconds": 380,
            "outside_contact_name": "Sam Patel",
            "outside_contact_relation": "friend",
            "alert_categories": ["wellness_mental_health"],
            "keyword_hits": ["panic", "hopeless"],
            "alert_confidence": 0.84,
            "review_status": "unreviewed",
        },
        {
            "conversation_id": "conv_005",
            "timestamp": "2025-12-15T11:00:00Z",
            "prisoner_ids": ["P006"],
            "prisoner_names": ["Morgan Davis"],
            "transcript": "Discussion about accounts and drops - possible fraud.",
            "facility_id": "FAC-001",
            "facility_name": "North River Correctional Center",
            "communication_type": "inmate_call",
            "call_duration_seconds": 450,
            "outside_contact_name": "Ari Johnson",
            "outside_contact_relation": "cousin",
            "alert_categories": ["fraud_identity_theft", "general_intel"],
            "keyword_hits": ["accounts", "drops"],
            "alert_confidence": 0.77,
            "review_status": "unreviewed",
        },
    ]
    conversation_api.load_conversations(sample_conversations)
    return TestClient(app)


class TestConversationAPI:
    """Test cases for Conversation API."""

    def test_health_sanity_list(self, client):
        """Test 1: Health sanity (list) - GET /conversations?limit=5&offset=0"""
        response = client.get("/conversations?limit=5&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        for item in data["items"]:
            assert "conversation_id" in item
            assert "timestamp" in item
            assert "prisoner_ids" in item
            assert "transcript" in item

    def test_exact_id_lookup(self, client):
        """Test 2: Exact ID lookup - GET /conversations/conv_001"""
        response = client.get("/conversations/conv_001")
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "conv_001"

    def test_unknown_id(self, client):
        """Test 3: Unknown ID - GET /conversations/conv_does_not_exist"""
        response = client.get("/conversations/conv_does_not_exist")
        assert response.status_code == 404

    def test_filter_by_prisoner_id(self, client):
        """Test 4: Filter by prisoner_id - GET /conversations?prisoner_id=P004"""
        response = client.get("/conversations?prisoner_id=P004")
        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert "P004" in item["prisoner_ids"]

    def test_filter_by_prisoner_name(self, client):
        """Test 5: Filter by prisoner_name - GET /conversations?prisoner_name=Taylor"""
        response = client.get("/conversations?prisoner_name=Casey")
        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert any("casey" in name.lower() for name in item["prisoner_names"])

    def test_filter_by_alert_category(self, client):
        """Test 6: Filter by alert_category - GET /conversations?alert_category=contraband_drugs"""
        response = client.get("/conversations?alert_category=contraband_drugs")
        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert "contraband_drugs" in item["alert_categories"]

    def test_time_range_filtering(self, client):
        """Test 7: Time range filtering"""
        response = client.get(
            "/conversations?start_time=2026-01-09T00:00:00Z&end_time=2026-01-10T00:00:00Z"
        )
        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["timestamp"] >= "2026-01-09T00:00:00Z"
            assert item["timestamp"] <= "2026-01-10T00:00:00Z"

    def test_invalid_time_range(self, client):
        """Test 8: Invalid time range - start after end"""
        response = client.get(
            "/conversations?start_time=2026-01-10T00:00:00Z&end_time=2026-01-09T00:00:00Z"
        )
        assert response.status_code == 400

    def test_similar_text_search_without_vector_store(self, client):
        """Test 9: Similar-text search (without vector store falls back to list)"""
        response = client.get("/conversations?query=drug%20use&top_k=5")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data

    def test_pagination(self, client):
        """Test 10: Pagination - pages should differ"""
        response1 = client.get("/conversations?limit=2&offset=0")
        response2 = client.get("/conversations?limit=2&offset=2")
        assert response1.status_code == 200
        assert response2.status_code == 200
        data1 = response1.json()
        data2 = response2.json()
        ids1 = {item["conversation_id"] for item in data1["items"]}
        ids2 = {item["conversation_id"] for item in data2["items"]}
        assert ids1 != ids2 or data1["total"] < 4

    def test_sorting(self, client):
        """Test 11: Sorting - GET /conversations?sort_by=timestamp&sort_order=desc&limit=5"""
        response = client.get("/conversations?sort_by=timestamp&sort_order=desc&limit=5")
        assert response.status_code == 200
        data = response.json()
        items = data["items"]
        for i in range(len(items) - 1):
            assert items[i]["timestamp"] >= items[i + 1]["timestamp"]
