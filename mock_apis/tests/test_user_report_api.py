"""Tests for User Report API."""
import pytest
from fastapi.testclient import TestClient

from mock_apis.internal_apis.app import app
from mock_apis.internal_apis import user_report_api


@pytest.fixture(scope="module")
def client():
    """Create test client with sample data."""
    sample_reports = [
        {
            "report_id": "rpt_001",
            "created_at": "2026-01-09T16:10:00Z",
            "title": "Shift Brief: Recent risks",
            "summary": "Recent communications indicate elevated tension with contraband pressure and drug use concerns.",
            "raw_text": "Analyst shift brief compiling key monitored-communication signals.",
            "linked_prisoner_ids": ["P004", "P005"],
            "linked_prisoner_names": ["Casey Garcia", "Riley Miller"],
            "linked_conversation_ids": ["conv_001", "conv_002", "conv_003", "conv_004"],
            "tags": ["facility_security", "contraband_drugs", "prea_related", "wellness"],
            "report_type": "alert_digest",
            "trigger_type": "keyword_alert",
            "risk_level": "high",
            "confidence": 0.78,
            "alert_categories": ["facility_security", "contraband_drugs", "prea_related", "wellness_mental_health"],
            "recommended_actions": ["Prioritize human review", "Notify shift leadership"],
            "key_excerpts": ["Unit B is short staffed", "pills are moving around"],
            "audit_note": "For demo use only",
        },
        {
            "report_id": "rpt_002",
            "created_at": "2026-01-09T12:00:00Z",
            "title": "Case Summary: Casey Garcia - 30 Day Themes",
            "summary": "Case summary aggregates the last 30 days of flagged communications.",
            "raw_text": "Subject-focused summary for Casey Garcia.",
            "linked_prisoner_ids": ["P004", "P005", "P006"],
            "linked_prisoner_names": ["Casey Garcia", "Riley Miller", "Morgan Davis"],
            "linked_conversation_ids": ["conv_001", "conv_002", "conv_006", "conv_007"],
            "tags": ["case_summary", "themes", "citations"],
            "report_type": "case_summary",
            "trigger_type": "case_followup",
            "risk_level": "medium",
            "confidence": 0.7,
            "alert_categories": ["general_intel", "facility_security"],
            "recommended_actions": ["Confirm transfer references", "Review associated calls"],
            "key_excerpts": ["getting pulled into that group again"],
            "audit_note": "Case packaging; validate all conclusions",
        },
        {
            "report_id": "rpt_003",
            "created_at": "2026-01-09T15:30:00Z",
            "title": "Incident Addendum: INC-017 Supporting Communications",
            "summary": "Addendum links contraband-pressure communication as potential precursors.",
            "raw_text": "Addendum for investigators.",
            "linked_prisoner_ids": ["P004"],
            "linked_prisoner_names": ["Casey Garcia"],
            "linked_conversation_ids": ["conv_001", "conv_002"],
            "tags": ["incident_addendum", "citations", "contraband_drugs"],
            "report_type": "case_summary",
            "trigger_type": "case_followup",
            "risk_level": "high",
            "confidence": 0.74,
            "alert_categories": ["contraband_drugs", "facility_security"],
            "recommended_actions": ["Correlate incident timeline"],
            "key_excerpts": ["pills are moving around"],
            "audit_note": "Incident addendum for demo",
        },
    ]
    user_report_api.load_user_reports(sample_reports)
    return TestClient(app)


class TestUserReportAPI:
    """Test cases for User Report API."""

    def test_list_sanity(self, client):
        """Test 1: List sanity - GET /user-reports?limit=5&offset=0"""
        response = client.get("/user-reports?limit=5&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        for item in data["items"]:
            assert "report_id" in item
            assert "created_at" in item
            assert "title" in item
            assert "summary" in item

    def test_exact_id_lookup(self, client):
        """Test 2: Exact ID lookup - GET /user-reports/rpt_001"""
        response = client.get("/user-reports/rpt_001")
        assert response.status_code == 200
        data = response.json()
        assert data["report_id"] == "rpt_001"

    def test_unknown_id(self, client):
        """Test 3: Unknown ID - GET /user-reports/rpt_does_not_exist"""
        response = client.get("/user-reports/rpt_does_not_exist")
        assert response.status_code == 404

    def test_filter_by_prisoner_id(self, client):
        """Test 4: Filter by prisoner_id - GET /user-reports?prisoner_id=P004"""
        response = client.get("/user-reports?prisoner_id=P004")
        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert "P004" in item["linked_prisoner_ids"]

    def test_filter_by_conversation_id(self, client):
        """Test 5: Filter by conversation_id - GET /user-reports?conversation_id=conv_001"""
        response = client.get("/user-reports?conversation_id=conv_001")
        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert "conv_001" in item["linked_conversation_ids"]

    def test_risk_level_filter(self, client):
        """Test 6: Risk level filter - GET /user-reports?risk_level=high"""
        response = client.get("/user-reports?risk_level=high")
        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["risk_level"] == "high"

    def test_time_range_filtering(self, client):
        """Test 7: Time range filtering"""
        response = client.get(
            "/user-reports?start_time=2026-01-09T00:00:00Z&end_time=2026-01-10T00:00:00Z"
        )
        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["created_at"] >= "2026-01-09T00:00:00Z"
            assert item["created_at"] <= "2026-01-10T00:00:00Z"

    def test_similar_text_search_without_vector_store(self, client):
        """Test 8: Similar-text search (without vector store falls back to list)"""
        response = client.get("/user-reports?query=contraband&top_k=5")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data

    def test_pagination_and_sorting(self, client):
        """Test 9: Pagination + sorting"""
        response = client.get("/user-reports?sort_by=created_at&sort_order=desc&limit=3&offset=0")
        assert response.status_code == 200
        data = response.json()
        items = data["items"]
        for i in range(len(items) - 1):
            assert items[i]["created_at"] >= items[i + 1]["created_at"]
