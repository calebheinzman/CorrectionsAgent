"""Tests for Prisoner Info API."""
import pytest
from fastapi.testclient import TestClient

from mock_apis.internal_apis.app import app
from mock_apis.internal_apis import prisoner_info_api


@pytest.fixture(scope="module")
def client():
    """Create test client with sample data."""
    sample_prisoners = [
        {"prisoner_id": "P001", "name": "Alex Smith"},
        {"prisoner_id": "P002", "name": "Jordan Johnson"},
        {"prisoner_id": "P003", "name": "Taylor Brown"},
        {"prisoner_id": "P004", "name": "Casey Garcia"},
        {"prisoner_id": "P005", "name": "Riley Miller"},
    ]
    prisoner_info_api.load_prisoners(sample_prisoners)
    return TestClient(app)


class TestPrisonerInfoAPI:
    """Test cases for Prisoner Info API."""

    def test_list_sanity(self, client):
        """Test 1: List sanity - GET /prisoners?limit=5&offset=0"""
        response = client.get("/prisoners?limit=5&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert len(data["items"]) <= 5
        for item in data["items"]:
            assert "prisoner_id" in item
            assert "name" in item

    def test_exact_id_lookup(self, client):
        """Test 2: Exact ID lookup - GET /prisoners/P001"""
        response = client.get("/prisoners/P001")
        assert response.status_code == 200
        data = response.json()
        assert data["prisoner_id"] == "P001"

    def test_unknown_id(self, client):
        """Test 3: Unknown ID - GET /prisoners/P999"""
        response = client.get("/prisoners/P999")
        assert response.status_code == 404

    def test_name_search(self, client):
        """Test 4: Name search - GET /prisoners?name=Smith"""
        response = client.get("/prisoners?name=Smith")
        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert "smith" in item["name"].lower()

    def test_pagination(self, client):
        """Test 5: Pagination - pages should differ"""
        response1 = client.get("/prisoners?limit=2&offset=0")
        response2 = client.get("/prisoners?limit=2&offset=2")
        assert response1.status_code == 200
        assert response2.status_code == 200
        data1 = response1.json()
        data2 = response2.json()
        ids1 = {item["prisoner_id"] for item in data1["items"]}
        ids2 = {item["prisoner_id"] for item in data2["items"]}
        assert ids1 != ids2 or data1["total"] < 4
