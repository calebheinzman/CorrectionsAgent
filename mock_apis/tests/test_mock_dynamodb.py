"""Tests for Mock DynamoDB."""
import tempfile
from pathlib import Path

import pytest

from mock_apis.cloud_apis import mock_dynamodb


@pytest.fixture
def temp_dynamodb_dir():
    """Create a temporary directory for DynamoDB storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_dynamodb.reset_store(Path(tmpdir))
        yield Path(tmpdir)
        mock_dynamodb.reset_store(None)


class TestMockDynamoDB:
    """Test cases for Mock DynamoDB."""

    def test_put_get_item(self, temp_dynamodb_dir):
        """Test 1: Put + Get item."""
        item = {
            "pk": "req#1",
            "sk": "event#1",
            "type": "tool_call",
            "ts": "2026-01-01T00:00:00Z",
        }
        mock_dynamodb.put_item("audit_events", item)
        result = mock_dynamodb.get_item("audit_events", {"pk": "req#1", "sk": "event#1"})
        assert result is not None
        assert result["pk"] == "req#1"
        assert result["sk"] == "event#1"
        assert result["type"] == "tool_call"

    def test_get_missing_item(self, temp_dynamodb_dir):
        """Test 2: Get missing item returns None."""
        result = mock_dynamodb.get_item("audit_events", {"pk": "missing", "sk": "missing"})
        assert result is None

    def test_query_basic(self, temp_dynamodb_dir):
        """Test 3: Query basic - items matching prefix."""
        mock_dynamodb.put_item("audit_events", {"pk": "req#1", "sk": "event#1", "data": "a"})
        mock_dynamodb.put_item("audit_events", {"pk": "req#1", "sk": "event#2", "data": "b"})
        mock_dynamodb.put_item("audit_events", {"pk": "req#2", "sk": "event#1", "data": "c"})
        mock_dynamodb.put_item("audit_events", {"pk": "other#1", "sk": "event#1", "data": "d"})

        results = mock_dynamodb.query("audit_events", key_prefix="req#", limit=50)
        assert len(results) >= 3
        for item in results:
            assert item["pk"].startswith("req#")

    def test_query_respects_limit(self, temp_dynamodb_dir):
        """Test query respects limit parameter."""
        for i in range(10):
            mock_dynamodb.put_item("test_table", {"pk": f"item#{i}", "sk": "data", "value": i})

        results = mock_dynamodb.query("test_table", limit=3)
        assert len(results) <= 3

    def test_deterministic_persistence(self, temp_dynamodb_dir):
        """Test 4: Deterministic persistence (file-backed)."""
        mock_dynamodb.put_item("persist_test", {"pk": "persist#1", "sk": "data", "value": "test"})

        mock_dynamodb.reset_store(temp_dynamodb_dir)

        results = mock_dynamodb.query("persist_test", key_prefix="persist#")
        assert len(results) >= 1
        assert results[0]["value"] == "test"
