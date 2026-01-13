"""Tests for Mock S3."""
import tempfile
from pathlib import Path

import pytest

from mock_apis.cloud_apis import mock_s3


@pytest.fixture
def temp_s3_dir():
    """Create a temporary directory for S3 storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_s3.reset_store(Path(tmpdir))
        yield Path(tmpdir)
        mock_s3.reset_store(None)


class TestMockS3:
    """Test cases for Mock S3."""

    def test_put_get_roundtrip(self, temp_s3_dir):
        """Test 1: Put + Get roundtrip."""
        mock_s3.put_object(bucket="demo", key="a/b.txt", data=b"hello")
        result = mock_s3.get_object(bucket="demo", key="a/b.txt")
        assert result == b"hello"

    def test_get_missing_object(self, temp_s3_dir):
        """Test 2: Get missing object returns None."""
        result = mock_s3.get_object(bucket="demo", key="missing.txt")
        assert result is None

    def test_list_by_prefix(self, temp_s3_dir):
        """Test 3: List by prefix."""
        mock_s3.put_object(bucket="demo", key="a_1.txt", data=b"one")
        mock_s3.put_object(bucket="demo", key="a_2.txt", data=b"two")
        mock_s3.put_object(bucket="demo", key="b_1.txt", data=b"three")

        keys = mock_s3.list_objects(bucket="demo", prefix="a_")
        assert len([k for k in keys if "a_" in k]) >= 2
        assert not any("b_1.txt" in k for k in keys)

    def test_deterministic_persistence(self, temp_s3_dir):
        """Test 4: Deterministic persistence (file-backed)."""
        mock_s3.put_object(bucket="test", key="persist.txt", data=b"persistent data")

        mock_s3.reset_store(temp_s3_dir)

        result = mock_s3.get_object(bucket="test", key="persist.txt")
        assert result == b"persistent data"

    def test_delete_object(self, temp_s3_dir):
        """Test delete object functionality."""
        mock_s3.put_object(bucket="demo", key="delete_me.txt", data=b"to delete")
        assert mock_s3.get_object(bucket="demo", key="delete_me.txt") == b"to delete"

        deleted = mock_s3.delete_object(bucket="demo", key="delete_me.txt")
        assert deleted is True

        result = mock_s3.get_object(bucket="demo", key="delete_me.txt")
        assert result is None
