"""Tests for Mock CloudWatch."""
import tempfile
from pathlib import Path

import pytest

from mock_apis.cloud_apis import mock_cloud_watch


@pytest.fixture
def temp_cloudwatch_dir():
    """Create a temporary directory for CloudWatch storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_cloud_watch.reset_log(Path(tmpdir))
        yield Path(tmpdir)
        mock_cloud_watch.reset_log(None)


class TestMockCloudWatch:
    """Test cases for Mock CloudWatch."""

    def test_put_metric(self, temp_cloudwatch_dir):
        """Test 1: Put a metric."""
        mock_cloud_watch.put_metric("latency_ms", 123.0, dimensions={"service": "agent"})

        metrics = mock_cloud_watch.get_metrics()
        assert len(metrics) >= 1
        latest = metrics[-1]
        assert latest["type"] == "metric"
        assert latest["name"] == "latency_ms"
        assert latest["value"] == 123.0
        assert latest["dimensions"]["service"] == "agent"

    def test_put_log_event(self, temp_cloudwatch_dir):
        """Test 2: Put a log event."""
        mock_cloud_watch.put_log({"level": "info", "msg": "hello", "request_id": "req-1"})

        logs = mock_cloud_watch.get_logs()
        assert len(logs) >= 1
        latest = logs[-1]
        assert latest["type"] == "log"
        assert latest["event"]["level"] == "info"
        assert latest["event"]["msg"] == "hello"
        assert latest["event"]["request_id"] == "req-1"

    def test_large_payload_handling(self, temp_cloudwatch_dir):
        """Test 3: Large-ish payload handling."""
        large_data = {"key": "x" * 10000, "nested": {"a": list(range(100))}}
        mock_cloud_watch.put_log(large_data)

        logs = mock_cloud_watch.get_logs()
        assert len(logs) >= 1

    def test_multiple_metrics(self, temp_cloudwatch_dir):
        """Test multiple metrics can be recorded."""
        mock_cloud_watch.put_metric("cpu_usage", 45.5, dimensions={"host": "server1"})
        mock_cloud_watch.put_metric("memory_usage", 78.2, dimensions={"host": "server1"})
        mock_cloud_watch.put_metric("disk_io", 120.0, dimensions={"host": "server1"})

        metrics = mock_cloud_watch.get_metrics()
        assert len(metrics) >= 3

    def test_metric_without_dimensions(self, temp_cloudwatch_dir):
        """Test metric without dimensions."""
        mock_cloud_watch.put_metric("simple_metric", 42.0)

        metrics = mock_cloud_watch.get_metrics()
        assert len(metrics) >= 1
        latest = metrics[-1]
        assert latest["dimensions"] == {}
