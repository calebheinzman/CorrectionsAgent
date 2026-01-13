"""Tests for Mock Model Registry."""
import json
import tempfile
from pathlib import Path

import pytest

from mock_apis.cloud_apis import mock_model_registry


@pytest.fixture
def temp_registry_dir():
    """Create a temporary directory for registry storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_model_registry.reset_registry(Path(tmpdir))
        yield Path(tmpdir)
        mock_model_registry.reset_registry(None)


class TestMockModelRegistry:
    """Test cases for Mock Model Registry."""

    def test_get_current_model(self, temp_registry_dir):
        """Test 1: Get current model."""
        model_data = {
            "provider": "google",
            "model_id": "gemini-2.5-flash",
            "version": "1.0.0",
        }
        (temp_registry_dir / "current_model.json").write_text(
            json.dumps(model_data), encoding="utf-8"
        )

        result = mock_model_registry.get_current_model()
        assert result is not None
        assert result["provider"] == "google"
        assert result["model_id"] == "gemini-2.5-flash"
        assert result["version"] == "1.0.0"

    def test_get_policy_bundle(self, temp_registry_dir):
        """Test 2: Get policy bundle."""
        bundle_data = {
            "bundle_id": "safety-policy-v1",
            "version": "2.0.0",
        }
        (temp_registry_dir / "policy_bundle.json").write_text(
            json.dumps(bundle_data), encoding="utf-8"
        )

        result = mock_model_registry.get_policy_bundle()
        assert result is not None
        assert result["bundle_id"] == "safety-policy-v1"
        assert result["version"] == "2.0.0"

    def test_missing_registry_data(self, temp_registry_dir):
        """Test 3: Missing registry data returns None."""
        result_model = mock_model_registry.get_current_model()
        result_bundle = mock_model_registry.get_policy_bundle()
        assert result_model is None
        assert result_bundle is None

    def test_set_current_model(self, temp_registry_dir):
        """Test set_current_model helper."""
        mock_model_registry.set_current_model(
            provider="openai", model_id="gpt-4", version="3.0.0"
        )
        result = mock_model_registry.get_current_model()
        assert result is not None
        assert result["provider"] == "openai"
        assert result["model_id"] == "gpt-4"
        assert result["version"] == "3.0.0"

    def test_task_scoped_current_models(self, temp_registry_dir):
        """Task-scoped current models should not overwrite each other."""
        mock_model_registry.set_current_model(
            provider="local_nb",
            model_id="relevance_check_123",
            version="1",
            task="relevance_check",
        )
        mock_model_registry.set_current_model(
            provider="local_nb",
            model_id="safety_check_456",
            version="1",
            task="safety_check",
        )

        rel = mock_model_registry.get_current_model(task="relevance_check")
        saf = mock_model_registry.get_current_model(task="safety_check")
        assert rel is not None
        assert saf is not None
        assert rel["model_id"] == "relevance_check_123"
        assert saf["model_id"] == "safety_check_456"

    def test_set_policy_bundle(self, temp_registry_dir):
        """Test set_policy_bundle helper."""
        mock_model_registry.set_policy_bundle(bundle_id="test-bundle", version="1.0.0")
        result = mock_model_registry.get_policy_bundle()
        assert result is not None
        assert result["bundle_id"] == "test-bundle"
        assert result["version"] == "1.0.0"
