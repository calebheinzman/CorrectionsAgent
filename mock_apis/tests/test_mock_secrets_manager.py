"""Tests for Mock Secrets Manager."""
import os
import pytest

from mock_apis.cloud_apis import mock_secrets_manager


class TestMockSecretsManager:
    """Test cases for Mock Secrets Manager."""

    def test_get_existing_api_key(self):
        """Test 1: Get existing API key (if set in environment)."""
        mock_secrets_manager.reload_env()
        result = mock_secrets_manager.get_secret("GEMINI_API_KEY")
        if os.getenv("GEMINI_API_KEY"):
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0

    def test_fallback_key_name(self):
        """Test 2: Fallback key name - GOOGLE_API_KEY."""
        mock_secrets_manager.reload_env()
        result = mock_secrets_manager.get_secret("GOOGLE_API_KEY")
        if os.getenv("GOOGLE_API_KEY"):
            assert result is not None
            assert isinstance(result, str)

    def test_missing_key(self):
        """Test 3: Missing key returns None."""
        mock_secrets_manager.reload_env()
        result = mock_secrets_manager.get_secret("DOES_NOT_EXIST_KEY_12345")
        assert result is None

    def test_no_crash_on_missing_env(self):
        """Test 4: No crash when .env is missing or empty."""
        mock_secrets_manager.reload_env()
        result = mock_secrets_manager.get_secret("SOME_RANDOM_KEY")
        assert result is None
