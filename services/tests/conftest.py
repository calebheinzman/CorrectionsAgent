"""Pytest configuration for services tests."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    import services.orchestrator.settings as settings
    import services.orchestrator.clients.safety_check_client as safety_client
    import services.orchestrator.clients.relevance_check_client as relevance_client
    import services.orchestrator.clients.agent_client as agent_client
    import services.orchestrator.clients.audit_store_client as audit_client
    import services.orchestrator.clients.registry_client as registry_client
    import services.orchestrator.handler as handler

    settings._settings = None
    safety_client._client_instance = None
    relevance_client._client_instance = None
    agent_client._client_instance = None
    audit_client._client_instance = None
    registry_client._client_instance = None
    handler._handler_instance = None

    yield

    settings._settings = None
    safety_client._client_instance = None
    relevance_client._client_instance = None
    agent_client._client_instance = None
    audit_client._client_instance = None
    registry_client._client_instance = None
    handler._handler_instance = None
