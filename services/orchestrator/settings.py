"""Configuration settings for Orchestrator service."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServiceURLs:
    """URLs for downstream services."""

    safety_check: str = field(default_factory=lambda: os.getenv("SAFETY_CHECK_URL", "http://localhost:8010"))
    relevance_check: str = field(default_factory=lambda: os.getenv("RELEVANCE_CHECK_URL", "http://localhost:8011"))
    agent: str = field(default_factory=lambda: os.getenv("AGENT_URL", "http://localhost:8012"))


@dataclass
class TimeoutSettings:
    """Timeout settings for downstream calls."""

    safety_check_timeout: float = field(default_factory=lambda: float(os.getenv("SAFETY_CHECK_TIMEOUT", "5.0")))
    relevance_check_timeout: float = field(default_factory=lambda: float(os.getenv("RELEVANCE_CHECK_TIMEOUT", "5.0")))
    agent_timeout: float = field(default_factory=lambda: float(os.getenv("AGENT_TIMEOUT", "60.0")))


@dataclass
class RetrySettings:
    """Retry settings for downstream calls."""

    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "2")))
    retry_delay: float = field(default_factory=lambda: float(os.getenv("RETRY_DELAY", "0.5")))


@dataclass
class AuditSettings:
    """Audit store settings."""

    table_name: str = field(default_factory=lambda: os.getenv("AUDIT_TABLE_NAME", "orchestrator_audit"))
    enabled: bool = field(default_factory=lambda: os.getenv("AUDIT_ENABLED", "true").lower() == "true")


@dataclass
class OrchestratorSettings:
    """All orchestrator settings."""

    service_urls: ServiceURLs = field(default_factory=ServiceURLs)
    timeouts: TimeoutSettings = field(default_factory=TimeoutSettings)
    retries: RetrySettings = field(default_factory=RetrySettings)
    audit: AuditSettings = field(default_factory=AuditSettings)
    safety_enabled: bool = field(
        default_factory=lambda: os.getenv("ORCHESTRATOR_SAFETY_ENABLED", "true").lower() == "true"
    )
    relevance_enabled: bool = field(
        default_factory=lambda: os.getenv("ORCHESTRATOR_RELEVANCE_ENABLED", "true").lower() == "true"
    )
    port: int = field(default_factory=lambda: int(os.getenv("ORCHESTRATOR_PORT", "8000")))


_settings: Optional[OrchestratorSettings] = None


def get_settings() -> OrchestratorSettings:
    """Get the singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = OrchestratorSettings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
