"""Safety policy definitions and category metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

POLICY_VERSION = "v1.0"


@dataclass(frozen=True)
class SafetyCategory:
    """A safety category definition."""

    name: str
    description: str
    keywords: List[str]


SAFETY_CATEGORIES: Dict[str, SafetyCategory] = {
    "sensitive_info": SafetyCategory(
        name="sensitive_info",
        description="Requests for sensitive system information like API keys, credentials, or internal configurations",
        keywords=[
            "api key",
            "api_key",
            "apikey",
            "secret",
            "password",
            "credential",
            "token",
            "private key",
            "access key",
            "secret key",
            "auth token",
            "bearer token",
            "jwt",
            "connection string",
            "database password",
            "admin password",
            "root password",
        ],
    ),
    "system_internals": SafetyCategory(
        name="system_internals",
        description="Requests about internal system architecture, implementation details, or security mechanisms",
        keywords=[
            "system prompt",
            "internal architecture",
            "source code",
            "implementation detail",
            "security mechanism",
            "bypass security",
            "ignore instructions",
            "override",
            "jailbreak",
            "prompt injection",
        ],
    ),
    "harmful_intent": SafetyCategory(
        name="harmful_intent",
        description="Requests that could facilitate harm to individuals or the system",
        keywords=[
            "hack",
            "exploit",
            "vulnerability",
            "attack",
            "breach",
            "unauthorized access",
            "delete all",
            "drop table",
            "sql injection",
        ],
    ),
}


def get_policy_version() -> str:
    """Return the current policy version."""
    return POLICY_VERSION


def get_all_categories() -> List[str]:
    """Return all category names."""
    return list(SAFETY_CATEGORIES.keys())


def get_category(name: str) -> SafetyCategory | None:
    """Get a category by name."""
    return SAFETY_CATEGORIES.get(name)
