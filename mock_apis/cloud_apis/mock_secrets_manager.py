"""Mock Secrets Manager - pulls secrets from .env file."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


_env_loaded = False


def _ensure_env_loaded() -> None:
    """Load .env file if not already loaded."""
    global _env_loaded
    if not _env_loaded:
        repo_root = Path(__file__).resolve().parents[2]
        env_path = repo_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        _env_loaded = True


def get_secret(name: str) -> Optional[str]:
    """
    Get a secret by name from environment variables.
    
    Returns None if the secret is not found. Never invents secrets.
    """
    _ensure_env_loaded()
    value = os.getenv(name)
    if value is not None and value.strip():
        return value
    return None


def reload_env() -> None:
    """Force reload of .env file."""
    global _env_loaded
    _env_loaded = False
    _ensure_env_loaded()
