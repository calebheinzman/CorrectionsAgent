"""Mock DynamoDB - file-backed document store."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .transport import JsonStore


_store: Optional[JsonStore] = None


def _get_store() -> JsonStore:
    """Get or create the JSON store."""
    global _store
    if _store is None:
        base_dir = Path(__file__).parent / "data" / "dynamodb"
        env_path = os.getenv("MOCK_DYNAMODB_DIR")
        if env_path:
            base_dir = Path(env_path)
        _store = JsonStore(base_dir)
    return _store


def put_item(table: str, item: dict) -> None:
    """Put an item into a table."""
    store = _get_store()
    store.put_item(table, item)


def get_item(table: str, key: Dict[str, Any]) -> Optional[dict]:
    """Get an item by key. Returns None if not found."""
    store = _get_store()
    return store.get_item(table, key)


def query(
    table: str, *, key_prefix: Optional[str] = None, limit: int = 50
) -> List[dict]:
    """Query items from a table."""
    store = _get_store()
    return store.query(table, key_prefix=key_prefix, limit=limit)


def scan(table: str) -> List[dict]:
    """Scan all items from a table (no limit)."""
    store = _get_store()
    return store._load_table(table)


def reset_store(base_dir: Optional[Path] = None) -> None:
    """Reset the store (useful for testing)."""
    global _store
    if base_dir:
        _store = JsonStore(base_dir)
    else:
        _store = None
