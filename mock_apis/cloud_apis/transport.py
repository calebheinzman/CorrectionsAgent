"""Shared transport abstraction for mock cloud services."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class FileBackedStore:
    """Simple file-backed key-value store."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, *parts: str) -> Path:
        """Get file path for given key parts."""
        path = self.base_dir
        for part in parts:
            safe_part = part.replace("/", "_").replace("\\", "_")
            path = path / safe_part
        return path

    def put(self, *key_parts: str, data: bytes) -> None:
        """Store bytes at the given key path."""
        path = self._get_path(*key_parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, *key_parts: str) -> Optional[bytes]:
        """Retrieve bytes from the given key path."""
        path = self._get_path(*key_parts)
        if path.exists() and path.is_file():
            return path.read_bytes()
        return None

    def delete(self, *key_parts: str) -> bool:
        """Delete the file at the given key path."""
        path = self._get_path(*key_parts)
        if path.exists() and path.is_file():
            path.unlink()
            return True
        return False

    def list_keys(self, *prefix_parts: str) -> list[str]:
        """List all keys under the given prefix."""
        path = self._get_path(*prefix_parts) if prefix_parts else self.base_dir
        if not path.exists():
            return []
        keys = []
        for item in path.rglob("*"):
            if item.is_file():
                rel = item.relative_to(self.base_dir)
                keys.append(str(rel))
        return keys


class JsonStore:
    """JSON document store backed by files."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, table: str) -> Path:
        """Get file path for a table."""
        return self.base_dir / f"{table}.json"

    def _load_table(self, table: str) -> list[dict]:
        """Load all items from a table."""
        path = self._get_path(table)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return []

    def _save_table(self, table: str, items: list[dict]) -> None:
        """Save all items to a table."""
        path = self._get_path(table)
        path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    def put_item(self, table: str, item: dict) -> None:
        """Put an item into the table."""
        items = self._load_table(table)
        items.append(item)
        self._save_table(table, items)

    def get_item(self, table: str, key: Dict[str, Any]) -> Optional[dict]:
        """Get an item by key."""
        items = self._load_table(table)
        for item in items:
            if all(item.get(k) == v for k, v in key.items()):
                return item
        return None

    def query(
        self, table: str, *, key_prefix: Optional[str] = None, limit: int = 50
    ) -> list[dict]:
        """Query items from a table."""
        items = self._load_table(table)
        if key_prefix:
            pk_field = "pk"
            items = [i for i in items if str(i.get(pk_field, "")).startswith(key_prefix)]
        return items[:limit]


class AppendOnlyLog:
    """Append-only log backed by JSONL files."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, log_name: str) -> Path:
        """Get file path for a log."""
        return self.base_dir / f"{log_name}.jsonl"

    def append(self, log_name: str, event: dict) -> None:
        """Append an event to the log."""
        path = self._get_path(log_name)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def read_all(self, log_name: str) -> list[dict]:
        """Read all events from a log."""
        path = self._get_path(log_name)
        if not path.exists():
            return []
        events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events
