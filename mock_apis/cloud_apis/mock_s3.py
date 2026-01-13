"""Mock S3 - file-backed object storage."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from .transport import FileBackedStore


_store: Optional[FileBackedStore] = None


def _get_store() -> FileBackedStore:
    """Get or create the file-backed store."""
    global _store
    if _store is None:
        base_dir = Path(__file__).parent / "data" / "s3"
        env_path = os.getenv("MOCK_S3_DIR")
        if env_path:
            base_dir = Path(env_path)
        _store = FileBackedStore(base_dir)
    return _store


def put_object(bucket: str, key: str, data: bytes) -> None:
    """Store an object."""
    store = _get_store()
    store.put(bucket, key, data=data)


def get_object(bucket: str, key: str) -> Optional[bytes]:
    """Retrieve an object. Returns None if not found."""
    store = _get_store()
    return store.get(bucket, key)


def list_objects(bucket: str, prefix: str = "") -> List[str]:
    """List object keys in a bucket, optionally filtered by prefix."""
    store = _get_store()
    all_keys = store.list_keys(bucket)
    if prefix:
        filtered = []
        for k in all_keys:
            key_without_bucket = k[len(bucket) + 1:] if k.startswith(bucket + "/") else k
            if key_without_bucket.startswith(prefix):
                filtered.append(k)
        return filtered
    return all_keys


def delete_object(bucket: str, key: str) -> bool:
    """Delete an object. Returns True if deleted."""
    store = _get_store()
    return store.delete(bucket, key)


def reset_store(base_dir: Optional[Path] = None) -> None:
    """Reset the store (useful for testing)."""
    global _store
    if base_dir:
        _store = FileBackedStore(base_dir)
    else:
        _store = None
