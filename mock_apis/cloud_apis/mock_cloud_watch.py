"""Mock CloudWatch - file-backed metrics and logging."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .transport import AppendOnlyLog


_log: Optional[AppendOnlyLog] = None


def _get_log() -> AppendOnlyLog:
    """Get or create the append-only log."""
    global _log
    if _log is None:
        base_dir = Path(__file__).parent / "data" / "cloudwatch"
        env_path = os.getenv("MOCK_CLOUDWATCH_DIR")
        if env_path:
            base_dir = Path(env_path)
        _log = AppendOnlyLog(base_dir)
    return _log


def put_metric(
    name: str, value: float, dimensions: Optional[Dict[str, str]] = None
) -> None:
    """
    Record a metric.
    
    Args:
        name: Metric name
        value: Metric value
        dimensions: Optional dimension key-value pairs
    """
    log = _get_log()
    event = {
        "type": "metric",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "value": value,
        "dimensions": dimensions or {},
    }
    log.append("metrics", event)


def put_log(event: dict) -> None:
    """
    Record a log event.
    
    Args:
        event: Log event dictionary
    """
    log = _get_log()
    record = {
        "type": "log",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
    }
    log.append("logs", record)


def get_metrics() -> list[dict]:
    """Read all recorded metrics."""
    log = _get_log()
    return log.read_all("metrics")


def get_logs() -> list[dict]:
    """Read all recorded log events."""
    log = _get_log()
    return log.read_all("logs")


def reset_log(base_dir: Optional[Path] = None) -> None:
    """Reset the log (useful for testing)."""
    global _log
    if base_dir:
        _log = AppendOnlyLog(base_dir)
    else:
        _log = None
