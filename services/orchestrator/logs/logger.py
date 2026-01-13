"""JSON structured logging configuration."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..middleware.request_id import get_request_id
from services.cloudwatch_client import get_cloudwatch_client


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = get_request_id()
        if request_id:
            log_data["request_id"] = request_id

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


_cloudwatch_client = None


def get_cloudwatch() -> Any:
    """Get the CloudWatch client for orchestrator."""
    global _cloudwatch_client
    if _cloudwatch_client is None:
        _cloudwatch_client = get_cloudwatch_client("orchestrator")
    return _cloudwatch_client


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up JSON structured logging."""
    logger = logging.getLogger("orchestrator")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    get_cloudwatch()

    return logger


def get_logger() -> logging.Logger:
    """Get the orchestrator logger."""
    return logging.getLogger("orchestrator")


def log_request(
    question: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Log an incoming request."""
    logger = get_logger()
    logger.info(
        "Request received",
        extra={
            "extra_data": {
                "event": "request_received",
                "question_length": len(question),
                "user_id": user_id,
                "session_id": session_id,
            }
        },
    )
    
    cw = get_cloudwatch()
    cw.log_info(
        "Request received",
        request_id=get_request_id(),
        question_length=len(question),
        user_id=user_id,
        session_id=session_id,
    )


def log_guardrail_decision(
    guardrail: str,
    decision: bool,
    reason: str,
) -> None:
    """Log a guardrail decision."""
    logger = get_logger()
    logger.info(
        f"Guardrail {guardrail} decision: {decision}",
        extra={
            "extra_data": {
                "event": "guardrail_decision",
                "guardrail": guardrail,
                "decision": decision,
                "reason": reason,
            }
        },
    )
    
    cw = get_cloudwatch()
    cw.log_guardrail_decision(
        guardrail_type=guardrail,
        decision=decision,
        reason=reason,
        request_id=get_request_id(),
    )


def log_agent_call(
    tool_calls_count: int,
    latency_ms: float,
) -> None:
    """Log an agent call."""
    logger = get_logger()
    logger.info(
        f"Agent call completed with {tool_calls_count} tool calls",
        extra={
            "extra_data": {
                "event": "agent_call",
                "tool_calls_count": tool_calls_count,
                "latency_ms": latency_ms,
            }
        },
    )
    
    cw = get_cloudwatch()
    cw.log_info(
        f"Agent call completed with {tool_calls_count} tool calls",
        request_id=get_request_id(),
        tool_calls_count=tool_calls_count,
        latency_ms=latency_ms,
    )
    cw.metric("agent_call.duration_ms", latency_ms)
    cw.metric("agent_call.tool_calls", float(tool_calls_count))


def log_error(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an error."""
    logger = get_logger()
    logger.error(
        message,
        extra={
            "extra_data": {
                "event": "error",
                "error_code": error_code,
                "details": details,
            }
        },
    )
    
    cw = get_cloudwatch()
    cw.log_error(
        message,
        request_id=get_request_id(),
        error_code=error_code,
        details=details,
    )
