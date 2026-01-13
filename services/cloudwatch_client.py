"""Centralized CloudWatch client for logging and tracing across all services."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional

from mock_apis.cloud_apis import mock_cloud_watch


class CloudWatchClient:
    """Client for logging and tracing to CloudWatch."""

    def __init__(self, service_name: str):
        """
        Initialize CloudWatch client.
        
        Args:
            service_name: Name of the service (e.g., 'orchestrator', 'agent', 'safety_check')
        """
        self.service_name = service_name
        self._enabled = os.getenv("CLOUDWATCH_ENABLED", "true").lower() == "true"

    def log(
        self,
        level: str,
        message: str,
        request_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """
        Log an event to CloudWatch.
        
        Args:
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            message: Log message
            request_id: Optional request ID for correlation
            **extra: Additional structured data to include
        """
        if not self._enabled:
            return

        event = {
            "service": self.service_name,
            "level": level,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if request_id:
            event["request_id"] = request_id

        if extra:
            event["extra"] = extra

        mock_cloud_watch.put_log(event)

    def log_info(self, message: str, request_id: Optional[str] = None, **extra: Any) -> None:
        """Log an INFO level message."""
        self.log("INFO", message, request_id, **extra)

    def log_warning(self, message: str, request_id: Optional[str] = None, **extra: Any) -> None:
        """Log a WARNING level message."""
        self.log("WARNING", message, request_id, **extra)

    def log_error(self, message: str, request_id: Optional[str] = None, **extra: Any) -> None:
        """Log an ERROR level message."""
        self.log("ERROR", message, request_id, **extra)

    def log_debug(self, message: str, request_id: Optional[str] = None, **extra: Any) -> None:
        """Log a DEBUG level message."""
        self.log("DEBUG", message, request_id, **extra)

    def metric(
        self,
        name: str,
        value: float,
        dimensions: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a metric to CloudWatch.
        
        Args:
            name: Metric name
            value: Metric value
            dimensions: Optional dimension key-value pairs
        """
        if not self._enabled:
            return

        dims = dimensions or {}
        dims["service"] = self.service_name

        mock_cloud_watch.put_metric(name, value, dims)

    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        request_id: Optional[str] = None,
        **extra: Any,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager for tracing an operation with timing.
        
        Args:
            operation_name: Name of the operation being traced
            request_id: Optional request ID for correlation
            **extra: Additional context to log
            
        Yields:
            Context dict that can be updated during the operation
            
        Example:
            with cw.trace_operation("agent_call", request_id="123") as ctx:
                result = do_work()
                ctx["result_size"] = len(result)
        """
        start_time = time.time()
        context: Dict[str, Any] = {}

        self.log_info(
            f"Starting {operation_name}",
            request_id=request_id,
            operation=operation_name,
            **extra,
        )

        try:
            yield context
            duration_ms = (time.time() - start_time) * 1000

            self.log_info(
                f"Completed {operation_name}",
                request_id=request_id,
                operation=operation_name,
                duration_ms=duration_ms,
                status="success",
                **context,
                **extra,
            )

            self.metric(
                f"{operation_name}.duration_ms",
                duration_ms,
                dimensions={"status": "success"},
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            self.log_error(
                f"Failed {operation_name}: {str(e)}",
                request_id=request_id,
                operation=operation_name,
                duration_ms=duration_ms,
                status="error",
                error_type=type(e).__name__,
                error_message=str(e),
                **context,
                **extra,
            )

            self.metric(
                f"{operation_name}.duration_ms",
                duration_ms,
                dimensions={"status": "error"},
            )

            raise

    def log_request(
        self,
        endpoint: str,
        request_id: str,
        **extra: Any,
    ) -> None:
        """
        Log an incoming HTTP request.
        
        Args:
            endpoint: API endpoint being called
            request_id: Request ID
            **extra: Additional request metadata
        """
        self.log_info(
            f"Request received: {endpoint}",
            request_id=request_id,
            event_type="request_received",
            endpoint=endpoint,
            **extra,
        )

    def log_response(
        self,
        endpoint: str,
        request_id: str,
        status_code: int,
        duration_ms: float,
        **extra: Any,
    ) -> None:
        """
        Log an HTTP response.
        
        Args:
            endpoint: API endpoint
            request_id: Request ID
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
            **extra: Additional response metadata
        """
        level = "INFO" if status_code < 400 else "ERROR"
        self.log(
            level,
            f"Response sent: {endpoint} [{status_code}]",
            request_id=request_id,
            event_type="response_sent",
            endpoint=endpoint,
            status_code=status_code,
            duration_ms=duration_ms,
            **extra,
        )

        self.metric(
            "http.request.duration_ms",
            duration_ms,
            dimensions={
                "endpoint": endpoint,
                "status_code": str(status_code),
            },
        )

    def log_guardrail_decision(
        self,
        guardrail_type: str,
        decision: bool,
        reason: str,
        request_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """
        Log a guardrail decision.
        
        Args:
            guardrail_type: Type of guardrail (safety, relevance)
            decision: Whether the check passed
            reason: Reason for the decision
            request_id: Request ID
            **extra: Additional context
        """
        self.log_info(
            f"Guardrail decision: {guardrail_type}={decision}",
            request_id=request_id,
            event_type="guardrail_decision",
            guardrail_type=guardrail_type,
            decision=decision,
            reason=reason,
            **extra,
        )

        self.metric(
            f"guardrail.{guardrail_type}.decision",
            1.0 if decision else 0.0,
            dimensions={"decision": "pass" if decision else "fail"},
        )

    def log_tool_call(
        self,
        tool_name: str,
        request_id: Optional[str] = None,
        success: bool = True,
        duration_ms: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """
        Log a tool call from the agent.
        
        Args:
            tool_name: Name of the tool called
            request_id: Request ID
            success: Whether the tool call succeeded
            duration_ms: Duration of the tool call
            **extra: Additional context
        """
        self.log_info(
            f"Tool call: {tool_name}",
            request_id=request_id,
            event_type="tool_call",
            tool_name=tool_name,
            success=success,
            duration_ms=duration_ms,
            **extra,
        )

        if duration_ms is not None:
            self.metric(
                "tool.call.duration_ms",
                duration_ms,
                dimensions={
                    "tool_name": tool_name,
                    "status": "success" if success else "error",
                },
            )


_clients: Dict[str, CloudWatchClient] = {}


def get_cloudwatch_client(service_name: str) -> CloudWatchClient:
    """
    Get or create a CloudWatch client for a service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        CloudWatchClient instance
    """
    if service_name not in _clients:
        _clients[service_name] = CloudWatchClient(service_name)
    return _clients[service_name]
