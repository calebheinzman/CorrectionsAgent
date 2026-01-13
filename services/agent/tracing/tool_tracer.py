"""Tool call tracing for the agent."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..schemas import ToolCallRecord


@dataclass
class ToolTrace:
    """A trace of tool calls made during agent execution."""

    request_id: str
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def record_call(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        output: Any,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record a tool call."""
        output_str = json.dumps(output) if output else ""
        self.tool_calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                inputs=inputs,
                output_size=len(output_str),
                latency_ms=latency_ms,
                success=success,
                error=error,
            )
        )

    def finish(self) -> None:
        """Mark the trace as finished."""
        self.end_time = time.time()

    def get_total_latency_ms(self) -> float:
        """Get total latency of all tool calls."""
        return sum(tc.latency_ms for tc in self.tool_calls)

    def get_tool_call_count(self) -> int:
        """Get the number of tool calls."""
        return len(self.tool_calls)


class ToolTracer:
    """Tracer for recording tool calls."""

    def __init__(self):
        self._traces: Dict[str, ToolTrace] = {}

    def start_trace(self, request_id: str) -> ToolTrace:
        """Start a new trace for a request."""
        trace = ToolTrace(request_id=request_id)
        self._traces[request_id] = trace
        return trace

    def get_trace(self, request_id: str) -> Optional[ToolTrace]:
        """Get an existing trace."""
        return self._traces.get(request_id)

    def finish_trace(self, request_id: str) -> Optional[ToolTrace]:
        """Finish and return a trace."""
        trace = self._traces.get(request_id)
        if trace:
            trace.finish()
        return trace

    def clear_trace(self, request_id: str) -> None:
        """Clear a trace from memory."""
        self._traces.pop(request_id, None)


_global_tracer: Optional[ToolTracer] = None


def get_tracer() -> ToolTracer:
    """Get the global tool tracer."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = ToolTracer()
    return _global_tracer
