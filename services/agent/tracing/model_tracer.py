"""Model call tracing for the agent."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelCall:
    """A record of a model invocation."""

    model_name: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class ModelTrace:
    """A trace of model calls made during agent execution."""

    request_id: str
    model_calls: List[ModelCall] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def record_call(
        self,
        model_name: str,
        latency_ms: float,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record a model call."""
        self.model_calls.append(
            ModelCall(
                model_name=model_name,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=success,
                error=error,
            )
        )

    def finish(self) -> None:
        """Mark the trace as finished."""
        self.end_time = time.time()

    def get_total_latency_ms(self) -> float:
        """Get total latency of all model calls."""
        return sum(mc.latency_ms for mc in self.model_calls)

    def get_model_call_count(self) -> int:
        """Get the number of model calls."""
        return len(self.model_calls)


class ModelTracer:
    """Tracer for recording model calls."""

    def __init__(self):
        self._traces: Dict[str, ModelTrace] = {}

    def start_trace(self, request_id: str) -> ModelTrace:
        """Start a new trace for a request."""
        trace = ModelTrace(request_id=request_id)
        self._traces[request_id] = trace
        return trace

    def get_trace(self, request_id: str) -> Optional[ModelTrace]:
        """Get an existing trace."""
        return self._traces.get(request_id)

    def finish_trace(self, request_id: str) -> Optional[ModelTrace]:
        """Finish and return a trace."""
        trace = self._traces.get(request_id)
        if trace:
            trace.finish()
        return trace

    def clear_trace(self, request_id: str) -> None:
        """Clear a trace from memory."""
        self._traces.pop(request_id, None)


_global_tracer: Optional[ModelTracer] = None


def get_tracer() -> ModelTracer:
    """Get the global model tracer."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = ModelTracer()
    return _global_tracer
