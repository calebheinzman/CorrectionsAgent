"""Minimal rate limiting middleware for MVP."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

SKIP_RATE_LIMIT_PATHS = {"/healthz", "/docs", "/openapi.json", "/redoc"}


class TokenBucket:
    """Simple token bucket rate limiter."""

    def __init__(self, capacity: int, refill_rate: float):
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._buckets: Dict[str, Tuple[float, float]] = defaultdict(
            lambda: (float(capacity), time.time())
        )

    def consume(self, key: str, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Returns True if tokens were consumed, False if rate limited.
        """
        current_tokens, last_update = self._buckets[key]
        now = time.time()

        elapsed = now - last_update
        current_tokens = min(self._capacity, current_tokens + elapsed * self._refill_rate)

        if current_tokens >= tokens:
            self._buckets[key] = (current_tokens - tokens, now)
            return True
        else:
            self._buckets[key] = (current_tokens, now)
            return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""

    def __init__(self, app, capacity: int = 100, refill_rate: float = 10.0, enabled: bool = False):
        super().__init__(app)
        self._bucket = TokenBucket(capacity, refill_rate)
        self._enabled = enabled

    def _get_client_key(self, request: Request) -> str:
        """Get a key to identify the client."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    async def dispatch(self, request: Request, call_next):
        if not self._enabled or request.url.path in SKIP_RATE_LIMIT_PATHS:
            return await call_next(request)

        client_key = self._get_client_key(request)
        if not self._bucket.consume(client_key):
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "code": "RATE_LIMITED",
                    "message": "Too many requests",
                    "request_id": getattr(request.state, "request_id", "unknown"),
                },
            )

        return await call_next(request)
