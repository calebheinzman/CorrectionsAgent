"""Minimal auth middleware for MVP."""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

AUTH_HEADER = "X-Auth-Token"
SKIP_AUTH_PATHS = {"/healthz", "/docs", "/openapi.json", "/redoc"}


class AuthMiddleware(BaseHTTPMiddleware):
    """Minimal auth middleware that checks for presence of auth header."""

    def __init__(self, app, require_auth: bool = False):
        super().__init__(app)
        self._require_auth = require_auth

    async def dispatch(self, request: Request, call_next):
        if request.url.path in SKIP_AUTH_PATHS:
            return await call_next(request)

        if self._require_auth:
            auth_token = request.headers.get(AUTH_HEADER)
            if not auth_token:
                return JSONResponse(
                    status_code=401,
                    content={
                        "status": "error",
                        "code": "UNAUTHORIZED",
                        "message": "Missing authentication token",
                        "request_id": getattr(request.state, "request_id", "unknown"),
                    },
                )

        return await call_next(request)
