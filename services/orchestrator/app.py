"""FastAPI application for Orchestrator service."""
from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .handler import handle_query
from .logs.logger import setup_logging
from .middleware.auth import AuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.request_id import RequestIdMiddleware, generate_request_id
from .schemas import QueryRequest, QueryResponse
from .settings import get_settings

app = FastAPI(
    title="Orchestrator Service",
    description="Main orchestration service for query processing",
    version="0.1.0",
)

cors_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

cors_allow_origin_regex = os.getenv(
    "CORS_ALLOW_ORIGIN_REGEX",
    r"^https?://(localhost|127\.0\.0\.1):\d+$",
).strip()

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_allow_origin_regex if cors_allow_origin_regex else None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestIdMiddleware)
app.add_middleware(AuthMiddleware, require_auth=False)
app.add_middleware(RateLimitMiddleware, enabled=False)


@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    from services.cloudwatch_client import get_cloudwatch_client
    cw = get_cloudwatch_client("orchestrator")
    cw.log_info("Orchestrator service starting up")
    setup_logging()
    cw.log_info("Orchestrator service started successfully")


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/v1/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, http_request: Request) -> QueryResponse:
    """
    Process a query through the full pipeline.

    Flow:
    1. Safety check - deny unsafe queries
    2. Relevance check - deny out-of-domain queries
    3. Agent call - answer approved queries
    4. Return response with guardrail decisions and answer
    """
    request_id = getattr(http_request.state, "request_id", None) or generate_request_id()
    return handle_query(request, request_id)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
