"""FastAPI application for Relevance Check service."""
from __future__ import annotations

import os

from fastapi import FastAPI

from .schemas import RelevanceCheckRequest, RelevanceCheckResponse
from .service import check_relevance, get_service
from services.cloudwatch_client import get_cloudwatch_client

app = FastAPI(
    title="Relevance Check Service",
    description="Guardrail service for checking question relevance to domain",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    cw = get_cloudwatch_client("relevance_check")
    cw.log_info("Relevance check service starting up")
    get_service()
    cw.log_info("Relevance check service started successfully")


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    svc = get_service()
    return {
        "status": "ok",
        "has_model": getattr(svc, "_model", None) is not None,
        "model_id": getattr(svc, "_model_id", None),
        "load_error": getattr(svc, "_load_error", None),
    }


@app.post("/v1/relevance/check", response_model=RelevanceCheckResponse)
async def relevance_check_endpoint(request: RelevanceCheckRequest) -> RelevanceCheckResponse:
    """
    Check if a question is relevant to the domain.

    Returns a decision with reason and matched domains.
    """
    return check_relevance(request)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("RELEVANCE_CHECK_PORT", "8011"))
    uvicorn.run(app, host="0.0.0.0", port=port)
