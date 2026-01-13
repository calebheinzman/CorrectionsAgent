"""FastAPI application for Safety Check service."""
from __future__ import annotations

import os

from fastapi import FastAPI

from .schemas import SafetyCheckRequest, SafetyCheckResponse
from .service import check_safety, get_service
from services.cloudwatch_client import get_cloudwatch_client

app = FastAPI(
    title="Safety Check Service",
    description="Guardrail service for checking question safety",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    cw = get_cloudwatch_client("safety_check")
    cw.log_info("Safety check service starting up")
    get_service()
    cw.log_info("Safety check service started successfully")


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/v1/safety/check", response_model=SafetyCheckResponse)
async def safety_check_endpoint(request: SafetyCheckRequest) -> SafetyCheckResponse:
    """
    Check if a question is safe to process.

    Returns a decision with policy, reason, and matched categories.
    """
    return check_safety(request)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SAFETY_CHECK_PORT", "8010"))
    uvicorn.run(app, host="0.0.0.0", port=port)
