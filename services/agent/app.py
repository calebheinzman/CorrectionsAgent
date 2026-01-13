"""FastAPI application for Agent service."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from .schemas import AgentRequest, AgentResponse
from .service import answer, get_service
from services.cloudwatch_client import get_cloudwatch_client

app = FastAPI(
    title="Agent Service",
    description="LLM-powered agent for answering investigative questions",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    cw = get_cloudwatch_client("agent")
    cw.log_info("Agent service starting up")
    
    service = get_service()
    if not service.is_available():
        agent_error = getattr(service, "get_init_error", lambda: None)()
        llm_error = getattr(service._client, "get_init_error", lambda: None)()
        details = agent_error or llm_error
        if details:
            cw.log_warning("Agent service started but not available", details=details)
            print(
                "WARNING: Agent service started but is not available. "
                f"Details: {details}"
            )
        else:
            cw.log_warning("Agent service started but not available - check API key")
            print("WARNING: Agent service started but is not available. Check API key configuration.")
    else:
        cw.log_info("Agent service started successfully")


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/v1/answer", response_model=AgentResponse)
async def answer_endpoint(request: AgentRequest) -> AgentResponse:
    """
    Answer a question using the agent.

    The agent will use available tools to search for relevant information
    and provide a grounded answer with citations.
    """
    try:
        return answer(request)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("AGENT_PORT", "8012"))
    uvicorn.run(app, host="0.0.0.0", port=port)
