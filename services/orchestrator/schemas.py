"""Pydantic schemas for Orchestrator service."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GuardrailDecision(BaseModel):
    """Decision from a guardrail service."""

    allowed: Optional[bool] = Field(default=None, description="Whether allowed (safety)")
    relevant: Optional[bool] = Field(default=None, description="Whether relevant (relevance)")
    reason: str = Field(default="", description="Explanation for the decision")
    policy: Optional[str] = Field(default=None, description="Policy applied (safety)")
    categories: List[str] = Field(default_factory=list, description="Matched categories")
    matched_domains: List[str] = Field(default_factory=list, description="Matched domains (relevance)")
    model_id: Optional[str] = Field(default=None, description="Model identifier used by the guardrail")


class AgentTrace(BaseModel):
    """Trace information from agent execution."""

    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Tool calls made")
    model_info: Optional[Dict[str, str]] = Field(default=None, description="Model information")


class Citation(BaseModel):
    """A citation to source data."""

    source_type: str = Field(..., description="Type of source")
    source_id: str = Field(..., description="ID of the source document")
    excerpt: Optional[str] = Field(default=None, description="Relevant excerpt")


class QueryRequest(BaseModel):
    """Request schema for query endpoint."""

    question: str = Field(..., description="The question to answer")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")
    session_id: Optional[str] = Field(default=None, description="Optional session identifier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class QueryResponse(BaseModel):
    """Response schema for query endpoint."""

    request_id: str = Field(..., description="Unique request identifier")
    status: str = Field(..., description="Status of the request (success, denied, error)")
    answer: Optional[str] = Field(default=None, description="The generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Citations to source data")
    safety: Optional[GuardrailDecision] = Field(default=None, description="Safety check decision")
    relevance: Optional[GuardrailDecision] = Field(default=None, description="Relevance check decision")
    agent_trace: Optional[AgentTrace] = Field(default=None, description="Agent execution trace")


class ErrorResponse(BaseModel):
    """Error response schema."""

    status: str = Field(default="error", description="Status indicator")
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    request_id: str = Field(..., description="Request identifier")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
