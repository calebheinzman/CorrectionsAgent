"""Pydantic schemas for Agent service."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    """Record of a single tool call made by the agent."""

    tool_name: str = Field(..., description="Name of the tool called")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input parameters")
    output_size: int = Field(default=0, description="Size of output in characters")
    latency_ms: float = Field(default=0.0, description="Latency in milliseconds")
    success: bool = Field(default=True, description="Whether the call succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class Citation(BaseModel):
    """A citation to source data."""

    source_type: str = Field(..., description="Type of source (conversation, incident, report, prisoner)")
    source_id: str = Field(..., description="ID of the source document")
    excerpt: Optional[str] = Field(default=None, description="Relevant excerpt from source")


class ToolConfig(BaseModel):
    """Configuration for tool behavior."""

    enabled_tools: Optional[List[str]] = Field(default=None, description="List of enabled tools (None = all)")
    max_tool_calls: int = Field(default=10, description="Maximum number of tool calls per request")
    tool_timeout_seconds: float = Field(default=30.0, description="Timeout for each tool call")


class AgentRequest(BaseModel):
    """Request schema for agent endpoint."""

    request_id: str = Field(..., description="Unique request identifier")
    question: str = Field(..., description="The question to answer")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")
    session_id: Optional[str] = Field(default=None, description="Optional session identifier for conversation state")
    tool_config: Optional[ToolConfig] = Field(default=None, description="Optional tool configuration")


class ModelInfo(BaseModel):
    """Information about the model used."""

    model_name: str = Field(..., description="Name of the model")
    provider: str = Field(..., description="Model provider")


class AgentResponse(BaseModel):
    """Response schema for agent endpoint."""

    request_id: str = Field(..., description="Echoed request identifier")
    answer: str = Field(..., description="The generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Citations to source data")
    tool_calls: List[ToolCallRecord] = Field(default_factory=list, description="Record of tool calls made")
    model_info: Optional[ModelInfo] = Field(default=None, description="Information about the model used")
