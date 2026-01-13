"""Pydantic schemas for Relevance Check service."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RelevanceCheckRequest(BaseModel):
    """Request schema for relevance check endpoint."""

    request_id: str = Field(..., description="Unique request identifier")
    question: str = Field(..., description="The question to check for relevance")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")


class RelevanceCheckResponse(BaseModel):
    """Response schema for relevance check endpoint."""

    request_id: str = Field(..., description="Echoed request identifier")
    relevant: bool = Field(..., description="Whether the question is relevant to the domain")
    reason: str = Field(..., description="Explanation for the decision")
    matched_domains: List[str] = Field(default_factory=list, description="Matched domain categories")
    model_id: Optional[str] = Field(default=None, description="ID of the model used for classification")
