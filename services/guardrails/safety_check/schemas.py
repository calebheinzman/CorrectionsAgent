"""Pydantic schemas for Safety Check service."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SafetyCheckRequest(BaseModel):
    """Request schema for safety check endpoint."""

    request_id: str = Field(..., description="Unique request identifier")
    question: str = Field(..., description="The question to check for safety")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")


class SafetyCheckResponse(BaseModel):
    """Response schema for safety check endpoint."""

    request_id: str = Field(..., description="Echoed request identifier")
    allowed: bool = Field(..., description="Whether the question is allowed")
    policy: str = Field(..., description="Policy that was applied")
    reason: str = Field(..., description="Explanation for the decision")
    categories: List[str] = Field(default_factory=list, description="Matched safety categories")
    model_id: Optional[str] = Field(default=None, description="ID of the model used for classification")
