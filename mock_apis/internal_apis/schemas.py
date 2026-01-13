"""Shared Pydantic schemas for internal mock APIs."""
from __future__ import annotations

from typing import Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T")


class Prisoner(BaseModel):
    prisoner_id: str
    name: str


class Conversation(BaseModel):
    conversation_id: str
    timestamp: str
    prisoner_ids: List[str]
    prisoner_names: List[str]
    transcript: str

    facility_id: str = ""
    facility_name: str = ""
    communication_type: Literal["inmate_call", "inmate_text"] = "inmate_call"
    call_duration_seconds: int = 0
    outside_contact_name: str = ""
    outside_contact_relation: str = ""

    alert_categories: List[str] = Field(default_factory=list)
    keyword_hits: List[str] = Field(default_factory=list)
    alert_confidence: float = 0.0
    review_status: Literal["unreviewed", "reviewed"] = "unreviewed"


class UserReport(BaseModel):
    report_id: str
    created_at: str
    title: str
    summary: str
    raw_text: str
    linked_prisoner_ids: List[str]
    linked_prisoner_names: List[str]
    linked_conversation_ids: List[str]
    tags: List[str] = Field(default_factory=list)

    report_type: Literal["alert_digest", "case_summary", "wellness_triage"] = "alert_digest"
    trigger_type: Literal["keyword_alert", "analyst_query", "case_followup"] = "keyword_alert"
    risk_level: Literal["low", "medium", "high"] = "low"
    confidence: float = 0.0
    alert_categories: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    key_excerpts: List[str] = Field(default_factory=list)
    audit_note: str = ""


class IncidentReport(BaseModel):
    incident_id: str
    date: str
    type: str
    severity: str
    description: str
    involved_prisoner_ids: List[str]
    involved_prisoner_names: List[str]

    facility_id: str = ""
    facility_name: str = ""
    location: str = ""
    shift: Literal["day", "evening", "night"] = "day"
    outcome: str = ""
    linked_conversation_ids: List[str] = Field(default_factory=list)
    linked_report_ids: List[str] = Field(default_factory=list)


class Page(BaseModel, Generic[T]):
    items: List[T]
    total: int
    limit: int
    offset: int


class TextSearchQuery(BaseModel):
    query: str
    top_k: int = 10
    min_score: Optional[float] = None


class TimeRange(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None


class SortSpec(BaseModel):
    sort_by: str = "timestamp"
    sort_order: Literal["asc", "desc"] = "desc"


class ErrorResponse(BaseModel):
    detail: str
