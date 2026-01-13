"""User Report API endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from .schemas import Page, UserReport

router = APIRouter(prefix="/user-reports", tags=["user-reports"])

_user_reports: List[UserReport] = []
_vector_store = None


def load_user_reports(data: List[dict]) -> None:
    """Load user report data from JSON."""
    global _user_reports
    _user_reports = [UserReport.model_validate(r) for r in data]


def set_vector_store(vs) -> None:
    """Set the vector store for similarity search."""
    global _vector_store
    _vector_store = vs


def get_all_user_reports() -> List[UserReport]:
    """Return all loaded user reports."""
    return _user_reports


def _parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


@router.get("/{report_id}", response_model=UserReport)
def get_user_report_by_id(report_id: str) -> UserReport:
    """Get a single user report by ID."""
    for r in _user_reports:
        if r.report_id == report_id:
            return r
    raise HTTPException(status_code=404, detail=f"User report {report_id} not found")


@router.get("", response_model=Page[UserReport])
def list_user_reports(
    report_id: Optional[str] = Query(None, description="Exact report ID"),
    prisoner_id: Optional[List[str]] = Query(None, description="Filter by linked prisoner ID(s)"),
    prisoner_name: Optional[str] = Query(None, description="Case-insensitive prisoner name contains"),
    conversation_id: Optional[str] = Query(None, description="Filter by linked conversation ID"),
    risk_level: Optional[str] = Query(None, description="low, medium, or high"),
    alert_category: Optional[List[str]] = Query(None, description="Filter by alert category"),
    tag: Optional[List[str]] = Query(None, description="Filter by tag"),
    start_time: Optional[str] = Query(None, description="ISO timestamp start (inclusive)"),
    end_time: Optional[str] = Query(None, description="ISO timestamp end (inclusive)"),
    query: Optional[str] = Query(None, description="Free-text similarity search over summary"),
    top_k: int = Query(10, ge=1, le=100, description="Max results for similarity search"),
    min_score: Optional[float] = Query(None, description="Min similarity score"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_order: str = Query("desc", description="asc or desc"),
) -> Page[UserReport]:
    """List user reports with filters, similarity search, pagination, and sorting."""

    if start_time and end_time:
        try:
            start_dt = _parse_timestamp(start_time)
            end_dt = _parse_timestamp(end_time)
            if start_dt > end_dt:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid time range: start_time must be before end_time",
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {e}")

    if query and _vector_store:
        try:
            docs = _vector_store.similarity_search_with_score(query, k=top_k)
            report_ids = []
            for doc, score in docs:
                if min_score is not None and score < min_score:
                    continue
                rid = doc.metadata.get("report_id")
                if rid:
                    report_ids.append(rid)
            results = [r for r in _user_reports if r.report_id in report_ids]
            results_order = {rid: i for i, rid in enumerate(report_ids)}
            results.sort(key=lambda r: results_order.get(r.report_id, 999))
        except Exception:
            results = list(_user_reports)
    else:
        results = list(_user_reports)

    if report_id:
        results = [r for r in results if r.report_id == report_id]

    if prisoner_id:
        results = [
            r for r in results if any(pid in r.linked_prisoner_ids for pid in prisoner_id)
        ]

    if prisoner_name:
        name_lower = prisoner_name.lower()
        results = [
            r
            for r in results
            if any(name_lower in pn.lower() for pn in r.linked_prisoner_names)
        ]

    if conversation_id:
        results = [r for r in results if conversation_id in r.linked_conversation_ids]

    if risk_level:
        results = [r for r in results if r.risk_level == risk_level]

    if alert_category:
        results = [
            r for r in results if any(cat in r.alert_categories for cat in alert_category)
        ]

    if tag:
        results = [r for r in results if any(t in r.tags for t in tag)]

    if start_time:
        start_dt = _parse_timestamp(start_time)
        results = [r for r in results if _parse_timestamp(r.created_at) >= start_dt]

    if end_time:
        end_dt = _parse_timestamp(end_time)
        results = [r for r in results if _parse_timestamp(r.created_at) <= end_dt]

    if not query:
        reverse = sort_order == "desc"
        if sort_by == "created_at":
            results.sort(key=lambda r: r.created_at, reverse=reverse)
        elif sort_by == "risk_level":
            level_order = {"low": 0, "medium": 1, "high": 2}
            results.sort(key=lambda r: level_order.get(r.risk_level, 0), reverse=reverse)
        elif sort_by == "report_id":
            results.sort(key=lambda r: r.report_id, reverse=reverse)

    total = len(results)
    items = results[offset : offset + limit]

    return Page(items=items, total=total, limit=limit, offset=offset)
