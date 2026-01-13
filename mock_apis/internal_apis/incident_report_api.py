"""Incident Report API endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from .schemas import IncidentReport, Page

router = APIRouter(prefix="/incidents", tags=["incidents"])

_incident_reports: List[IncidentReport] = []
_vector_store = None


def load_incident_reports(data: List[dict]) -> None:
    """Load incident report data from JSON."""
    global _incident_reports
    _incident_reports = [IncidentReport.model_validate(i) for i in data]


def set_vector_store(vs) -> None:
    """Set the vector store for similarity search."""
    global _vector_store
    _vector_store = vs


def get_all_incident_reports() -> List[IncidentReport]:
    """Return all loaded incident reports."""
    return _incident_reports


def _parse_date(d: str) -> datetime:
    """Parse YYYY-MM-DD date string to datetime."""
    return datetime.strptime(d, "%Y-%m-%d")


@router.get("/{incident_id}", response_model=IncidentReport)
def get_incident_by_id(incident_id: str) -> IncidentReport:
    """Get a single incident report by ID."""
    for i in _incident_reports:
        if i.incident_id == incident_id:
            return i
    raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")


@router.get("", response_model=Page[IncidentReport])
def list_incidents(
    incident_id: Optional[str] = Query(None, description="Exact incident ID"),
    prisoner_id: Optional[List[str]] = Query(None, description="Filter by involved prisoner ID(s)"),
    prisoner_name: Optional[str] = Query(None, description="Case-insensitive prisoner name contains"),
    conversation_id: Optional[str] = Query(None, description="Filter by linked conversation ID"),
    report_id: Optional[str] = Query(None, description="Filter by linked report ID"),
    type: Optional[str] = Query(None, description="Incident type"),
    severity: Optional[str] = Query(None, description="Severity level"),
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD start date (inclusive)"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD end date (inclusive)"),
    query: Optional[str] = Query(None, description="Free-text similarity search over description"),
    top_k: int = Query(10, ge=1, le=100, description="Max results for similarity search"),
    min_score: Optional[float] = Query(None, description="Min similarity score"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("date", description="Field to sort by"),
    sort_order: str = Query("desc", description="asc or desc"),
) -> Page[IncidentReport]:
    """List incident reports with filters, similarity search, pagination, and sorting."""

    if start_date and end_date:
        try:
            start_dt = _parse_date(start_date)
            end_dt = _parse_date(end_date)
            if start_dt > end_dt:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid date range: start_date must be before end_date",
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    if query and _vector_store:
        try:
            docs = _vector_store.similarity_search_with_score(query, k=top_k)
            incident_ids = []
            for doc, score in docs:
                if min_score is not None and score < min_score:
                    continue
                iid = doc.metadata.get("incident_id")
                if iid:
                    incident_ids.append(iid)
            results = [i for i in _incident_reports if i.incident_id in incident_ids]
            results_order = {iid: idx for idx, iid in enumerate(incident_ids)}
            results.sort(key=lambda i: results_order.get(i.incident_id, 999))
        except Exception:
            results = list(_incident_reports)
    else:
        results = list(_incident_reports)

    if incident_id:
        results = [i for i in results if i.incident_id == incident_id]

    if prisoner_id:
        results = [
            i for i in results if any(pid in i.involved_prisoner_ids for pid in prisoner_id)
        ]

    if prisoner_name:
        name_lower = prisoner_name.lower()
        results = [
            i
            for i in results
            if any(name_lower in pn.lower() for pn in i.involved_prisoner_names)
        ]

    if conversation_id:
        results = [i for i in results if conversation_id in i.linked_conversation_ids]

    if report_id:
        results = [i for i in results if report_id in i.linked_report_ids]

    if type:
        results = [i for i in results if i.type == type]

    if severity:
        results = [i for i in results if i.severity == severity]

    if start_date:
        start_dt = _parse_date(start_date)
        results = [i for i in results if _parse_date(i.date) >= start_dt]

    if end_date:
        end_dt = _parse_date(end_date)
        results = [i for i in results if _parse_date(i.date) <= end_dt]

    if not query:
        reverse = sort_order == "desc"
        if sort_by == "date":
            results.sort(key=lambda i: i.date, reverse=reverse)
        elif sort_by == "severity":
            sev_order = {"low": 0, "medium": 1, "high": 2}
            results.sort(key=lambda i: sev_order.get(i.severity, 0), reverse=reverse)
        elif sort_by == "incident_id":
            results.sort(key=lambda i: i.incident_id, reverse=reverse)

    total = len(results)
    items = results[offset : offset + limit]

    return Page(items=items, total=total, limit=limit, offset=offset)
