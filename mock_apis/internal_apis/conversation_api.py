"""Conversation API endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from .schemas import Conversation, Page

router = APIRouter(prefix="/conversations", tags=["conversations"])

_conversations: List[Conversation] = []
_vector_store = None


def load_conversations(data: List[dict]) -> None:
    """Load conversation data from JSON."""
    global _conversations
    _conversations = [Conversation.model_validate(c) for c in data]


def set_vector_store(vs) -> None:
    """Set the vector store for similarity search."""
    global _vector_store
    _vector_store = vs


def get_all_conversations() -> List[Conversation]:
    """Return all loaded conversations."""
    return _conversations


def _parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


@router.get("/{conversation_id}", response_model=Conversation)
def get_conversation_by_id(conversation_id: str) -> Conversation:
    """Get a single conversation by ID."""
    for c in _conversations:
        if c.conversation_id == conversation_id:
            return c
    raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")


@router.get("", response_model=Page[Conversation])
def list_conversations(
    conversation_id: Optional[str] = Query(None, description="Exact conversation ID"),
    prisoner_id: Optional[List[str]] = Query(None, description="Filter by prisoner ID(s)"),
    prisoner_name: Optional[str] = Query(None, description="Case-insensitive prisoner name contains"),
    alert_category: Optional[List[str]] = Query(None, description="Filter by alert category"),
    keyword: Optional[List[str]] = Query(None, description="Filter by keyword hits"),
    min_alert_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_alert_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    review_status: Optional[str] = Query(None, description="unreviewed or reviewed"),
    start_time: Optional[str] = Query(None, description="ISO timestamp start (inclusive)"),
    end_time: Optional[str] = Query(None, description="ISO timestamp end (inclusive)"),
    query: Optional[str] = Query(None, description="Free-text similarity search"),
    top_k: int = Query(10, ge=1, le=100, description="Max results for similarity search"),
    min_score: Optional[float] = Query(None, description="Min similarity score"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("timestamp", description="Field to sort by"),
    sort_order: str = Query("desc", description="asc or desc"),
) -> Page[Conversation]:
    """List conversations with filters, similarity search, pagination, and sorting."""

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
            conv_ids = []
            for doc, score in docs:
                if min_score is not None and score < min_score:
                    continue
                cid = doc.metadata.get("conversation_id")
                if cid:
                    conv_ids.append(cid)
            results = [c for c in _conversations if c.conversation_id in conv_ids]
            results_order = {cid: i for i, cid in enumerate(conv_ids)}
            results.sort(key=lambda c: results_order.get(c.conversation_id, 999))
        except Exception:
            results = list(_conversations)
    else:
        results = list(_conversations)

    if conversation_id:
        results = [c for c in results if c.conversation_id == conversation_id]

    if prisoner_id:
        results = [c for c in results if any(pid in c.prisoner_ids for pid in prisoner_id)]

    if prisoner_name:
        name_lower = prisoner_name.lower()
        results = [
            c for c in results if any(name_lower in pn.lower() for pn in c.prisoner_names)
        ]

    if alert_category:
        results = [
            c for c in results if any(cat in c.alert_categories for cat in alert_category)
        ]

    if keyword:
        results = [c for c in results if any(kw in c.keyword_hits for kw in keyword)]

    if min_alert_confidence is not None:
        results = [c for c in results if c.alert_confidence >= min_alert_confidence]

    if max_alert_confidence is not None:
        results = [c for c in results if c.alert_confidence <= max_alert_confidence]

    if review_status:
        results = [c for c in results if c.review_status == review_status]

    if start_time:
        start_dt = _parse_timestamp(start_time)
        results = [c for c in results if _parse_timestamp(c.timestamp) >= start_dt]

    if end_time:
        end_dt = _parse_timestamp(end_time)
        results = [c for c in results if _parse_timestamp(c.timestamp) <= end_dt]

    if not query:
        reverse = sort_order == "desc"
        if sort_by == "timestamp":
            results.sort(key=lambda c: c.timestamp, reverse=reverse)
        elif sort_by == "alert_confidence":
            results.sort(key=lambda c: c.alert_confidence, reverse=reverse)
        elif sort_by == "conversation_id":
            results.sort(key=lambda c: c.conversation_id, reverse=reverse)

    total = len(results)
    items = results[offset : offset + limit]

    return Page(items=items, total=total, limit=limit, offset=offset)
