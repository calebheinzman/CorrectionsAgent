"""Conversation tool adapter for the agent."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

from ..schemas import Citation, ToolCallRecord

TOOL_API_BASE = os.getenv("TOOL_API_BASE", "http://localhost:8001")


def search_conversations(
    query: Optional[str] = None,
    prisoner_id: Optional[str] = None,
    prisoner_name: Optional[str] = None,
    alert_category: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Search conversations with various filters.

    Args:
        query: Free-text semantic search query
        prisoner_id: Filter by prisoner ID
        prisoner_name: Filter by prisoner name (case-insensitive contains)
        alert_category: Filter by alert category
        keyword: Filter by keyword hit
        limit: Maximum number of results

    Returns:
        Dictionary with 'items' list and metadata
    """
    start_time = time.time()
    params: Dict[str, Any] = {"limit": limit}

    if query:
        params["query"] = query
        params["top_k"] = limit
    if prisoner_id:
        params["prisoner_id"] = prisoner_id
    if prisoner_name:
        params["prisoner_name"] = prisoner_name
    if alert_category:
        params["alert_category"] = alert_category
    if keyword:
        params["keyword"] = keyword

    if os.getenv("EVAL_IN_PROCESS") == "1":
        try:
            from mock_apis.internal_apis.conversation_api import list_conversations

            prisoner_ids: Optional[List[str]] = [prisoner_id] if prisoner_id else None
            alert_categories: Optional[List[str]] = [alert_category] if alert_category else None
            keywords: Optional[List[str]] = [keyword] if keyword else None

            page = list_conversations(
                conversation_id=None,
                prisoner_id=prisoner_ids,
                prisoner_name=prisoner_name,
                alert_category=alert_categories,
                keyword=keywords,
                min_alert_confidence=None,
                max_alert_confidence=None,
                review_status=None,
                start_time=None,
                end_time=None,
                query=query,
                top_k=limit,
                min_score=None,
                limit=limit,
                offset=0,
                sort_by="timestamp",
                sort_order="desc",
            )
            latency_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "data": page.model_dump(),
                "latency_ms": latency_ms,
            }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "data": {"items": [], "total": 0, "limit": limit, "offset": 0},
                "latency_ms": latency_ms,
            }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{TOOL_API_BASE}/conversations", params=params)
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "data": data,
                "latency_ms": latency_ms,
            }
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return {
            "success": False,
            "error": str(e),
            "data": {"items": [], "total": 0},
            "latency_ms": latency_ms,
        }


def get_conversation_by_id(conversation_id: str) -> Dict[str, Any]:
    """
    Get a specific conversation by ID.

    Args:
        conversation_id: The conversation ID to retrieve

    Returns:
        Dictionary with conversation data or error
    """
    start_time = time.time()
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{TOOL_API_BASE}/conversations/{conversation_id}")
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "data": data,
                "latency_ms": latency_ms,
            }
    except httpx.HTTPStatusError as e:
        latency_ms = (time.time() - start_time) * 1000
        if e.response.status_code == 404:
            return {
                "success": False,
                "error": f"Conversation {conversation_id} not found",
                "data": None,
                "latency_ms": latency_ms,
            }
        return {
            "success": False,
            "error": str(e),
            "data": None,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return {
            "success": False,
            "error": str(e),
            "data": None,
            "latency_ms": latency_ms,
        }


def extract_citations(conversations: List[Dict[str, Any]]) -> List[Citation]:
    """Extract citations from conversation results."""
    citations = []
    for conv in conversations:
        conv_id = conv.get("conversation_id", "")
        transcript = conv.get("transcript", "")
        excerpt = transcript[:200] + "..." if len(transcript) > 200 else transcript
        citations.append(
            Citation(
                source_type="conversation",
                source_id=conv_id,
                excerpt=excerpt,
            )
        )
    return citations
