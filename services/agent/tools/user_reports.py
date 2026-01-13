"""User reports tool adapter for the agent."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

from ..schemas import Citation

TOOL_API_BASE = os.getenv("TOOL_API_BASE", "http://localhost:8001")


def search_user_reports(
    query: Optional[str] = None,
    prisoner_id: Optional[str] = None,
    prisoner_name: Optional[str] = None,
    risk_level: Optional[str] = None,
    alert_category: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Search user reports with various filters.

    Args:
        query: Free-text semantic search query over summary
        prisoner_id: Filter by linked prisoner ID
        prisoner_name: Filter by linked prisoner name (case-insensitive contains)
        risk_level: Filter by risk level (low, medium, high)
        alert_category: Filter by alert category
        tag: Filter by tag
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
    if risk_level:
        params["risk_level"] = risk_level
    if alert_category:
        params["alert_category"] = alert_category
    if tag:
        params["tag"] = tag

    if os.getenv("EVAL_IN_PROCESS") == "1":
        try:
            from mock_apis.internal_apis.user_report_api import list_user_reports

            page = list_user_reports(
                report_id=None,
                prisoner_id=None,
                prisoner_name=prisoner_name,
                conversation_id=None,
                risk_level=risk_level,
                alert_category=None,
                tag=None,
                start_time=None,
                end_time=None,
                query=query,
                top_k=limit,
                min_score=None,
                limit=limit,
                offset=0,
                sort_by="created_at",
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
            response = client.get(f"{TOOL_API_BASE}/user-reports", params=params)
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


def get_user_report_by_id(report_id: str) -> Dict[str, Any]:
    """
    Get a specific user report by ID.

    Args:
        report_id: The report ID to retrieve

    Returns:
        Dictionary with report data or error
    """
    start_time = time.time()
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{TOOL_API_BASE}/user-reports/{report_id}")
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
                "error": f"User report {report_id} not found",
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


def extract_citations(reports: List[Dict[str, Any]]) -> List[Citation]:
    """Extract citations from user report results."""
    citations = []
    for report in reports:
        report_id = report.get("report_id", "")
        summary = report.get("summary", "")
        excerpt = summary[:200] + "..." if len(summary) > 200 else summary
        citations.append(
            Citation(
                source_type="report",
                source_id=report_id,
                excerpt=excerpt,
            )
        )
    return citations
