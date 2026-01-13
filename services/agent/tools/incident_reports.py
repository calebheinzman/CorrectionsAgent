"""Incident reports tool adapter for the agent."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

from ..schemas import Citation

TOOL_API_BASE = os.getenv("TOOL_API_BASE", "http://localhost:8001")


def search_incidents(
    query: Optional[str] = None,
    prisoner_id: Optional[str] = None,
    prisoner_name: Optional[str] = None,
    incident_type: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Search incident reports with various filters.

    Args:
        query: Free-text semantic search query over description
        prisoner_id: Filter by involved prisoner ID
        prisoner_name: Filter by involved prisoner name (case-insensitive contains)
        incident_type: Filter by incident type
        severity: Filter by severity level (low, medium, high)
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
    if incident_type:
        params["type"] = incident_type
    if severity:
        params["severity"] = severity

    if os.getenv("EVAL_IN_PROCESS") == "1":
        try:
            from mock_apis.internal_apis.incident_report_api import list_incidents

            page = list_incidents(
                incident_id=None,
                prisoner_id=None,
                prisoner_name=prisoner_name,
                conversation_id=None,
                report_id=None,
                type=incident_type,
                severity=severity,
                start_date=None,
                end_date=None,
                query=query,
                top_k=limit,
                min_score=None,
                limit=limit,
                offset=0,
                sort_by="date",
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
            response = client.get(f"{TOOL_API_BASE}/incidents", params=params)
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


def get_incident_by_id(incident_id: str) -> Dict[str, Any]:
    """
    Get a specific incident report by ID.

    Args:
        incident_id: The incident ID to retrieve

    Returns:
        Dictionary with incident data or error
    """
    start_time = time.time()
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{TOOL_API_BASE}/incidents/{incident_id}")
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
                "error": f"Incident {incident_id} not found",
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


def extract_citations(incidents: List[Dict[str, Any]]) -> List[Citation]:
    """Extract citations from incident results."""
    citations = []
    for incident in incidents:
        incident_id = incident.get("incident_id", "")
        description = incident.get("description", "")
        excerpt = description[:200] + "..." if len(description) > 200 else description
        citations.append(
            Citation(
                source_type="incident",
                source_id=incident_id,
                excerpt=excerpt,
            )
        )
    return citations
