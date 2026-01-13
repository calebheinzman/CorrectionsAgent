"""Prisoner info tool adapter for the agent."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

from ..schemas import Citation

TOOL_API_BASE = os.getenv("TOOL_API_BASE", "http://localhost:8001")


def search_prisoners(
    prisoner_id: Optional[str] = None,
    name: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Search prisoners with filters.

    Args:
        prisoner_id: Exact prisoner ID match
        name: Case-insensitive name contains
        limit: Maximum number of results

    Returns:
        Dictionary with 'items' list and metadata
    """
    start_time = time.time()
    params: Dict[str, Any] = {"limit": limit}

    if prisoner_id:
        params["prisoner_id"] = prisoner_id
    if name:
        params["name"] = name

    if os.getenv("EVAL_IN_PROCESS") == "1":
        try:
            from mock_apis.internal_apis.prisoner_info_api import list_prisoners

            page = list_prisoners(
                prisoner_id=prisoner_id,
                name=name,
                limit=limit,
                offset=0,
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
            response = client.get(f"{TOOL_API_BASE}/prisoners", params=params)
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


def get_prisoner_by_id(prisoner_id: str) -> Dict[str, Any]:
    """
    Get a specific prisoner by ID.

    Args:
        prisoner_id: The prisoner ID to retrieve

    Returns:
        Dictionary with prisoner data or error
    """
    start_time = time.time()
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{TOOL_API_BASE}/prisoners/{prisoner_id}")
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
                "error": f"Prisoner {prisoner_id} not found",
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


def extract_citations(prisoners: List[Dict[str, Any]]) -> List[Citation]:
    """Extract citations from prisoner results."""
    citations = []
    for prisoner in prisoners:
        prisoner_id = prisoner.get("prisoner_id", "")
        name = prisoner.get("name", "")
        citations.append(
            Citation(
                source_type="prisoner",
                source_id=prisoner_id,
                excerpt=f"Prisoner: {name}",
            )
        )
    return citations
