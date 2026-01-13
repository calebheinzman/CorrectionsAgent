"""Prisoner Info API endpoints."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from .schemas import Page, Prisoner

router = APIRouter(prefix="/prisoners", tags=["prisoners"])

_prisoners: List[Prisoner] = []


def load_prisoners(data: List[dict]) -> None:
    """Load prisoner data from JSON."""
    global _prisoners
    _prisoners = [Prisoner.model_validate(p) for p in data]


def get_all_prisoners() -> List[Prisoner]:
    """Return all loaded prisoners."""
    return _prisoners


@router.get("/{prisoner_id}", response_model=Prisoner)
def get_prisoner_by_id(prisoner_id: str) -> Prisoner:
    """Get a single prisoner by ID."""
    for p in _prisoners:
        if p.prisoner_id == prisoner_id:
            return p
    raise HTTPException(status_code=404, detail=f"Prisoner {prisoner_id} not found")


@router.get("", response_model=Page[Prisoner])
def list_prisoners(
    prisoner_id: Optional[str] = Query(None, description="Exact prisoner ID match"),
    name: Optional[str] = Query(None, description="Case-insensitive name contains"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Page[Prisoner]:
    """List prisoners with optional filters."""
    results = _prisoners

    if prisoner_id:
        results = [p for p in results if p.prisoner_id == prisoner_id]

    if name:
        name_lower = name.lower()
        results = [p for p in results if name_lower in p.name.lower()]

    total = len(results)
    items = results[offset : offset + limit]

    return Page(items=items, total=total, limit=limit, offset=offset)
