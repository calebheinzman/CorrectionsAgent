"""JSONL read/write helpers for evaluation datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Type, TypeVar, Union

from pydantic import BaseModel

from .schemas import (
    DatapointPairsRow,
    QuestionListRow,
    RelevanceRow,
    SafetyRow,
    ToolPairsRow,
)

T = TypeVar("T", bound=BaseModel)

EvalRow = Union[RelevanceRow, SafetyRow, ToolPairsRow, DatapointPairsRow, QuestionListRow]

DATASET_SCHEMAS: dict[str, Type[EvalRow]] = {
    "relevance": RelevanceRow,
    "safety": SafetyRow,
    "tool_pairs": ToolPairsRow,
    "datapoint_pairs": DatapointPairsRow,
    "question_list": QuestionListRow,
}


def write_jsonl(path: Path, rows: List[BaseModel]) -> None:
    """Write a list of Pydantic models to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path, schema: Type[T]) -> List[T]:
    """Read a JSONL file into a list of Pydantic models."""
    rows: List[T] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                rows.append(schema.model_validate(data))
    return rows


def read_jsonl_raw(path: Path) -> List[dict]:
    """Read a JSONL file into a list of dicts (no validation)."""
    rows: List[dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dataset(path: Path) -> List[EvalRow]:
    """Load a dataset JSONL file, auto-detecting the schema from the first row."""
    raw = read_jsonl_raw(path)
    if not raw:
        return []
    
    first_dataset = raw[0].get("dataset", "")
    schema = DATASET_SCHEMAS.get(first_dataset)
    if schema is None:
        raise ValueError(f"Unknown dataset type: {first_dataset}")
    
    return [schema.model_validate(r) for r in raw]


def find_latest_dataset(data_dir: Path, prefix: str) -> Path | None:
    """Find the latest dataset file matching a prefix (by timestamp in filename)."""
    matches = sorted(data_dir.glob(f"{prefix}_*.jsonl"), reverse=True)
    return matches[0] if matches else None
