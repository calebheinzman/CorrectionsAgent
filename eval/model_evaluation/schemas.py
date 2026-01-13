"""Pydantic schemas for evaluation datasets and results."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvalRowMeta(BaseModel):
    """Metadata embedded in each evaluation row."""
    seed: int = 0
    generator_model: str = ""
    generated_at: str = Field(default_factory=_utc_now_iso)


class RelevanceRow(BaseModel):
    """Dataset A: Relevance classification row."""
    id: str
    dataset: Literal["relevance"] = "relevance"
    input: "RelevanceInput"
    labels: "RelevanceLabels"
    meta: EvalRowMeta = Field(default_factory=EvalRowMeta)


class RelevanceInput(BaseModel):
    text: str


class RelevanceLabels(BaseModel):
    relevance: Literal["relevant", "irrelevant"]


class SafetyRow(BaseModel):
    """Dataset B: Safety classification row."""
    id: str
    dataset: Literal["safety"] = "safety"
    input: "SafetyInput"
    labels: "SafetyLabels"
    meta: EvalRowMeta = Field(default_factory=EvalRowMeta)


class SafetyInput(BaseModel):
    text: str


class SafetyLabels(BaseModel):
    safety: Literal["safe", "unsafe"]


class ToolPairsRow(BaseModel):
    """Dataset C: Question -> Tool set pairs row."""
    id: str
    dataset: Literal["tool_pairs"] = "tool_pairs"
    input: "ToolPairsInput"
    labels: "ToolPairsLabels"
    meta: EvalRowMeta = Field(default_factory=EvalRowMeta)


class ToolPairsInput(BaseModel):
    tools_allowed: List[str]
    text: str


class ToolPairsLabels(BaseModel):
    tools_required: List[str]


class DatapointPairsRow(BaseModel):
    """Dataset D: Question -> Data point pairs row."""
    id: str
    dataset: Literal["datapoint_pairs"] = "datapoint_pairs"
    input: "DatapointPairsInput"
    labels: "DatapointPairsLabels"
    meta: EvalRowMeta = Field(default_factory=EvalRowMeta)


class DatapointPairsInput(BaseModel):
    text: str
    target_datapoint: Dict[str, Any]
    datapoint_source: Literal["prisoners", "conversations", "user_reports", "incident_reports"]


class DatapointPairsLabels(BaseModel):
    must_contain: Dict[str, Any]


class QuestionListRow(BaseModel):
    """Dataset E: Broad questions row."""
    id: str
    dataset: Literal["question_list"] = "question_list"
    input: "QuestionListInput"
    labels: "QuestionListLabels"
    meta: EvalRowMeta = Field(default_factory=EvalRowMeta)


class QuestionListInput(BaseModel):
    text: str


class QuestionListLabels(BaseModel):
    topic_tags: List[str] = Field(default_factory=list)


RelevanceRow.model_rebuild()
SafetyRow.model_rebuild()
ToolPairsRow.model_rebuild()
DatapointPairsRow.model_rebuild()
QuestionListRow.model_rebuild()


class GeminiRelevanceOutput(BaseModel):
    """Structured output schema for Gemini relevance generation."""
    text: str
    relevance: Literal["relevant", "irrelevant"]


class GeminiSafetyOutput(BaseModel):
    """Structured output schema for Gemini safety generation."""
    text: str
    safety: Literal["safe", "unsafe"]


class GeminiToolPairsOutput(BaseModel):
    """Structured output schema for Gemini tool pairs generation."""
    text: str


class GeminiDatapointPairsOutput(BaseModel):
    """Structured output schema for Gemini datapoint pairs generation."""
    text: str


class GeminiQuestionListOutput(BaseModel):
    """Structured output schema for Gemini question list generation."""
    text: str
    topic_tags: List[str] = Field(default_factory=list)


class EvalRunResult(BaseModel):
    """Result of a single evaluation run (one question)."""
    row_id: str
    dataset: str
    model_id: str
    prompt_version: str
    input_text: str
    output_text: str
    predicted: Optional[Any] = None
    expected: Optional[Any] = None
    correct: Optional[bool] = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None
    retrieved_contexts: List[str] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    tool_calls: List[str] = Field(default_factory=list)
    run_index: int = 0


class DatasetMetrics(BaseModel):
    """Aggregated metrics for a dataset."""
    dataset: str
    model_id: str
    prompt_version: str
    total_samples: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_macro: float = 0.0
    f1_micro: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    total_cost_usd: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0


class ConsistencyMetrics(BaseModel):
    """Consistency metrics for multiple runs of the same question."""
    row_id: str
    model_id: str
    prompt_version: str
    num_runs: int = 0
    answer_similarity: float = 0.0
    retrieval_overlap: float = 0.0
    citation_overlap: float = 0.0


class JudgeMetrics(BaseModel):
    """LLM-as-judge metrics for a single response."""
    row_id: str
    model_id: str
    prompt_version: str
    faithfulness: float = 0.0
    context_precision: float = 0.0
    completeness: float = 0.0
    clarity: float = 0.0
    relevance: float = 0.0
    answer_relevance_embedding: float = 0.0
    context_utilization: float = 0.0
    citation_correctness: float = 0.0


class EvalSummary(BaseModel):
    """Summary of an evaluation run."""
    run_id: str
    created_at: str = Field(default_factory=_utc_now_iso)
    datasets: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)
    prompt_versions: List[str] = Field(default_factory=list)
    total_samples: int = 0
    total_runs: int = 0
    metrics_by_config: Dict[str, DatasetMetrics] = Field(default_factory=dict)
