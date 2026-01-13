"""Evaluation runner for 3 models x 3 prompts x datasets.

This runner evaluates the full orchestrator->guardrails->agent pipeline,
not standalone API calls.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    HAS_EMBEDDINGS = True
except Exception:
    HAS_EMBEDDINGS = False

from mock_apis.cloud_apis import mock_model_registry, mock_s3
from mock_apis.cloud_apis.mock_secrets_manager import get_secret
from mock_apis.internal_apis import app as internal_app

from services.guardrails.safety_check.schemas import SafetyCheckRequest
from services.guardrails.safety_check.service import check_safety as safety_check_service
from services.guardrails.safety_check.service import get_service as get_safety_service

from services.guardrails.relevance_check.schemas import RelevanceCheckRequest
from services.guardrails.relevance_check.service import check_relevance as relevance_check_service
from services.guardrails.relevance_check.service import get_service as get_relevance_service

from services.agent.schemas import AgentRequest
from services.agent.service import answer as agent_answer_service
from services.agent.service import get_service as get_agent_service
from services.agent.service import reset_service as reset_agent_service

from .datasets import find_latest_dataset, load_dataset, read_jsonl_raw
from .metrics import (
    check_datapoint_match,
    compute_consistency_metrics,
    compute_dataset_metrics,
    compute_embedding_similarity,
    extract_tool_calls_from_response,
)
from .reporting import (
    generate_report_md,
    generate_summary,
    write_consistency_metrics,
    write_judge_metrics,
    write_metrics_summary,
    write_raw_results,
    write_report_md,
    write_summary_json,
)
from .schemas import (
    ConsistencyMetrics,
    DatapointPairsRow,
    DatasetMetrics,
    EvalRunResult,
    EvalSummary,
    JudgeMetrics,
    QuestionListRow,
    RelevanceRow,
    SafetyRow,
    ToolPairsRow,
)
from .judging import compute_all_judge_metrics


def _utc_compact_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


_services_initialized = False


def initialize_services() -> None:
    """Initialize all services for in-process evaluation."""
    global _services_initialized
    if _services_initialized:
        return

    print("Initializing services...")

    print("  Loading mock data from internal APIs...")
    internal_app.load_data()

    # Ensure agent tool adapters use in-process mock APIs instead of requiring an HTTP server.
    os.environ.setdefault("EVAL_IN_PROCESS", "1")

    print("  Initializing safety check service...")
    get_safety_service()

    print("  Initializing relevance check service...")
    get_relevance_service()

    print("  Initializing agent service...")
    agent_svc = get_agent_service()
    if not agent_svc.is_available():
        error = agent_svc.get_init_error()
        print(f"  WARNING: Agent service not fully available: {error}")
        print("  (Will use fallback mode)")

    _services_initialized = True
    print("  Services initialized.")


class PipelineResult:
    """Result from running the full pipeline."""

    def __init__(
        self,
        answer: str,
        status: str,
        safety_allowed: bool,
        safety_reason: str,
        relevance_relevant: Optional[bool],
        relevance_reason: Optional[str],
        tool_calls: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        contexts: List[str],
        latency_ms: float,
        error: Optional[str] = None,
    ):
        self.answer = answer
        self.status = status
        self.safety_allowed = safety_allowed
        self.safety_reason = safety_reason
        self.relevance_relevant = relevance_relevant
        self.relevance_reason = relevance_reason
        self.tool_calls = tool_calls
        self.citations = citations
        self.contexts = contexts
        self.latency_ms = latency_ms
        self.error = error


def _extract_contexts_from_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[str]:
    """Extract text contexts from tool call outputs for judge metrics."""
    contexts = []
    
    for tc in tool_calls:
        tool_name = tc.get("tool_name", "")
        
        if not tc.get("success", True):
            continue
        
        try:
            from services.agent.tracing.tool_tracer import get_tracer
            tracer = get_tracer()
            
            traces = tracer.get_traces_for_request(tc.get("request_id", ""))
            for trace in traces:
                if trace.tool_name == tool_name and trace.output_data:
                    try:
                        output_data = json.loads(trace.output_data) if isinstance(trace.output_data, str) else trace.output_data
                        
                        if isinstance(output_data, list):
                            for item in output_data[:5]:
                                if isinstance(item, dict):
                                    text_parts = []
                                    for key in ["transcript", "description", "summary", "notes", "content"]:
                                        if key in item and item[key]:
                                            text_parts.append(f"{key}: {item[key]}")
                                    if text_parts:
                                        contexts.append(" | ".join(text_parts))
                        elif isinstance(output_data, dict):
                            text_parts = []
                            for key in ["transcript", "description", "summary", "notes", "content"]:
                                if key in output_data and output_data[key]:
                                    text_parts.append(f"{key}: {output_data[key]}")
                            if text_parts:
                                contexts.append(" | ".join(text_parts))
                    except Exception:
                        pass
        except Exception:
            pass
    
    return contexts[:10]


def _contexts_and_ids_from_citations(citations: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    contexts: List[str] = []
    ids: List[str] = []
    for c in citations:
        source_id = str(c.get("source_id", "") or "")
        source_type = str(c.get("source_type", "") or "")
        excerpt = str(c.get("excerpt", "") or "")
        if source_id:
            ids.append(source_id)
        if excerpt:
            prefix = f"[{source_type}:{source_id}] " if (source_type or source_id) else ""
            contexts.append(prefix + excerpt)
    return contexts, ids


def run_full_pipeline(
    question: str,
    request_id: str,
) -> PipelineResult:
    """
    Run a question through the full orchestrator->guardrails->agent pipeline.

    Flow:
    1. Safety check - deny unsafe queries
    2. Relevance check - deny out-of-domain queries
    3. Agent call - answer approved queries
    """
    start_time = time.time()

    safety_req = SafetyCheckRequest(request_id=request_id, question=question)
    safety_resp = safety_check_service(safety_req)

    if not safety_resp.allowed:
        latency_ms = (time.time() - start_time) * 1000
        return PipelineResult(
            answer=f"Request denied by safety check: {safety_resp.reason}",
            status="denied_safety",
            safety_allowed=False,
            safety_reason=safety_resp.reason,
            relevance_relevant=None,
            relevance_reason=None,
            tool_calls=[],
            citations=[],
            contexts=[],
            latency_ms=latency_ms,
        )

    relevance_req = RelevanceCheckRequest(request_id=request_id, question=question)
    relevance_resp = relevance_check_service(relevance_req)

    if not relevance_resp.relevant:
        latency_ms = (time.time() - start_time) * 1000
        return PipelineResult(
            answer=f"Request denied by relevance check: {relevance_resp.reason}",
            status="denied_relevance",
            safety_allowed=True,
            safety_reason=safety_resp.reason,
            relevance_relevant=False,
            relevance_reason=relevance_resp.reason,
            tool_calls=[],
            citations=[],
            contexts=[],
            latency_ms=latency_ms,
        )

    try:
        agent_req = AgentRequest(request_id=request_id, question=question)
        agent_resp = agent_answer_service(agent_req)

        tool_calls = [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in agent_resp.tool_calls]
        citations = [c.model_dump() if hasattr(c, "model_dump") else c for c in agent_resp.citations]

        # Prefer citation excerpts as the retrieval contexts; fall back to tool-call based extraction.
        contexts, _ = _contexts_and_ids_from_citations(citations)
        if not contexts:
            contexts = _extract_contexts_from_tool_calls(tool_calls)

        latency_ms = (time.time() - start_time) * 1000
        return PipelineResult(
            answer=agent_resp.answer,
            status="success",
            safety_allowed=True,
            safety_reason=safety_resp.reason,
            relevance_relevant=True,
            relevance_reason=relevance_resp.reason,
            tool_calls=tool_calls,
            citations=citations,
            contexts=contexts,
            latency_ms=latency_ms,
        )
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return PipelineResult(
            answer=f"Agent error: {str(e)}",
            status="error",
            safety_allowed=True,
            safety_reason=safety_resp.reason,
            relevance_relevant=True,
            relevance_reason=relevance_resp.reason,
            tool_calls=[],
            citations=[],
            contexts=[],
            latency_ms=latency_ms,
            error=str(e),
        )


EVAL_MODELS = ["eval_model_1", "eval_model_2", "eval_model_3"]
PROMPT_VERSIONS = ["v1.0.0", "v1.1.0", "v1.2.0"]
PROMPT_BUNDLE_ID = "investigative-assistant"
S3_BUCKET = "agent-configs"
TOOLS_BUNDLE_ID = "investigative-tools"
TOOLS_VERSION = "v1.0.0"


DEFAULT_PROMPTS = {
    "v1.0.0": {
        "system_prompt": """You are a prison intelligence assistant. Answer questions about prisoners, 
conversations, incidents, and reports accurately and concisely. Always cite specific IDs when referencing data.""",
        "tool_instructions": "Use the available tools to retrieve relevant information before answering.",
    },
    "v1.1.0": {
        "system_prompt": """You are an expert prison intelligence analyst assistant. Your role is to help 
investigators by providing accurate, well-sourced information about prisoners, their communications, 
incidents, and analyst reports. Always:
1. Cite specific record IDs (prisoner_id, conversation_id, incident_id, report_id)
2. Summarize key findings clearly
3. Flag any concerning patterns or risks""",
        "tool_instructions": "Carefully select and use the most relevant tools. Retrieve specific records when IDs are mentioned.",
    },
    "v1.2.0": {
        "system_prompt": """You are a specialized prison intelligence assistant designed to support 
correctional facility investigators. You have access to prisoner records, monitored communications, 
incident reports, and analyst-generated intelligence reports.

Guidelines:
- Be precise and factual
- Always cite source records with their IDs
- Highlight risk indicators and patterns
- Provide actionable intelligence summaries
- Never fabricate information not present in the data""",
        "tool_instructions": """Tool usage guidelines:
- prisoner_info: Use for demographic lookups by ID or name
- conversations: Use for communication records, supports filtering by prisoner, time, alerts
- incident_reports: Use for security incidents, fights, contraband
- user_reports: Use for analyst-generated intelligence summaries""",
    },
}


def setup_prompt_bundles() -> None:
    """Upload prompt bundles to mock S3."""
    for version, content in DEFAULT_PROMPTS.items():
        key = f"prompts/{PROMPT_BUNDLE_ID}/{version}.yaml"
        yaml_content = yaml.dump(content, default_flow_style=False)
        mock_s3.put_object(S3_BUCKET, key, yaml_content.encode("utf-8"))
        print(f"  Uploaded prompt bundle: {key}")

    tools_src = (
        Path(__file__).resolve().parents[2]
        / "mock_apis"
        / "cloud_apis"
        / "data"
        / "s3"
        / S3_BUCKET
        / "tools"
        / TOOLS_BUNDLE_ID
        / f"{TOOLS_VERSION}.yaml"
    )
    if tools_src.exists():
        tools_key = f"tools/{TOOLS_BUNDLE_ID}/{TOOLS_VERSION}.yaml"
        mock_s3.put_object(S3_BUCKET, tools_key, tools_src.read_bytes())
        print(f"  Uploaded tools bundle: {tools_key}")


def setup_model_registry() -> None:
    """Set up model registry entries for evaluation models."""
    models = [
        ("eval_model_1", "google", "gemini-2.0-flash-lite", "latest"),
        ("eval_model_2", "google", "gemini-2.0-flash", "latest"),
        ("eval_model_3", "google", "gemini-2.5-flash-lite", "latest"),
    ]

    for task, provider, model_id, version in models:
        mock_model_registry.set_current_model(provider, model_id, version, task=task)
        print(f"  Registered model: {task} -> {provider}/{model_id}")


def get_prompt_bundle(version: str) -> Dict[str, str]:
    """Load a prompt bundle from mock S3."""
    key = f"prompts/{PROMPT_BUNDLE_ID}/{version}.yaml"
    data = mock_s3.get_object(S3_BUCKET, key)
    if data is None:
        return DEFAULT_PROMPTS.get(version, DEFAULT_PROMPTS["v1.0.0"])
    return yaml.safe_load(data.decode("utf-8"))


def get_model_info(task: str) -> Tuple[str, str]:
    """Get model provider and ID from registry."""
    info = mock_model_registry.get_current_model(task=task)
    if info is None:
        return "google", "gemini-2.0-flash-lite"
    return info["provider"], info["model_id"]


def set_agent_config_for_prompt(prompt_version: str) -> None:
    mock_model_registry.set_agent_config(
        prompt_bundle_id=PROMPT_BUNDLE_ID,
        prompt_version=prompt_version,
        tools_bundle_id=TOOLS_BUNDLE_ID,
        tools_version=TOOLS_VERSION,
        task="agent",
    )


def evaluate_relevance_dataset(
    prompt_version: str,
    rows: List[RelevanceRow],
) -> List[EvalRunResult]:
    """
    Evaluate relevance classification using the full pipeline.
    
    The relevance guardrail service determines if questions are relevant.
    """
    results = []

    _, model_id = get_model_info("agent")

    for row in rows:
        request_id = f"eval_rel_{row.id}"
        result = run_full_pipeline(row.input.text, request_id)

        if result.relevance_relevant is not None:
            predicted = "relevant" if result.relevance_relevant else "irrelevant"
        else:
            predicted = "relevant" if result.safety_allowed else "irrelevant"

        expected = row.labels.relevance
        correct = predicted == expected

        results.append(
            EvalRunResult(
                row_id=row.id,
                dataset="relevance",
                model_id=model_id,
                prompt_version=prompt_version,
                input_text=row.input.text,
                output_text=result.answer,
                predicted=predicted,
                expected=expected,
                correct=correct,
                latency_ms=result.latency_ms,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                error=result.error,
                retrieved_contexts=[],
                tool_calls=[tc.get("tool_name", "") for tc in result.tool_calls],
            )
        )

    return results


def evaluate_safety_dataset(
    prompt_version: str,
    rows: List[SafetyRow],
) -> List[EvalRunResult]:
    """
    Evaluate safety classification using the full pipeline.
    
    The safety guardrail service determines if questions are safe.
    """
    results = []

    _, model_id = get_model_info("agent")

    for row in rows:
        request_id = f"eval_saf_{row.id}"
        result = run_full_pipeline(row.input.text, request_id)

        predicted = "safe" if result.safety_allowed else "unsafe"

        expected = row.labels.safety
        correct = predicted == expected

        results.append(
            EvalRunResult(
                row_id=row.id,
                dataset="safety",
                model_id=model_id,
                prompt_version=prompt_version,
                input_text=row.input.text,
                output_text=result.answer,
                predicted=predicted,
                expected=expected,
                correct=correct,
                latency_ms=result.latency_ms,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                error=result.error,
                retrieved_contexts=[],
                tool_calls=[tc.get("tool_name", "") for tc in result.tool_calls],
            )
        )

    return results


def evaluate_tool_pairs_dataset(
    prompt_version: str,
    rows: List[ToolPairsRow],
) -> List[EvalRunResult]:
    """
    Evaluate tool selection using the full pipeline.
    
    The agent's tool calls are compared against expected tools.
    """
    results = []

    _, model_id = get_model_info("agent")

    for row in rows:
        request_id = f"eval_tool_{row.id}"
        result = run_full_pipeline(row.input.text, request_id)

        predicted_tools = []
        for tc in result.tool_calls:
            tool_name = tc.get("tool_name", "") if isinstance(tc, dict) else str(tc)
            if "conversation" in tool_name.lower():
                predicted_tools.append("conversations")
            elif "prisoner" in tool_name.lower():
                predicted_tools.append("prisoner_info")
            elif "incident" in tool_name.lower():
                predicted_tools.append("incident_reports")
            elif "user_report" in tool_name.lower() or "report" in tool_name.lower():
                predicted_tools.append("user_reports")

        predicted_tools = list(set(predicted_tools))

        expected_tools = row.labels.tools_required
        correct = set(predicted_tools) == set(expected_tools)

        results.append(
            EvalRunResult(
                row_id=row.id,
                dataset="tool_pairs",
                model_id=model_id,
                prompt_version=prompt_version,
                input_text=row.input.text,
                output_text=result.answer,
                predicted=predicted_tools,
                expected=expected_tools,
                correct=correct,
                latency_ms=result.latency_ms,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                error=result.error,
                retrieved_contexts=[],
                tool_calls=[tc.get("tool_name", "") for tc in result.tool_calls],
            )
        )

    return results


def evaluate_datapoint_pairs_dataset(
    prompt_version: str,
    rows: List[DatapointPairsRow],
) -> List[EvalRunResult]:
    """
    Evaluate datapoint retrieval using the full pipeline.
    
    Checks if the agent's response contains the required datapoint.
    """
    results = []

    _, model_id = get_model_info("agent")

    for row in rows:
        request_id = f"eval_dp_{row.id}"
        result = run_full_pipeline(row.input.text, request_id)

        correct = check_datapoint_match(result.answer, row.labels.must_contain)

        results.append(
            EvalRunResult(
                row_id=row.id,
                dataset="datapoint_pairs",
                model_id=model_id,
                prompt_version=prompt_version,
                input_text=row.input.text,
                output_text=result.answer,
                predicted=correct,
                expected=True,
                correct=correct,
                latency_ms=result.latency_ms,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                error=result.error,
                retrieved_contexts=[],
                tool_calls=[tc.get("tool_name", "") for tc in result.tool_calls],
            )
        )

    return results


def evaluate_question_list_dataset(
    prompt_version: str,
    rows: List[QuestionListRow],
    run_index: int = 0,
) -> List[EvalRunResult]:
    """
    Evaluate question list using the full pipeline (system-level metrics).
    """
    results = []

    _, model_id = get_model_info("agent")

    for row in rows:
        request_id = f"eval_ql_{row.id}_r{run_index}"
        result = run_full_pipeline(row.input.text, request_id)

        citations = result.citations
        contexts, retrieved_chunk_ids = _contexts_and_ids_from_citations(citations)
        if not contexts:
            contexts = result.contexts

        results.append(
            EvalRunResult(
                row_id=row.id,
                dataset="question_list",
                model_id=model_id,
                prompt_version=prompt_version,
                input_text=row.input.text,
                output_text=result.answer,
                predicted=None,
                expected=None,
                correct=result.status == "success",
                latency_ms=result.latency_ms,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                error=result.error,
                retrieved_contexts=contexts,
                citations=citations,
                retrieved_chunk_ids=retrieved_chunk_ids,
                tool_calls=[tc.get("tool_name", "") for tc in result.tool_calls],
                run_index=run_index,
            )
        )

    return results


def compute_embeddings(texts: List[str]) -> List[List[float]]:
    """Compute embeddings for a list of texts."""
    if not HAS_EMBEDDINGS:
        return [[0.0] * 768 for _ in texts]

    try:
        api_key = (
            get_secret("GEMINI_API_KEY")
            or get_secret("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
        )
        return embeddings_model.embed_documents(texts)
    except Exception:
        return [[0.0] * 768 for _ in texts]


def run_evaluation(
    data_dir: Path,
    results_dir: Path,
    datasets: List[str],
    runs_per_question: int = 1,
    compute_judge: bool = True,
) -> EvalSummary:
    """Run the full evaluation matrix using the orchestrator->guardrails->agent pipeline."""
    initialize_services()

    all_metrics: Dict[str, List[DatasetMetrics]] = {}
    all_consistency: List[ConsistencyMetrics] = []
    all_judge_metrics: List[JudgeMetrics] = []
    metrics_by_config: Dict[str, DatasetMetrics] = {}

    models_used: List[str] = []
    prompts_used = list(PROMPT_VERSIONS)

    for model_task in EVAL_MODELS:
        provider, model_id = get_model_info(model_task)
        models_used.append(model_id)

        for prompt_version in PROMPT_VERSIONS:
            print(f"\n=== Evaluating {model_id} with prompt: {prompt_version} ===")

            # Configure agent prompt + tools, and set the active agent model for this run.
            set_agent_config_for_prompt(prompt_version)
            mock_model_registry.set_current_model(provider, model_id, "latest", task="agent")
            reset_agent_service()
            initialize_services()

            for dataset_name in datasets:
                print(f"  Dataset: {dataset_name}")

                dataset_path = find_latest_dataset(data_dir, dataset_name)
                if dataset_path is None:
                    print(f"    WARNING: No dataset found for {dataset_name}")
                    continue

                rows = load_dataset(dataset_path)
                if not rows:
                    print(f"    WARNING: Empty dataset {dataset_name}")
                    continue

                print(f"    Loaded {len(rows)} rows from {dataset_path.name}")

                if dataset_name == "relevance":
                    results = evaluate_relevance_dataset(prompt_version, rows)  # type: ignore
                    is_multiclass = False
                elif dataset_name == "safety":
                    results = evaluate_safety_dataset(prompt_version, rows)  # type: ignore
                    is_multiclass = False
                elif dataset_name == "tool_pairs":
                    results = evaluate_tool_pairs_dataset(prompt_version, rows)  # type: ignore
                    is_multiclass = True
                elif dataset_name == "datapoint_pairs":
                    results = evaluate_datapoint_pairs_dataset(prompt_version, rows)  # type: ignore
                    is_multiclass = False
                elif dataset_name == "question_list":
                    all_run_results = []
                    all_run_embeddings = []

                    for run_idx in range(runs_per_question):
                        print(f"    Run {run_idx + 1}/{runs_per_question}")
                        run_results = evaluate_question_list_dataset(
                            prompt_version, rows, run_idx  # type: ignore
                        )
                        all_run_results.append(run_results)

                        if runs_per_question > 1:
                            texts = [r.output_text for r in run_results]
                            embeddings = compute_embeddings(texts)
                            all_run_embeddings.append(embeddings)

                    results = all_run_results[0]

                    if runs_per_question > 1:
                        consistency = compute_consistency_metrics(
                            all_run_results, all_run_embeddings
                        )
                        all_consistency.extend(consistency)

                    if compute_judge:
                        print("    Computing judge metrics...")
                        for r in results:
                            judge_m = compute_all_judge_metrics(
                                question=r.input_text,
                                contexts=r.retrieved_contexts,
                                answer=r.output_text,
                                row_id=r.row_id,
                                model_id=model_id,
                                prompt_version=prompt_version,
                                citations=r.citations,
                                retrieved_chunk_ids=r.retrieved_chunk_ids,
                            )
                            all_judge_metrics.append(judge_m)
                        print(f"    Judge metrics computed for {len(results)} samples")

                    is_multiclass = False
                else:
                    print(f"    Unknown dataset type: {dataset_name}")
                    continue

                write_raw_results(results_dir, dataset_name, model_id, prompt_version, results)

                metrics = compute_dataset_metrics(
                    results, dataset_name, model_id, prompt_version, is_multiclass
                )

                config_key = f"{dataset_name}|{model_id}|{prompt_version}"
                metrics_by_config[config_key] = metrics

                if dataset_name not in all_metrics:
                    all_metrics[dataset_name] = []
                all_metrics[dataset_name].append(metrics)

                print(f"    Accuracy: {metrics.accuracy:.3f}, F1-macro: {metrics.f1_macro:.3f}")

    for dataset_name, metrics_list in all_metrics.items():
        write_metrics_summary(results_dir, dataset_name, metrics_list)

    if all_consistency:
        write_consistency_metrics(results_dir, all_consistency)

    if all_judge_metrics:
        write_judge_metrics(results_dir, all_judge_metrics)

    run_id = _utc_compact_ts()
    summary = generate_summary(
        run_id=run_id,
        metrics_by_config=metrics_by_config,
        datasets=datasets,
        models=models_used,
        prompt_versions=prompts_used,
    )

    write_summary_json(results_dir, summary)

    report_md = generate_report_md(
        summary,
        all_metrics,
        all_consistency if all_consistency else None,
        all_judge_metrics if all_judge_metrics else None,
    )
    write_report_md(results_dir, report_md)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).parent / "data" / "synthetic"),
        help="Directory containing synthetic datasets",
    )
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).parent / "results"),
        help="Directory for evaluation results",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated list of datasets or 'all'",
    )
    parser.add_argument(
        "--runs-per-question",
        type=int,
        default=1,
        help="Number of runs per question for consistency metrics (dataset E)",
    )
    parser.add_argument(
        "--upload-prompts",
        action="store_true",
        help="Upload prompt bundles to mock S3 and exit",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Set up prompt bundles and model registry",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-judge metrics computation",
    )
    args = parser.parse_args()

    if args.upload_prompts or args.setup:
        print("Setting up prompt bundles in mock S3...")
        setup_prompt_bundles()
        print("\nSetting up model registry...")
        setup_model_registry()
        if args.upload_prompts:
            print("\nSetup complete.")
            return 0

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir) / _utc_compact_ts()
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets == "all":
        datasets = ["relevance", "safety", "tool_pairs", "datapoint_pairs", "question_list"]
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]

    compute_judge = not args.no_judge

    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Datasets: {datasets}")
    print(f"Runs per question: {args.runs_per_question}")
    print(f"Compute judge metrics: {compute_judge}")

    summary = run_evaluation(
        data_dir=data_dir,
        results_dir=results_dir,
        datasets=datasets,
        runs_per_question=args.runs_per_question,
        compute_judge=compute_judge,
    )

    print(f"\n=== Evaluation Complete ===")
    print(f"Run ID: {summary.run_id}")
    print(f"Total samples: {summary.total_samples}")
    print(f"Results written to: {results_dir}")
    print(f"Report: {results_dir / 'report.md'}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        print(f"Error: {e}")
        raise SystemExit(1)
