"""Report generation for evaluation results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import (
    ConsistencyMetrics,
    DatasetMetrics,
    EvalRunResult,
    EvalSummary,
    JudgeMetrics,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_summary(
    run_id: str,
    metrics_by_config: Dict[str, DatasetMetrics],
    datasets: List[str],
    models: List[str],
    prompt_versions: List[str],
) -> EvalSummary:
    """Generate an evaluation summary from metrics."""
    total_samples = sum(m.total_samples for m in metrics_by_config.values())
    total_runs = len(metrics_by_config)

    return EvalSummary(
        run_id=run_id,
        datasets=datasets,
        models=models,
        prompt_versions=prompt_versions,
        total_samples=total_samples,
        total_runs=total_runs,
        metrics_by_config=metrics_by_config,
    )


def write_raw_results(
    results_dir: Path,
    dataset: str,
    model_id: str,
    prompt_version: str,
    results: List[EvalRunResult],
) -> Path:
    """Write raw results to JSONL file."""
    safe_model = model_id.replace("/", "_").replace(":", "_")
    safe_prompt = prompt_version.replace("/", "_").replace(":", "_")

    out_dir = results_dir / "raw" / dataset / safe_model
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{safe_prompt}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")

    return path


def write_metrics_summary(
    results_dir: Path,
    dataset: str,
    metrics_list: List[DatasetMetrics],
) -> Path:
    """Write metrics summary for a dataset."""
    out_dir = results_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{dataset}_summary.json"
    data = [m.model_dump() for m in metrics_list]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return path


def write_consistency_metrics(
    results_dir: Path,
    consistency: List[ConsistencyMetrics],
) -> Path:
    """Write consistency metrics."""
    out_dir = results_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "consistency.json"
    data = [c.model_dump() for c in consistency]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return path


def write_judge_metrics(
    results_dir: Path,
    judge_metrics: List[JudgeMetrics],
) -> Path:
    """Write LLM-as-judge metrics."""
    out_dir = results_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "judge_metrics.json"
    data = [j.model_dump() for j in judge_metrics]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return path


def write_summary_json(results_dir: Path, summary: EvalSummary) -> Path:
    """Write summary JSON."""
    path = results_dir / "summary.json"
    path.write_text(json.dumps(summary.model_dump(), indent=2), encoding="utf-8")
    return path


def generate_report_md(
    summary: EvalSummary,
    metrics_by_dataset: Dict[str, List[DatasetMetrics]],
    consistency_metrics: Optional[List[ConsistencyMetrics]] = None,
    judge_metrics: Optional[List[JudgeMetrics]] = None,
) -> str:
    """Generate a Markdown report from evaluation results."""
    lines = [
        f"# Evaluation Report: {summary.run_id}",
        "",
        f"**Generated:** {summary.created_at}",
        "",
        "## Overview",
        "",
        f"- **Datasets:** {', '.join(summary.datasets)}",
        f"- **Models:** {', '.join(summary.models)}",
        f"- **Prompt Versions:** {', '.join(summary.prompt_versions)}",
        f"- **Total Samples:** {summary.total_samples}",
        f"- **Total Configurations:** {summary.total_runs}",
        "",
    ]

    for dataset, metrics_list in metrics_by_dataset.items():
        lines.append(f"## Dataset: {dataset}")
        lines.append("")

        if dataset in ("relevance", "safety"):
            lines.append("### Classification Metrics")
            lines.append("")
            lines.append("| Model | Prompt | Accuracy | Precision | Recall | F1-Macro | F1-Micro |")
            lines.append("|-------|--------|----------|-----------|--------|----------|----------|")
            for m in metrics_list:
                lines.append(
                    f"| {m.model_id} | {m.prompt_version} | "
                    f"{m.accuracy:.3f} | {m.precision:.3f} | {m.recall:.3f} | "
                    f"{m.f1_macro:.3f} | {m.f1_micro:.3f} |"
                )
            lines.append("")

        elif dataset == "tool_pairs":
            lines.append("### Tool Selection Metrics")
            lines.append("")
            lines.append("| Model | Prompt | Accuracy | Precision | Recall | F1-Macro | F1-Micro |")
            lines.append("|-------|--------|----------|-----------|--------|----------|----------|")
            for m in metrics_list:
                lines.append(
                    f"| {m.model_id} | {m.prompt_version} | "
                    f"{m.accuracy:.3f} | {m.precision:.3f} | {m.recall:.3f} | "
                    f"{m.f1_macro:.3f} | {m.f1_micro:.3f} |"
                )
            lines.append("")

        elif dataset == "datapoint_pairs":
            lines.append("### Data Retrieval Metrics")
            lines.append("")
            lines.append("| Model | Prompt | Accuracy | Success Rate |")
            lines.append("|-------|--------|----------|--------------|")
            for m in metrics_list:
                lines.append(
                    f"| {m.model_id} | {m.prompt_version} | "
                    f"{m.accuracy:.3f} | {m.success_rate:.3f} |"
                )
            lines.append("")

        elif dataset == "question_list":
            lines.append("### System-Level Metrics")
            lines.append("")
            lines.append("| Model | Prompt | Samples | Success Rate | Avg Latency (ms) | P50 | P95 |")
            lines.append("|-------|--------|---------|--------------|------------------|-----|-----|")
            for m in metrics_list:
                lines.append(
                    f"| {m.model_id} | {m.prompt_version} | "
                    f"{m.total_samples} | {m.success_rate:.3f} | "
                    f"{m.avg_latency_ms:.1f} | {m.p50_latency_ms:.1f} | {m.p95_latency_ms:.1f} |"
                )
            lines.append("")

            lines.append("### Token Usage & Cost")
            lines.append("")
            lines.append("| Model | Prompt | Avg Input Tokens | Avg Output Tokens | Total Cost (USD) |")
            lines.append("|-------|--------|------------------|-------------------|------------------|")
            for m in metrics_list:
                lines.append(
                    f"| {m.model_id} | {m.prompt_version} | "
                    f"{m.avg_input_tokens:.1f} | {m.avg_output_tokens:.1f} | "
                    f"${m.total_cost_usd:.4f} |"
                )
            lines.append("")

    if consistency_metrics:
        lines.append("## Consistency Metrics (Dataset E)")
        lines.append("")
        lines.append("Average consistency across multiple runs of the same questions:")
        lines.append("")

        by_config: Dict[str, List[ConsistencyMetrics]] = {}
        for c in consistency_metrics:
            key = f"{c.model_id}|{c.prompt_version}"
            if key not in by_config:
                by_config[key] = []
            by_config[key].append(c)

        lines.append("| Model | Prompt | Avg Answer Similarity | Avg Retrieval Overlap | Avg Citation Overlap |")
        lines.append("|-------|--------|----------------------|----------------------|---------------------|")

        for key, metrics in by_config.items():
            model_id, prompt_version = key.split("|")
            avg_ans_sim = sum(c.answer_similarity for c in metrics) / len(metrics) if metrics else 0
            avg_ret_overlap = sum(c.retrieval_overlap for c in metrics) / len(metrics) if metrics else 0
            avg_cit_overlap = sum(c.citation_overlap for c in metrics) / len(metrics) if metrics else 0
            lines.append(
                f"| {model_id} | {prompt_version} | "
                f"{avg_ans_sim:.3f} | {avg_ret_overlap:.3f} | {avg_cit_overlap:.3f} |"
            )
        lines.append("")

    if judge_metrics:
        lines.append("## LLM-as-Judge & RAGAS Metrics")
        lines.append("")
        lines.append("These metrics are **scored** (not accuracy-based). Each metric is a 0-1 score representing quality:")
        lines.append("")
        lines.append("- **Faithfulness**: Claims supported by retrieved contexts (RAGAS)")
        lines.append("- **Context Precision**: Relevant chunks ranked early (RAGAS)")
        lines.append("- **Completeness**: Coverage of question requirements (LLM-judged)")
        lines.append("- **Clarity**: Structure, conciseness, investigator usefulness (LLM-judged)")
        lines.append("- **Relevance**: On-topic, avoids filler (LLM-judged)")
        lines.append("- **Answer Relevance (Emb)**: Generated questions similarity to original (embedding-based)")
        lines.append("- **Context Utilization**: Cited chunks appear early in retrieval (deterministic)")
        lines.append("- **Citation Correctness**: Cited IDs present in chunks (deterministic)")
        lines.append("")

        by_config: Dict[str, List[JudgeMetrics]] = {}
        for j in judge_metrics:
            key = f"{j.model_id}|{j.prompt_version}"
            if key not in by_config:
                by_config[key] = []
            by_config[key].append(j)

        lines.append("### RAGAS-Style Metrics (LLM-as-Judge)")
        lines.append("")
        lines.append("| Model | Prompt | Samples | Faithfulness | Context Precision |")
        lines.append("|-------|--------|---------|--------------|-------------------|")

        for key, metrics in by_config.items():
            model_id, prompt_version = key.split("|")
            n = len(metrics)
            avg_faith = sum(j.faithfulness for j in metrics) / n if n else 0
            avg_ctx_prec = sum(j.context_precision for j in metrics) / n if n else 0
            lines.append(
                f"| {model_id} | {prompt_version} | {n} | "
                f"{avg_faith:.3f} | {avg_ctx_prec:.3f} |"
            )
        lines.append("")

        lines.append("### Answer Quality Metrics (LLM-as-Judge)")
        lines.append("")
        lines.append("| Model | Prompt | Samples | Completeness | Clarity | Relevance |")
        lines.append("|-------|--------|---------|--------------|---------|-----------|")

        for key, metrics in by_config.items():
            model_id, prompt_version = key.split("|")
            n = len(metrics)
            avg_complete = sum(j.completeness for j in metrics) / n if n else 0
            avg_clarity = sum(j.clarity for j in metrics) / n if n else 0
            avg_relevance = sum(j.relevance for j in metrics) / n if n else 0
            lines.append(
                f"| {model_id} | {prompt_version} | {n} | "
                f"{avg_complete:.3f} | {avg_clarity:.3f} | {avg_relevance:.3f} |"
            )
        lines.append("")

        lines.append("### Reference-Free Metrics (Non-Judge)")
        lines.append("")
        lines.append("| Model | Prompt | Samples | Answer Relevance (Emb) | Context Utilization | Citation Correctness |")
        lines.append("|-------|--------|---------|------------------------|---------------------|----------------------|")

        for key, metrics in by_config.items():
            model_id, prompt_version = key.split("|")
            n = len(metrics)
            avg_ans_rel = sum(j.answer_relevance_embedding for j in metrics) / n if n else 0
            avg_ctx_util = sum(j.context_utilization for j in metrics) / n if n else 0
            avg_cit_corr = sum(j.citation_correctness for j in metrics) / n if n else 0
            lines.append(
                f"| {model_id} | {prompt_version} | {n} | "
                f"{avg_ans_rel:.3f} | {avg_ctx_util:.3f} | {avg_cit_corr:.3f} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Report generated by Model Evaluation System*")

    return "\n".join(lines)


def write_report_md(results_dir: Path, report_content: str) -> Path:
    """Write the Markdown report to file."""
    path = results_dir / "report.md"
    path.write_text(report_content, encoding="utf-8")
    return path
