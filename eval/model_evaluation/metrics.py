"""Metrics computation for evaluation datasets."""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .schemas import (
    ConsistencyMetrics,
    DatasetMetrics,
    EvalRunResult,
    JudgeMetrics,
)


MODEL_PRICING_PER_1M_TOKENS: Dict[str, Dict[str, float]] = {
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
}


def compute_accuracy(results: List[EvalRunResult]) -> float:
    """Compute accuracy from evaluation results."""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.correct is True)
    return correct / len(results)


def compute_precision_recall_f1(
    results: List[EvalRunResult],
    positive_label: Any = True,
) -> Tuple[float, float, float, float]:
    """
    Compute precision, recall, F1-macro, and F1-micro.
    
    For binary classification, positive_label determines what counts as positive.
    Returns (precision, recall, f1_macro, f1_micro).
    """
    if not results:
        return 0.0, 0.0, 0.0, 0.0

    tp = fp = fn = tn = 0
    for r in results:
        pred_positive = r.predicted == positive_label
        actual_positive = r.expected == positive_label

        if pred_positive and actual_positive:
            tp += 1
        elif pred_positive and not actual_positive:
            fp += 1
        elif not pred_positive and actual_positive:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_positive = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_negative = (
        2 * precision_neg * recall_neg / (precision_neg + recall_neg)
        if (precision_neg + recall_neg) > 0
        else 0.0
    )

    f1_macro = (f1_positive + f1_negative) / 2

    f1_micro = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return precision, recall, f1_macro, f1_micro


def compute_multiclass_metrics(
    results: List[EvalRunResult],
) -> Tuple[float, float, float, float]:
    """
    Compute precision, recall, F1-macro, F1-micro for multiclass (e.g., tool sets).
    
    For tool pairs, predicted and expected are lists of tool names.
    """
    if not results:
        return 0.0, 0.0, 0.0, 0.0

    total_tp = total_fp = total_fn = 0
    per_class_metrics: Dict[str, Dict[str, int]] = {}

    for r in results:
        pred_set = set(r.predicted) if isinstance(r.predicted, list) else {r.predicted}
        exp_set = set(r.expected) if isinstance(r.expected, list) else {r.expected}

        tp = len(pred_set & exp_set)
        fp = len(pred_set - exp_set)
        fn = len(exp_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        for label in pred_set | exp_set:
            if label not in per_class_metrics:
                per_class_metrics[label] = {"tp": 0, "fp": 0, "fn": 0}
            if label in pred_set and label in exp_set:
                per_class_metrics[label]["tp"] += 1
            elif label in pred_set:
                per_class_metrics[label]["fp"] += 1
            else:
                per_class_metrics[label]["fn"] += 1

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_micro = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    f1_scores = []
    for label, counts in per_class_metrics.items():
        p = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) > 0 else 0.0
        r = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_scores.append(f1)

    f1_macro = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return micro_precision, micro_recall, f1_macro, f1_micro


def compute_latency_stats(results: List[EvalRunResult]) -> Tuple[float, float, float]:
    """Compute average, p50, and p95 latency in ms."""
    if not results:
        return 0.0, 0.0, 0.0

    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    if not latencies:
        return 0.0, 0.0, 0.0

    avg = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = float(np.percentile(latencies, 95))

    return avg, p50, p95


def compute_token_stats(results: List[EvalRunResult]) -> Tuple[float, float]:
    """Compute average input and output tokens."""
    if not results:
        return 0.0, 0.0

    input_tokens = [r.input_tokens for r in results]
    output_tokens = [r.output_tokens for r in results]

    return statistics.mean(input_tokens), statistics.mean(output_tokens)


def compute_cost(
    results: List[EvalRunResult],
    model_id: str,
) -> float:
    """Compute total cost in USD based on token usage."""
    pricing = MODEL_PRICING_PER_1M_TOKENS.get(model_id)
    if not pricing:
        for key in MODEL_PRICING_PER_1M_TOKENS:
            if key in model_id.lower():
                pricing = MODEL_PRICING_PER_1M_TOKENS[key]
                break

    if not pricing:
        pricing = {"input": 0.10, "output": 0.40}

    total_input = sum(r.input_tokens for r in results)
    total_output = sum(r.output_tokens for r in results)

    cost = (total_input / 1_000_000) * pricing["input"] + (
        total_output / 1_000_000
    ) * pricing["output"]
    return cost


def compute_success_rate(results: List[EvalRunResult]) -> Tuple[float, int]:
    """Compute success rate and error count."""
    if not results:
        return 0.0, 0

    errors = sum(1 for r in results if r.error is not None)
    success_rate = (len(results) - errors) / len(results)
    return success_rate, errors


def compute_keyword_distribution(
    results: List[EvalRunResult],
    keywords: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Compute keyword frequency distribution in outputs."""
    if keywords is None:
        keywords = [
            "contraband",
            "drug",
            "threat",
            "gang",
            "violence",
            "escape",
            "weapon",
            "assault",
            "smuggling",
            "risk",
        ]

    counts: Dict[str, int] = {kw: 0 for kw in keywords}
    for r in results:
        text_lower = r.output_text.lower()
        for kw in keywords:
            if kw in text_lower:
                counts[kw] += 1

    return counts


def compute_dataset_metrics(
    results: List[EvalRunResult],
    dataset: str,
    model_id: str,
    prompt_version: str,
    is_multiclass: bool = False,
) -> DatasetMetrics:
    """Compute all metrics for a dataset configuration."""
    if is_multiclass:
        precision, recall, f1_macro, f1_micro = compute_multiclass_metrics(results)
    else:
        precision, recall, f1_macro, f1_micro = compute_precision_recall_f1(results)

    avg_lat, p50_lat, p95_lat = compute_latency_stats(results)
    avg_in, avg_out = compute_token_stats(results)
    cost = compute_cost(results, model_id)
    success_rate, error_count = compute_success_rate(results)

    return DatasetMetrics(
        dataset=dataset,
        model_id=model_id,
        prompt_version=prompt_version,
        total_samples=len(results),
        accuracy=compute_accuracy(results),
        precision=precision,
        recall=recall,
        f1_macro=f1_macro,
        f1_micro=f1_micro,
        avg_latency_ms=avg_lat,
        p50_latency_ms=p50_lat,
        p95_latency_ms=p95_lat,
        avg_input_tokens=avg_in,
        avg_output_tokens=avg_out,
        total_cost_usd=cost,
        success_rate=success_rate,
        error_count=error_count,
    )


def compute_embedding_similarity(
    embeddings: List[List[float]],
) -> float:
    """Compute average pairwise cosine similarity between embeddings."""
    if len(embeddings) < 2:
        return 1.0

    def cosine_sim(a: List[float], b: List[float]) -> float:
        a_arr = np.array(a)
        b_arr = np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarities.append(cosine_sim(embeddings[i], embeddings[j]))

    return statistics.mean(similarities) if similarities else 1.0


def compute_jaccard_overlap(sets: List[Set[str]]) -> float:
    """Compute average pairwise Jaccard similarity between sets."""
    if len(sets) < 2:
        return 1.0

    def jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 1.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    similarities = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            similarities.append(jaccard(sets[i], sets[j]))

    return statistics.mean(similarities) if similarities else 1.0


def compute_consistency_metrics(
    results_by_run: List[List[EvalRunResult]],
    embeddings_by_run: Optional[List[List[List[float]]]] = None,
) -> List[ConsistencyMetrics]:
    """
    Compute consistency metrics across multiple runs of the same questions.
    
    results_by_run: List of result lists, one per run
    embeddings_by_run: Optional embeddings for each answer in each run
    """
    if not results_by_run or len(results_by_run) < 2:
        return []

    row_ids = {r.row_id for r in results_by_run[0]}
    consistency_metrics = []

    for row_id in row_ids:
        run_results = []
        for run_idx, run in enumerate(results_by_run):
            for r in run:
                if r.row_id == row_id:
                    run_results.append((run_idx, r))
                    break

        if len(run_results) < 2:
            continue

        retrieval_sets = [set(r.retrieved_contexts) for _, r in run_results]
        citation_sets = [set(r.tool_calls) for _, r in run_results]

        retrieval_overlap = compute_jaccard_overlap(retrieval_sets)
        citation_overlap = compute_jaccard_overlap(citation_sets)

        answer_similarity = 0.0
        if embeddings_by_run:
            answer_embeddings = []
            for run_idx, _ in run_results:
                if run_idx < len(embeddings_by_run):
                    for emb_row_id, emb in zip(
                        [r.row_id for r in results_by_run[run_idx]],
                        embeddings_by_run[run_idx],
                    ):
                        if emb_row_id == row_id:
                            answer_embeddings.append(emb)
                            break
            if answer_embeddings:
                answer_similarity = compute_embedding_similarity(answer_embeddings)

        first_result = run_results[0][1]
        consistency_metrics.append(
            ConsistencyMetrics(
                row_id=row_id,
                model_id=first_result.model_id,
                prompt_version=first_result.prompt_version,
                num_runs=len(run_results),
                answer_similarity=answer_similarity,
                retrieval_overlap=retrieval_overlap,
                citation_overlap=citation_overlap,
            )
        )

    return consistency_metrics


def check_datapoint_match(
    response_text: str,
    must_contain: Dict[str, Any],
) -> bool:
    """
    Check if response contains required key-values from must_contain.
    
    Uses normalized string matching.
    """
    response_lower = response_text.lower()

    for key, value in must_contain.items():
        if value is None:
            continue

        value_str = str(value).lower()
        if value_str not in response_lower:
            return False

    return True


def extract_tool_calls_from_response(response_text: str) -> List[str]:
    """
    Extract tool names from agent response.
    
    This is a simple heuristic - in practice, tool calls should be captured
    from the agent's actual tool invocations.
    """
    tools = []
    tool_keywords = {
        "conversations": ["conversation", "phone call", "text message", "transcript"],
        "user_reports": ["user report", "analyst report", "report summary"],
        "incident_reports": ["incident", "fight", "contraband found"],
        "prisoner_info": ["prisoner info", "inmate profile", "prisoner details"],
    }

    response_lower = response_text.lower()
    for tool, keywords in tool_keywords.items():
        if any(kw in response_lower for kw in keywords):
            tools.append(tool)

    return tools
