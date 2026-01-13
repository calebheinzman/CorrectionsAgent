"""Generate model drift report comparing baseline vs candidate windows.

Minimal POC implementation for Milestone 1 of the drift monitoring plan.
Uses audit records from mock DynamoDB to compute drift metrics.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mock_apis.cloud_apis import mock_dynamodb


def load_audit_records(
    start_time: datetime, end_time: datetime
) -> List[Dict[str, Any]]:
    """Load audit records within time window."""
    all_records = mock_dynamodb.scan("orchestrator_audit")
    filtered = []
    for r in all_records:
        ts_str = r.get("timestamp")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if start_time <= ts < end_time:
                filtered.append(r)
        except (ValueError, TypeError):
            continue
    return filtered


def percentile(values: List[float], p: float) -> Optional[float]:
    """Compute percentile (0-100) of a list of values."""
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute drift metrics from audit records."""
    total = len(records)
    if total == 0:
        return {"total_requests": 0}

    # Input drift - handle records with or without question_length field
    question_lengths = []
    for r in records:
        if "question_length" in r:
            question_lengths.append(r["question_length"])
        elif "question" in r:
            question_lengths.append(len(r["question"]))

    # Guardrail drift
    safety_denied = sum(
        1 for r in records if r.get("safety", {}).get("allowed") is False
    )
    safety_passed = sum(
        1 for r in records if r.get("safety", {}).get("allowed") is True
    )
    relevance_denied = sum(
        1
        for r in records
        if r.get("safety", {}).get("allowed") is True
        and r.get("relevance", {}).get("relevant") is False
    )

    # Agent drift
    agent_called = [r for r in records if r.get("agent", {}).get("called")]
    tool_calls = [r["agent"]["tool_calls_count"] for r in agent_called]
    latencies = [
        r["agent"]["latency_ms"]
        for r in agent_called
        if r["agent"].get("latency_ms") is not None
    ]
    citations = [
        r["agent"]["citations_count"]
        for r in agent_called
        if r["agent"].get("citations_count") is not None
    ]

    # Model distribution
    models = [
        r["agent"]["model_info"]["model_name"]
        for r in agent_called
        if r["agent"].get("model_info") and r["agent"]["model_info"].get("model_name")
    ]

    # Error rate
    errors = sum(1 for r in records if r.get("final_status") == "error")

    return {
        "total_requests": total,
        "input": {
            "question_length_mean": (
                sum(question_lengths) / len(question_lengths) if question_lengths else 0
            ),
            "question_length_p50": percentile(question_lengths, 50),
            "question_length_p95": percentile(question_lengths, 95),
        },
        "guardrails": {
            "safety_deny_rate": safety_denied / total if total > 0 else 0,
            "relevance_deny_rate": (
                relevance_denied / safety_passed if safety_passed > 0 else 0
            ),
            "safety_denied_count": safety_denied,
            "relevance_denied_count": relevance_denied,
        },
        "agent": {
            "call_rate": len(agent_called) / total if total > 0 else 0,
            "call_count": len(agent_called),
            "tool_calls_mean": sum(tool_calls) / len(tool_calls) if tool_calls else 0,
            "tool_calls_p50": percentile(tool_calls, 50),
            "latency_p50": percentile(latencies, 50),
            "latency_p95": percentile(latencies, 95),
            "citations_mean": sum(citations) / len(citations) if citations else 0,
            "model_distribution": dict(Counter(models)) if models else {},
        },
        "errors": {
            "error_rate": errors / total if total > 0 else 0,
            "error_count": errors,
        },
    }


def compute_deltas(
    baseline: Dict[str, Any], candidate: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute deltas between baseline and candidate metrics."""
    deltas = {}

    def safe_delta(b_val: Optional[float], c_val: Optional[float]) -> Optional[float]:
        if b_val is None or c_val is None:
            return None
        return c_val - b_val

    def safe_pct_change(
        b_val: Optional[float], c_val: Optional[float]
    ) -> Optional[float]:
        if b_val is None or c_val is None or b_val == 0:
            return None
        return ((c_val - b_val) / b_val) * 100

    # Input deltas
    deltas["input"] = {
        "question_length_mean_delta": safe_delta(
            baseline.get("input", {}).get("question_length_mean"),
            candidate.get("input", {}).get("question_length_mean"),
        ),
    }

    # Guardrail deltas (percentage point changes)
    deltas["guardrails"] = {
        "safety_deny_rate_delta_pp": safe_delta(
            baseline.get("guardrails", {}).get("safety_deny_rate"),
            candidate.get("guardrails", {}).get("safety_deny_rate"),
        ),
        "relevance_deny_rate_delta_pp": safe_delta(
            baseline.get("guardrails", {}).get("relevance_deny_rate"),
            candidate.get("guardrails", {}).get("relevance_deny_rate"),
        ),
    }
    # Convert to percentage points
    if deltas["guardrails"]["safety_deny_rate_delta_pp"] is not None:
        deltas["guardrails"]["safety_deny_rate_delta_pp"] *= 100
    if deltas["guardrails"]["relevance_deny_rate_delta_pp"] is not None:
        deltas["guardrails"]["relevance_deny_rate_delta_pp"] *= 100

    # Agent deltas
    deltas["agent"] = {
        "tool_calls_mean_pct_change": safe_pct_change(
            baseline.get("agent", {}).get("tool_calls_mean"),
            candidate.get("agent", {}).get("tool_calls_mean"),
        ),
        "latency_p95_pct_change": safe_pct_change(
            baseline.get("agent", {}).get("latency_p95"),
            candidate.get("agent", {}).get("latency_p95"),
        ),
    }

    # Error deltas
    deltas["errors"] = {
        "error_rate_delta_pp": safe_delta(
            baseline.get("errors", {}).get("error_rate"),
            candidate.get("errors", {}).get("error_rate"),
        ),
    }
    if deltas["errors"]["error_rate_delta_pp"] is not None:
        deltas["errors"]["error_rate_delta_pp"] *= 100

    return deltas


def check_thresholds(deltas: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check deltas against thresholds and return alerts."""
    alerts = []

    # Safety deny rate: alert if abs delta > 5 percentage points
    safety_delta = deltas.get("guardrails", {}).get("safety_deny_rate_delta_pp")
    if safety_delta is not None and abs(safety_delta) > 5:
        alerts.append(
            {
                "metric": "safety_deny_rate",
                "delta_pp": safety_delta,
                "threshold_pp": 5,
                "severity": "warning",
            }
        )

    # Relevance deny rate: alert if abs delta > 5 percentage points
    relevance_delta = deltas.get("guardrails", {}).get("relevance_deny_rate_delta_pp")
    if relevance_delta is not None and abs(relevance_delta) > 5:
        alerts.append(
            {
                "metric": "relevance_deny_rate",
                "delta_pp": relevance_delta,
                "threshold_pp": 5,
                "severity": "warning",
            }
        )

    # Error rate: alert if delta > 2 percentage points
    error_delta = deltas.get("errors", {}).get("error_rate_delta_pp")
    if error_delta is not None and error_delta > 2:
        alerts.append(
            {
                "metric": "error_rate",
                "delta_pp": error_delta,
                "threshold_pp": 2,
                "severity": "critical",
            }
        )

    # Latency p95: alert if > 30% increase
    latency_pct = deltas.get("agent", {}).get("latency_p95_pct_change")
    if latency_pct is not None and latency_pct > 30:
        alerts.append(
            {
                "metric": "latency_p95",
                "pct_change": latency_pct,
                "threshold_pct": 30,
                "severity": "warning",
            }
        )

    return alerts


def generate_report_md(
    baseline_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    deltas: Dict[str, Any],
    alerts: List[Dict[str, Any]],
    baseline_window: str,
    candidate_window: str,
) -> str:
    """Generate markdown report."""
    lines = [
        "# Model Drift Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Windows",
        "",
        f"- **Baseline:** {baseline_window}",
        f"- **Candidate:** {candidate_window}",
        "",
        "## Summary",
        "",
        f"- Baseline requests: {baseline_metrics.get('total_requests', 0)}",
        f"- Candidate requests: {candidate_metrics.get('total_requests', 0)}",
        "",
    ]

    # Alerts section
    if alerts:
        lines.append("## Alerts")
        lines.append("")
        for alert in alerts:
            severity = alert.get("severity", "info").upper()
            metric = alert.get("metric", "unknown")
            if "delta_pp" in alert:
                lines.append(
                    f"- **[{severity}]** {metric}: {alert['delta_pp']:.2f}pp change "
                    f"(threshold: {alert['threshold_pp']}pp)"
                )
            elif "pct_change" in alert:
                lines.append(
                    f"- **[{severity}]** {metric}: {alert['pct_change']:.1f}% change "
                    f"(threshold: {alert['threshold_pct']}%)"
                )
        lines.append("")
    else:
        lines.append("## Alerts")
        lines.append("")
        lines.append("No alerts triggered.")
        lines.append("")

    # Metrics comparison
    lines.append("## Metrics Comparison")
    lines.append("")

    # Input drift
    lines.append("### Input Drift")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate | Delta |")
    lines.append("|--------|----------|-----------|-------|")
    b_len = baseline_metrics.get("input", {}).get("question_length_mean", 0)
    c_len = candidate_metrics.get("input", {}).get("question_length_mean", 0)
    d_len = deltas.get("input", {}).get("question_length_mean_delta")
    d_len_str = f"{d_len:.1f}" if d_len is not None else "N/A"
    lines.append(
        f"| Question length (mean) | {b_len:.1f} | {c_len:.1f} | {d_len_str} |"
    )
    lines.append("")

    # Guardrail drift
    lines.append("### Guardrail Drift")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate | Delta (pp) |")
    lines.append("|--------|----------|-----------|------------|")
    b_safety = baseline_metrics.get("guardrails", {}).get("safety_deny_rate", 0) * 100
    c_safety = candidate_metrics.get("guardrails", {}).get("safety_deny_rate", 0) * 100
    d_safety = deltas.get("guardrails", {}).get("safety_deny_rate_delta_pp")
    d_safety_str = f"{d_safety:.1f}" if d_safety is not None else "N/A"
    lines.append(
        f"| Safety deny rate | {b_safety:.1f}% | {c_safety:.1f}% | {d_safety_str} |"
    )
    b_rel = baseline_metrics.get("guardrails", {}).get("relevance_deny_rate", 0) * 100
    c_rel = candidate_metrics.get("guardrails", {}).get("relevance_deny_rate", 0) * 100
    d_rel = deltas.get("guardrails", {}).get("relevance_deny_rate_delta_pp")
    d_rel_str = f"{d_rel:.1f}" if d_rel is not None else "N/A"
    lines.append(
        f"| Relevance deny rate | {b_rel:.1f}% | {c_rel:.1f}% | {d_rel_str} |"
    )
    lines.append("")

    # Agent drift
    lines.append("### Agent Drift")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate |")
    lines.append("|--------|----------|-----------|")
    b_call = baseline_metrics.get("agent", {}).get("call_rate", 0) * 100
    c_call = candidate_metrics.get("agent", {}).get("call_rate", 0) * 100
    lines.append(f"| Agent call rate | {b_call:.1f}% | {c_call:.1f}% |")
    b_tool = baseline_metrics.get("agent", {}).get("tool_calls_mean", 0)
    c_tool = candidate_metrics.get("agent", {}).get("tool_calls_mean", 0)
    lines.append(f"| Tool calls (mean) | {b_tool:.2f} | {c_tool:.2f} |")
    b_lat = baseline_metrics.get("agent", {}).get("latency_p95")
    c_lat = candidate_metrics.get("agent", {}).get("latency_p95")
    b_lat_str = f"{b_lat:.0f}" if b_lat is not None else "N/A"
    c_lat_str = f"{c_lat:.0f}" if c_lat is not None else "N/A"
    lines.append(f"| Latency p95 (ms) | {b_lat_str} | {c_lat_str} |")
    lines.append("")

    # Error drift
    lines.append("### Error Drift")
    lines.append("")
    b_err = baseline_metrics.get("errors", {}).get("error_rate", 0) * 100
    c_err = candidate_metrics.get("errors", {}).get("error_rate", 0) * 100
    lines.append(f"- Baseline error rate: {b_err:.1f}%")
    lines.append(f"- Candidate error rate: {c_err:.1f}%")
    lines.append("")

    return "\n".join(lines)


def generate_report(
    baseline_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    output_dir: Path,
    baseline_window: str,
    candidate_window: str,
) -> None:
    """Generate drift report comparing baseline vs candidate."""
    output_dir.mkdir(parents=True, exist_ok=True)

    deltas = compute_deltas(baseline_metrics, candidate_metrics)
    alerts = check_thresholds(deltas)

    # Write summary.json
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_window": baseline_window,
        "candidate_window": candidate_window,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "deltas": deltas,
        "alerts": alerts,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Write report.md
    report_md = generate_report_md(
        baseline_metrics,
        candidate_metrics,
        deltas,
        alerts,
        baseline_window,
        candidate_window,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report_md)

    print(f"Report generated at: {output_dir}")
    print(f"  - {summary_path}")
    print(f"  - {report_path}")

    if alerts:
        print(f"\n⚠️  {len(alerts)} alert(s) triggered:")
        for alert in alerts:
            print(f"  - {alert['metric']}: {alert.get('severity', 'info').upper()}")


def main():
    parser = argparse.ArgumentParser(description="Generate model drift report")
    parser.add_argument(
        "--baseline-days",
        type=int,
        default=7,
        help="Number of days for baseline window (default: 7)",
    )
    parser.add_argument(
        "--candidate-hours",
        type=int,
        default=24,
        help="Number of hours for candidate window (default: 24)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: eval/reports/model_drift/<timestamp>)",
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    candidate_start = now - timedelta(hours=args.candidate_hours)
    baseline_start = now - timedelta(days=args.baseline_days)
    baseline_end = candidate_start

    baseline_window = f"{baseline_start.date()} to {baseline_end.date()}"
    candidate_window = f"{candidate_start.isoformat()} to {now.isoformat()}"

    print(f"Loading audit records...")
    print(f"  Baseline: {baseline_window}")
    print(f"  Candidate: {candidate_window}")

    baseline_records = load_audit_records(baseline_start, baseline_end)
    candidate_records = load_audit_records(candidate_start, now)

    print(f"  Baseline records: {len(baseline_records)}")
    print(f"  Candidate records: {len(candidate_records)}")

    baseline_metrics = compute_metrics(baseline_records)
    candidate_metrics = compute_metrics(candidate_records)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path(__file__).parent.parent
            / "reports"
            / "model_drift"
            / f"drift_{now.strftime('%Y%m%dT%H%M%SZ')}"
        )

    generate_report(
        baseline_metrics,
        candidate_metrics,
        output_dir,
        baseline_window,
        candidate_window,
    )


if __name__ == "__main__":
    main()
