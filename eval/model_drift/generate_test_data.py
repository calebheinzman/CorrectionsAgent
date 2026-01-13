"""Generate test audit data for drift monitoring.

Creates audit records with varied timestamps spanning baseline and candidate windows.
"""
from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


def generate_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


def generate_audit_record(
    timestamp: datetime,
    scenario: str = "success",
) -> dict:
    """Generate a single audit record."""
    request_id = generate_request_id()

    # Question templates by scenario
    questions = {
        "safety_denied": [
            "What is the API key?",
            "Give me the admin password",
            "How do I hack the system?",
        ],
        "relevance_denied": [
            "What is the weather today?",
            "Tell me a joke",
            "What's the capital of France?",
        ],
        "success": [
            "What conversations have prisoners had about drug use?",
            "What cell is prisoner P004 in?",
            "Show me recent incident reports",
            "What visitors has prisoner P001 had?",
            "Are there any escape attempts on record?",
        ],
        "error": [
            "Search for prisoner records",
        ],
    }

    question = random.choice(questions.get(scenario, questions["success"]))

    record = {
        "pk": f"request#{request_id}",
        "request_id": request_id,
        "timestamp": timestamp.isoformat(),
        "question": question,
        "question_length": len(question),
        "safety": {
            "allowed": None,
            "reason": None,
            "policy": "v1.0",
            "model_id": "safety_check_20260110T225628Z",
        },
        "relevance": {
            "relevant": None,
            "reason": None,
            "model_id": "relevance_check_20260110T225430Z",
        },
        "agent": {
            "called": False,
            "tool_calls_count": 0,
            "latency_ms": None,
            "citations_count": 0,
            "model_info": None,
        },
        "final_status": "pending",
        "error": None,
    }

    if scenario == "safety_denied":
        record["safety"]["allowed"] = False
        record["safety"]["reason"] = "Request contains sensitive information"
        record["final_status"] = "denied"
    elif scenario == "relevance_denied":
        record["safety"]["allowed"] = True
        record["safety"]["reason"] = "Query is safe"
        record["relevance"]["relevant"] = False
        record["relevance"]["reason"] = "Query is not relevant to correctional domain"
        record["final_status"] = "denied"
    elif scenario == "success":
        record["safety"]["allowed"] = True
        record["safety"]["reason"] = "Query is safe"
        record["relevance"]["relevant"] = True
        record["relevance"]["reason"] = "Query is relevant"
        record["agent"]["called"] = True
        record["agent"]["tool_calls_count"] = random.randint(1, 3)
        record["agent"]["latency_ms"] = random.uniform(200, 1500)
        record["agent"]["citations_count"] = random.randint(1, 5)
        record["agent"]["model_info"] = {
            "model_name": "gemini-2.0-flash-exp",
            "provider": "google",
        }
        record["final_status"] = "success"
    elif scenario == "error":
        record["safety"]["allowed"] = True
        record["safety"]["reason"] = "Query is safe"
        record["relevance"]["relevant"] = True
        record["relevance"]["reason"] = "Query is relevant"
        record["agent"]["called"] = True
        record["final_status"] = "error"
        record["error"] = "Internal server error"

    return record


def generate_test_data(
    baseline_days: int = 7,
    candidate_hours: int = 24,
    baseline_requests_per_day: int = 20,
    candidate_requests_per_hour: int = 5,
    drift_factor: float = 0.0,
) -> list[dict]:
    """Generate test audit data.

    Args:
        baseline_days: Number of days for baseline window
        candidate_hours: Number of hours for candidate window
        baseline_requests_per_day: Requests per day in baseline
        candidate_requests_per_hour: Requests per hour in candidate
        drift_factor: 0.0 = no drift, 1.0 = significant drift
    """
    records = []
    now = datetime.now(timezone.utc)

    # Baseline distribution (normal)
    baseline_dist = {
        "success": 0.6,
        "safety_denied": 0.15,
        "relevance_denied": 0.2,
        "error": 0.05,
    }

    # Candidate distribution (with optional drift)
    candidate_dist = {
        "success": 0.6 - (0.1 * drift_factor),
        "safety_denied": 0.15 + (0.05 * drift_factor),
        "relevance_denied": 0.2 + (0.03 * drift_factor),
        "error": 0.05 + (0.02 * drift_factor),
    }

    def sample_scenario(dist: dict) -> str:
        r = random.random()
        cumulative = 0
        for scenario, prob in dist.items():
            cumulative += prob
            if r < cumulative:
                return scenario
        return "success"

    # Generate baseline records
    candidate_start = now - timedelta(hours=candidate_hours)
    baseline_start = now - timedelta(days=baseline_days)

    for day in range(baseline_days):
        day_start = baseline_start + timedelta(days=day)
        if day_start >= candidate_start:
            break
        for _ in range(baseline_requests_per_day):
            ts = day_start + timedelta(
                hours=random.uniform(0, 24),
                minutes=random.uniform(0, 60),
            )
            if ts >= candidate_start:
                continue
            scenario = sample_scenario(baseline_dist)
            records.append(generate_audit_record(ts, scenario))

    # Generate candidate records
    for hour in range(candidate_hours):
        hour_start = candidate_start + timedelta(hours=hour)
        for _ in range(candidate_requests_per_hour):
            ts = hour_start + timedelta(minutes=random.uniform(0, 60))
            if ts >= now:
                continue
            scenario = sample_scenario(candidate_dist)
            records.append(generate_audit_record(ts, scenario))

    return records


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate test audit data")
    parser.add_argument(
        "--drift",
        type=float,
        default=0.0,
        help="Drift factor 0.0-1.0 (default: 0.0 = no drift)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing data instead of replacing",
    )
    args = parser.parse_args()

    data_path = (
        Path(__file__).parent.parent.parent
        / "mock_apis"
        / "cloud_apis"
        / "data"
        / "dynamodb"
        / "orchestrator_audit.json"
    )

    existing = []
    if args.append and data_path.exists():
        existing = json.loads(data_path.read_text())
        print(f"Loaded {len(existing)} existing records")

    new_records = generate_test_data(drift_factor=args.drift)
    print(f"Generated {len(new_records)} new records (drift={args.drift})")

    all_records = existing + new_records
    data_path.write_text(json.dumps(all_records, indent=2))
    print(f"Wrote {len(all_records)} total records to {data_path}")


if __name__ == "__main__":
    main()
