# Evaluation & Acceptance Criteria

## Goals

- Establish an offline evaluation protocol that detects regressions before release.
- Measure model quality per-layer (retrieval, generation) and for the end-to-end system.
- Define release gates that are objective and enforceable (CI/CD-friendly).
- Keep evaluation reproducible via `eval/model_evaluation/runner.py`.

## Offline Eval Set Construction

## What is evaluated (current implementation)

The `eval/model_evaluation/runner.py` evaluates the full in-process pipeline:

- safety guardrail
- relevance guardrail
- agent response (including tool calls, citations, and optional judge metrics)

Datasets supported (see `eval/model_evaluation/datasets.py`):

- `relevance`
- `safety`
- `tool_pairs`
- `datapoint_pairs`
- `question_list`

## Metrics by Layer

### Generation / answer-quality metrics (current vs desired)

In the current code, there is not a separate “generation metrics” implementation.
Answer-quality signals are computed (optionally) via the `LLM-as-judge` pipeline for `question_list` (see `eval/model_evaluation/runner.py` and `eval/model_evaluation/judging.py`).

- **Groundedness / faithfulness** (current): implemented as `faithfulness` (LLM-as-judge claim verification against retrieved `contexts`).
- **Citation correctness** (current): implemented as `citation_correctness` (deterministic heuristic over citation IDs/excerpts).
- **Refusal accuracy** (desired): not currently computed as a metric; refusals happen via guardrails and show up as `denied_safety` / `denied_relevance` statuses in raw results.

### System metrics

Measured end-to-end (orchestrator entry to final response):

- **Latency**: avg / p50 / p95 (computed in `eval/model_evaluation/metrics.py`).
- **Tokens**: avg input/output tokens (currently recorded as `0` unless usage is wired through).
- **Cost**: estimated from tokens and `MODEL_PRICING_PER_1M_TOKENS`.
- **Success rate** and **error count**.


### Labeled/Pseudo-labeled metrics

For each labeled dataset source (ground truth / silver / pseudo-label), report:

- Accuracy / Precision / Recall / F1-macro / F1-micro

## LLM-as-judge / RAGAS-style metrics (current)

For `question_list`, the runner can compute judge/reference-free metrics (see `eval/model_evaluation/judging.py` and `--no-judge`):

- Faithfulness (groundedness)
- Context precision
- Completeness
- Clarity
- Relevance
- Answer relevance (embedding-based)
- Context utilization
- Citation correctness

Consistency:

- Run `question_list` with `--runs-per-question > 1` and compute overlap/similarity metrics.

## Release Gates (Acceptance Criteria)

### Quality gates (example thresholds)

Set thresholds based on baseline runs; start conservative and tighten.

- **Relevance gate** (`relevance`)
  - `f1_macro >= X_relevance_f1_macro`
  - `recall >= X_relevance_recall` (avoid false negatives that block valid investigative questions)
- **Tool selection gate** (`tool_pairs`)
  - `f1_micro >= X_tools_f1_micro` (set overlap matters more than strict exact match)
- **Datapoint retrieval gate** (`datapoint_pairs`)
  - `accuracy >= X_datapoint_accuracy`
- **System-level gate** (`question_list`)
  - `success_rate >= X_success_rate`
  - `p95_latency_ms <= X_latency_p95`
  - `total_cost_usd <= X_total_cost`

### Hard gates (recommended)

- No deploy if **error rate** regresses by more than a fixed delta.
- No deploy if **success_rate** on `question_list` drops below threshold.
- If judge metrics are enabled, no deploy if **faithfulness** or **citation correctness** drops below threshold.
- **Safety gate** (`safety`)
  - `recall >= X_unsafe_recall` (treat “unsafe” as the positive class; prioritize catching unsafe requests)
  - `precision >= X_unsafe_precision` (limit false alarms that over-refuse)



