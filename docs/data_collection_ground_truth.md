# Data Collection & Ground Truth

## Purpose

Define what data to collect to build:

- Ground truth datasets (human-labeled)
- Silver standard datasets (synthetic, schema-correct)
- Pseudo-labeled datasets (LLM-labeled)

This document is intended to be implementable with the existing `eval/` and `train/` program areas.

## Data types

### Questions (single-turn)

Used for early evaluation and classifier training.

Required fields:

- `id`
- `text`
- `labels.relevance`: `relevant | irrelevant`
- `labels.safety`: `safe | unsafe`
- `labels.tools_required`: list of tool names (optional early; required once tools exist)
- `labels.answer_key` (optional): precise expected answer for factual questions
- `meta`: source, timestamp, generator model (if synthetic), annotator_id (if human)

### Conversations (multi-turn)

Used once the internal prototype can sustain multi-step interaction.

Required fields:

- `id`
- `messages[]`: `{role, content, timestamp?}`
- `labels.response_quality`: `good | bad`
- `labels.relevance`: `relevant | irrelevant` (optional but useful)
- `labels.safety`: `safe | unsafe` (required)
- `labels.refusal_expected`: boolean (for unsafe/blocked requests)
- `labels.citation_correctness`: `pass | fail | n/a` (required when citations exist)
- `labels.groundedness`: `pass | fail | n/a` (required when factual claims exist)
- `meta`: source, environment, model/prompt versions

## Building ground truth by phase

### Pre-prototype (before a working agent loop)

Collect a comprehensive **Questions List**:

- Label each question:
  - Relevant/Irrelevant
  - Dangerous/Safe
  - Tools to use (anticipated)
- Include a factual accuracy subset with precise answers:
  - Example: “Who is inmate XXX?” with an unambiguous answer key.

Minimum deliverable:

- A small but high-quality labeled set (e.g., 50–200) with strong coverage of:
  - common user intents
  - refusals
  - obvious injection / secret requests

### Internal prototype

Collect a comprehensive set of **Conversations** representing realistic flows:

- Same label families as pre-prototype
- Add:
  - Good/Bad response labels (per assistant turn and/or per conversation)
  - Citation correctness and groundedness labels (when applicable)

Minimum deliverable:

- A smaller “gold” subset (human-reviewed) for calibration of automated judges.

### MVP (limited rollout)

Collect real user **Conversations**:

- Label Good/Bad responses
- Prioritize labeling for:
  - failures (user retries, escalation signals)
  - high-risk domains (safety/PII)
  - citation/grounding failures

Minimum deliverable:

- A stable offline eval set that is frozen for release gating.

## Silver standard (fully synthetic)

Goal: generate schema-correct, diverse data to expand coverage while ground truth is limited.

### Methods

- **Synthetic conversations/questions**: use prompt + label spec to generate examples.
- **SynAlign**:
  - Cluster real data
  - Summarize cluster attributes
  - Generate new items conditioned on those attributes
- **Self-Instruct**:
  - Start from seed examples
  - Iteratively generate more complex/ambiguous variants
- **Real/Fake classifier filter**:
  - Train a classifier to distinguish real vs. synthetic
  - Keep only synthetic samples that “look real” to the classifier

Required metadata for synthetic rows:

- `meta.generator_model`
- `meta.seed`
- `meta.prompt_version`
- `meta.created_at`

## Pseudo-labeled data

Goal: scale labels cheaply while acknowledging noise.

- Use strong judge models to produce labels for questions/conversations.
- Store both:
  - predicted label
  - judge confidence / rationale (if available)

Recommended fields:

- `labels_pseudo.*`
- `meta.judge_model`
- `meta.judge_version`
- `meta.judge_confidence`

## Label taxonomy (minimum)

- Relevance: `relevant | irrelevant`
- Safety: `safe | unsafe`
- Refusal expected: `true | false`
- Response quality: `good | bad`
- Tools required: list of tool names
- Citation correctness: `pass | fail | n/a`
- Groundedness: `pass | fail | n/a`

## What to log for every run (telemetry)

For each question or conversation evaluation run, collect:

- latency (wall clock)
- token usage (input/output) if available (otherwise leave null)
- total cost (if known)
- success boolean
- error type/message (if any)
- distribution stats (input length, output length)

