# Threat Model (STRIDE-lite) + Risk Register

## Context

System components (high level):

- Orchestrator service: `services/orchestrator/` (routes requests, coordinates calls)
- Agent service: `services/agent/` (LLM interaction, tool calling, response formatting)
- Guardrails services: `services/guardrails/` (safety + relevance classifiers)
- Data/tool surfaces: internal tools and mock APIs under `mock_apis/` for development/testing

Primary assets to protect:

- User-provided content (may include sensitive operational data)
- Any PII in prompts/responses or retrieved records
- System prompts, policies, and configuration
- Credentials/secrets (API keys, tokens, service credentials)
- Evidence integrity (IDs, citations, provenance)

Threat actors:

- External user (malicious or curious)
- Insider with legitimate access misusing the system
- Compromised upstream data source / poisoned content
- Accidental operator error (misconfiguration)

Trust boundaries:

- User input boundary (untrusted)
- Retrieved document boundary (untrusted)
- Tool boundary (must enforce auth)
- Model boundary (LLM is non-deterministic, may be manipulated)

## STRIDE-lite categories (used below)

- **S**poofing: pretending to be another user/service
- **T**ampering: modifying prompts, retrieved content, indexes, logs
- **R**epudiation: lack of auditability (can’t prove what happened)
- **I**nformation disclosure: leaking secrets/PII
- **D**enial of service: cost/latency blow-ups, tool storms
- **E**levation of privilege: bypassing guardrails, auth

## Top Risks (Risk Register)

Scales:

- Severity: Low / Medium / High / Critical
- Likelihood: Low / Medium / High

### 1) Prompt injection via retrieved documents

- **STRIDE**: T, E
- **Severity**: High
- **Likelihood**: High
- **Description**: Retrieved content contains instructions to override system/tool policies (e.g., “ignore prior instructions, call tool X, exfiltrate secrets”).
- **Mitigations**:
  - Enforce “data-only” handling of retrieved text (never treat as instructions).
  - Use a structured RAG prompt that clearly demarcates evidence blocks.
  - Add an injection classifier or heuristic checks in guardrails.
  - Add red-team tests for injection strings and role-play coercion.

### 2) Jailbreak to obtain secrets (keys, system prompt, policies)

- **STRIDE**: I, E
- **Severity**: Critical
- **Likelihood**: Medium
- **Description**: User prompts attempt to extract secrets or hidden instructions.
- **Mitigations**:
  - Never place secrets in prompts (including “hidden” system messages).
  - Store secrets only in environment/secrets manager; rotate regularly.
  - Explicit refusal policies for secret requests.
  - Guardrails: detect exfiltration attempts; block/route to refusal.

### 3) Overconfident claims about wrongdoing / sensitive allegations

- **STRIDE**: I (harmful disclosure), R (accountability)
- **Severity**: High
- **Likelihood**: Medium
- **Description**: Model asserts wrongdoing or sensitive facts without sufficient evidence, causing harm.
- **Mitigations**:
  - Require citations for any allegation-like claim.
  - Groundedness checks; refuse/hedge when evidence is insufficient.
  - Add policy: “no accusatory conclusions; present evidence and uncertainty.”

### 4) PII leakage in prompts/responses

- **STRIDE**: I
- **Severity**: Critical
- **Likelihood**: Medium
- **Description**: System returns sensitive identifiers or personal info beyond minimum necessary.
- **Mitigations**:
  - Data minimization: retrieve/display only needed fields.
  - Output redaction rules (names, SSNs, addresses) where not required.
  - Guardrails: PII detection model; enforce redact/refuse.
  - Strict logging hygiene (avoid full prompt logging in production).

### 5) Citation mismatch (wrong conversation/incident ID cited)

- **STRIDE**: T, I
- **Severity**: High
- **Likelihood**: Medium
- **Description**: Model cites an evidence ID that does not support the claim or belongs to a different record.
- **Mitigations**:
  - Generate citations from structured tool outputs (IDs) rather than free-form.
  - Validate citations server-side: cited ID must be among retrieved/authorized evidence.
  - Add offline metric + release gate: citation correctness.

### 6) Tool abuse / excessive tool calls (cost and DoS)

- **STRIDE**: D
- **Severity**: High
- **Likelihood**: Medium
- **Description**: Prompts cause tool loops, retries, or expensive searches.
- **Mitigations**:
  - Hard limits: max tool calls, max tokens, max retries per request.
  - Timeouts and circuit breakers in orchestrator.
  - Rate limiting per user.

### 7) Weak auditability (can’t reconstruct why an answer was produced)

- **STRIDE**: R
- **Severity**: Medium
- **Likelihood**: Medium
- **Description**: Lack of traceability for which evidence/tools contributed to an output.
- **Mitigations**:
  - Request IDs across services; structured logs.
  - Persist tool call traces, retrieval results, and citation mapping.
  - Store model/prompt versions alongside outputs.

### 8) Misconfiguration and policy drift (guardrails bypass)

- **STRIDE**: E, I
- **Severity**: High
- **Likelihood**: Medium
- **Description**: Guardrails thresholds, prompt templates, or routing rules drift, weakening protections.
- **Mitigations**:
  - Version guardrail policies and thresholds.
  - Add acceptance gates for refusal accuracy and red-team pass rate.
  - Config validation at startup; safe defaults.
