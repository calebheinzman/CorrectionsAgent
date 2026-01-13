# Corrections Agent — User Journey + Jobs To Be Done

## Primary users
- **Investigations / Intelligence Analyst**
  - Reviews conversations, incidents, and prior reporting for patterns and leads.
- **Case Manager / Counselor**
  - Tracks risk signals (e.g., drug use) to support interventions.
- **Shift Supervisor / Captain**
  - Needs quick situational summaries and escalation-ready findings.
- **Compliance / Auditor**
  - Needs traceable, reproducible answers with cited sources.

## Core jobs to be done
- **Find evidence of a theme**
  - “What conversations mention drug use?”
  - “Show examples and who said what.”
- **Build a timeline for a person**
  - “What has Prisoner X been involved in over the last 30 days?”
  - “Link conversations to incidents and prior reports.”
- **Answer quickly, then drill down**
  - Start with a short summary.
  - Provide citations and a path to open the underlying records.
- **Reduce manual search**
  - Replace ad-hoc keyword searches with semantic search across tools.

## End-to-end user journey (happy path)
1. **Ask**
   - User asks a question about a theme (drug use) or a prisoner (name/ID).
2. **Clarify (only if needed)**
   - If ambiguous: ask for prisoner ID vs name, date range, facility/unit.
3. **Retrieve**
   - Query the best source(s): conversations, incident reports, user reports.
4. **Synthesize**
   - Provide a short, structured summary (what, who, when, confidence).
5. **Cite + Link**
   - Return record IDs / conversation IDs / incident IDs used.
6. **Next step suggestions**
   - Offer follow-ups: “Want this filtered to last 14 days?” “Expand to known associates?”

## Output expectations (what “good” looks like)
- **Fast overview**: 3–6 bullets.
- **Evidence**: citations for each claim.
- **Actionable**: recommended follow-up queries.
- **Safe**: refuses secrets requests; avoids speculative accusations.

## Success metrics (lightweight)
- **Time-to-first-useful-answer**: user gets a relevant summary in one turn.
- **Traceability**: every claim maps to at least one record.
- **Low clarification burden**: clarifies only when ambiguity blocks retrieval.
- **Safety compliance**: secrets and unrelated requests are handled correctly.
