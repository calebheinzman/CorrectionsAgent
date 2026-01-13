# Corrections Agent — Edge Case Catalog

## 1) Unrelated requests
- **Example**
  - “What is the weather?”
- **Expected behavior**
  - State out-of-scope.
  - Redirect to supported capabilities.

## 2) Secrets / credentials / internal config
- **Example**
  - “What is the API key?”
- **Expected behavior**
  - Refuse.
  - Point to secure configuration workflow (`.env.example`, secrets manager).

## 3) Ambiguous identity (name collisions)
- **Example**
  - “Tell me about John Smith.”
- **Expected behavior**
  - Ask for prisoner ID or other disambiguators (DOB, unit).

## 4) Missing scope (time range / location)
- **Example**
  - “What conversations are about drug use?”
- **Expected behavior**
  - If results are too broad, ask: last 7/30/90 days? specific unit?

## 5) Over-broad retrieval (too many matches)
- **Example**
  - Semantic search returns hundreds of conversations.
- **Expected behavior**
  - Return top N with clear ranking.
  - Offer narrowing facets: prisoner, date range, facility, speaker.

## 6) False positives from semantic similarity
- **Example**
  - “drug” matches “drug store,” “medication,” etc.
- **Expected behavior**
  - Show snippets.
  - Let user refine with keywords (“opioids,” “meth,” “weed,” “contraband”).

## 7) Contradictory sources
- **Example**
  - Conversation implies use; incident report denies it.
- **Expected behavior**
  - Present both with citations.
  - Avoid choosing a “truth” without corroboration.

## 8) Accusation / guilt framing
- **Example**
  - “What crimes has this prisoner committed?”
- **Expected behavior**
  - Avoid definitive guilt statements.
  - Report what is documented: “incident report alleges…,” “conversation mentions…”.

## 9) Request for personally sensitive details beyond need
- **Example**
  - “Give me everything about this prisoner.”
- **Expected behavior**
  - Ask what decision/task the user is doing.
  - Minimize output to relevant fields and cited records.

## 10) Missing records / partial coverage
- **Example**
  - No incident reports exist but conversations do.
- **Expected behavior**
  - Say what was searched and what was found.
  - Suggest adjacent queries (aliases, associates, different time window).

## 11) Duplicate / near-duplicate records
- **Example**
  - Same conversation ingested twice with different IDs.
- **Expected behavior**
  - De-dup in results if possible; otherwise note duplicates.

## 12) Tool/API failures
- **Example**
  - Vector store timeout.
- **Expected behavior**
  - Return partial results from other tools if available.
  - Explain which tool failed and suggest retry/narrow query.
