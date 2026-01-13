# Corrections Agent — Costs 

Quick document to outline costs of this system. Most of these are just back of the napkin estimates. rather than real world numbers. 
## What costs money 
- **LLM inference**: input/output tokens for each user request.
- **Embeddings**: tokens embedded for semantic search + indexing.
- **Vector search/storage**: where embeddings live and how they’re queried (local FAISS vs managed service).
- **Compute**: running the API/services (CPU/RAM) and any autoscaling.
- **Logging/metrics**: centralized logs/metrics/traces (e.g., CloudWatch) and retention.
- **Data storage**: storing conversations/reports (object storage) and metadata (DB).

## Usage 
| Agency Tier | Est. # of Agencies (Corrections Agent) | Est. Users per Agency | Avg. Daily Requests per User | Total Daily Requests |
|---|---:|---:|---:|---:|
| Tier 1 (State DOCs) | 5 | 100 | 20 | 10,000 |
| Tier 2 (Large County) | 15 | 10 | 8 | 1,200 |
| Tier 3 (Small/Pilot) | 20 | 2 | 5 | 200 |
| **TOTAL** | **40** |  |  | **11,400** |

Derived:
- **Monthly requests (30d)**: `11,400 * 30` = **342,000**
- **Yearly requests (365d)**: `11,400 * 365` = **4,161,000**

## Total cost estimate (LLM + query embedding only)
This is a single, simple estimate for *variable* model usage costs.

Assumptions:
- **Model**: Gemini 2.0 Flash-Lite (standard)
- **Pricing** (public list): $0.075 / 1M input tokens, $0.30 / 1M output tokens
- **Embeddings**: Gemini Embedding at $0.15 / 1M tokens
- **Per request tokens**: 2,500 input + 500 output + 100 embedding tokens

Computed variable cost per request:
- **~$0.0003525 / request**

## Variable cost estimate including agentic work
Assumption:
- **Agentic multiplier**: each incoming request triggers internal “thinking + actions + interpretation” that totals ~**12x** the token usage of a single pass.

Computed variable cost per incoming request:
- **~$0.00423 / request** (`0.0003525 * 12`)

Totals (with 12x multiplier):
- **Daily**: `11,400 * 0.00423` = **~$48.22 / day**
- **Monthly (30d)**: `342,000 * 0.00423` = **~$1,446.66 / month**
- **Yearly (365d)**: `4,161,000 * 0.00423` = **~$17,601.03 / year**

## AWS services + scaling costs to handle daily demand
This is the *non-LLM* AWS cost to serve the request volume above.

Included (per architecture diagram):
- **Frontend + entry**: Amplify + API Gateway
- **Orchestration**: Lambda handler
- **Compute**: autoscaled EC2 for Guardrails + Agent services
- **State + storage**: DynamoDB (audit logs), S3 (artifacts)
- **Ops**: CloudWatch (logs/metrics), Secrets Manager
- **Registry**: model/prompt registry

Traffic sizing (sanity check):
- **Average**: **~0.13 req/s** (11,400/day)
- **Peak planning**: **~1–2 req/s** (bursty daytime usage)

Rough total (infra only, excluding LLM):
- **~$250–$900/mo**

Combined monthly estimate (12x variable LLM + infra):
- **~$1,700–$2,400/mo** (`$1,446.66` variable + infra)
