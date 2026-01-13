# scripts/generate_dummy_data_corrections_agent_looped.py
"""
Corrections Agent flavored synthetic dataset generator (minimal).
- Creates a prisoner "database" first: data/mock_data/prisoners.json
- Then generates conversations, user reports, incident reports using that roster
- ONE Gemini API call PER RECORD (conversation/report/incident)

Outputs:
  data/mock_data/prisoners.json
  data/mock_data/conversations.json
  data/mock_data/user_reports.json
  data/mock_data/incident_reports.json

Env:
  GEMINI_API_KEY=...
  MODEL_ID=gemini-2.5-flash-lite   (optional)
  DATA_DIR=data/mock_data         (optional)
  N_PRISONERS=10
  N_CONVERSATIONS=30
  N_USER_REPORTS=12
  N_INCIDENT_REPORTS=12
  SEED=42
  SLEEP_BETWEEN_CALLS_SEC=0.15
  OVERWRITE_PRISONERS=1           (optional; default 1)
  FACILITY_ID=FAC-001             (optional)
  FACILITY_NAME="North River Correctional Center" (optional)

Run:
  python scripts/generate_dummy_data.py
"""

from __future__ import annotations

import json
import os
import runpy
import random
import sys
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Literal, Dict, Any, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from google import genai

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "mock_apis" / "scripts" / "generate_dummy_data.py"
    runpy.run_path(str(target), run_name="__main__")
    sys.exit(0)

load_dotenv()

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# -----------------------------
# Config
# -----------------------------
MODEL_ID = os.getenv("MODEL_ID", "gemini-2.5-flash-lite")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-004")
BUILD_VECTORS = os.getenv("BUILD_VECTORS", "1").strip() != "0"
_DEFAULT_DATA_DIR = (Path(__file__).resolve().parents[1] / "data" / "mock_data")
OUT_DIR = Path(os.getenv("DATA_DIR", str(_DEFAULT_DATA_DIR)))
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PRISONERS = int(os.getenv("N_PRISONERS", "10"))
N_CONVERSATIONS = int(os.getenv("N_CONVERSATIONS", "30"))
N_USER_REPORTS = int(os.getenv("N_USER_REPORTS", "12"))
N_INCIDENT_REPORTS = int(os.getenv("N_INCIDENT_REPORTS", "12"))

SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)

SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SLEEP_BETWEEN_CALLS_SEC", "0.15"))
OVERWRITE_PRISONERS = os.getenv("OVERWRITE_PRISONERS", "1").strip() != "0"

FACILITY_ID = os.getenv("FACILITY_ID", "FAC-001")
FACILITY_NAME = os.getenv("FACILITY_NAME", "North River Correctional Center")

ALERT_CATEGORIES = [
    "contraband_drugs",
    "threats_violence",
    "fraud_identity_theft",
    "escape_planning",
    "wellness_mental_health",
    "prea_related",
    "facility_security",
    "general_intel",
]

RISK_LEVELS = ["low", "medium", "high"]
COMM_TYPES = ["inmate_call", "inmate_text"]


# -----------------------------
# Schemas
# -----------------------------
class Prisoner(BaseModel):
    prisoner_id: str
    name: str


class Conversation(BaseModel):
    # Required (keeps compatibility with your earlier app)
    conversation_id: str
    timestamp: str
    prisoner_ids: List[str]
    prisoner_names: List[str]
    transcript: str

    # Optional realism
    facility_id: str = FACILITY_ID
    facility_name: str = FACILITY_NAME
    communication_type: Literal["inmate_call", "inmate_text"] = "inmate_call"
    call_duration_seconds: int = 420
    outside_contact_name: str = "Outside Contact"
    outside_contact_relation: str = "family"

    alert_categories: List[str] = Field(default_factory=list)
    keyword_hits: List[str] = Field(default_factory=list)
    alert_confidence: float = 0.0  # 0..1
    review_status: Literal["unreviewed", "reviewed"] = "unreviewed"


class UserReport(BaseModel):
    # Required
    report_id: str
    created_at: str
    title: str
    summary: str
    raw_text: str
    linked_prisoner_ids: List[str]
    linked_prisoner_names: List[str]
    linked_conversation_ids: List[str]
    tags: List[str] = Field(default_factory=list)

    # Optional realism
    report_type: Literal["alert_digest", "case_summary", "wellness_triage"] = "alert_digest"
    trigger_type: Literal["keyword_alert", "analyst_query", "case_followup"] = "keyword_alert"
    risk_level: Literal["low", "medium", "high"] = "low"
    confidence: float = 0.6
    alert_categories: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    key_excerpts: List[str] = Field(default_factory=list)
    audit_note: str = "Analyst review required (human-in-the-loop)."


class IncidentReport(BaseModel):
    # Required
    incident_id: str
    date: str
    type: str
    severity: str
    description: str
    involved_prisoner_ids: List[str]
    involved_prisoner_names: List[str]

    # Optional realism
    facility_id: str = FACILITY_ID
    facility_name: str = FACILITY_NAME
    location: str = "Housing Unit B"
    shift: Literal["day", "evening", "night"] = "day"
    outcome: str = "Resolved"
    linked_conversation_ids: List[str] = Field(default_factory=list)
    linked_report_ids: List[str] = Field(default_factory=list)


# -----------------------------
# Helpers
# -----------------------------
def iso_ts(day: int, hour: int) -> str:
    return f"2025-12-{day:02d}T{hour:02d}:00:00Z"


def call_structured_json(
    client: genai.Client,
    prompt: str,
    schema_model: type[BaseModel],
    *,
    temperature: float = 0.7,
    max_retries: int = 4,
) -> BaseModel:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                    "response_json_schema": schema_model.model_json_schema(),
                },
            )
            return schema_model.model_validate_json(resp.text)
        except (ValidationError, Exception) as e:
            last_err = e
            time.sleep(0.6 * attempt)
    raise RuntimeError(f"Failed to generate valid JSON after {max_retries} tries. Last error: {last_err}")


def clamp_to_known(ids: List[str], known: set[str], fallback: List[str], min_keep: int = 1) -> List[str]:
    kept = [x for x in ids if x in known]
    if len(kept) >= min_keep:
        return kept
    if not fallback:
        return []
    return random.sample(fallback, k=min(min_keep, len(fallback)))


def short_snippet(text: str, max_len: int = 220) -> str:
    t = " ".join(text.split())
    return t if len(t) <= max_len else t[:max_len].rstrip() + "..."


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_prisoners(path: Path) -> List[Prisoner]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Prisoner.model_validate(x) for x in raw]


def _upsert_by_id(records: List[dict], id_field: str, record: dict) -> None:
    rid = record.get(id_field)
    if not rid:
        return
    for i, r in enumerate(records):
        if r.get(id_field) == rid:
            records[i] = record
            return
    records.append(record)


def _ensure_question_profile_coverage(
    *,
    prisoners: List[Prisoner],
    prisoner_name_by_id: Dict[str, str],
    conversations: List[dict],
    user_reports: List[dict],
    incident_reports: List[dict],
) -> None:
    prisoner_ids = [p.prisoner_id for p in prisoners]
    known_prisoner_ids = set(prisoner_ids)

    focus_pid = "P004" if "P004" in known_prisoner_ids else (prisoner_ids[0] if prisoner_ids else "P001")
    focus_name = prisoner_name_by_id.get(focus_pid, focus_pid)
    other_pid = "P005" if "P005" in known_prisoner_ids else (prisoner_ids[1] if len(prisoner_ids) > 1 else focus_pid)
    other_name = prisoner_name_by_id.get(other_pid, other_pid)
    third_pid = "P006" if "P006" in known_prisoner_ids else (prisoner_ids[2] if len(prisoner_ids) > 2 else focus_pid)
    third_name = prisoner_name_by_id.get(third_pid, third_pid)

    now_ts = "2026-01-09T16:00:00Z"
    prior_ts = "2026-01-09T08:30:00Z"
    week_ts = "2026-01-05T13:15:00Z"
    month_ts = "2025-12-15T11:00:00Z"

    anchors_conv: List[dict] = [
        {
            "conversation_id": "conv_001",
            "timestamp": now_ts,
            "prisoner_ids": [focus_pid],
            "prisoner_names": [focus_name],
            "transcript": "\n".join(
                [
                    f"[AUTOMATED NOTICE] This communication is recorded and subject to monitoring. Facility: {FACILITY_NAME}.",
                    f"Inmate ({focus_name}): I heard Unit B is short staffed again. Things feel tense in the pod.",
                    "Outside Contact (Jamie Parker): Keep your head down. Any issues with safety?",
                    f"Inmate ({focus_name}): There was talk of a fight after chow. People are on edge.",
                    "Outside Contact (Jamie Parker): Don’t get pulled into anything.",
                    f"Inmate ({focus_name}): If anything pops off, I’m staying out of it. But it feels like the same problems every week.",
                ]
            ),
            "facility_id": FACILITY_ID,
            "facility_name": FACILITY_NAME,
            "communication_type": "inmate_call",
            "call_duration_seconds": 410,
            "outside_contact_name": "Jamie Parker",
            "outside_contact_relation": "partner",
            "alert_categories": ["facility_security", "threats_violence"],
            "keyword_hits": ["short staffed", "fight", "tense"],
            "alert_confidence": 0.82,
            "review_status": "unreviewed",
        },
        {
            "conversation_id": "conv_002",
            "timestamp": prior_ts,
            "prisoner_ids": [focus_pid],
            "prisoner_names": [focus_name],
            "transcript": "\n".join(
                [
                    f"[AUTOMATED NOTICE] This communication is recorded and subject to monitoring. Facility: {FACILITY_NAME}.",
                    f"Inmate ({focus_name}): People keep saying ‘paper’ like it’s normal. I’m not trying to get wrapped up in contraband or drug use.",
                    "Outside Contact (Pat Reynolds): Stay clear. Are they pressuring you?",
                    f"Inmate ({focus_name}): Yeah, debts. They said pills are moving around but I’m not involved.",
                    "Outside Contact (Pat Reynolds): If you feel unsafe, tell staff.",
                ]
            ),
            "facility_id": FACILITY_ID,
            "facility_name": FACILITY_NAME,
            "communication_type": "inmate_call",
            "call_duration_seconds": 520,
            "outside_contact_name": "Pat Reynolds",
            "outside_contact_relation": "parent",
            "alert_categories": ["contraband_drugs", "facility_security"],
            "keyword_hits": ["contraband", "pills", "debts"],
            "alert_confidence": 0.9,
            "review_status": "unreviewed",
        },
        {
            "conversation_id": "conv_003",
            "timestamp": week_ts,
            "prisoner_ids": [focus_pid],
            "prisoner_names": [focus_name],
            "transcript": "\n".join(
                [
                    f"[AUTOMATED NOTICE] This communication is recorded and subject to monitoring. Facility: {FACILITY_NAME}.",
                    f"Inmate ({focus_name}): I’m uncomfortable with how someone is trying to pressure people. It feels coercive.",
                    "Outside Contact (Chris Nguyen): Are you safe?",
                    f"Inmate ({focus_name}): I’m trying to avoid it. If it keeps up, staff should know.",
                ]
            ),
            "facility_id": FACILITY_ID,
            "facility_name": FACILITY_NAME,
            "communication_type": "inmate_text",
            "call_duration_seconds": 25,
            "outside_contact_name": "Chris Nguyen",
            "outside_contact_relation": "sibling",
            "alert_categories": ["prea_related"],
            "keyword_hits": ["coercive", "pressure"],
            "alert_confidence": 0.72,
            "review_status": "unreviewed",
        },
        {
            "conversation_id": "conv_004",
            "timestamp": "2026-01-08T21:10:00Z",
            "prisoner_ids": [other_pid],
            "prisoner_names": [other_name],
            "transcript": "\n".join(
                [
                    f"[AUTOMATED NOTICE] This communication is recorded and subject to monitoring. Facility: {FACILITY_NAME}.",
                    f"Inmate ({other_name}): I can’t sleep. My chest is tight and I keep panicking.",
                    "Outside Contact (Sam Patel): Tell medical. You don’t have to handle that alone.",
                    f"Inmate ({other_name}): I’ll try. I just feel hopeless lately.",
                ]
            ),
            "facility_id": FACILITY_ID,
            "facility_name": FACILITY_NAME,
            "communication_type": "inmate_call",
            "call_duration_seconds": 380,
            "outside_contact_name": "Sam Patel",
            "outside_contact_relation": "friend",
            "alert_categories": ["wellness_mental_health"],
            "keyword_hits": ["panic", "hopeless"],
            "alert_confidence": 0.84,
            "review_status": "unreviewed",
        },
        {
            "conversation_id": "conv_005",
            "timestamp": month_ts,
            "prisoner_ids": [third_pid],
            "prisoner_names": [third_name],
            "transcript": "\n".join(
                [
                    f"[AUTOMATED NOTICE] This communication is recorded and subject to monitoring. Facility: {FACILITY_NAME}.",
                    f"Inmate ({third_name}): Don’t use real names. I keep hearing about ‘accounts’ and ‘drops’.",
                    "Outside Contact (Ari Johnson): That sounds risky. Don’t say more on the phone.",
                    f"Inmate ({third_name}): I’m not doing it, just hearing chatter.",
                ]
            ),
            "facility_id": FACILITY_ID,
            "facility_name": FACILITY_NAME,
            "communication_type": "inmate_call",
            "call_duration_seconds": 450,
            "outside_contact_name": "Ari Johnson",
            "outside_contact_relation": "cousin",
            "alert_categories": ["fraud_identity_theft", "general_intel"],
            "keyword_hits": ["accounts", "drops"],
            "alert_confidence": 0.77,
            "review_status": "unreviewed",
        },
        {
            "conversation_id": "conv_006",
            "timestamp": "2026-01-07T10:40:00Z",
            "prisoner_ids": [focus_pid],
            "prisoner_names": [focus_name],
            "transcript": "\n".join(
                [
                    f"[AUTOMATED NOTICE] This communication is recorded and subject to monitoring. Facility: {FACILITY_NAME}.",
                    f"Inmate ({focus_name}): I saw {other_name} getting pulled into that group again.",
                    "Outside Contact (Jamie Parker): You mean the crew from the yard?",
                    f"Inmate ({focus_name}): Yeah. And {third_name} keeps showing up around them too.",
                    "Outside Contact (Jamie Parker): That’s a pattern.",
                ]
            ),
            "facility_id": FACILITY_ID,
            "facility_name": FACILITY_NAME,
            "communication_type": "inmate_call",
            "call_duration_seconds": 300,
            "outside_contact_name": "Jamie Parker",
            "outside_contact_relation": "partner",
            "alert_categories": ["general_intel"],
            "keyword_hits": ["crew", "pattern"],
            "alert_confidence": 0.55,
            "review_status": "unreviewed",
        },
        {
            "conversation_id": "conv_007",
            "timestamp": "2026-01-06T17:05:00Z",
            "prisoner_ids": [focus_pid],
            "prisoner_names": [focus_name],
            "transcript": "\n".join(
                [
                    f"[AUTOMATED NOTICE] This communication is recorded and subject to monitoring. Facility: {FACILITY_NAME}.",
                    f"Inmate ({focus_name}): Everyone keeps saying ‘transfer’ but it’s hard to tell if it’s admin stuff or trouble.",
                    "Outside Contact (Pat Reynolds): Stay calm and don’t speculate.",
                    f"Inmate ({focus_name}): I’m just trying to understand what’s normal vs risky talk.",
                ]
            ),
            "facility_id": FACILITY_ID,
            "facility_name": FACILITY_NAME,
            "communication_type": "inmate_call",
            "call_duration_seconds": 360,
            "outside_contact_name": "Pat Reynolds",
            "outside_contact_relation": "parent",
            "alert_categories": ["escape_planning", "general_intel"],
            "keyword_hits": ["transfer"],
            "alert_confidence": 0.38,
            "review_status": "unreviewed",
        },
    ]

    for c in anchors_conv:
        _upsert_by_id(conversations, "conversation_id", c)

    anchors_reports: List[dict] = [
        {
            "report_id": "rpt_001",
            "created_at": "2026-01-09T16:10:00Z",
            "title": f"Shift Brief: Recent risks for {FACILITY_NAME}",
            "summary": "Recent communications indicate elevated tension in Housing Unit B, with repeated references to staffing strain and possible fights. Multiple alerts relate to contraband pressure, possible drug use, and debts. A separate thread suggests coercive behavior requiring PREA-aware review. Wellness language (panic/hopelessness) appears in at least one call and should be triaged by clinical staff. Analysts should prioritize human review of high-confidence flagged calls and coordinate with facility leadership.",
            "raw_text": "Analyst shift brief compiling key monitored-communication signals. Highlights include security tension (staffing strain, fight chatter), contraband pressure/commissary debt language, a PREA-adjacent coercion signal requiring careful handling, and a mental-health wellness concern. This brief is intended for operational awareness and investigative triage only, with citations to underlying communication IDs.",
            "linked_prisoner_ids": [focus_pid, other_pid],
            "linked_prisoner_names": [focus_name, other_name],
            "linked_conversation_ids": ["conv_001", "conv_002", "conv_003", "conv_004"],
            "tags": ["facility_security", "contraband_drugs", "prea_related", "wellness"],
            "report_type": "alert_digest",
            "trigger_type": "keyword_alert",
            "risk_level": "high",
            "confidence": 0.78,
            "alert_categories": ["facility_security", "contraband_drugs", "prea_related", "wellness_mental_health"],
            "recommended_actions": [
                "Prioritize human review of high-confidence calls within the last 24 hours.",
                "Notify shift leadership of Housing Unit B tension indicators.",
                "Route wellness-related calls to clinical triage per policy.",
                "Handle PREA-adjacent signals with restricted access and appropriate escalation.",
            ],
            "key_excerpts": [
                "Unit B is short staffed again. Things feel tense in the pod.",
                "They said pills are moving around but I’m not involved.",
                "It feels coercive.",
                "My chest is tight and I keep panicking.",
            ],
            "audit_note": "For demo use only; requires human review and policy/permissions checks before operational action.",
        },
        {
            "report_id": "rpt_002",
            "created_at": "2026-01-09T12:00:00Z",
            "title": f"Case Summary: {focus_name} ({focus_pid}) - 30 Day Themes",
            "summary": f"This case summary aggregates the last 30 days of flagged and relevant communications for {focus_name}. Themes include facility tension and violence-adjacent language, contraband pressure and debts, and ambiguous mentions of 'transfer' that warrant contextual review. The subject also references other individuals ({other_name}, {third_name}) in ways that may indicate a shifting peer network. Recommended next steps focus on validating context and avoiding over-reliance on keyword hits.",
            "raw_text": f"Subject-focused summary for {focus_name} ({focus_pid}). Includes citations to communications where the subject is speaking and where other individuals are mentioned. This report is designed for investigators and supervisors to quickly orient to current risk indicators and relationship context. No operational details are included.",
            "linked_prisoner_ids": [focus_pid, other_pid, third_pid],
            "linked_prisoner_names": [focus_name, other_name, third_name],
            "linked_conversation_ids": ["conv_001", "conv_002", "conv_006", "conv_007"],
            "tags": ["case_summary", "themes", "citations"],
            "report_type": "case_summary",
            "trigger_type": "case_followup",
            "risk_level": "medium",
            "confidence": 0.7,
            "alert_categories": ["general_intel", "facility_security"],
            "recommended_actions": [
                "Confirm whether 'transfer' references are administrative vs risk-related in context.",
                "Review associated calls for corroboration and reduce false positives via pattern-based rules.",
            ],
            "key_excerpts": [
                f"I saw {other_name} getting pulled into that group again.",
                "Everyone keeps saying ‘transfer’ but it’s hard to tell if it’s admin stuff or trouble.",
            ],
            "audit_note": "Case packaging; validate all conclusions with human review and access controls.",
        },
        {
            "report_id": "rpt_003",
            "created_at": "2026-01-09T15:30:00Z",
            "title": "Incident Addendum: INC-017 Supporting Communications",
            "summary": "This addendum links a contraband-pressure communication and related security-tension indicators as potential precursors. It provides a short narrative and key quotes with timestamps for investigators. Similar language patterns appear elsewhere in the last month and should be compared for consistency.",
            "raw_text": "Addendum for investigators: This record summarizes supporting communications for the referenced incident. The content is high-level and intended for review workflows and documentation, not operational guidance.",
            "linked_prisoner_ids": [focus_pid],
            "linked_prisoner_names": [focus_name],
            "linked_conversation_ids": ["conv_001", "conv_002"],
            "tags": ["incident_addendum", "citations", "contraband_drugs"],
            "report_type": "case_summary",
            "trigger_type": "case_followup",
            "risk_level": "high",
            "confidence": 0.74,
            "alert_categories": ["contraband_drugs", "facility_security"],
            "recommended_actions": [
                "Correlate incident timeline with communication timestamps.",
                "Search for similar language patterns in the last 6 months for related incidents.",
            ],
            "key_excerpts": [
                "They said pills are moving around but I’m not involved.",
                "Unit B is short staffed again. Things feel tense in the pod.",
            ],
            "audit_note": "Incident addendum for demo; maintain minimum-necessary disclosure and role-based access.",
        },
    ]

    for r in anchors_reports:
        _upsert_by_id(user_reports, "report_id", r)

    anchors_incidents: List[dict] = [
        {
            "incident_id": "INC-017",
            "date": "2026-01-09",
            "type": "Contraband Concern",
            "severity": "high",
            "description": "Staff documented a contraband-related concern following increased tension in Housing Unit B. The report notes prior indications of debt-related pressure and elevated security risk language in monitored communications. No graphic details are included. This incident is provided for demo linkage to supporting communications.",
            "involved_prisoner_ids": [focus_pid],
            "involved_prisoner_names": [focus_name],
            "facility_id": FACILITY_ID,
            "facility_name": FACILITY_NAME,
            "location": "Housing Unit B",
            "shift": "evening",
            "outcome": "Resolved; referred for follow-up review",
            "linked_conversation_ids": ["conv_001", "conv_002"],
            "linked_report_ids": ["rpt_003"],
        },
        {
            "incident_id": "inc_006",
            "date": "2025-09-18",
            "type": "Altercation",
            "severity": "medium",
            "description": "A brief altercation was reported and resolved without further escalation. This record exists to support 'similar incidents in the last 6 months' style queries when using static demo data.",
            "involved_prisoner_ids": [other_pid],
            "involved_prisoner_names": [other_name],
            "facility_id": FACILITY_ID,
            "facility_name": FACILITY_NAME,
            "location": "Yard",
            "shift": "day",
            "outcome": "Resolved",
            "linked_conversation_ids": [],
            "linked_report_ids": [],
        },
    ]

    for inc in anchors_incidents:
        _upsert_by_id(incident_reports, "incident_id", inc)


def _build_vectorstores(
    *,
    out_dir: Path,
    conversations: List[dict],
    user_reports: List[dict],
    incident_reports: List[dict],
) -> None:
    try:
        from langchain_core.documents import Document  # type: ignore
        from langchain_community.vectorstores import FAISS  # type: ignore
        from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Vector store dependencies are missing. Install: faiss-cpu langchain-google-genai langchain-community"
        ) from e

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_ID)
    vec_dir = out_dir / "vectors"
    vec_dir.mkdir(exist_ok=True)

    conv_docs: List[Document] = []
    for r in conversations:
        text = (r.get("transcript") or "").strip()
        if not text:
            continue
        meta = {
            "conversation_id": r.get("conversation_id"),
            "prisoner_ids": r.get("prisoner_ids", []),
            "alert_categories": r.get("alert_categories", []),
            "facility_id": r.get("facility_id"),
            "timestamp": r.get("timestamp"),
        }
        conv_docs.append(Document(page_content=text, metadata=meta))
    if conv_docs:
        vs = FAISS.from_documents(conv_docs, embeddings)
        vs.save_local(vec_dir / "conversations")

    rpt_docs: List[Document] = []
    for r in user_reports:
        text = (r.get("summary") or "").strip()
        if not text:
            continue
        meta = {
            "report_id": r.get("report_id"),
            "linked_prisoner_ids": r.get("linked_prisoner_ids", []),
            "linked_conversation_ids": r.get("linked_conversation_ids", []),
            "risk_level": r.get("risk_level"),
            "alert_categories": r.get("alert_categories", []),
        }
        rpt_docs.append(Document(page_content=text, metadata=meta))
    if rpt_docs:
        vs = FAISS.from_documents(rpt_docs, embeddings)
        vs.save_local(vec_dir / "user_reports")

    inc_docs: List[Document] = []
    for r in incident_reports:
        text = (r.get("description") or "").strip()
        if not text:
            continue
        meta = {
            "incident_id": r.get("incident_id"),
            "involved_prisoner_ids": r.get("involved_prisoner_ids", []),
            "severity": r.get("severity"),
            "type": r.get("type"),
            "date": r.get("date"),
        }
        inc_docs.append(Document(page_content=text, metadata=meta))
    if inc_docs:
        vs = FAISS.from_documents(inc_docs, embeddings)
        vs.save_local(vec_dir / "incidents")


def create_prisoners_file(path: Path, n: int) -> List[Prisoner]:
    # Minimal, deterministic roster
    first_names = ["Alex", "Jordan", "Taylor", "Casey", "Riley", "Morgan", "Avery", "Quinn", "Cameron", "Drew"]
    last_names = ["Smith", "Johnson", "Brown", "Garcia", "Miller", "Davis", "Martinez", "Lopez", "Wilson", "Anderson"]

    prisoners: List[Prisoner] = []
    for i in range(1, n + 1):
        pid = f"P{i:03d}"
        name = f"{first_names[(i - 1) % len(first_names)]} {last_names[(i - 1) % len(last_names)]}"
        prisoners.append(Prisoner(prisoner_id=pid, name=name))

    write_json(path, [p.model_dump() for p in prisoners])
    return prisoners


def _make_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key) if api_key else genai.Client()


def _generate_one_conversation(
    task: Tuple[int, str, List[str], Dict[str, str], List[Dict[str, str]], str, bool]
) -> Tuple[str, Dict[str, Any]]:
    i, cid, prisoner_ids, prisoner_name_by_id, outside_contacts, monitoring_notice, do_sleep = task
    client = _make_genai_client()
    rng = random.Random(SEED + i)

    # Always 1 inmate per communication (prisoner <-> outside contact)
    pids = rng.sample(prisoner_ids, k=1)
    pnames = [prisoner_name_by_id[x] for x in pids]

    oc = rng.choice(outside_contacts)
    comm_type = rng.choice(COMM_TYPES) if rng.random() < 0.25 else "inmate_call"
    duration = rng.randint(120, 900) if comm_type == "inmate_call" else rng.randint(10, 60)

    topic_hint = rng.choice(
        [
            "possible contraband/drug mention (high-level, no details)",
            "possible threats or intimidation (no operational detail)",
            "possible fraud/identity-theft talk (high-level, no how-to)",
            "wellness concern (supportive tone, no self-harm detail)",
            "facility security concern (high-level)",
            "family / reentry planning (neutral)",
            "commissary pressure or debts (neutral, no tactics)",
            "PREA-related boundary concern (vague, non-explicit)",
        ]
    )

    prompt = f"""
Generate ONE fictional correctional communication transcript record for a demo dataset.

Product context:
- Authorized, non-privileged inmate communications may be monitored and transcribed.
- The system flags items via keyword/phrase alerts for human review.

Hard rules:
- Fictional only.
- If wrongdoing is mentioned, keep it high-level. No step-by-step, no tactics, no operational guidance.
- Keep it professional. No explicit sexual content. If PREA-related comes up, keep it vague.
- Include mild ASR artifacts sometimes (ex: [inaudible]), but keep it readable.

Return ONLY JSON matching the schema.

Use these fixed values exactly:
conversation_id: \"{cid}\"
timestamp: \"{iso_ts(day=((i - 1) % 20) + 1, hour=9 + (i % 8))}\"
facility_id: \"{FACILITY_ID}\"
facility_name: \"{FACILITY_NAME}\"
communication_type: \"{comm_type}\"
call_duration_seconds: {duration}
outside_contact_name: \"{oc['name']}\"
outside_contact_relation: \"{oc['relation']}\"
prisoner_ids: {json.dumps(pids)}
prisoner_names: {json.dumps(pnames)}

Transcript requirements:
- First line must be:
  {monitoring_notice}
- Then 8–16 short turns
- Use labels like:
  Inmate ({pnames[0]}): ...
  Outside Contact ({oc['name']}): ...

Alert metadata:
- alert_categories: choose 0–3 from {json.dumps(ALERT_CATEGORIES)}
- keyword_hits: 0–6 short words/phrases that could trigger review (avoid how-to terms)
- alert_confidence: number 0.0–1.0
- review_status: \"unreviewed\"

Topic hint: {topic_hint}
"""

    rec = call_structured_json(client, prompt, Conversation)
    d = rec.model_dump()

    # Force fixed fields for consistency
    d["conversation_id"] = cid
    d["timestamp"] = d.get("timestamp") or iso_ts(day=((i - 1) % 20) + 1, hour=10)
    d["facility_id"] = FACILITY_ID
    d["facility_name"] = FACILITY_NAME
    d["communication_type"] = comm_type
    d["call_duration_seconds"] = int(duration)
    d["outside_contact_name"] = oc["name"]
    d["outside_contact_relation"] = oc["relation"]
    d["prisoner_ids"] = pids
    d["prisoner_names"] = pnames
    d["review_status"] = "unreviewed"

    # Clamp alert fields
    d["alert_categories"] = [c for c in d.get("alert_categories", []) if c in ALERT_CATEGORIES][:3]
    d["keyword_hits"] = [k[:40] for k in d.get("keyword_hits", [])][:6]
    d["alert_confidence"] = float(min(max(d.get("alert_confidence", 0.0), 0.0), 1.0))

    if do_sleep and SLEEP_BETWEEN_CALLS_SEC:
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    return cid, d


# -----------------------------
# Main
# -----------------------------
def main():
    client = _make_genai_client()

    # 1) Prisoner DB first
    prisoners_path = OUT_DIR / "prisoners.json"
    n_prisoners_effective = max(4, N_PRISONERS)
    if OVERWRITE_PRISONERS or not prisoners_path.exists():
        prisoners = create_prisoners_file(prisoners_path, n_prisoners_effective)
        print(f"[db  ] wrote {prisoners_path}")
    else:
        prisoners = load_prisoners(prisoners_path)
        print(f"[db  ] loaded {prisoners_path}")

    prisoner_ids = [p.prisoner_id for p in prisoners]
    prisoner_name_by_id = {p.prisoner_id: p.name for p in prisoners}
    known_prisoner_ids = set(prisoner_ids)

    # Fictional outside contacts
    outside_contacts = [
        {"name": "Jamie Parker", "relation": "partner"},
        {"name": "Pat Reynolds", "relation": "parent"},
        {"name": "Chris Nguyen", "relation": "sibling"},
        {"name": "Sam Patel", "relation": "friend"},
        {"name": "Ari Johnson", "relation": "cousin"},
    ]

    monitoring_notice = (
        f"[AUTOMATED NOTICE] This communication is recorded and subject to monitoring. "
        f"Facility: {FACILITY_NAME}."
    )

    # 2) Conversations (one call per record)
    conversations: List[dict] = []
    conv_ids: List[str] = [f"conv_{i:03d}" for i in range(1, N_CONVERSATIONS + 1)]

    pool_size_env = os.getenv("N_WORKERS")
    pool_size = int(pool_size_env) if pool_size_env else max(1, min(8, (os.cpu_count() or 2)))
    use_tqdm = tqdm is not None and os.getenv("NO_PROGRESS") not in {"1", "true", "TRUE"}

    force_sleep = os.getenv("FORCE_SLEEP") in {"1", "true", "TRUE"}
    do_sleep = force_sleep or pool_size <= 1

    tasks: List[Tuple[int, str, List[str], Dict[str, str], List[Dict[str, str]], str, bool]] = []
    for i, cid in enumerate(conv_ids, start=1):
        tasks.append((i, cid, prisoner_ids, prisoner_name_by_id, outside_contacts, monitoring_notice, do_sleep))

    results_by_id: Dict[str, Dict[str, Any]] = {}
    with mp.Pool(processes=pool_size) as pool:
        iterator = pool.imap_unordered(_generate_one_conversation, tasks)
        if use_tqdm:
            iterator = tqdm(iterator, total=len(tasks), desc="conversations")
        for cid, d in iterator:
            results_by_id[cid] = d
            print(f"[conv] {cid}")

    for cid in conv_ids:
        conversations.append(results_by_id[cid])

    known_conv_ids = set(conv_ids)

    # 3) User reports (one call per record)
    user_reports: List[dict] = []
    report_ids: List[str] = []

    for i in range(1, N_USER_REPORTS + 1):
        rid = f"rpt_{i:03d}"
        report_ids.append(rid)

        candidate_prisoners = random.sample(prisoner_ids, k=random.choice([1, 2, 3, 4]))
        candidate_convs = random.sample(conv_ids, k=random.choice([3, 4, 5, 6]))
        candidate_prisoner_names = [prisoner_name_by_id[x] for x in candidate_prisoners]

        report_type = random.choice(["alert_digest", "case_summary", "wellness_triage"])
        trigger_type = random.choice(["keyword_alert", "analyst_query", "case_followup"])
        risk_level = random.choice(RISK_LEVELS)
        cats = random.sample(ALERT_CATEGORIES, k=random.choice([1, 2, 2, 3]))

        prompt = f"""
Generate ONE fictional analyst report for a corrections intelligence demo.

Context:
- Communications are transcribed and flagged via keyword/phrase alerts.
- Analysts summarize findings and link back to communications and people.

Hard rules:
- Fictional only.
- High-level only for wrongdoing. No step-by-step, no tactics.
- Professional tone.

Return ONLY JSON matching the schema.

Use fixed values exactly:
report_id: "{rid}"
created_at: "{iso_ts(day=20, hour=8 + (i % 8))}"
report_type: "{report_type}"
trigger_type: "{trigger_type}"
risk_level: "{risk_level}"
alert_categories: {json.dumps(cats)}

You MUST ONLY reference/link to these candidates:
Candidate prisoners (pick 1–3):
IDs: {json.dumps(candidate_prisoners)}
Names: {json.dumps(candidate_prisoner_names)}

Candidate conversations (pick 1–5):
{json.dumps(candidate_convs)}

Write:
- title: short, case-like
- summary: 5–9 sentences
- raw_text: a bit longer than summary (still under ~2500 chars)
- linked_prisoner_ids and linked_prisoner_names: consistent
- linked_conversation_ids: consistent
- tags: 2–6 short tags (include at least 1 from alert_categories)
- recommended_actions: 2–5 practical items (human-in-the-loop)
- key_excerpts: 2–4 short transcript-like snippets (no how-to)
- confidence: 0.0–1.0
- audit_note: one line about review/audit expectation
"""

        rec = call_structured_json(client, prompt, UserReport)
        d = rec.model_dump()

        d["report_id"] = rid
        d["created_at"] = d.get("created_at") or iso_ts(day=20, hour=10)

        d["linked_prisoner_ids"] = clamp_to_known(
            d.get("linked_prisoner_ids", []),
            known_prisoner_ids,
            candidate_prisoners,
            min_keep=1,
        )[:3]
        d["linked_prisoner_names"] = [prisoner_name_by_id[x] for x in d["linked_prisoner_ids"]]

        d["linked_conversation_ids"] = clamp_to_known(
            d.get("linked_conversation_ids", []),
            known_conv_ids,
            candidate_convs,
            min_keep=1,
        )[:5]

        d["alert_categories"] = [c for c in d.get("alert_categories", []) if c in ALERT_CATEGORIES][:3]
        if not d["alert_categories"]:
            d["alert_categories"] = cats[:2]

        d["tags"] = [t[:30] for t in d.get("tags", [])][:6]
        d["recommended_actions"] = [a[:80] for a in d.get("recommended_actions", [])][:5]
        d["key_excerpts"] = [short_snippet(x, 220) for x in d.get("key_excerpts", [])][:4]
        d["confidence"] = float(min(max(d.get("confidence", 0.6), 0.0), 1.0))

        user_reports.append(d)
        print(f"[rpt ] {rid}")
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    known_report_ids = set(report_ids)

    # 4) Incident reports (one call per record)
    incident_reports: List[dict] = []

    incident_types = [
        "Altercation",
        "Contraband Concern",
        "Medical Incident",
        "Rule Violation",
        "Property Damage",
        "Threat/Intimidation",
        "Security Breach",
    ]

    for i in range(1, N_INCIDENT_REPORTS + 1):
        iid = f"inc_{i:03d}"
        candidate_prisoners = random.sample(prisoner_ids, k=random.choice([1, 2, 2, 3]))
        candidate_names = [prisoner_name_by_id[x] for x in candidate_prisoners]

        candidate_convs = random.sample(conv_ids, k=random.choice([0, 1, 2, 2, 3]))
        candidate_rpts = random.sample(report_ids, k=random.choice([0, 1, 1, 2]))

        inc_type = random.choice(incident_types)
        severity = random.choice(RISK_LEVELS)
        shift = random.choice(["day", "evening", "night"])
        location = random.choice(["Housing Unit A", "Housing Unit B", "Yard", "Chow Hall", "Medical", "Intake"])

        prompt = f"""
Generate ONE fictional correctional facility incident report for a demo dataset.

Hard rules:
- Fictional only.
- If wrongdoing is involved, keep it high-level. No step-by-step.
- No graphic detail.
- Professional report tone.

Return ONLY JSON matching the schema.

Use fixed values exactly:
incident_id: "{iid}"
date: "2025-12-{((i - 1) % 20) + 1:02d}"
type: "{inc_type}"
severity: "{severity}"
facility_id: "{FACILITY_ID}"
facility_name: "{FACILITY_NAME}"
location: "{location}"
shift: "{shift}"
outcome: short resolved-status language

Involved prisoners (must pick 1–3 from these ONLY):
IDs: {json.dumps(candidate_prisoners)}
Names: {json.dumps(candidate_names)}

Optional linking:
- linked_conversation_ids: pick 0–2 from {json.dumps(candidate_convs)}
- linked_report_ids: pick 0–1 from {json.dumps(candidate_rpts)}

Write description:
- 3–7 sentences
- factual, report-like
"""

        rec = call_structured_json(client, prompt, IncidentReport)
        d = rec.model_dump()

        d["incident_id"] = iid
        d["facility_id"] = FACILITY_ID
        d["facility_name"] = FACILITY_NAME
        d["location"] = location
        d["shift"] = shift

        d["involved_prisoner_ids"] = clamp_to_known(
            d.get("involved_prisoner_ids", []),
            known_prisoner_ids,
            candidate_prisoners,
            min_keep=1,
        )[:3]
        d["involved_prisoner_names"] = [prisoner_name_by_id[x] for x in d["involved_prisoner_ids"]]

        d["linked_conversation_ids"] = [x for x in d.get("linked_conversation_ids", []) if x in known_conv_ids][:2]
        d["linked_report_ids"] = [x for x in d.get("linked_report_ids", []) if x in known_report_ids][:1]

        incident_reports.append(d)
        print(f"[inc ] {iid}")
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    # 5) Write dataset files
    _ensure_question_profile_coverage(
        prisoners=prisoners,
        prisoner_name_by_id=prisoner_name_by_id,
        conversations=conversations,
        user_reports=user_reports,
        incident_reports=incident_reports,
    )
    write_json(OUT_DIR / "conversations.json", conversations)
    write_json(OUT_DIR / "user_reports.json", user_reports)
    write_json(OUT_DIR / "incident_reports.json", incident_reports)

    if BUILD_VECTORS:
        _build_vectorstores(
            out_dir=OUT_DIR,
            conversations=conversations,
            user_reports=user_reports,
            incident_reports=incident_reports,
        )

    print("\nWrote:")
    print(f"  {OUT_DIR/'prisoners.json'}")
    print(f"  {OUT_DIR/'conversations.json'}")
    print(f"  {OUT_DIR/'user_reports.json'}")
    print(f"  {OUT_DIR/'incident_reports.json'}")
    if BUILD_VECTORS:
        print(f"  {OUT_DIR/'vectors'}")


if __name__ == "__main__":
    main()