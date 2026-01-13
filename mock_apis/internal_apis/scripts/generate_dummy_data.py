"""
Synthetic dataset generator.

Generates coherent datasets with cross-linked IDs for:
- prisoners.json
- conversations.json
- user_reports.json
- incident_reports.json

Optionally builds FAISS vector indexes for similarity search.

Environment variables:
    GEMINI_API_KEY or GOOGLE_API_KEY - API key for Gemini
    MODEL_ID - Model to use (default: gemini-2.5-flash-lite)
    EMBEDDING_MODEL_ID - Embedding model (default: text-embedding-004)
    DATA_DIR - Output directory (default: mock_apis/internal_apis/data)
    N_PRISONERS - Number of prisoners (default: 10)
    N_CONVERSATIONS - Number of conversations (default: 30)
    N_USER_REPORTS - Number of user reports (default: 12)
    N_INCIDENT_REPORTS - Number of incident reports (default: 12)
    SEED - Random seed (default: 42)
    BUILD_VECTORS - Build vector indexes (default: 1)
    FACILITY_ID - Facility ID (default: FAC-001)
    FACILITY_NAME - Facility name (default: North River Correctional Center)

Run:
    python -m mock_apis.internal_apis.scripts.generate_dummy_data
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

MODEL_ID = os.getenv("MODEL_ID", "gemini-2.5-flash-lite")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-004")
BUILD_VECTORS = os.getenv("BUILD_VECTORS", "1").strip() != "0"

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
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


class Prisoner(BaseModel):
    prisoner_id: str
    name: str


class Conversation(BaseModel):
    conversation_id: str
    timestamp: str
    prisoner_ids: List[str]
    prisoner_names: List[str]
    transcript: str
    facility_id: str = FACILITY_ID
    facility_name: str = FACILITY_NAME
    communication_type: Literal["inmate_call", "inmate_text"] = "inmate_call"
    call_duration_seconds: int = 420
    outside_contact_name: str = "Outside Contact"
    outside_contact_relation: str = "family"
    alert_categories: List[str] = Field(default_factory=list)
    keyword_hits: List[str] = Field(default_factory=list)
    alert_confidence: float = 0.0
    review_status: Literal["unreviewed", "reviewed"] = "unreviewed"


class UserReport(BaseModel):
    report_id: str
    created_at: str
    title: str
    summary: str
    raw_text: str
    linked_prisoner_ids: List[str]
    linked_prisoner_names: List[str]
    linked_conversation_ids: List[str]
    tags: List[str] = Field(default_factory=list)
    report_type: Literal["alert_digest", "case_summary", "wellness_triage"] = "alert_digest"
    trigger_type: Literal["keyword_alert", "analyst_query", "case_followup"] = "keyword_alert"
    risk_level: Literal["low", "medium", "high"] = "low"
    confidence: float = 0.6
    alert_categories: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    key_excerpts: List[str] = Field(default_factory=list)
    audit_note: str = "Analyst review required (human-in-the-loop)."


class IncidentReport(BaseModel):
    incident_id: str
    date: str
    type: str
    severity: str
    description: str
    involved_prisoner_ids: List[str]
    involved_prisoner_names: List[str]
    facility_id: str = FACILITY_ID
    facility_name: str = FACILITY_NAME
    location: str = "Housing Unit B"
    shift: Literal["day", "evening", "night"] = "day"
    outcome: str = "Resolved"
    linked_conversation_ids: List[str] = Field(default_factory=list)
    linked_report_ids: List[str] = Field(default_factory=list)


def iso_ts(day: int, hour: int) -> str:
    return f"2025-12-{day:02d}T{hour:02d}:00:00Z"


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_prisoners(path: Path) -> List[Prisoner]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Prisoner.model_validate(x) for x in raw]


def create_prisoners_file(path: Path, n: int) -> List[Prisoner]:
    """Create deterministic prisoner roster."""
    first_names = ["Alex", "Jordan", "Taylor", "Casey", "Riley", "Morgan", "Avery", "Quinn", "Cameron", "Drew"]
    last_names = ["Smith", "Johnson", "Brown", "Garcia", "Miller", "Davis", "Martinez", "Lopez", "Wilson", "Anderson"]

    prisoners: List[Prisoner] = []
    for i in range(1, n + 1):
        pid = f"P{i:03d}"
        name = f"{first_names[(i - 1) % len(first_names)]} {last_names[(i - 1) % len(last_names)]}"
        prisoners.append(Prisoner(prisoner_id=pid, name=name))

    write_json(path, [p.model_dump() for p in prisoners])
    return prisoners


def _upsert_by_id(records: List[dict], id_field: str, record: dict) -> None:
    """Upsert a record by ID field."""
    rid = record.get(id_field)
    if not rid:
        return
    for i, r in enumerate(records):
        if r.get(id_field) == rid:
            records[i] = record
            return
    records.append(record)


def _ensure_anchor_records(
    *,
    prisoners: List[Prisoner],
    prisoner_name_by_id: Dict[str, str],
    conversations: List[dict],
    user_reports: List[dict],
    incident_reports: List[dict],
) -> None:
    """Insert anchor records for demo queries."""
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

    anchors_conv = [
        {
            "conversation_id": "conv_001",
            "timestamp": now_ts,
            "prisoner_ids": [focus_pid],
            "prisoner_names": [focus_name],
            "transcript": f"[AUTOMATED NOTICE] This communication is recorded. Facility: {FACILITY_NAME}.\n"
                         f"Inmate ({focus_name}): I heard Unit B is short staffed again. Things feel tense.\n"
                         f"Outside Contact: Keep your head down. Any issues with safety?\n"
                         f"Inmate ({focus_name}): There was talk of a fight after chow.",
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
            "transcript": f"[AUTOMATED NOTICE] This communication is recorded. Facility: {FACILITY_NAME}.\n"
                         f"Inmate ({focus_name}): People keep saying 'paper' like it's normal. I'm not trying to get wrapped up in contraband or drug use.\n"
                         f"Outside Contact: Stay clear. Are they pressuring you?\n"
                         f"Inmate ({focus_name}): Yeah, debts. They said pills are moving around but I'm not involved.",
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
            "transcript": f"[AUTOMATED NOTICE] This communication is recorded. Facility: {FACILITY_NAME}.\n"
                         f"Inmate ({focus_name}): I'm uncomfortable with how someone is trying to pressure people. It feels coercive.\n"
                         f"Outside Contact: Are you safe?\n"
                         f"Inmate ({focus_name}): I'm trying to avoid it.",
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
            "transcript": f"[AUTOMATED NOTICE] This communication is recorded. Facility: {FACILITY_NAME}.\n"
                         f"Inmate ({other_name}): I can't sleep. My chest is tight and I keep panicking.\n"
                         f"Outside Contact: Tell medical. You don't have to handle that alone.\n"
                         f"Inmate ({other_name}): I'll try. I just feel hopeless lately.",
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
            "transcript": f"[AUTOMATED NOTICE] This communication is recorded. Facility: {FACILITY_NAME}.\n"
                         f"Inmate ({third_name}): Don't use real names. I keep hearing about 'accounts' and 'drops'.\n"
                         f"Outside Contact: That sounds risky. Don't say more on the phone.\n"
                         f"Inmate ({third_name}): I'm not doing it, just hearing chatter.",
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
    ]

    for c in anchors_conv:
        _upsert_by_id(conversations, "conversation_id", c)

    anchors_reports = [
        {
            "report_id": "rpt_001",
            "created_at": "2026-01-09T16:10:00Z",
            "title": f"Shift Brief: Recent risks for {FACILITY_NAME}",
            "summary": "Recent communications indicate elevated tension with contraband pressure and drug use concerns.",
            "raw_text": "Analyst shift brief compiling key monitored-communication signals.",
            "linked_prisoner_ids": [focus_pid, other_pid],
            "linked_prisoner_names": [focus_name, other_name],
            "linked_conversation_ids": ["conv_001", "conv_002", "conv_003", "conv_004"],
            "tags": ["facility_security", "contraband_drugs", "prea_related", "wellness"],
            "report_type": "alert_digest",
            "trigger_type": "keyword_alert",
            "risk_level": "high",
            "confidence": 0.78,
            "alert_categories": ["facility_security", "contraband_drugs", "prea_related", "wellness_mental_health"],
            "recommended_actions": ["Prioritize human review", "Notify shift leadership"],
            "key_excerpts": ["Unit B is short staffed", "pills are moving around"],
            "audit_note": "For demo use only",
        },
        {
            "report_id": "rpt_002",
            "created_at": "2026-01-09T12:00:00Z",
            "title": f"Case Summary: {focus_name} ({focus_pid}) - 30 Day Themes",
            "summary": f"Case summary aggregates the last 30 days of flagged communications for {focus_name}.",
            "raw_text": f"Subject-focused summary for {focus_name} ({focus_pid}).",
            "linked_prisoner_ids": [focus_pid, other_pid, third_pid],
            "linked_prisoner_names": [focus_name, other_name, third_name],
            "linked_conversation_ids": ["conv_001", "conv_002"],
            "tags": ["case_summary", "themes", "citations"],
            "report_type": "case_summary",
            "trigger_type": "case_followup",
            "risk_level": "medium",
            "confidence": 0.7,
            "alert_categories": ["general_intel", "facility_security"],
            "recommended_actions": ["Confirm transfer references", "Review associated calls"],
            "key_excerpts": ["getting pulled into that group again"],
            "audit_note": "Case packaging; validate all conclusions",
        },
        {
            "report_id": "rpt_003",
            "created_at": "2026-01-09T15:30:00Z",
            "title": "Incident Addendum: INC-017 Supporting Communications",
            "summary": "Addendum links contraband-pressure communication as potential precursors.",
            "raw_text": "Addendum for investigators.",
            "linked_prisoner_ids": [focus_pid],
            "linked_prisoner_names": [focus_name],
            "linked_conversation_ids": ["conv_001", "conv_002"],
            "tags": ["incident_addendum", "citations", "contraband_drugs"],
            "report_type": "case_summary",
            "trigger_type": "case_followup",
            "risk_level": "high",
            "confidence": 0.74,
            "alert_categories": ["contraband_drugs", "facility_security"],
            "recommended_actions": ["Correlate incident timeline"],
            "key_excerpts": ["pills are moving around"],
            "audit_note": "Incident addendum for demo",
        },
    ]

    for r in anchors_reports:
        _upsert_by_id(user_reports, "report_id", r)

    anchors_incidents = [
        {
            "incident_id": "INC-017",
            "date": "2026-01-09",
            "type": "Contraband Concern",
            "severity": "high",
            "description": "Staff documented a contraband-related concern following increased tension in Housing Unit B.",
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
            "description": "A brief altercation was reported and resolved without further escalation.",
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
    """Build FAISS vector stores for similarity search."""
    try:
        from langchain_core.documents import Document
        from langchain_community.vectorstores import FAISS
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as e:
        print(f"Skipping vector store build: {e}")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_ID)
    vec_dir = out_dir / "vectors"
    vec_dir.mkdir(exist_ok=True)

    conv_docs = []
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
        vs.save_local(str(vec_dir / "conversations"))

    rpt_docs = []
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
        vs.save_local(str(vec_dir / "user_reports"))

    inc_docs = []
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
        vs.save_local(str(vec_dir / "incidents"))


def generate_static_data() -> None:
    """Generate static anchor data without LLM calls."""
    prisoners_path = OUT_DIR / "prisoners.json"
    n_prisoners_effective = max(10, N_PRISONERS)

    if OVERWRITE_PRISONERS or not prisoners_path.exists():
        prisoners = create_prisoners_file(prisoners_path, n_prisoners_effective)
        print(f"[db  ] wrote {prisoners_path}")
    else:
        prisoners = load_prisoners(prisoners_path)
        print(f"[db  ] loaded {prisoners_path}")

    prisoner_name_by_id = {p.prisoner_id: p.name for p in prisoners}

    conversations: List[dict] = []
    user_reports: List[dict] = []
    incident_reports: List[dict] = []

    _ensure_anchor_records(
        prisoners=prisoners,
        prisoner_name_by_id=prisoner_name_by_id,
        conversations=conversations,
        user_reports=user_reports,
        incident_reports=incident_reports,
    )

    write_json(OUT_DIR / "conversations.json", conversations)
    write_json(OUT_DIR / "user_reports.json", user_reports)
    write_json(OUT_DIR / "incident_reports.json", incident_reports)

    print(f"\nWrote static data to {OUT_DIR}:")
    print(f"  - prisoners.json ({len(prisoners)} records)")
    print(f"  - conversations.json ({len(conversations)} records)")
    print(f"  - user_reports.json ({len(user_reports)} records)")
    print(f"  - incident_reports.json ({len(incident_reports)} records)")

    if BUILD_VECTORS:
        print("\nBuilding vector stores...")
        _build_vectorstores(
            out_dir=OUT_DIR,
            conversations=conversations,
            user_reports=user_reports,
            incident_reports=incident_reports,
        )
        print(f"  - vectors/")


def main():
    """Main entry point."""
    generate_static_data()


if __name__ == "__main__":
    main()
