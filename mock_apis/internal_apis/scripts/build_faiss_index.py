"""
Build FAISS vector indexes for similarity search.

This script builds vector indexes over the text fields in the mock data:
- Conversations: embed `transcript`
- User reports: embed `summary`
- Incidents: embed `description`

Environment variables:
    GEMINI_API_KEY or GOOGLE_API_KEY - API key for Gemini embeddings
    EMBEDDING_MODEL_ID - Embedding model (default: text-embedding-004)
    DATA_DIR - Data directory (default: mock_apis/internal_apis/data)

Run:
    python -m mock_apis.internal_apis.scripts.build_faiss_index
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-004")
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR = Path(os.getenv("DATA_DIR", str(_DEFAULT_DATA_DIR)))


def load_json(path: Path) -> list:
    """Load JSON file if it exists."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def build_indexes() -> None:
    """Build FAISS indexes for all datasets."""
    try:
        from langchain_core.documents import Document
        from langchain_community.vectorstores import FAISS
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as e:
        print(f"Error: Missing dependencies. Install: faiss-cpu langchain-google-genai langchain-community")
        print(f"Details: {e}")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_ID)
    vec_dir = DATA_DIR / "vectors"
    vec_dir.mkdir(exist_ok=True)

    conversations = load_json(DATA_DIR / "conversations.json")
    if conversations:
        print(f"Building conversation index ({len(conversations)} records)...")
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
            print(f"  Saved to {vec_dir / 'conversations'}")

    user_reports = load_json(DATA_DIR / "user_reports.json")
    if user_reports:
        print(f"Building user reports index ({len(user_reports)} records)...")
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
            print(f"  Saved to {vec_dir / 'user_reports'}")

    incident_reports = load_json(DATA_DIR / "incident_reports.json")
    if incident_reports:
        print(f"Building incident reports index ({len(incident_reports)} records)...")
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
            print(f"  Saved to {vec_dir / 'incidents'}")

    print("\nDone!")


def main():
    """Main entry point."""
    build_indexes()


if __name__ == "__main__":
    main()
