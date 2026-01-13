"""FastAPI application for internal mock APIs."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from . import conversation_api, incident_report_api, prisoner_info_api, user_report_api


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    load_data()
    yield


app = FastAPI(
    title="Internal Mock APIs",
    description="Mock internal dataset APIs for agent tooling",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(prisoner_info_api.router)
app.include_router(conversation_api.router)
app.include_router(user_report_api.router)
app.include_router(incident_report_api.router)


def _get_data_dir() -> Path:
    """Get the data directory path."""
    env_path = os.getenv("MOCK_DATA_DIR")
    if env_path:
        return Path(env_path)
    return Path(__file__).parent / "data"


def _load_json(path: Path) -> list:
    """Load JSON file if it exists."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _load_vector_store(vec_dir: Path, name: str) -> Optional[object]:
    """Load a FAISS vector store if it exists."""
    store_path = vec_dir / name
    if not store_path.exists():
        return None
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embedding_model = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-004")
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        return FAISS.load_local(str(store_path), embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None


def load_data() -> None:
    """Load all datasets and vector stores."""
    data_dir = _get_data_dir()

    prisoners = _load_json(data_dir / "prisoners.json")
    prisoner_info_api.load_prisoners(prisoners)

    conversations = _load_json(data_dir / "conversations.json")
    conversation_api.load_conversations(conversations)

    user_reports = _load_json(data_dir / "user_reports.json")
    user_report_api.load_user_reports(user_reports)

    incident_reports = _load_json(data_dir / "incident_reports.json")
    incident_report_api.load_incident_reports(incident_reports)

    vec_dir = data_dir / "vectors"
    if vec_dir.exists():
        conv_vs = _load_vector_store(vec_dir, "conversations")
        if conv_vs:
            conversation_api.set_vector_store(conv_vs)

        rpt_vs = _load_vector_store(vec_dir, "user_reports")
        if rpt_vs:
            user_report_api.set_vector_store(rpt_vs)

        inc_vs = _load_vector_store(vec_dir, "incidents")
        if inc_vs:
            incident_report_api.set_vector_store(inc_vs)




@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
