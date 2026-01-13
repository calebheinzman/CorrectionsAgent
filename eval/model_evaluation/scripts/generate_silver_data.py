"""Generate silver evaluation datasets using Gemini."""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

try:
    from google import genai

    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from mock_apis.cloud_apis.mock_secrets_manager import get_secret
from mock_apis.internal_apis import app as internal_app
from mock_apis.internal_apis import (
    conversation_api,
    incident_report_api,
    prisoner_info_api,
    user_report_api,
)

from eval.model_evaluation.datasets import write_jsonl
from eval.model_evaluation.schemas import (
    DatapointPairsInput,
    DatapointPairsLabels,
    DatapointPairsRow,
    EvalRowMeta,
    GeminiDatapointPairsOutput,
    GeminiQuestionListOutput,
    GeminiRelevanceOutput,
    GeminiSafetyOutput,
    GeminiToolPairsOutput,
    QuestionListInput,
    QuestionListLabels,
    QuestionListRow,
    RelevanceInput,
    RelevanceLabels,
    RelevanceRow,
    SafetyInput,
    SafetyLabels,
    SafetyRow,
    ToolPairsInput,
    ToolPairsLabels,
    ToolPairsRow,
)
from eval.model_evaluation.scripts.gen_prompts.prompts import (
    DATAPOINT_PAIRS_PROMPT_CONVERSATION,
    DATAPOINT_PAIRS_PROMPT_INCIDENT,
    DATAPOINT_PAIRS_PROMPT_PRISONER,
    DATAPOINT_PAIRS_PROMPT_USER_REPORT,
    QUESTION_LIST_PROMPT_OPEN,
    QUESTION_LIST_PROMPT_WITH_DATA,
    RELEVANCE_PROMPT,
    SAFETY_PROMPT,
    TOOL_PAIRS_PROMPT,
)

MODEL_ID = os.getenv("MODEL_ID", "gemini-2.0-flash-lite")

TOOL_UNIVERSE = ["conversations", "user_reports", "incident_reports", "prisoner_info"]

QUESTION_CATEGORIES = ["entity_specific", "pattern_seeking", "investigative", "ambiguous"]


def _utc_compact_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _make_genai_client() -> "genai.Client":
    if not HAS_GENAI:
        raise RuntimeError(
            "Gemini SDK unavailable. Install the Google GenAI SDK (google-genai) and retry."
        )

    api_key = (
        get_secret("GEMINI_API_KEY")
        or get_secret("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Gemini API key unavailable. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment "
            "or configure it in the secrets manager mock under the same key name."
        )

    return genai.Client(api_key=api_key)


def _call_structured_json(
    client: "genai.Client",
    *,
    prompt: str,
    schema_model: type[BaseModel],
    temperature: float = 0.7,
    max_retries: int = 4,
) -> BaseModel:
    last_err: Optional[Exception] = None
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
    raise RuntimeError(
        f"Failed to generate valid JSON after {max_retries} tries. Last error: {last_err}"
    )


def _gen_relevance_row(
    client: "genai.Client", idx: int, seed: int, model_id: str
) -> RelevanceRow:
    rng = random.Random(seed + 1000 + idx)
    label: Literal["relevant", "irrelevant"] = (
        "relevant" if rng.random() < 0.5 else "irrelevant"
    )

    prompt = RELEVANCE_PROMPT.format(label=label)
    out = _call_structured_json(client, prompt=prompt, schema_model=GeminiRelevanceOutput)

    return RelevanceRow(
        id=f"rel_{idx:05d}",
        input=RelevanceInput(text=out.text.strip()),
        labels=RelevanceLabels(relevance=out.relevance),
        meta=EvalRowMeta(seed=seed, generator_model=model_id),
    )


def _gen_safety_row(
    client: "genai.Client", idx: int, seed: int, model_id: str
) -> SafetyRow:
    rng = random.Random(seed + 2000 + idx)
    label: Literal["safe", "unsafe"] = "unsafe" if rng.random() < 0.5 else "safe"

    prompt = SAFETY_PROMPT.format(label=label)
    out = _call_structured_json(client, prompt=prompt, schema_model=GeminiSafetyOutput)

    return SafetyRow(
        id=f"saf_{idx:05d}",
        input=SafetyInput(text=out.text.strip()),
        labels=SafetyLabels(safety=out.safety),
        meta=EvalRowMeta(seed=seed, generator_model=model_id),
    )


def _gen_tool_pairs_row(
    client: "genai.Client", idx: int, seed: int, model_id: str
) -> ToolPairsRow:
    rng = random.Random(seed + 3000 + idx)
    subset_size = rng.randint(1, 2)
    tools = rng.sample(TOOL_UNIVERSE, subset_size)

    prompt = TOOL_PAIRS_PROMPT.format(tools=", ".join(tools))
    out = _call_structured_json(client, prompt=prompt, schema_model=GeminiToolPairsOutput)

    return ToolPairsRow(
        id=f"tool_{idx:05d}",
        input=ToolPairsInput(tools_allowed=tools, text=out.text.strip()),
        labels=ToolPairsLabels(tools_required=tools),
        meta=EvalRowMeta(seed=seed, generator_model=model_id),
    )


def _get_datapoint_prompt_and_must_contain(
    source: str, datapoint: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    if source == "prisoners":
        prompt = DATAPOINT_PAIRS_PROMPT_PRISONER.format(datapoint=json.dumps(datapoint, indent=2))
        must_contain = {
            "prisoner_id": datapoint.get("prisoner_id"),
            "name": datapoint.get("name"),
        }
    elif source == "conversations":
        prompt = DATAPOINT_PAIRS_PROMPT_CONVERSATION.format(
            datapoint=json.dumps(datapoint, indent=2)
        )
        must_contain = {
            "conversation_id": datapoint.get("conversation_id"),
        }
    elif source == "user_reports":
        prompt = DATAPOINT_PAIRS_PROMPT_USER_REPORT.format(
            datapoint=json.dumps(datapoint, indent=2)
        )
        must_contain = {
            "report_id": datapoint.get("report_id"),
        }
    elif source == "incident_reports":
        prompt = DATAPOINT_PAIRS_PROMPT_INCIDENT.format(datapoint=json.dumps(datapoint, indent=2))
        must_contain = {
            "incident_id": datapoint.get("incident_id"),
        }
    else:
        raise ValueError(f"Unknown source: {source}")

    return prompt, must_contain


def _gen_datapoint_pairs_row(
    client: "genai.Client",
    idx: int,
    seed: int,
    model_id: str,
    all_data: Dict[str, List[Dict[str, Any]]],
) -> DatapointPairsRow:
    rng = random.Random(seed + 4000 + idx)

    sources_with_data = [s for s in all_data if all_data[s]]
    if not sources_with_data:
        raise RuntimeError("No mock data available for datapoint pairs generation")

    source = rng.choice(sources_with_data)
    datapoint = rng.choice(all_data[source])

    prompt, must_contain = _get_datapoint_prompt_and_must_contain(source, datapoint)
    out = _call_structured_json(
        client, prompt=prompt, schema_model=GeminiDatapointPairsOutput
    )

    source_literal: Literal["prisoners", "conversations", "user_reports", "incident_reports"]
    source_literal = source  # type: ignore

    return DatapointPairsRow(
        id=f"dp_{idx:05d}",
        input=DatapointPairsInput(
            text=out.text.strip(),
            target_datapoint=datapoint,
            datapoint_source=source_literal,
        ),
        labels=DatapointPairsLabels(must_contain=must_contain),
        meta=EvalRowMeta(seed=seed, generator_model=model_id),
    )


def _gen_question_list_row(
    client: "genai.Client",
    idx: int,
    seed: int,
    model_id: str,
    all_data: Dict[str, List[Dict[str, Any]]],
    embed_data_pct: float = 0.3,
) -> QuestionListRow:
    rng = random.Random(seed + 5000 + idx)

    use_data = rng.random() < embed_data_pct
    sources_with_data = [s for s in all_data if all_data[s]]

    if use_data and sources_with_data:
        source = rng.choice(sources_with_data)
        datapoint = rng.choice(all_data[source])
        prompt = QUESTION_LIST_PROMPT_WITH_DATA.format(
            datapoint=json.dumps(datapoint, indent=2),
            source_type=source.replace("_", " "),
        )
    else:
        category = rng.choice(QUESTION_CATEGORIES)
        prompt = QUESTION_LIST_PROMPT_OPEN.format(category=category)

    out = _call_structured_json(
        client, prompt=prompt, schema_model=GeminiQuestionListOutput
    )

    return QuestionListRow(
        id=f"ql_{idx:05d}",
        input=QuestionListInput(text=out.text.strip()),
        labels=QuestionListLabels(topic_tags=out.topic_tags),
        meta=EvalRowMeta(seed=seed, generator_model=model_id),
    )


_SHARED_DATA: Dict[str, List[Dict[str, Any]]] = {}


def _init_worker(data: Dict[str, List[Dict[str, Any]]]) -> None:
    global _SHARED_DATA
    _SHARED_DATA = data


def _gen_relevance_worker(args: Tuple[int, int, float, str]) -> dict:
    idx, seed, sleep_sec, model_id = args
    client = _make_genai_client()
    row = _gen_relevance_row(client, idx, seed, model_id)
    if sleep_sec:
        time.sleep(sleep_sec)
    return row.model_dump()


def _gen_safety_worker(args: Tuple[int, int, float, str]) -> dict:
    idx, seed, sleep_sec, model_id = args
    client = _make_genai_client()
    row = _gen_safety_row(client, idx, seed, model_id)
    if sleep_sec:
        time.sleep(sleep_sec)
    return row.model_dump()


def _gen_tool_pairs_worker(args: Tuple[int, int, float, str]) -> dict:
    idx, seed, sleep_sec, model_id = args
    client = _make_genai_client()
    row = _gen_tool_pairs_row(client, idx, seed, model_id)
    if sleep_sec:
        time.sleep(sleep_sec)
    return row.model_dump()


def _gen_datapoint_pairs_worker(args: Tuple[int, int, float, str]) -> dict:
    idx, seed, sleep_sec, model_id = args
    client = _make_genai_client()
    row = _gen_datapoint_pairs_row(client, idx, seed, model_id, _SHARED_DATA)
    if sleep_sec:
        time.sleep(sleep_sec)
    return row.model_dump()


def _gen_question_list_worker(args: Tuple[int, int, float, str]) -> dict:
    idx, seed, sleep_sec, model_id = args
    client = _make_genai_client()
    row = _gen_question_list_row(client, idx, seed, model_id, _SHARED_DATA)
    if sleep_sec:
        time.sleep(sleep_sec)
    return row.model_dump()


def _load_mock_data() -> Dict[str, List[Dict[str, Any]]]:
    internal_app.load_data()

    prisoners = [p.model_dump() for p in prisoner_info_api.get_all_prisoners()]
    conversations = [c.model_dump() for c in conversation_api.get_all_conversations()]
    user_reports = [r.model_dump() for r in user_report_api.get_all_user_reports()]
    incidents = [i.model_dump() for i in incident_report_api.get_all_incident_reports()]

    return {
        "prisoners": prisoners,
        "conversations": conversations,
        "user_reports": user_reports,
        "incident_reports": incidents,
    }


def main() -> int:
    global MODEL_ID
    
    parser = argparse.ArgumentParser(description="Generate silver evaluation datasets")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "synthetic"),
    )
    parser.add_argument("--n-relevance", type=int, default=int(os.getenv("N_RELEVANCE", "200")))
    parser.add_argument("--n-safety", type=int, default=int(os.getenv("N_SAFETY", "200")))
    parser.add_argument("--n-tool-pairs", type=int, default=int(os.getenv("N_TOOL_PAIRS", "200")))
    parser.add_argument(
        "--n-datapoint-pairs", type=int, default=int(os.getenv("N_DATAPOINT_PAIRS", "200"))
    )
    parser.add_argument(
        "--n-question-list", type=int, default=int(os.getenv("N_QUESTION_LIST", "100"))
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    parser.add_argument("--sleep-sec", type=float, default=float(os.getenv("SLEEP_SEC", "0.15")))
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", "0")),
        help="Number of parallel workers (0 = cpu_count())",
    )
    parser.add_argument("--model-id", default=MODEL_ID, help="Gemini model ID for generation")
    args = parser.parse_args()

    MODEL_ID = args.model_id

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = _utc_compact_ts()

    _make_genai_client()

    print("Loading mock data from internal APIs...")
    mock_data = _load_mock_data()
    print(
        f"  Loaded: {len(mock_data['prisoners'])} prisoners, "
        f"{len(mock_data['conversations'])} conversations, "
        f"{len(mock_data['user_reports'])} user reports, "
        f"{len(mock_data['incident_reports'])} incidents"
    )

    workers = args.workers if args.workers and args.workers > 0 else (os.cpu_count() or 2)
    ctx = mp.get_context("spawn")

    def run_generation(
        name: str,
        worker_fn,
        n: int,
        schema: type[BaseModel],
        filename: str,
    ) -> None:
        if n <= 0:
            return
        print(f"\nGenerating {n} {name} examples...")
        tasks = [(i, args.seed, args.sleep_sec, args.model_id) for i in range(n)]

        with ctx.Pool(
            processes=workers, initializer=_init_worker, initargs=(mock_data,)
        ) as pool:
            it = pool.imap_unordered(worker_fn, tasks)
            if tqdm is not None:
                it = tqdm(it, total=n, desc=name)
            results = list(it)

        rows = [schema.model_validate(d) for d in results]
        path = out_dir / f"{filename}_{dataset_id}.jsonl"
        write_jsonl(path, rows)
        print(f"  Wrote {len(rows)} rows to {path}")

    run_generation("relevance", _gen_relevance_worker, args.n_relevance, RelevanceRow, "relevance")
    run_generation("safety", _gen_safety_worker, args.n_safety, SafetyRow, "safety")
    run_generation("tool_pairs", _gen_tool_pairs_worker, args.n_tool_pairs, ToolPairsRow, "tool_pairs")
    run_generation(
        "datapoint_pairs",
        _gen_datapoint_pairs_worker,
        args.n_datapoint_pairs,
        DatapointPairsRow,
        "datapoint_pairs",
    )
    run_generation(
        "question_list",
        _gen_question_list_worker,
        args.n_question_list,
        QuestionListRow,
        "question_list",
    )

    print(f"\nDone! All datasets written to {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        print(f"Error: {e}")
        raise SystemExit(1)
