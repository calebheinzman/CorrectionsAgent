"""LLM-as-judge and reference-free metrics for evaluation.

Implements metrics from docs/eval/judging.md:
- A) RAGAS-style LLM-as-judge: Faithfulness, Context Precision
- B) LLM-as-judge quality: Completeness, Clarity, Relevance
- C) Non-judge metrics: Answer Relevance, Context Utilization, Citation Correctness, Consistency
"""
from __future__ import annotations

import os
import re
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

try:
    from google import genai

    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    HAS_EMBEDDINGS = True
except Exception:
    HAS_EMBEDDINGS = False

from mock_apis.cloud_apis.mock_secrets_manager import get_secret

from .schemas import JudgeMetrics


MODEL_ID = os.getenv("JUDGE_MODEL_ID", "gemini-2.0-flash-lite")


def _get_api_key() -> Optional[str]:
    return (
        get_secret("GEMINI_API_KEY")
        or get_secret("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


def _make_genai_client() -> Optional["genai.Client"]:
    if not HAS_GENAI:
        return None
    api_key = _get_api_key()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _call_judge(
    client: "genai.Client",
    prompt: str,
    temperature: float = 0.0,
) -> str:
    """Call the judge LLM and return the response text."""
    try:
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config={"temperature": temperature},
        )
        return resp.text or ""
    except Exception:
        return ""


def _parse_score(text: str, default: float = 0.0) -> float:
    """Parse a score from judge response text."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:/\s*(?:10|1\.0|1))?", text)
    if match:
        score = float(match.group(1))
        if score > 1.0:
            score = score / 10.0
        return min(1.0, max(0.0, score))
    
    text_lower = text.lower()
    if "yes" in text_lower or "true" in text_lower or "supported" in text_lower:
        return 1.0
    if "no" in text_lower or "false" in text_lower or "not supported" in text_lower:
        return 0.0
    
    return default


def compute_faithfulness(
    question: str,
    contexts: List[str],
    answer: str,
    client: Optional["genai.Client"] = None,
) -> float:
    """
    Compute faithfulness/groundedness score.
    
    Splits answer into claims and checks if each is supported by contexts.
    Score = supported_claims / total_claims
    """
    if not answer.strip():
        return 0.0
    
    if client is None:
        client = _make_genai_client()
    if client is None:
        return 0.0
    
    claims_prompt = f"""Extract all factual claims from the following answer. 
List each claim on a separate line, numbered.
Only include factual assertions, not opinions or hedged statements.

Answer: {answer}

Claims:"""
    
    claims_text = _call_judge(client, claims_prompt)
    claims = [line.strip() for line in claims_text.split("\n") if line.strip() and re.match(r"^\d+[\.\)]", line.strip())]
    
    if not claims:
        sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
        claims = sentences[:10]
    
    if not claims:
        return 1.0
    
    context_text = "\n\n".join(f"Context {i+1}: {c}" for i, c in enumerate(contexts))
    
    supported_count = 0
    for claim in claims:
        verify_prompt = f"""Determine if the following claim is supported by the provided contexts.
A claim is supported if the contexts contain evidence that directly supports it.

Contexts:
{context_text}

Claim: {claim}

Is this claim supported by the contexts? Answer only "Yes" or "No"."""
        
        response = _call_judge(client, verify_prompt)
        if "yes" in response.lower():
            supported_count += 1
    
    return supported_count / len(claims) if claims else 1.0


def compute_context_precision(
    question: str,
    contexts: List[str],
    answer: str,
    client: Optional["genai.Client"] = None,
) -> float:
    """
    Compute context precision score.
    
    For each context chunk, judge assigns relevance label.
    Score rewards relevant chunks appearing earlier in the ranked list.
    """
    if not contexts:
        return 0.0
    
    if client is None:
        client = _make_genai_client()
    if client is None:
        return 0.0
    
    relevance_labels = []
    for i, context in enumerate(contexts):
        prompt = f"""Determine if the following context is relevant for answering the question.
A context is relevant if it contains information that helps answer the question.

Question: {question}

Context: {context}

Is this context relevant? Answer only "Yes" or "No"."""
        
        response = _call_judge(client, prompt)
        is_relevant = "yes" in response.lower()
        relevance_labels.append(1 if is_relevant else 0)
    
    if sum(relevance_labels) == 0:
        return 0.0
    
    precision_at_k = []
    relevant_so_far = 0
    for k, label in enumerate(relevance_labels, 1):
        relevant_so_far += label
        if label == 1:
            precision_at_k.append(relevant_so_far / k)
    
    return sum(precision_at_k) / sum(relevance_labels) if precision_at_k else 0.0


def compute_completeness(
    question: str,
    answer: str,
    client: Optional["genai.Client"] = None,
) -> float:
    """
    Compute completeness score.
    
    Extracts required constraints from question and scores coverage.
    """
    if not answer.strip():
        return 0.0
    
    if client is None:
        client = _make_genai_client()
    if client is None:
        return 0.0
    
    prompt = f"""Evaluate how completely the answer addresses all aspects of the question.

Question: {question}

Answer: {answer}

Consider:
1. Does the answer address all entities mentioned in the question?
2. Does it cover any time ranges or constraints specified?
3. Does it provide the type of information requested (IDs, summaries, lists, etc.)?
4. Are there any parts of the question left unanswered?

Rate the completeness on a scale of 0 to 10, where:
- 0 = completely misses the question
- 5 = partially addresses the question
- 10 = fully addresses all aspects of the question

Provide only the numeric score."""
    
    response = _call_judge(client, prompt)
    return _parse_score(response, 0.5)


def compute_clarity(
    answer: str,
    client: Optional["genai.Client"] = None,
) -> float:
    """
    Compute clarity score.
    
    Scores structure, conciseness, and usefulness for an investigator.
    """
    if not answer.strip():
        return 0.0
    
    if client is None:
        client = _make_genai_client()
    if client is None:
        return 0.0
    
    prompt = f"""Evaluate the clarity and readability of the following answer for a prison investigator.

Answer: {answer}

Consider:
1. Is it well-structured (uses bullets, clear sections)?
2. Does it cite specific IDs (prisoner IDs, conversation IDs, report IDs)?
3. Is it concise without unnecessary filler?
4. Is the attribution clear (which information comes from which source)?
5. Would an investigator find this actionable?

Rate the clarity on a scale of 0 to 10, where:
- 0 = completely unclear, unusable
- 5 = somewhat clear but could be improved
- 10 = excellent clarity, well-structured, actionable

Provide only the numeric score."""
    
    response = _call_judge(client, prompt)
    return _parse_score(response, 0.5)


def compute_answer_relevance_judge(
    question: str,
    answer: str,
    client: Optional["genai.Client"] = None,
) -> float:
    """
    Compute answer relevance score (LLM-as-judge version).
    
    Penalizes off-topic additions not requested by the question.
    """
    if not answer.strip():
        return 0.0
    
    if client is None:
        client = _make_genai_client()
    if client is None:
        return 0.0
    
    prompt = f"""Evaluate how relevant the answer is to the question asked.

Question: {question}

Answer: {answer}

Consider:
1. Does the answer directly address what was asked?
2. Is there off-topic information that wasn't requested?
3. Does it stay focused on the question's scope?

Rate the relevance on a scale of 0 to 10, where:
- 0 = completely off-topic
- 5 = partially relevant with some off-topic content
- 10 = perfectly on-topic, no filler

Provide only the numeric score."""
    
    response = _call_judge(client, prompt)
    return _parse_score(response, 0.5)


def compute_answer_relevance_embedding(
    question: str,
    answer: str,
    num_questions: int = 3,
) -> float:
    """
    Compute answer relevance using embeddings (non-judge).
    
    Generates questions from the answer and compares to original question.
    """
    if not HAS_EMBEDDINGS or not HAS_GENAI:
        return 0.0
    
    api_key = _get_api_key()
    if not api_key:
        return 0.0
    
    client = _make_genai_client()
    if client is None:
        return 0.0
    
    gen_prompt = f"""Based on the following answer, generate {num_questions} questions that this answer would be a good response to.
List each question on a separate line, numbered.

Answer: {answer}

Questions:"""
    
    response = _call_judge(client, gen_prompt)
    generated_questions = [
        line.strip() for line in response.split("\n") 
        if line.strip() and re.match(r"^\d+[\.\)]", line.strip())
    ]
    
    if not generated_questions:
        return 0.0
    
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
        )
        
        all_texts = [question] + generated_questions
        embeddings = embeddings_model.embed_documents(all_texts)
        
        question_emb = embeddings[0]
        generated_embs = embeddings[1:]
        
        import numpy as np
        similarities = []
        for gen_emb in generated_embs:
            q_arr = np.array(question_emb)
            g_arr = np.array(gen_emb)
            dot = np.dot(q_arr, g_arr)
            norm_q = np.linalg.norm(q_arr)
            norm_g = np.linalg.norm(g_arr)
            if norm_q > 0 and norm_g > 0:
                similarities.append(dot / (norm_q * norm_g))
        
        return statistics.mean(similarities) if similarities else 0.0
    except Exception:
        return 0.0


def compute_context_utilization(
    retrieved_chunk_ids: List[str],
    cited_chunk_ids: List[str],
) -> float:
    """
    Compute context utilization score.
    
    Measures whether cited chunks appear early in ranked retrieval.
    """
    if not retrieved_chunk_ids or not cited_chunk_ids:
        return 0.0
    
    cited_set = set(cited_chunk_ids)
    
    precision_at_k = []
    cited_so_far = 0
    for k, chunk_id in enumerate(retrieved_chunk_ids, 1):
        if chunk_id in cited_set:
            cited_so_far += 1
            precision_at_k.append(cited_so_far / k)
    
    if not precision_at_k:
        return 0.0
    
    return sum(precision_at_k) / len(cited_set)


def compute_citation_correctness(
    answer: str,
    citations: List[Dict[str, Any]],
    cited_chunks: Dict[str, str],
) -> float:
    """
    Compute citation correctness score.
    
    Verifies that cited IDs/entities are present in cited chunk text.
    """
    if not citations:
        return 1.0

    # If no external chunk store is provided, fall back to using citation excerpts.
    if not cited_chunks:
        cited_chunks = {
            str(c.get("source_id", "") or ""): str(c.get("excerpt", "") or "")
            for c in citations
            if c.get("source_id")
        }
    
    correct_count = 0
    for citation in citations:
        source_id = citation.get("source_id", "")
        source_type = citation.get("source_type", "")
        
        if source_id in cited_chunks:
            chunk_text = (cited_chunks[source_id] or "").lower()
            
            if source_id.lower() in chunk_text:
                correct_count += 1
            elif source_type and source_type.lower() in chunk_text:
                correct_count += 1
            else:
                correct_count += 0.5
        else:
            correct_count += 0
    
    return correct_count / len(citations)


def extract_cited_ids_from_answer(answer: str) -> List[str]:
    """Extract cited IDs from answer text using regex patterns."""
    patterns = [
        r"(?:prisoner_id|prisoner|inmate)[:\s]+([A-Z0-9-]+)",
        r"(?:conversation_id|conversation)[:\s]+([A-Z0-9-]+)",
        r"(?:incident_id|incident)[:\s]+([A-Z0-9-]+)",
        r"(?:report_id|report)[:\s]+([A-Z0-9-]+)",
        r"\b([A-Z]{2,4}-\d{3,6})\b",
    ]
    
    cited_ids = []
    answer_upper = answer.upper()
    for pattern in patterns:
        matches = re.findall(pattern, answer_upper, re.IGNORECASE)
        cited_ids.extend(matches)
    
    return list(set(cited_ids))


def compute_all_judge_metrics(
    question: str,
    contexts: List[str],
    answer: str,
    row_id: str,
    model_id: str,
    prompt_version: str,
    citations: Optional[List[Dict[str, Any]]] = None,
    retrieved_chunk_ids: Optional[List[str]] = None,
    cited_chunks: Optional[Dict[str, str]] = None,
) -> JudgeMetrics:
    """
    Compute all judge metrics for a single response.
    
    Args:
        question: The user question
        contexts: Retrieved context chunks (text)
        answer: The model's answer
        row_id: Evaluation row ID
        model_id: Model identifier
        prompt_version: Prompt version
        citations: List of citation dicts with source_id, source_type
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs
        cited_chunks: Mapping of chunk ID to chunk text
    """
    client = _make_genai_client()
    
    faithfulness = compute_faithfulness(question, contexts, answer, client)
    context_precision = compute_context_precision(question, contexts, answer, client)
    completeness = compute_completeness(question, answer, client)
    clarity = compute_clarity(answer, client)
    relevance = compute_answer_relevance_judge(question, answer, client)
    
    answer_relevance_embedding = compute_answer_relevance_embedding(question, answer)
    
    cited_ids: List[str] = []
    if citations:
        cited_ids = [str(c.get("source_id", "") or "") for c in citations if c.get("source_id")]
    if not cited_ids:
        cited_ids = extract_cited_ids_from_answer(answer)

    context_utilization = compute_context_utilization(retrieved_chunk_ids or [], cited_ids)

    citation_correctness = compute_citation_correctness(
        answer,
        citations or [],
        cited_chunks or {},
    )
    
    return JudgeMetrics(
        row_id=row_id,
        model_id=model_id,
        prompt_version=prompt_version,
        faithfulness=faithfulness,
        context_precision=context_precision,
        completeness=completeness,
        clarity=clarity,
        relevance=relevance,
        answer_relevance_embedding=answer_relevance_embedding,
        context_utilization=context_utilization,
        citation_correctness=citation_correctness,
    )


def compute_batch_judge_metrics(
    results: List[Dict[str, Any]],
    model_id: str,
    prompt_version: str,
) -> List[JudgeMetrics]:
    """
    Compute judge metrics for a batch of results.
    
    Args:
        results: List of dicts with keys: row_id, question, contexts, answer, citations
        model_id: Model identifier
        prompt_version: Prompt version
    """
    metrics = []
    for r in results:
        m = compute_all_judge_metrics(
            question=r.get("question", r.get("input_text", "")),
            contexts=r.get("contexts", r.get("retrieved_contexts", [])),
            answer=r.get("answer", r.get("output_text", "")),
            row_id=r.get("row_id", ""),
            model_id=model_id,
            prompt_version=prompt_version,
            citations=r.get("citations"),
            retrieved_chunk_ids=r.get("retrieved_chunk_ids"),
            cited_chunks=r.get("cited_chunks"),
        )
        metrics.append(m)
    return metrics
