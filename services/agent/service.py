"""Agent service implementation."""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import tool

from .config_loader import AgentConfigLoader
from .model.client_factory import create_client_from_registry
from .schemas import AgentRequest, AgentResponse, Citation, ModelInfo, ToolCallRecord
from .tools import conversations, incident_reports, prisoner_info, user_reports
from .tracing.tool_tracer import ToolTrace, get_tracer
from services.cloudwatch_client import get_cloudwatch_client


@tool
def search_conversations_tool(
    query: Optional[str] = None,
    prisoner_id: Optional[str] = None,
    prisoner_name: Optional[str] = None,
    alert_category: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Search prisoner conversations with various filters.
    
    Args:
        query: Free-text semantic search query (e.g., "drug use", "escape plan")
        prisoner_id: Filter by specific prisoner ID
        prisoner_name: Filter by prisoner name (case-insensitive)
        alert_category: Filter by alert category
        keyword: Filter by keyword hit
        limit: Maximum results (default 10)
    """
    result = conversations.search_conversations(
        query=query,
        prisoner_id=prisoner_id,
        prisoner_name=prisoner_name,
        alert_category=alert_category,
        keyword=keyword,
        limit=limit,
    )
    return json.dumps(result.get("data", {}), indent=2)


@tool
def search_prisoners_tool(
    prisoner_id: Optional[str] = None,
    name: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Search prisoner information.
    
    Args:
        prisoner_id: Exact prisoner ID match
        name: Case-insensitive name search
        limit: Maximum results (default 10)
    """
    result = prisoner_info.search_prisoners(
        prisoner_id=prisoner_id,
        name=name,
        limit=limit,
    )
    return json.dumps(result.get("data", {}), indent=2)


@tool
def search_incidents_tool(
    query: Optional[str] = None,
    prisoner_id: Optional[str] = None,
    prisoner_name: Optional[str] = None,
    incident_type: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Search incident reports.
    
    Args:
        query: Free-text semantic search over descriptions
        prisoner_id: Filter by involved prisoner ID
        prisoner_name: Filter by involved prisoner name
        incident_type: Filter by incident type
        severity: Filter by severity (low, medium, high)
        limit: Maximum results (default 10)
    """
    result = incident_reports.search_incidents(
        query=query,
        prisoner_id=prisoner_id,
        prisoner_name=prisoner_name,
        incident_type=incident_type,
        severity=severity,
        limit=limit,
    )
    return json.dumps(result.get("data", {}), indent=2)


@tool
def search_user_reports_tool(
    query: Optional[str] = None,
    prisoner_id: Optional[str] = None,
    prisoner_name: Optional[str] = None,
    risk_level: Optional[str] = None,
    alert_category: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Search analyst reports.
    
    Args:
        query: Free-text semantic search over summaries
        prisoner_id: Filter by linked prisoner ID
        prisoner_name: Filter by linked prisoner name
        risk_level: Filter by risk level (low, medium, high)
        alert_category: Filter by alert category
        tag: Filter by tag
        limit: Maximum results (default 10)
    """
    result = user_reports.search_user_reports(
        query=query,
        prisoner_id=prisoner_id,
        prisoner_name=prisoner_name,
        risk_level=risk_level,
        alert_category=alert_category,
        tag=tag,
        limit=limit,
    )
    return json.dumps(result.get("data", {}), indent=2)


ALL_TOOLS = [
    search_conversations_tool,
    search_prisoners_tool,
    search_incidents_tool,
    search_user_reports_tool,
]


class AgentService:
    """Service for answering questions using LangChain agent with tools."""

    def __init__(self, task: str = "agent"):
        self._task = task
        self._config_loader = AgentConfigLoader(task=task)
        self._client = None
        self._tools = ALL_TOOLS
        self._agent = None
        self._init_error: Optional[str] = None
        self._cw = get_cloudwatch_client("agent")
        self._initialize_agent()

    def _initialize_agent(self) -> None:
        """Initialize the LangChain agent with tools."""
        self._cw.log_info("Initializing agent service")
        if not self._config_loader.load():
            self._agent = None
            self._init_error = f"config_load_failed: {self._config_loader.get_load_error()}"
            self._cw.log_error("Agent config load failed", error=self._init_error)
            return
        
        self._client, client_error = create_client_from_registry(task=self._task)
        if not self._client or not self._client.is_available():
            self._agent = None
            self._init_error = client_error or "llm_client_unavailable"
            self._cw.log_error("LLM client unavailable", error=self._init_error)
            return

        try:
            from langchain.agents import create_agent  # type: ignore

            prompt_config = self._config_loader.get_prompt_config()
            if not prompt_config:
                raise RuntimeError("Prompt config not loaded")
            
            system_prompt = prompt_config.get_system_prompt()
            llm = self._client.get_llm()
            
            self._agent = create_agent(
                model=llm,
                tools=self._tools,
                system_prompt=system_prompt,
            )
            self._init_error = None
            self._cw.log_info("Agent initialized successfully")
        except Exception as e:
            self._agent = None
            self._init_error = f"agent_init_failed: {type(e).__name__}: {e}"
            self._cw.log_error("Agent initialization failed", error=str(e), error_type=type(e).__name__)

    def is_available(self) -> bool:
        """Check if the agent is properly initialized."""
        return self._agent is not None

    def get_init_error(self) -> Optional[str]:
        """Return initialization error details, if any."""
        return self._init_error

    def answer(self, request: AgentRequest) -> AgentResponse:
        """
        Answer a question using the agent.

        Args:
            request: The agent request with question and metadata

        Returns:
            AgentResponse with answer, citations, and tool call records
        """
        self._cw.log_info("Agent answer request received", request_id=request.request_id, question_length=len(request.question))
        tracer = get_tracer()
        trace = tracer.start_trace(request.request_id)

        if not self.is_available():
            llm_error = None
            if hasattr(self._client, "get_init_error"):
                llm_error = self._client.get_init_error()

            details = self._init_error or llm_error
            if details:
                msg = f"Agent is not available. Details: {details}"
            else:
                msg = "Agent is not available. The API key may be missing or invalid."
            self._cw.log_error("Agent unavailable", request_id=request.request_id, details=details)
            trace.finish()
            raise RuntimeError(msg)

        try:
            start_time = time.time()
            with self._cw.trace_operation("agent_invoke", request_id=request.request_id) as ctx:
                result = self._agent.invoke({"messages": [("user", request.question)]})
            total_time_ms = (time.time() - start_time) * 1000

            # LangChain 1.2.2 CompiledStateGraph returns messages in result
            answer = ""
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    answer = last_message.content
                elif isinstance(last_message, dict):
                    answer = last_message.get("content", "")
                else:
                    answer = str(last_message)

            tool_calls: List[ToolCallRecord] = []
            all_citations: List[Citation] = []

            # Extract tool calls from messages
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get("name", "unknown") if isinstance(tool_call, dict) else getattr(tool_call, "name", "unknown")
                        tool_input = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                        
                        tool_calls.append(ToolCallRecord(
                            tool_name=tool_name,
                            inputs=tool_input if isinstance(tool_input, dict) else {"input": str(tool_input)},
                            output_size=0,
                            latency_ms=0,
                            success=True,
                        ))
                        
                        self._cw.log_tool_call(
                            tool_name=tool_name,
                            request_id=request.request_id,
                            success=True,
                            output_size=0,
                        )
                
                # Extract tool results for citations
                if hasattr(msg, "type") and msg.type == "tool":
                    tool_name = getattr(msg, "name", "")
                    content = getattr(msg, "content", "")
                    
                    try:
                        obs_data = json.loads(content) if isinstance(content, str) else content
                        items = obs_data.get("items", []) if isinstance(obs_data, dict) else []
                        
                        if tool_name == "search_conversations_tool":
                            all_citations.extend(conversations.extract_citations(items))
                        elif tool_name == "search_prisoners_tool":
                            all_citations.extend(prisoner_info.extract_citations(items))
                        elif tool_name == "search_incidents_tool":
                            all_citations.extend(incident_reports.extract_citations(items))
                        elif tool_name == "search_user_reports_tool":
                            all_citations.extend(user_reports.extract_citations(items))
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        pass

            answer_with_citations = answer
            if all_citations:
                answer_with_citations = self._append_citations_to_answer(answer, all_citations[:20])

            trace.finish()
            
            self._cw.log_info(
                "Agent answer completed",
                request_id=request.request_id,
                tool_calls_count=len(tool_calls),
                citations_count=len(all_citations),
                total_time_ms=total_time_ms,
            )
            self._cw.metric("agent.answer.duration_ms", total_time_ms)
            self._cw.metric("agent.answer.tool_calls", float(len(tool_calls)))

            return AgentResponse(
                request_id=request.request_id,
                answer=answer_with_citations,
                citations=all_citations[:20],
                tool_calls=tool_calls,
                model_info=ModelInfo(
                    model_name=self._client.get_model_name(),
                    provider=self._client.get_provider(),
                ),
            )

        except Exception as e:
            trace.finish()
            self._cw.log_error(
                f"Agent answer failed: {str(e)}",
                request_id=request.request_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            return AgentResponse(
                request_id=request.request_id,
                answer=f"An error occurred while processing your request: {str(e)}",
                citations=[],
                tool_calls=[],
                model_info=ModelInfo(
                    model_name=self._client.get_model_name(),
                    provider=self._client.get_provider(),
                ),
            )

    def _append_citations_to_answer(self, answer: str, citations: List[Citation]) -> str:
        normalized = (answer or "").strip()
        if not citations:
            return normalized

        if "\nsources:\n" in f"\n{normalized.lower()}\n":
            return normalized

        lines: List[str] = []
        for idx, c in enumerate(citations, start=1):
            label = f"{c.source_type}:{c.source_id}" if c.source_type and c.source_id else (c.source_id or c.source_type or "source")
            if c.excerpt:
                lines.append(f"{idx}. {label} — {c.excerpt}")
            else:
                lines.append(f"{idx}. {label}")

        if not lines:
            return normalized

        inline_markers = "".join([f"[{i}]" for i in range(1, len(lines) + 1)])
        answer_with_markers = normalized
        if answer_with_markers and inline_markers and inline_markers not in answer_with_markers:
            answer_with_markers = f"{answer_with_markers} {inline_markers}".strip()

        suffix = "\n\nSources:\n" + "\n".join(lines)
        if answer_with_markers:
            return answer_with_markers + suffix
        return (inline_markers + suffix).lstrip()


_service_instance: Optional[AgentService] = None


def get_service() -> AgentService:
    """Get or create the singleton service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = AgentService()
    return _service_instance


def reset_service() -> None:
    """Reset the singleton service instance (forces re-init on next get_service)."""
    global _service_instance
    _service_instance = None


def answer(request: AgentRequest) -> AgentResponse:
    """Convenience function to answer using the singleton service."""
    return get_service().answer(request)
