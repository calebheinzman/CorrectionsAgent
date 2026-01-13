"""Request handler for orchestrator service."""
from __future__ import annotations

import time
from typing import Optional

import httpx

from .clients import agent_client, audit_store_client, relevance_check_client, safety_check_client
from .clients.audit_store_client import AuditRecord
from .logs import logger
from .middleware.request_id import get_request_id
from .schemas import AgentTrace, Citation, GuardrailDecision, QueryRequest, QueryResponse
from .settings import get_settings


class QueryHandler:
    """Handler for processing query requests."""

    def __init__(self):
        self._settings = get_settings()

    def handle_query(self, request: QueryRequest, request_id: str) -> QueryResponse:
        """
        Handle a query request through the full pipeline.

        Flow:
        1. Safety check
        2. Relevance check (if safety passes)
        3. Agent call (if relevance passes)
        4. Store audit record
        5. Return response

        Args:
            request: The query request
            request_id: The request ID

        Returns:
            QueryResponse with the result
        """
        logger.log_request(request.question, request.user_id, request.session_id)

        audit_record = AuditRecord(
            request_id=request_id,
            question=request.question,
        )

        safety_decision: Optional[GuardrailDecision] = None
        relevance_decision: Optional[GuardrailDecision] = None
        agent_trace: Optional[AgentTrace] = None
        citations: list[Citation] = []
        answer: Optional[str] = None
        status = "pending"

        try:
            safety_decision = self._check_safety(request_id, request.question, request.user_id)
            audit_record.safety_allowed = safety_decision.allowed
            audit_record.safety_reason = safety_decision.reason
            audit_record.safety_policy = safety_decision.policy
            audit_record.safety_model_id = getattr(safety_decision, 'model_id', None)

            logger.log_guardrail_decision(
                "safety",
                safety_decision.allowed or False,
                safety_decision.reason,
            )

            if not safety_decision.allowed:
                status = "denied"
                answer = f"Request denied by safety check: {safety_decision.reason}"
                audit_record.final_status = status
                audit_store_client.store_audit(audit_record)

                return QueryResponse(
                    request_id=request_id,
                    status=status,
                    answer=answer,
                    safety=safety_decision,
                )

            relevance_decision = self._check_relevance(request_id, request.question, request.user_id)
            audit_record.relevance_relevant = relevance_decision.relevant
            audit_record.relevance_reason = relevance_decision.reason
            audit_record.relevance_model_id = getattr(relevance_decision, 'model_id', None)

            logger.log_guardrail_decision(
                "relevance",
                relevance_decision.relevant or False,
                relevance_decision.reason,
            )

            if not relevance_decision.relevant:
                status = "denied"
                answer = f"Request denied by relevance check: {relevance_decision.reason}"
                audit_record.final_status = status
                audit_store_client.store_audit(audit_record)

                return QueryResponse(
                    request_id=request_id,
                    status=status,
                    answer=answer,
                    safety=safety_decision,
                    relevance=relevance_decision,
                )

            start_time = time.time()
            agent_result = self._call_agent(
                request_id,
                request.question,
                request.user_id,
                request.session_id,
            )
            latency_ms = (time.time() - start_time) * 1000

            audit_record.agent_called = True
            audit_record.agent_tool_calls = agent_result.trace.tool_calls
            audit_record.agent_latency_ms = latency_ms
            audit_record.agent_citations_count = len(agent_result.citations)
            if agent_result.trace.model_info:
                audit_record.agent_model_info = agent_result.trace.model_info
            logger.log_agent_call(len(agent_result.trace.tool_calls), latency_ms)

            answer = agent_result.answer
            citations = agent_result.citations
            agent_trace = agent_result.trace
            status = "success"
            audit_record.final_status = status

            audit_store_client.store_audit(audit_record)

            return QueryResponse(
                request_id=request_id,
                status=status,
                answer=answer,
                citations=citations,
                safety=safety_decision,
                relevance=relevance_decision,
                agent_trace=agent_trace,
            )

        except httpx.TimeoutException as e:
            status = "error"
            error_msg = f"Timeout error: {str(e)}"
            logger.log_error("TIMEOUT", error_msg)
            audit_record.final_status = status
            audit_record.error = error_msg
            audit_store_client.store_audit(audit_record)

            return QueryResponse(
                request_id=request_id,
                status=status,
                answer=error_msg,
                safety=safety_decision,
                relevance=relevance_decision,
            )

        except httpx.HTTPStatusError as e:
            status = "error"
            detail = None
            try:
                data = e.response.json()
                if isinstance(data, dict):
                    detail = data.get("detail")
            except Exception:
                try:
                    detail = e.response.text
                except Exception:
                    detail = None

            error_msg = f"Service error: {e.response.status_code}"
            if detail:
                error_msg = f"{error_msg} - {detail}"
            logger.log_error(
                "SERVICE_ERROR",
                error_msg,
                {"status_code": e.response.status_code, "detail": detail},
            )
            audit_record.final_status = status
            audit_record.error = error_msg
            audit_store_client.store_audit(audit_record)

            return QueryResponse(
                request_id=request_id,
                status=status,
                answer=error_msg,
                safety=safety_decision,
                relevance=relevance_decision,
            )

        except Exception as e:
            status = "error"
            error_msg = f"Unexpected error: {str(e)}"
            logger.log_error("UNEXPECTED_ERROR", error_msg)
            audit_record.final_status = status
            audit_record.error = error_msg
            audit_store_client.store_audit(audit_record)

            return QueryResponse(
                request_id=request_id,
                status=status,
                answer=error_msg,
                safety=safety_decision,
                relevance=relevance_decision,
            )

    def _check_safety(
        self,
        request_id: str,
        question: str,
        user_id: Optional[str],
    ) -> GuardrailDecision:
        """Call the safety check service."""
        if not self._settings.safety_enabled:
            return GuardrailDecision(
                allowed=True,
                reason="Safety check disabled (ORCHESTRATOR_SAFETY_ENABLED=false)",
            )
        return safety_check_client.check_safety(request_id, question, user_id)

    def _check_relevance(
        self,
        request_id: str,
        question: str,
        user_id: Optional[str],
    ) -> GuardrailDecision:
        """Call the relevance check service."""
        if not self._settings.relevance_enabled:
            return GuardrailDecision(
                relevant=True,
                reason="Relevance check disabled (ORCHESTRATOR_RELEVANCE_ENABLED=false)",
            )
        return relevance_check_client.check_relevance(request_id, question, user_id)

    def _call_agent(
        self,
        request_id: str,
        question: str,
        user_id: Optional[str],
        session_id: Optional[str],
    ) -> agent_client.AgentResult:
        """Call the agent service."""
        return agent_client.get_answer(request_id, question, user_id, session_id)


_handler_instance: Optional[QueryHandler] = None


def get_handler() -> QueryHandler:
    """Get or create the singleton handler instance."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = QueryHandler()
    return _handler_instance


def handle_query(request: QueryRequest, request_id: str) -> QueryResponse:
    """Convenience function to handle a query using the singleton handler."""
    return get_handler().handle_query(request, request_id)
