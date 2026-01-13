"""Gemini LLM client using LangChain."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from mock_apis.cloud_apis import mock_secrets_manager

from .interfaces import LLMClient


class GeminiClient(LLMClient):
    """LangChain-based Gemini client."""

    def __init__(self, model_name: str = "gemini-flash-2.0-flash-lite"):
        self._model_name = model_name
        self._provider = "google"
        self._llm = None
        self._init_error: Optional[str] = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the LangChain Gemini model."""
        self._init_error = None
        try:
            mock_secrets_manager.reload_env()
        except Exception:
            pass

        candidate_names = [
            "GOOGLE_API_KEY",
            "GOOGLE_GENAI_API_KEY",
            "GEMINI_API_KEY",
        ]

        api_key = None
        for name in candidate_names:
            api_key = mock_secrets_manager.get_secret(name)
            if api_key:
                break
            api_key = os.getenv(name)
            if api_key:
                break

        if not api_key:
            self._llm = None
            self._init_error = (
                "missing_api_key: set one of GOOGLE_API_KEY, GOOGLE_GENAI_API_KEY, or GEMINI_API_KEY in .env"
            )
            return

        # Prefer explicit env override for model id.
        env_model = os.getenv("GEMINI_MODEL") or os.getenv("GEMINI_MODEL_ID") or os.getenv("MODEL_ID")
        model_candidates = [m for m in [env_model, self._model_name] if m]
        model_candidates.extend(
            [
                "gemini-2.0-flash-lite",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
            ]
        )

        # Deduplicate while preserving order
        seen = set()
        model_candidates = [m for m in model_candidates if not (m in seen or seen.add(m))]

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            last_error: Optional[Exception] = None
            for model in model_candidates:
                try:
                    self._llm = ChatGoogleGenerativeAI(
                        model=model,
                        google_api_key=api_key,
                        temperature=0.1,
                        max_output_tokens=2048,
                    )
                    self._model_name = model
                    return
                except Exception as e:
                    last_error = e
                    self._llm = None

            if last_error is not None:
                raise last_error
        except Exception as e:
            self._llm = None
            self._init_error = f"init_failed: {type(e).__name__}: {e}"

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Invoke the LLM with messages and return the response."""
        if self._llm is None:
            detail = self._init_error or "unknown"
            raise RuntimeError(f"Gemini client not initialized: {detail}")

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        response = self._llm.invoke(lc_messages)
        return response.content

    def get_model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def get_provider(self) -> str:
        """Return the provider name."""
        return self._provider

    def is_available(self) -> bool:
        """Check if the client is properly initialized."""
        return self._llm is not None

    def get_init_error(self) -> Optional[str]:
        """Return initialization error details, if any."""
        return self._init_error

    def get_llm(self):
        """Return the underlying LangChain LLM for tool binding."""
        return self._llm
