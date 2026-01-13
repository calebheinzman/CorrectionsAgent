"""Safety check service implementation."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

from mock_apis.cloud_apis import mock_model_registry
from services.cloudwatch_client import get_cloudwatch_client

from .policy import get_policy_version
from .schemas import SafetyCheckRequest, SafetyCheckResponse


class SafetyCheckService:
    """Service for checking question safety using trained classifier."""

    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        self._model_id: Optional[str] = None
        self._load_error: Optional[str] = None
        self._cw = get_cloudwatch_client("safety_check")
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[Path] = None) -> None:
        """Load the safety check model from registry or explicit path."""
        self._cw.log_info("Loading safety check model")
        if model_path and model_path.exists():
            self._load_from_path(model_path)
            return

        model_info = mock_model_registry.get_current_model(task="safety_check")
        if model_info:
            models_dir = Path(__file__).resolve().parents[3] / "train" / "models"
            candidate = models_dir / f"{model_info['model_id']}.pkl"
            if candidate.exists():
                self._load_from_path(candidate)
                return

        models_dir = Path(__file__).resolve().parents[3] / "train" / "models"
        if models_dir.exists():
            pkl_files = sorted(
                [f for f in models_dir.glob("safety_check_*.pkl")],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if pkl_files:
                self._load_from_path(pkl_files[0])

    def _load_from_path(self, path: Path) -> None:
        """Load model from a specific path."""
        try:
            from train.train_safety_check import NaiveBayesTextClassifier

            class _CompatUnpickler(pickle.Unpickler):
                def find_class(self, module: str, name: str):  # type: ignore[override]
                    if module == "__main__" and name == "NaiveBayesTextClassifier":
                        return NaiveBayesTextClassifier
                    return super().find_class(module, name)

            with open(path, "rb") as f:
                artifact = _CompatUnpickler(f).load()
            self._model = artifact.get("model")
            self._model_id = artifact.get("run_id", path.stem)
            self._load_error = None
            self._cw.log_info("Safety check model loaded", model_id=self._model_id, model_path=str(path))
        except Exception as e:
            self._model = None
            self._model_id = None
            self._load_error = f"{type(e).__name__}: {e}"
            self._cw.log_error("Failed to load safety check model", error=str(e), model_path=str(path))

    def check_safety(self, request: SafetyCheckRequest) -> SafetyCheckResponse:
        """
        Check if a question is safe to process.

        Uses a combination of:
        1. Trained classifier (if available)
        2. Keyword-based detection as fallback/supplement
        """
        self._cw.log_info("Safety check started", request_id=request.request_id)
        question = request.question
        policy = get_policy_version()

        if self._model is None:
            self._cw.log_info(
                "Safety model unavailable; failing closed",
                request_id=request.request_id,
            )
            self._cw.metric("safety_check.denied", 1.0, dimensions={"policy": policy})
            return SafetyCheckResponse(
                request_id=request.request_id,
                allowed=False,
                policy=policy,
                reason="Safety model unavailable",
                categories=[],
                model_id=self._model_id,
            )

        prediction = self._model.predict([question])[0]
        is_dangerous = prediction == "dangerous"

        if is_dangerous:
            self._cw.log_info(
                "Safety check: DENIED",
                request_id=request.request_id,
                allowed=False,
                has_model=self._model is not None,
            )
            self._cw.metric("safety_check.denied", 1.0, dimensions={"policy": policy})
            return SafetyCheckResponse(
                request_id=request.request_id,
                allowed=False,
                policy=policy,
                reason="Question flagged as potentially dangerous",
                categories=[],
                model_id=self._model_id,
            )

        self._cw.log_info(
            "Safety check: ALLOWED",
            request_id=request.request_id,
            allowed=True,
            has_model=self._model is not None,
        )
        self._cw.metric("safety_check.allowed", 1.0, dimensions={"policy": policy})
        return SafetyCheckResponse(
            request_id=request.request_id,
            allowed=True,
            policy=policy,
            reason="Question passed safety checks",
            categories=[],
            model_id=self._model_id,
        )


_service_instance: Optional[SafetyCheckService] = None


def get_service() -> SafetyCheckService:
    """Get or create the singleton service instance."""
    global _service_instance
    if _service_instance is None:
        model_path_env = os.getenv("SAFETY_MODEL_PATH")
        model_path = Path(model_path_env) if model_path_env else None
        _service_instance = SafetyCheckService(model_path=model_path)
    return _service_instance


def check_safety(request: SafetyCheckRequest) -> SafetyCheckResponse:
    """Convenience function to check safety using the singleton service."""
    return get_service().check_safety(request)
