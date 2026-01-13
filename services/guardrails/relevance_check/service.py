"""Relevance check service implementation."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

from mock_apis.cloud_apis import mock_model_registry
from services.cloudwatch_client import get_cloudwatch_client

from .schemas import RelevanceCheckRequest, RelevanceCheckResponse

class RelevanceCheckService:
    """Service for checking question relevance using trained classifier."""

    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        self._model_id: Optional[str] = None
        self._load_error: Optional[str] = None
        self._cw = get_cloudwatch_client("relevance_check")
        self._load_model(model_path)

    def _resolve_registry_model_path(self) -> Optional[Path]:
        model_info = mock_model_registry.get_current_model(task="relevance_check")
        if not model_info:
            return None

        models_dir = Path(__file__).resolve().parents[3] / "train" / "models"
        candidate = models_dir / f"{model_info['model_id']}.pkl"
        return candidate if candidate.exists() else None

    def _maybe_reload_model_from_registry(self) -> None:
        """Reload model if the registry points to a newer/different model than currently loaded."""
        desired = self._resolve_registry_model_path()
        if desired is None:
            return

        desired_id = desired.stem
        if self._model is None or self._model_id != desired_id:
            self._cw.log_info(
                "Reloading relevance check model from registry",
                current_model_id=self._model_id,
                desired_model_id=desired_id,
                desired_model_path=str(desired),
            )
            self._load_from_path(desired)

    def _load_model(self, model_path: Optional[Path] = None) -> None:
        """Load the relevance check model from registry or explicit path."""
        self._cw.log_info("Loading relevance check model")
        if model_path and model_path.exists():
            self._load_from_path(model_path)
            return

        candidate = self._resolve_registry_model_path()
        if candidate is not None:
            self._load_from_path(candidate)
            return

        models_dir = Path(__file__).resolve().parents[3] / "train" / "models"
        if models_dir.exists():
            pkl_files = sorted(
                [f for f in models_dir.glob("relevance_check_*.pkl")],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if pkl_files:
                self._load_from_path(pkl_files[0])

    def _load_from_path(self, path: Path) -> None:
        """Load model from a specific path."""
        try:
            from train.train_relevance_check import NaiveBayesTextClassifier

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
            self._cw.log_info("Relevance check model loaded", model_id=self._model_id, model_path=str(path))
        except Exception as e:
            self._model = None
            self._model_id = None
            self._load_error = f"{type(e).__name__}: {e}"
            self._cw.log_error("Failed to load relevance check model", error=str(e), model_path=str(path))

    def check_relevance(self, request: RelevanceCheckRequest) -> RelevanceCheckResponse:
        """
        Check if a question is relevant to the domain.

        Uses a combination of:
        1. Trained classifier (if available)
        2. Keyword-based detection as fallback/supplement
        """
        self._cw.log_info("Relevance check started", request_id=request.request_id)
        self._maybe_reload_model_from_registry()
        question = request.question

        if self._model is None:
            self._cw.log_info(
                "Relevance model unavailable; failing closed",
                request_id=request.request_id,
            )
            self._cw.metric("relevance_check.not_relevant", 1.0)
            return RelevanceCheckResponse(
                request_id=request.request_id,
                relevant=False,
                reason="Relevance model unavailable",
                matched_domains=[],
                model_id=self._model_id,
            )

        prediction = self._model.predict([question])[0]
        is_relevant = prediction == "relevant"

        if is_relevant:
            self._cw.log_info(
                "Relevance check: RELEVANT",
                request_id=request.request_id,
                relevant=True,
                has_model=self._model is not None,
            )
            self._cw.metric("relevance_check.relevant", 1.0)
            return RelevanceCheckResponse(
                request_id=request.request_id,
                relevant=True,
                reason="Question is relevant to the correctional facility domain",
                matched_domains=[],
                model_id=self._model_id,
            )

        self._cw.log_info(
            "Relevance check: NOT RELEVANT",
            request_id=request.request_id,
            relevant=False,
            has_model=self._model is not None,
        )
        self._cw.metric("relevance_check.not_relevant", 1.0)
        return RelevanceCheckResponse(
            request_id=request.request_id,
            relevant=False,
            reason="Question is not relevant to the correctional facility domain",
            matched_domains=[],
            model_id=self._model_id,
        )


_service_instance: Optional[RelevanceCheckService] = None


def get_service() -> RelevanceCheckService:
    """Get or create the singleton service instance."""
    global _service_instance
    if _service_instance is None:
        model_path_env = os.getenv("RELEVANCE_MODEL_PATH")
        model_path = Path(model_path_env) if model_path_env else None
        _service_instance = RelevanceCheckService(model_path=model_path)
    return _service_instance


def check_relevance(request: RelevanceCheckRequest) -> RelevanceCheckResponse:
    """Convenience function to check relevance using the singleton service."""
    return get_service().check_relevance(request)
