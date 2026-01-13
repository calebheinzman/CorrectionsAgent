"""Mock Model Registry - file-backed registry for model and policy metadata."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, TypedDict


class ModelInfo(TypedDict):
    provider: str
    model_id: str
    version: str


class PolicyBundle(TypedDict):
    bundle_id: str
    version: str


class AgentConfig(TypedDict):
    prompt_bundle_id: str
    prompt_version: str
    tools_bundle_id: str
    tools_version: str


class TaskedCurrentModel(TypedDict, total=False):
    tasks: Dict[str, ModelInfo]


_registry_dir: Optional[Path] = None


def _get_registry_dir() -> Path:
    """Get the registry data directory."""
    global _registry_dir
    if _registry_dir is None:
        _registry_dir = Path(__file__).parent / "data" / "registry"
        env_path = os.getenv("MOCK_REGISTRY_DIR")
        if env_path:
            _registry_dir = Path(env_path)
        _registry_dir.mkdir(parents=True, exist_ok=True)
    return _registry_dir


def _load_json(filename: str) -> Optional[dict]:
    """Load a JSON file from the registry directory."""
    path = _get_registry_dir() / filename
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return None
    return None


def get_current_model(task: Optional[str] = None) -> Optional[ModelInfo]:
    """
    Get the current model configuration.
    
    Returns None if not configured.
    """
    data = _load_json("current_model.json")
    if not data:
        return None

    if all(k in data for k in ("provider", "model_id", "version")):
        return ModelInfo(provider=data["provider"], model_id=data["model_id"], version=data["version"])

    if task is None:
        return None

    tasks = data.get("tasks") if isinstance(data, dict) else None
    if isinstance(tasks, dict):
        entry = tasks.get(task)
        if isinstance(entry, dict) and all(k in entry for k in ("provider", "model_id", "version")):
            return ModelInfo(
                provider=entry["provider"],
                model_id=entry["model_id"],
                version=entry["version"],
            )
    return None


def get_policy_bundle() -> Optional[PolicyBundle]:
    """
    Get the current policy bundle configuration.
    
    Returns None if not configured.
    """
    data = _load_json("policy_bundle.json")
    if data and all(k in data for k in ("bundle_id", "version")):
        return PolicyBundle(
            bundle_id=data["bundle_id"],
            version=data["version"],
        )
    return None


def set_current_model(provider: str, model_id: str, version: str, task: Optional[str] = None) -> None:
    """Set the current model configuration."""
    path = _get_registry_dir() / "current_model.json"
    if task is None:
        data = {"provider": provider, "model_id": model_id, "version": version}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return

    existing = _load_json("current_model.json")
    tasks: Dict[str, ModelInfo] = {}
    if isinstance(existing, dict):
        existing_tasks = existing.get("tasks")
        if isinstance(existing_tasks, dict):
            for k, v in existing_tasks.items():
                if isinstance(v, dict) and all(x in v for x in ("provider", "model_id", "version")):
                    tasks[str(k)] = ModelInfo(
                        provider=v["provider"],
                        model_id=v["model_id"],
                        version=v["version"],
                    )
    tasks[task] = ModelInfo(provider=provider, model_id=model_id, version=version)
    data = TaskedCurrentModel(tasks=tasks)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def set_policy_bundle(bundle_id: str, version: str) -> None:
    """Set the current policy bundle configuration."""
    path = _get_registry_dir() / "policy_bundle.json"
    data = {"bundle_id": bundle_id, "version": version}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_agent_config(task: str = "agent") -> Optional[AgentConfig]:
    """
    Get the agent configuration for prompts and tools.
    
    Args:
        task: The agent task name (default: "agent")
    
    Returns:
        AgentConfig with bundle IDs and versions, or None if not configured.
    """
    data = _load_json("agent_config.json")
    if not data:
        return None
    
    if isinstance(data, dict):
        tasks = data.get("tasks")
        if isinstance(tasks, dict):
            entry = tasks.get(task)
            if isinstance(entry, dict) and all(k in entry for k in ("prompt_bundle_id", "prompt_version", "tools_bundle_id", "tools_version")):
                return AgentConfig(
                    prompt_bundle_id=entry["prompt_bundle_id"],
                    prompt_version=entry["prompt_version"],
                    tools_bundle_id=entry["tools_bundle_id"],
                    tools_version=entry["tools_version"],
                )
    return None


def set_agent_config(
    prompt_bundle_id: str,
    prompt_version: str,
    tools_bundle_id: str,
    tools_version: str,
    task: str = "agent"
) -> None:
    """Set the agent configuration for prompts and tools."""
    path = _get_registry_dir() / "agent_config.json"
    
    existing = _load_json("agent_config.json")
    tasks: Dict[str, AgentConfig] = {}
    if isinstance(existing, dict):
        existing_tasks = existing.get("tasks")
        if isinstance(existing_tasks, dict):
            for k, v in existing_tasks.items():
                if isinstance(v, dict) and all(x in v for x in ("prompt_bundle_id", "prompt_version", "tools_bundle_id", "tools_version")):
                    tasks[str(k)] = AgentConfig(
                        prompt_bundle_id=v["prompt_bundle_id"],
                        prompt_version=v["prompt_version"],
                        tools_bundle_id=v["tools_bundle_id"],
                        tools_version=v["tools_version"],
                    )
    
    tasks[task] = AgentConfig(
        prompt_bundle_id=prompt_bundle_id,
        prompt_version=prompt_version,
        tools_bundle_id=tools_bundle_id,
        tools_version=tools_version,
    )
    data = {"tasks": tasks}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def reset_registry(base_dir: Optional[Path] = None) -> None:
    """Reset the registry directory (useful for testing)."""
    global _registry_dir
    _registry_dir = base_dir
