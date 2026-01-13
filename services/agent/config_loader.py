"""Config loader for agent prompts and tools from S3."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import yaml

from mock_apis.cloud_apis import mock_model_registry, mock_s3


class PromptConfig:
    """Prompt configuration loaded from S3."""
    
    def __init__(self, data: Dict[str, Any]):
        self.system_prompt = data.get("system_prompt", "")
        self.metadata = data.get("metadata", {})
    
    def get_system_prompt(self) -> str:
        """Get the system prompt text."""
        return self.system_prompt


class ToolParameter:
    """Tool parameter definition."""
    
    def __init__(self, data: Dict[str, Any]):
        self.name = data.get("name", "")
        self.type = data.get("type", "string")
        self.required = data.get("required", False)
        self.default = data.get("default")


class ToolDefinition:
    """Tool definition loaded from S3."""
    
    def __init__(self, data: Dict[str, Any]):
        self.name = data.get("name", "")
        self.description = data.get("description", "")
        self.module = data.get("module", "")
        self.function = data.get("function", "")
        self.parameters = [ToolParameter(p) for p in data.get("parameters", [])]


class ToolsConfig:
    """Tools configuration loaded from S3."""
    
    def __init__(self, data: Dict[str, Any]):
        self.tools = [ToolDefinition(t) for t in data.get("tools", [])]
        self.metadata = data.get("metadata", {})
    
    def get_tools(self) -> List[ToolDefinition]:
        """Get the list of tool definitions."""
        return self.tools


class AgentConfigLoader:
    """Loader for agent configuration from registry and S3."""
    
    def __init__(self, task: str = "agent", bucket: str = "agent-configs"):
        self.task = task
        self.bucket = bucket
        self._prompt_config: Optional[PromptConfig] = None
        self._tools_config: Optional[ToolsConfig] = None
        self._load_error: Optional[str] = None
    
    def load(self) -> bool:
        """
        Load agent configuration from registry and S3.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            agent_config = mock_model_registry.get_agent_config(self.task)
            if not agent_config:
                self._load_error = f"No agent config found in registry for task '{self.task}'"
                return False
            
            prompt_key = f"prompts/{agent_config['prompt_bundle_id']}/{agent_config['prompt_version']}.yaml"
            prompt_data = mock_s3.get_object(self.bucket, prompt_key)
            if not prompt_data:
                self._load_error = f"Prompt config not found in S3: {self.bucket}/{prompt_key}"
                return False
            
            prompt_yaml = yaml.safe_load(prompt_data.decode("utf-8"))
            self._prompt_config = PromptConfig(prompt_yaml)
            
            tools_key = f"tools/{agent_config['tools_bundle_id']}/{agent_config['tools_version']}.yaml"
            tools_data = mock_s3.get_object(self.bucket, tools_key)
            if not tools_data:
                self._load_error = f"Tools config not found in S3: {self.bucket}/{tools_key}"
                return False
            
            tools_yaml = yaml.safe_load(tools_data.decode("utf-8"))
            self._tools_config = ToolsConfig(tools_yaml)
            
            self._load_error = None
            return True
            
        except Exception as e:
            self._load_error = f"Failed to load agent config: {type(e).__name__}: {e}"
            return False
    
    def get_prompt_config(self) -> Optional[PromptConfig]:
        """Get the loaded prompt configuration."""
        return self._prompt_config
    
    def get_tools_config(self) -> Optional[ToolsConfig]:
        """Get the loaded tools configuration."""
        return self._tools_config
    
    def get_load_error(self) -> Optional[str]:
        """Get the load error message if any."""
        return self._load_error
    
    def is_loaded(self) -> bool:
        """Check if configuration is loaded."""
        return self._prompt_config is not None and self._tools_config is not None
