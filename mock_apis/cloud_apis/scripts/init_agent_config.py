"""Initialize agent configuration in registry and S3."""
from __future__ import annotations

from pathlib import Path

from mock_apis.cloud_apis import mock_model_registry, mock_s3


def init_agent_config():
    """Initialize agent configuration in registry and S3."""
    
    # Set model configuration for agent task
    mock_model_registry.set_current_model(
        provider="google",
        model_id="gemini-2.0-flash-lite",
        version="latest",
        task="agent"
    )
    
    # Set agent configuration (prompt and tools bundles)
    mock_model_registry.set_agent_config(
        prompt_bundle_id="investigative-assistant",
        prompt_version="v1.0.0",
        tools_bundle_id="investigative-tools",
        tools_version="v1.0.0",
        task="agent"
    )
    
    # Upload YAML configs to S3
    bucket = "agent-configs"
    
    # Upload prompt config
    prompt_path = Path(__file__).parent.parent / "data" / "s3" / bucket / "prompts" / "investigative-assistant" / "v1.0.0.yaml"
    if prompt_path.exists():
        prompt_data = prompt_path.read_bytes()
        mock_s3.put_object(bucket, "prompts/investigative-assistant/v1.0.0.yaml", prompt_data)
        print(f"✓ Uploaded prompt config: {bucket}/prompts/investigative-assistant/v1.0.0.yaml")
    else:
        print(f"✗ Prompt config not found at {prompt_path}")
    
    # Upload tools config
    tools_path = Path(__file__).parent.parent / "data" / "s3" / bucket / "tools" / "investigative-tools" / "v1.0.0.yaml"
    if tools_path.exists():
        tools_data = tools_path.read_bytes()
        mock_s3.put_object(bucket, "tools/investigative-tools/v1.0.0.yaml", tools_data)
        print(f"✓ Uploaded tools config: {bucket}/tools/investigative-tools/v1.0.0.yaml")
    else:
        print(f"✗ Tools config not found at {tools_path}")
    
    print("\n✓ Agent configuration initialized successfully")
    print(f"  - Model: google/gemini-2.0-flash-lite")
    print(f"  - Prompt bundle: investigative-assistant v1.0.0")
    print(f"  - Tools bundle: investigative-tools v1.0.0")


if __name__ == "__main__":
    init_agent_config()
