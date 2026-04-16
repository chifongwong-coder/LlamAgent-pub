"""Module presets: smart defaults so users don't need config.yaml knowledge."""

MODULE_PRESETS = {
    "memory": {
        "memory_mode": "hybrid",
        "memory_recall_mode": "auto",
        "memory_backend": "fs",
    },
    "reflection": {
        "reflection_write_mode": "auto",
        "reflection_read_mode": "auto",
        "reflection_backend": "fs",
    },
    "child_agent": {
        "child_agent_runner": "thread",
    },
    "compression": {
        "tool_result_strategy": "head",
        "strip_thinking": True,
    },
    "persistence": {
        "persistence_enabled": True,
        "persistence_auto_restore": True,
    },
}

# User-friendly descriptions -- no implementation jargon
MODULE_DESCRIPTIONS = {
    "safety": "Safety -- Filters harmful content and protects sensitive data",
    "tools": "Tools -- File operations, web search, command execution",
    "sandbox": "Sandbox -- Runs risky operations in isolated environment",
    "planning": "Planning -- Breaks complex tasks into manageable steps",
    "reflection": "Reflection -- Learns from mistakes, improves over time",
    "retrieval": "Retrieval -- Searches your documents for answers",
    "memory": "Memory -- Remembers important facts across conversations",
    "child_agent": "Child Agent -- Delegates sub-tasks to specialized agents",
    "mcp": "MCP -- Uses tools from external services (GitHub, Slack, etc.)",
    "skill": "Skill -- Runs step-by-step guides for specific tasks",
    "compression": "Compression -- Summarizes old messages to keep conversations fast",
    "persistence": "Persistence -- Saves conversation across restarts",
    "resilience": "Resilience -- Auto-retries and switches models on failure",
    "job": "Job -- Runs and manages background commands",
}

# Grouped for CLI/Web display
MODULE_GROUPS = {
    "Core": ["safety", "tools", "sandbox"],
    "Intelligence": ["planning", "reflection"],
    "Knowledge": ["retrieval", "memory"],
    "Collaboration": ["child_agent", "mcp"],
    "Workflow": ["skill", "job"],
    "System": ["compression", "persistence", "resilience"],
}


def apply_presets(config, module_names: list[str]):
    """Apply smart defaults for selected modules. Only sets values still at default."""
    from llamagent.core.config import Config
    defaults = Config()
    for name in module_names:
        for key, value in MODULE_PRESETS.get(name, {}).items():
            if hasattr(config, key) and getattr(config, key) == getattr(defaults, key):
                setattr(config, key, value)
