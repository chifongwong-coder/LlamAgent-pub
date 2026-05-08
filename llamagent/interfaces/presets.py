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
    "retrieval": {
        # Default to the FS knowledge backend — zero-dep, works without chromadb,
        # and is consistent with memory / reflection interface defaults.
        "retrieval_backend": "fs",
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
    "safety": "Safety -- Blocks harmful content and protects private data",
    "tools": "Tools -- Read/write files, search the web, run commands",
    "sandbox": "Sandbox -- Runs untrusted code safely in isolation",
    "planning": "Planning -- Breaks complex tasks into steps and tracks progress",
    "reflection": "Reflection -- Learns from past mistakes to avoid repeating them",
    "retrieval": "Retrieval -- Searches your uploaded documents for relevant answers",
    "memory": "Memory -- Remembers important facts across conversations",
    "child_agent": "Child Agent -- Delegates sub-tasks to specialized agents",
    "mcp": "MCP -- Uses tools from external services (GitHub, Slack, etc.)",
    "skill": "Skill -- Follows step-by-step guides for specific workflows",
    "compression": "Compression -- Summarizes older messages to save context space",
    "persistence": "Persistence -- Saves conversation across restarts",
    "resilience": "Resilience -- Auto-retries and switches models on failure",
    "job": "Job -- Runs long-running commands in the background",
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
    """Apply smart defaults for selected modules. Only sets values still at default.

    v3.8.1 R7-#24: build the reference defaults from a Config instance
    that bypasses ``_load_yaml`` / ``_apply_env_overrides``, so the
    'still at default' comparison reflects code-level dataclass defaults
    rather than the user's YAML / env overrides. Pre-fix calling
    ``Config()`` re-loaded the user's YAML, so any key the user set
    via YAML matched ``defaults`` and got silently overwritten by the
    preset.
    """
    from llamagent.core.config import Config
    # Bypass init's YAML / env override paths; only run _set_defaults.
    defaults = Config.__new__(Config)
    defaults._set_defaults()
    defaults._post_process()
    for name in module_names:
        for key, value in MODULE_PRESETS.get(name, {}).items():
            if hasattr(config, key) and getattr(config, key) == getattr(defaults, key):
                setattr(config, key, value)
