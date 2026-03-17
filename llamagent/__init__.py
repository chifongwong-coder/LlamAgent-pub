"""
LlamAgent — A modular AI Agent framework.

Core design:
- core/ provides a standalone base Agent (conversation, LLM calls)
- modules/ provides pluggable enhanced capabilities (tools, RAG, memory, reasoning, reflection, multi-agent, MCP, safety)
- interfaces/ provides multiple interaction methods (CLI, Web UI, API)

Even without loading any modules, SmartAgent is a fully functional conversational Agent.
Each module loaded grants the Agent a new capability.

Usage:
    from llamagent import SmartAgent, Config, Module
    agent = SmartAgent(Config())
    reply = agent.chat("Hello")
"""

__version__ = "1.1.0"

# Export commonly used classes from the core layer for external convenience
from llamagent.core import SmartAgent, Module, Config, LLMClient, Persona, PersonaManager

__all__ = [
    "SmartAgent",
    "Module",
    "Config",
    "LLMClient",
    "Persona",
    "PersonaManager",
    "__version__",
]
