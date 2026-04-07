"""
LlamAgent — A modular AI Agent framework.

Core design:
- core/ provides a standalone base Agent (conversation, LLM calls)
- modules/ provides 13 pluggable enhanced capabilities (tools, retrieval, memory, reasoning, reflection, multi-agent, MCP, safety, skill, sandbox, child agent, job, compression)
- interfaces/ provides multiple interaction methods (CLI, Web UI, API)

Even without loading any modules, LlamAgent is a fully functional conversational Agent.
Each module loaded grants the Agent a new capability.

Usage:
    from llamagent import LlamAgent, Config, Module
    agent = LlamAgent(Config())
    reply = agent.chat("Hello")
"""

__version__ = "2.3"

# Export commonly used classes from the core layer for external convenience
from llamagent.core import LlamAgent, Module, Config, LLMClient, Persona, PersonaManager

__all__ = [
    "LlamAgent",
    "Module",
    "Config",
    "LLMClient",
    "Persona",
    "PersonaManager",
    "__version__",
]
