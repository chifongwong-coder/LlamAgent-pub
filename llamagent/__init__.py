"""
LlamAgent — A modular AI Agent framework.

Core design:
- core/ provides a standalone base Agent (conversation, LLM calls,
  authorization engine, write-boundary primitives, persistence
  round-trip contracts).
- modules/ provides 15 pluggable enhanced capabilities (resilience,
  safety, compression, persistence, sandbox, tools, job, retrieval,
  memory, skill, reflection, reasoning/planning, mcp, child_agent,
  toolsmith). Loading a module is one line; modules are loosely
  coupled (graceful degradation when peers are absent).
- interfaces/ provides three interaction surfaces (CLI, Web UI, API)
  with shared module presets.

A bare LlamAgent is a fully functional conversational Agent. Each
module loaded grants a new capability.

v3.3 highlights:
- Model never sees a `zone` parameter or path prefix; the framework
  auto-classifies write paths into playground / project / rejected
  via classify_write.
- Long tool outputs (web_fetch, wait_job, child_agent return, large
  read_files) flow through a unified persistence contract: results
  are saved under llama_playground/tool_results/ and the model
  reads them back via read_files. read_files has an internal cap so
  re-reads can't cycle.
- Every typed write (write_files / apply_patch / move_path /
  copy_path / delete_path) is recorded as a Changeset and can be
  rolled back via revert_changes.

Usage:
    from llamagent import LlamAgent, Config, Module
    agent = LlamAgent(Config())
    reply = agent.chat("Hello")
"""

__version__ = "3.3"

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
