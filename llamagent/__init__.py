"""
LlamAgent — A modular AI Agent framework.

Core design:
- core/ provides a standalone base Agent (conversation, LLM calls,
  authorization engine, write-boundary primitives, persistence
  round-trip contracts).
- modules/ provides 14 pluggable enhanced capabilities (resilience,
  safety, compression, persistence, sandbox, tools, job, retrieval,
  memory, skill, reflection, reasoning/planning, mcp, child_agent).
  Loading a module is one line; modules are loosely coupled (graceful
  degradation when peers are absent). Toolsmith lives as a pack inside
  the tools module, not a separate module.
- interfaces/ provides three interaction surfaces (CLI, Web UI, API)
  with shared module presets.

A bare LlamAgent is a fully functional conversational Agent. Each
module loaded grants a new capability.

v3.5 highlights (child agent collaboration: summary + artifacts, not data passing):
- spawn_child returns structured text including child_dir so the
  parent's model can resolve relative artifact paths against the
  right directory. Hard break: wait_child no longer accepts
  include_history / include_logs (anti-pattern).
- Child completion report convention (Status / Summary / Artifacts).
  Two delivery templates, controlled by child_agent_report_template:
  "system_prompt" (default) | "auto" | "off". Framework does NOT
  parse record.result; the contract is model-to-model.
- Crash fallback: when a child crashes (BudgetExceededError,
  unhandled exception, SIGKILL, JSON decode error from process
  runner), the runner's finally block writes a v3.5-shaped
  fallback report into record.result so the parent has a
  consistent shape to read.
- Per-child runlog at <parent.playground>/child_runlogs/<task_id>.log
  (JSONL: reply / tool / end). Observability-only, not exposed to
  the parent agent through the tool surface.
- Cancellation cascade: cancel_child(task_id) walks descendants
  depth-first and runner-cancels each before the target.
- max_delegation_depth = 2 default (Hermes-style); enforced at
  spawn time.
- send_message accepts agent_id or task_id (resolves task_id →
  agent_id internally; the message_child tool was not added —
  one tool, two target shapes).

v3.4 highlights (terminology cleanup):
- rename_path(target, new_name) added to path-fallback pack for
  in-place renames; move_path now rejects same-parent calls.
- start_job cwd is path-only (None → scratch root, absolute → as-is,
  relative → project_dir). No special string literals.
- AgentExecutionPolicy.workspace_mode removed; replaced by
  share_parent_project_dir: bool (False = isolated child).
- Per-session scratch cache renamed: WorkspaceService → ScratchService;
  workspace_root → scratch_root; Config.workspace_id → scratch_id.
- builtin skill workspace-ops renamed to path-ops (hard break, no alias).

v3.3 highlights:
- Model never sees a `zone` parameter or path prefix; the framework
  auto-classifies write paths into playground / project / rejected
  via classify_write.
- Long tool outputs (web_fetch, wait_job, child_agent return, large
  read_files) flow through a unified persistence contract: results
  are saved under llama_playground/tool_results/ and the model
  reads them back via read_files. read_files has an internal cap so
  re-reads can't cycle.
- Every typed write (write_files / apply_patch / rename_path /
  move_path / copy_path / delete_path) is recorded as a Changeset and
  can be rolled back via revert_changes.

Usage:
    from llamagent import LlamAgent, Config, Module
    agent = LlamAgent(Config())
    reply = agent.chat("Hello")
"""

__version__ = "3.5"

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
