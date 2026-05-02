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

v3.7 highlights (parent-child shared persistent storage, read-only contract):
- ``Module.shareable: bool = False`` class attribute. Modules opt in
  by flipping to True. The child_agent factory consults this flag
  before invoking ``inherit_storage_from`` and raises ValueError on
  modules that aren't declared shareable. Reflection on the v3.7.1
  roadmap will be the second consumer of this contract — keeps the
  abstraction earning its keep per P6 ("reserve extension interfaces
  for foreseeable cases").
- ``Module.inherit_storage_from(other)`` lifecycle hook. Default
  raises NotImplementedError. Subclasses (today: ``MemoryModule``)
  override to copy ONLY data-layer handles (stores, vector pipelines)
  — NOT LLM-bound helpers (compilers, mergers). This keeps each
  agent's BudgetedLLM / model selection intact across the inherit.
- ``AgentExecutionPolicy.share_parent_modules: list[str] | None``.
  Per-spawn opt-in. Example: ``share_parent_modules=["memory"]`` lets
  a child read the parent's persistent memory store. Default None
  preserves the pre-v3.7 contract: child has no persistent modules.
- **Read-only contract**: shared children get only the read tools
  (``recall_memory`` / ``list_memories`` / ``read_memory``). Write
  tools (``save_memory`` / ``consolidate_memory``) are NOT registered
  on the shared child. The framework forces ``memory_mode = "off"``
  via ``_apply_shared_modules`` so the on_attach branch that registers
  write tools never fires. Bypasses the auth-scope leak that direct
  child writes to the parent's persona-keyed FS dir would otherwise
  open. Children that need to write must send a message to the parent
  and let the parent persist on their behalf.
- Parent recall mode is inherited (parent ``memory_recall_mode="off"``
  → child also off). Children never have permissions the parent
  itself has disabled.
- Concurrency safety relies on storage-layer primitives (Chroma's
  ``PerThreadPool`` + busy_timeout for vector backends; ``os.replace``
  atomic writes for FS backends). Pinned ``chromadb>=0.6.1`` so the
  PR #3335 LRU Segment Cache thread-safety fix is guaranteed. v3.8
  will add explicit per-store locks; module-level locks are NEVER
  appropriate (modules are not shared, stores are).
- Documentation: see ``docs/llamagent-v3.7-plan.md`` (private) for the
  full design rationale, three rounds of reviewer findings, and the
  per-decision audit. v3.6 plan ``docs/llamagent-v3.6-plan.md`` is
  the orthogonal predecessor (dispatch identity vs. storage ownership).

v3.6 highlights (tool-dispatch contract: agent identity is runtime, not closure):
- ``register_tool(takes_agent: bool = False)`` flag. When True, the
  framework dispatcher injects the calling agent as the first
  positional arg at every tool-invocation site (``call_tool``,
  ``_execute_with_timeout``, ``ToolExecutor.execute``). Tool functions
  read ``agent.project_dir`` / ``agent.write_root`` /
  ``agent._authorization_engine`` etc. from the param, so a child
  agent's tools resolve against the child's own state instead of
  closure-aliasing back to the parent that registered them.
- ``tools/module.py``'s 13 framework tools (read/write/patch/list/
  glob/search/rename/move/copy/delete/temp/revert) and ``sandbox``'s
  ``command`` tool migrated to ``takes_agent=True``. Path-extractor
  lambdas upgraded to ``(args, agent)`` signature; helpers
  (``_safe_extract_paths``, ``_writable_root_hint``, etc.) now take
  ``agent`` as first param.
- ``share_parent_project_dir=False`` (isolated child) actually works
  end-to-end now: child's ``write_files`` writes under the child's
  ``<parent.playground>/children/<task_id>/`` directory, not
  parent's project_dir. Verified via real-LLM RCC-V35-04.
- Persona-tools (toolsmith) and external custom modules default to
  ``takes_agent=False`` — backward compatible. No persona-tool
  signature change required.

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

__version__ = "3.7"

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
