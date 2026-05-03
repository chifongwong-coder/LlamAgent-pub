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

v3.7.2 highlights (factory + spawn-tool merge, structural debt cleanup):
- **Factory merge**: the twin private factories ``_create_short_child_agent``
  and ``_create_continuous_child_agent`` (~80% structurally identical;
  recurring source of v3.5–v3.7.1 sibling-miss bugs) collapse into one
  ``_create_child_agent(spec)`` that branches on ``spec.continuous`` only
  where the logic genuinely differs (set_mode + post-set_mode override,
  registry register + messaging-tool wiring, system_prompt template,
  spawn / messaging tool prune list).
- **Spawn-tool merge**: ``_spawn_child`` and ``_spawn_continuous_child``
  remain as public tool entry points but delegate to a shared
  ``_spawn_impl(continuous=...)`` body. Mode-specific guards
  (``agent.mode == "continuous"``, runner != ``inline``) and return-message
  formatting (continuous one-liner vs short multi-line ``child_dir``
  header) stay in the public wrappers; the rest is one path. Closes the
  v3.7.1 pre-check + RuntimeError-narrowing parallel maintenance.
- **`build_metrics` lift**: byte-identical helper from
  ``runners/inline.py`` and ``runners/thread.py`` now lives once in
  ``runners/runner.py`` next to ``format_fallback_report`` /
  ``maybe_request_completion_report``. Single source of truth across
  every runner backend.
- **Prophecy purge**: stale "v3.8 will / may add" + "deferred indefinitely"
  references removed from highlights, ``Module.inherit_storage_from``
  docstring, ``share_parent_modules`` field comment, and the factory's
  shareable-module fallthrough comment. The codebase now states what
  it does, not what a hypothetical future version was promised to do.

v3.7.1 highlights (post-merge hardening of v3.7):
- **Spawn-tool pre-validation** of ``share_parent_modules``: a new
  module-level helper ``_check_share_modules(parent, share_modules,
  runner_name)`` is called from ``_spawn_child`` /
  ``_spawn_continuous_child`` BEFORE ``controller.spawn_child``.
  Without this, the helper's ``ValueError`` (parent missing module /
  not shareable) was swallowed by the runners'
  ``except Exception`` block at ``runners/{inline,thread}.py`` and
  reached the model as a misleading ``"Spawned child agent.\n
  task_id: ...\nResult: ...execution error: ValueError..."`` (false
  success header). The same helper still backs
  ``_apply_shared_modules`` factory-side, so direct
  ``_create_child_agent`` callers (tests) keep their loud-failure
  contract.
- **Process runner refusal** (B2): the same pre-check returns a clean
  refusal string when ``runner_name=="process"`` and
  ``share_parent_modules`` is non-empty. The subprocess can't alias
  an in-process Python store handle, and chromadb's
  ``PersistentClient`` is not multi-process-safe — cross-process
  sharing is intentionally not supported. Pre-v3.7.1, this combination
  silently ignored ``share_parent_modules``; v3.7.1 makes it loud.
- **Tool-name role split** on ``MemoryModule``: now exposes
  ``_WRITE_TOOL_NAMES`` (save / consolidate) and
  ``_READ_TOOL_NAMES`` (recall / list / read), with
  ``_TOOL_NAMES`` as the union. Role-aware tests can assert against
  the narrow set without confusing the reader by using the union
  for everything.
- **BudgetedLLM invariant pinned on production wire-order**: v3.7
  commit-9's test bypassed ``_attach_runlog`` by leaving
  ``spec.runlog_path`` empty. v3.7.1 sets ``runlog_path`` explicitly
  so ``LoggingLLM`` actually wraps ``child.llm``, then asserts
  ``child_mem.llm.tracker is child.llm.tracker`` (proxies through
  ``LoggingLLM.__getattr__``) instead of LLM-instance identity.
- **Reverted v3.7 commit-8**'s broaden of the spawn-tool catch from
  ``RuntimeError`` to ``(RuntimeError, ValueError)``. With the new
  pre-check fronting the share path, the broaden is dead code (the
  share-related ``ValueError`` no longer reaches the catch site).
  Per P6, dead code is worse than no code. ``logger.exception`` is
  kept for the live ``RuntimeError`` path.
- **Documentation tightening**: ``_apply_shared_modules`` docstring
  + ``AgentExecutionPolicy.share_parent_modules`` field comment now
  explicitly state that the no-share default also strips
  parent's deepcopied memory tool entries (closes a pre-v3.7
  silent leak via parent-bound closures). v3.7 ``__init__.py``
  highlights captured this in commit-10; v3.7.1 finishes the
  doc-update across the adjacent surfaces.

v3.7 highlights (parent-child shared persistent storage, read-only contract):
- ``Module.shareable: bool = False`` class attribute. Modules opt in
  by flipping to True. The child_agent factory consults this flag
  before invoking ``inherit_storage_from`` and raises ValueError on
  modules that aren't declared shareable. Today's only in-tree
  consumer is ``MemoryModule``; the abstraction's extension-interface
  justification rests on P6 ("reserve extension interfaces for
  foreseeable cases") plus the first consumer being non-trivial.
- ``Module.inherit_storage_from(other)`` lifecycle hook. Default
  raises NotImplementedError. Subclasses (today: ``MemoryModule``)
  override to copy ONLY data-layer handles (stores, vector pipelines)
  — NOT LLM-bound helpers (compilers, mergers). This keeps each
  agent's BudgetedLLM / model selection intact across the inherit.
- ``AgentExecutionPolicy.share_parent_modules: list[str] | None``.
  Per-spawn opt-in. Example: ``share_parent_modules=["memory"]`` lets
  a child read the parent's persistent memory store. Default None
  results in the child having no persistent modules — and crucially,
  also has the framework strip any memory-tool entries that landed
  in the child's tool table via ``copy.deepcopy(parent._tools)``.
  Pre-v3.7, those deepcopied entries closed over the parent's
  MemoryModule instance, so a child of a memory-loaded parent could
  silently invoke memory tools and reach back into parent's store.
  v3.7 closes that leak — the no-share default is now genuinely
  isolated, not just config-disabled.
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
  PR #3335 LRU Segment Cache thread-safety fix is guaranteed.
  Module-level locks are NEVER appropriate (modules are not shared,
  stores are).
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

__version__ = "3.7.2"

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
