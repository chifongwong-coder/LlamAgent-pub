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

v3.7.8 highlights (v3.7-closing audit cleanup pack):
- **B1 sessions.py reads schema v=2**: ``interfaces/sessions.py`` listed
  v=1 only, so every session written since v3.7.5 (which bumped to v=2)
  was silently invisible to the CLI / Web UI session list. One-line fix
  to accept ``{1, 2}``.
- **B2 child factory rebinds service handles**: the v3.7 audit closed
  the parent-bound tool closure leak only for MemoryModule. ToolsModule's
  project-sync tools, JobModule's start/inspect/wait/cancel_job, and
  MCPModule's bridged tools all used to reach back into parent's
  service instances after a child was spawned. ``_apply_shared_modules``
  now strips parent-bound entries from ``child._tools`` and either
  inherits via ``inherit_storage_from`` (share-mode) or builds child-
  owned services and re-registers tools (isolated default). Each
  module declares ``_SERVICE_BOUND_TOOL_NAMES`` so the factory has a
  single source of truth.
- **B5 deprecation warnings**: ``AuthorizationEngine._switch_policy /
  _clear_all_scopes`` and ``LlamAgent._ask_confirmation`` (v3.7.3
  aliases scheduled for v3.8.1 removal) now emit ``DeprecationWarning``.
- **A-F25 ``_infer_parameters`` consolidation**: was implemented twice
  (LlamAgent, ToolRegistry), only LlamAgent supported ``skip_first_arg``.
  Extracted to a single module-level helper; ``@tool(takes_agent=True)``
  via decorator without explicit ``parameters=`` now produces a
  schema that omits the framework-injected first arg.
- **A-F26 ``_MODE_KEYS`` derived**: was hand-synchronized with
  ``_MODE_DEFAULTS``; now derived + import-time consistency check.
- **C-F4 YAML type-checking**: ``_load_yaml`` rejects wrong-type
  values for ``list`` / ``dict`` fields with a warning; pre-fix
  ``skill: { dirs: "string" }`` silently broke downstream iteration.
- **C-F6 SSRF guard now DNS-resolves hostnames**: ``web_fetch``
  rejects internal corporate hostnames whose DNS resolves to RFC1918,
  plus DNS-rebinding-style attacks. Unresolvable names default to
  reject (conservative).
- **C-F7 sandbox ``_DANGEROUS_BUILTINS`` extended**: agent-created
  tool code was checked against ``{exec, eval}`` only. Now also
  rejects ``compile``, ``__import__``, ``getattr/setattr/delattr``,
  ``globals/locals/vars``, ``breakpoint``, ``open``. Docstring
  downgraded from "restricted execution" to "best-effort sandbox".
- **C-F14 exception cause chain**: ``agent_tools._validate_syntax``
  / ``_compile`` re-raise with ``from e`` so debugging tool-creation
  failures preserves the original SyntaxError / Exception.
- **A-F1 dead test fixture attribute**: ``bare_agent`` continued to
  set ``agent._in_hook = False`` — production moved hook-reentry
  tracking to a class-level ``threading.local`` in v3.7.3. Removed
  the dead line.

The above are all v3.5-twin-mistake-class debt cleared by the
v3.7-closing full-codebase audit (memory rule: post-version review
covers the whole architecture, not just the diff).

v3.7.7 highlights (test fixture + log hygiene + child-contract docs):
- **`init_agent` test fixture** (additive, internal-only): a new
  ``init_agent`` fixture in ``tests_internal/conftest.py`` constructs a
  real ``LlamAgent(config)`` so tests covering paths bypassed by
  ``bare_agent`` (snapshot setup, auto_approve scope seeding, write_root
  resolution, hook registration) finally have coverage. Existing
  ``bare_agent`` references (~2848 of them) are NOT migrated; the new
  fixture is purely additive.
- **`print()` -> `logger` in library code**: 31 ``print()`` call sites
  in ``modules/safety/module.py``, ``modules/tools/module.py``,
  ``modules/mcp/module.py``, ``modules/mcp/client.py``, and
  ``modules/tools/agent_tools.py`` are replaced with module-level
  ``logger`` calls so hosts can route them through the standard logging
  configuration. Six entry-point files (``interfaces/cli.py``,
  ``interfaces/web_ui.py``, ``interfaces/api_server.py``, ``main.py``,
  ``modules/mcp/server_example.py``, ``tools/md_validator/validator.py``)
  keep their ``print()`` calls (65 sites) because they fire before
  logging is configured and target user stdout, not the log stream;
  each file's docstring carries an explicit "print() usage" note.
- **Parent -> child contract docs (ChildContract-2..7)**: a new
  architecture-doc section documents six edge cases discovered in the
  v3.7.6 deep audit — parent-bound tool closures in non-share modules,
  shallow Config dict aliasing, task scope longevity in shared
  children, factory fallback asymmetry, ``_persisted_files`` non-copy,
  and runtime hook non-propagation. Five are by-design / known-state
  with clear handling; v3.8 will decide opt-in mechanisms for the
  remaining two (`_persisted_files`, `inherit_runtime_hooks`).

> **Known issue, fixed in v3.7.8**: with ``auto_approve=True`` AND
> ``share_parent_project_dir=False`` (the researcher / writer / analyst /
> delegate roles' default), an isolated child agent's ``session_scopes``
> include both the parent's project directory AND the child's isolated
> directory — leaking parent-write permission through what should be a
> sandbox. Do NOT deploy v3.7.7 in that combination without the v3.7.8
> hotfix patch. v3.7.7 deployments that don't combine those two flags
> are unaffected.

v3.7.6 highlights (multi-tenant builtins via takes_agent):
- **`takes_agent` flag plumbed through the registry**: ``ToolInfo``,
  ``ToolRegistry.register``, the ``@tool`` decorator, and
  ``ToolsModule._bridge_to_core`` all carry the new
  ``takes_agent: bool = False`` field. ``LlamAgent.register_tool`` and
  the four dispatcher read sites (``call_tool`` paths in
  ``core/agent.py``, ``AuthorizationEngine``, sandbox executor) already
  honored the flag — this commit closes the write-side gap so a tool
  declared with ``takes_agent=True`` actually reaches the dispatcher
  through every registration path.
- **Per-agent ``_tool_state`` namespace**: ``LlamAgent._tool_state:
  dict[str, Any]`` is a per-agent dict for tools that need agent-scoped
  state. ``ToolsModule.on_attach`` writes the search backend (key
  ``"web_search_backend"``) and the interaction handler (key
  ``"ask_user_handler"``) here.
- **`web_search` / `ask_user` migrated**: both are now
  ``takes_agent=True`` and read state from ``agent._tool_state`` via
  the dispatcher-injected agent, instead of mutating module-level
  function attributes (``builtin.web_search._backend`` etc.). Two
  agents in the same process (e.g. API server's ``agent_sessions``)
  no longer alias each other's backend / handler. ``web_fetch`` is
  unchanged — it has no per-agent state, only the URL.
- **Child agents inherit ``_tool_state``**: the
  ``ChildAgentModule._create_child_agent`` factory shallow-copies
  ``parent._tool_state`` so a child whose ``_tools`` dict deepcopies
  builtin ``web_search`` / ``ask_user`` from the parent keeps the
  same search backend and interaction handler (service references
  shared, dict containers independent — same shape as the existing
  ``child.tool_executor = parent.tool_executor`` line). Pre-v3.7.6
  this fell out of having state on a process-global function
  attribute; the migration to per-agent storage required the factory
  to carry it forward explicitly.

v3.7.5 highlights (persistence forward-compat + compression marker):
- **Persistence schema v=2**: ``PersistenceModule._save`` now writes
  ``version=2`` and persists ``_delegation_depth`` + ``_active_packs``.
  ``_delegation_depth`` is the immediately-effective fix: in the niche
  but real case where a *child* agent runs Persistence and resumes
  after a crash, the depth cap survives the restart. ``_active_packs``
  is forward-compat groundwork for v3.8 Q3: as of v3.7.5 the first
  ``ToolsModule.on_input`` after restore wipes the set and re-derives
  state-driven packs from in-memory services (``JobService`` itself is
  not yet persisted), so the persisted set isn't observable until
  v3.8 lands JobService persistence. See PersistenceModule._save
  docstring for the contract. ``_load`` accepts version in {1, 2}
  and falls back to sane defaults for missing keys, so v=1 files
  still restore cleanly. Note: v=2 files cannot be downgraded —
  v3.7.4 and earlier reject them (silent skip with WARNING log;
  no exception, no partial state).
- **Structured persisted-file marker**: ``_truncate_observation``
  appends ``<<<llamagent:persisted:PATH>>>`` (literal token; PATH is
  the relative path to the persisted file) at message-end after the
  human-prose persistence hint. ``CompressionModule._compress_tool_result``
  extracts the marker (regex anchored with ``\\Z``) before its rewrite
  strategies (head / placeholder / llm_summary) and re-appends it
  after, so framework code can recover the persisted-file path even
  when the prose is replaced. Pre-fix the path was lost for
  ``start_job`` / ``web_fetch`` / ``write_files`` outputs once any
  compression strategy ran; only read-tool inputs were partially
  rescued by ``tool_calls.arguments`` introspection.

v3.7.4 highlights (config + lifecycle + DiD pass, second of v3.7.x):
- **Child-agent factory namespaced Config**: the nine hardcoded
  child-tightening defaults that lived in
  ``modules/child_agent/module.py:_create_child_agent`` (compress
  threshold/keep-turns, max observation tokens, max plan adjustments,
  short/continuous react steps + timeout, short/continuous context
  window) now read from named ``Config.child_agent_*`` fields. Defaults
  preserve the prior behavior; YAML (``child_agent.compress_threshold``
  etc.) and direct attribute assignment override per-deployment.
- **JobModule cancel bounded join**: ``JobHandle.cancel`` now
  bound-joins the worker thread after signalling. Previously
  ``JobService.shutdown`` flipped ``_cancelled`` and returned, racing
  with ``ToolsModule`` reverse-shutdown clearing the scratch
  directory while a worker still ran a ``run_command`` mid-IO. The
  join is bounded by ``Config.job_cancel_join_timeout`` (default 5s)
  and the worker is daemon, so a wedged worker can't block shutdown.
  Cancel-vs-result race semantics are unchanged; the deeper question
  is deferred to v3.8.
- **import_scopes trusted/external split**: ``AuthorizationEngine.import_scopes``
  takes a keyword-only ``source="trusted" | "external"``. Trusted
  (default) preserves the historical zero-validation path; external
  validates each dict against the ApprovalScope field whitelist, the
  scope/zone/actions value whitelists matching what the engine
  actually emits (incl. ``"playground"``), and pins ``path_prefixes``
  to the agent's ``write_root`` subtree using the same
  ``os.path.normpath`` + subtree-boundary semantics as the runtime
  matcher (``_path_in_prefixes``). Defense-in-depth for any future
  untrusted JSON entry point — production parent->child inheritance
  and persistence restore are unchanged.

v3.7.3 highlights (round-5 + round-6 audit cleanup, first of v3.7.x series):
- **Dead-code purge**: drop unreachable ``hasattr`` guards
  (``api_retry_count`` in child_agent factory; ``get_all_tool_schemas`` /
  ``set_execution_strategy`` fallbacks in PlanningModule; ``_on_complete``
  hasattr probe in ChildAgentController). All targets are unconditionally
  defined now; the guards were transitional from earlier versions.
- **runner ``record_failure`` helper**: byte-identical TaskRecord
  construction in ``InlineRunnerBackend`` and ``ThreadRunnerBackend``
  except blocks (BudgetExceededError + Exception) lifted into
  ``runners/runner.py`` next to ``build_metrics`` / ``format_fallback_report``.
  Single source of truth for the failure-record shape.
- **Engine-agent boundary public-rename + alias**: ``_switch_policy`` /
  ``_clear_all_scopes`` (on AuthorizationEngine) and ``_ask_confirmation``
  (on LlamAgent) are documented cross-component contracts; rename to
  public names while keeping underscore aliases for backward-compat.
  Aliases get a deprecation warning in v3.7.4-v3.7.7 and are removed in
  v3.8.1.
- **agent_runner subprocess KeyboardInterrupt path**: pre-fix, Ctrl-C
  reaching a child subprocess was caught as ``BaseException`` and
  reported to the parent as ``status='failed' result='execution error:
  KeyboardInterrupt'``. Now mapped to ``status='cancelled'`` with a
  proper user-interrupt fallback report.
- **Reserved HookEvent enum members documented**: ``PLAN_CREATED`` /
  ``STEP_START`` / ``STEP_END`` / ``REPLAN`` are codebase-verified
  zero-emit zero-reference, but kept on the public enum surface so
  future emit-or-remove decisions don't break existing handler
  registrations. v3.8 plan tracks the decision.

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

__version__ = "3.7.8"

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
