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

v3.8.3 highlights (ChildContract-7 closed via additive
``register_hook_factory`` API + v3.8.x trio migration consolidation):

- **`register_hook_factory(event, factory, *, matcher, priority, source)`**:
  new ``LlamAgent`` public API. Unlike ``register_hook`` (per-agent;
  handler binds to THIS agent's state, not inherited by child),
  factories are kept on a separate list and re-invoked at each child
  agent's construction. The factory receives the new agent instance
  and returns a fresh handler bound to THAT agent's state. Solves
  the LLM-spawn case where ``spawn_child`` creates a child dynamically
  — integrators previously had no injection point between agent
  construction and first hook event.

  Sidesteps Python's closure-rebind impossibility (``func.__closure__``
  cell vars are read-only at the public ABI). The factory pattern
  doesn't try to rebind a handler — it produces a fresh handler
  per agent.

- **`unregister_hook_factory(factory) -> int`**: per-agent removal.
  Already-spawned children retain the factory-generated handler in
  their own ``_hooks`` table — unregister is NOT retroactive.

- **`inherit_hook_factories_from(parent)` + child_agent step 4a**:
  ``_create_child_agent`` between step 4 marker and step 5 mode setup
  invokes ``child.inherit_hook_factories_from(parent)``. List-level
  shallow copy of parent's factory registrations + immediate replay
  on the child. Public API call, not private attribute poke (P5
  cleanliness — same hygiene as v3.8.2 P5-2's ``build_isolated_for``).

- **`_HookFactoryRegistration`** dataclass added to ``core/hooks.py``
  (mirrors ``HookRegistration``'s shape but holds factory not handler).

- **`_init_done` sentinel** on ``LlamAgent``: ``True`` at end of
  ``__init__`` after ``_invoke_pending_factories()`` drains any
  pre-init registrations. ``register_hook_factory`` uses it to
  decide between "queue for __init__-end invoke" (pre _init_done)
  vs "invoke immediately" (post _init_done — the common path for
  Module.on_attach-style integrator wiring).

- **Factory raise = log + skip**: a buggy factory cannot break agent
  init or sibling factory registrations. ``_invoke_factory`` wraps
  each call in try/except + warning log with traceback.

- **Stateless contract** (load-bearing, framework-not-enforced):
  factory body must be side-effect-free apart from creating the
  handler. The docstring carries a top-level ⚠️ section. A test in
  the private mock suite (``test_factory_with_state_documents_pollution``)
  intentionally demonstrates the anti-pattern to lock the contract
  on documentation, not on enforcement.

- **Persistence reload**: factories live on agent INSTANCE, not on
  ``Config``. After ``persistence_enabled=True`` reload, the new
  ``LlamAgent(config)`` has empty ``_hook_factories``; integrators
  MUST re-call ``register_hook_factory`` post-reload (same contract
  as ``register_hook``).

- **Trio integration test** (``tests/test_v2_features.py
  ::test_v38x_lifecycle_smoke``): public smoke that locks the
  cross-spec interaction across v3.8 + v3.8.1 + v3.8.2 + v3.8.3 —
  Config.project_dir wiring, SkillModule + MemoryModule attach,
  child_agent.build_isolated_for, register_hook_factory + step 4a
  inheritance, combined shutdown chain.

> Migrating to v3.8.x (consolidated trio note — covers v3.8.1 + v3.8.2
> + v3.8.3 in one place per round-trio audit recommendation):
>
> **API breaks**:
>   - **v3.8.2 E5**: ``Module.on_execute`` removed. Modules that
>     overrode it must register an ``ExecutionStrategy`` via
>     ``agent.set_execution_strategy(...)`` in ``on_attach`` (see
>     PlanningModule for the canonical pattern, or the migrated
>     ``tests/test_integration.py`` LegacyModule example).
>
> **Behavior changes**:
>   - **v3.8.1 R7-#11**: skills without ``pin_packs: true`` lose
>     their packs each turn (was: persist forever).
>   - **v3.8.1 R7-#20**: ``start_job(cwd=<absolute path outside
>     playground/write_root>)`` raises PermissionError (was:
>     silently accepted).
>
> **Import-path moves (warnings only)**:
>   - **v3.8.2 A1**: ``llamagent.modules.tools.snapshot`` is a
>     2-version deprecation shim re-exporting from
>     ``llamagent.core.snapshot``. Update imports; shim removed in
>     v3.10.
>
> **New deps**:
>   - **v3.8.1 R7-#12**: ``python-frontmatter>=1.0`` (PyYAML wrapper,
>     0 transitive new deps).
>
> **New public APIs**:
>   - **v3.8.2 P1-1**: ``LlamAgent.get_active_task_id`` /
>     ``TaskModeController.get_active_task_id`` accessors. If
>     reading ``agent._controller.state.task_id`` directly, switch.
>   - **v3.8.3**: ``LlamAgent.register_hook_factory`` /
>     ``unregister_hook_factory`` /
>     ``inherit_hook_factories_from``.
>
> **Log-warning emit**:
>   - **v3.8.1 R7-#12**: legacy v3.7.x permissive frontmatter files
>     emit ``[skill-migrate]`` log warnings on load. Files still
>     load via fallback parser. Run the ``migrate-skills`` builtin
>     skill (auto-triggers on natural-language queries about
>     "fix / migrate / repair / yaml") to upgrade in place.

v3.8.2 highlights (architecture cleanup — A1 + E5 + P1-1 + P5-2 + P6-4
audit, 7 git commits + 1 private audit doc, ~80 LOC net delete from
twin-factory consolidation):

- **A1 — SnapshotService P5 inversion**: ``modules/tools/snapshot.py``
  relocated to ``core/snapshot.py``. Pre-fix ``core/agent.py`` ran
  ``from llamagent.modules.tools.snapshot import SnapshotService``
  inside ``ensure_snapshot`` (a core→modules reverse import) and
  ``SnapshotService(self)`` held an agent ref. Now SnapshotService
  takes a ``SnapshotConfig`` value object and write_root / playground_dir
  primitives; agent.py constructs the config and passes it down. The
  old import path keeps working as a 2-version DeprecationWarning shim
  (slated for removal in v3.10).

- **E5 — `Module.on_execute` removed (API break)**: Pre-fix
  ``SimpleReAct.execute`` had a fallback loop calling
  ``Module.on_execute`` on every module to support un-migrated modules.
  PlanningModule was the last user; it now exclusively goes through
  ``agent.set_execution_strategy(...)`` (the pattern it has used since
  v3.7). Third-party modules that override ``on_execute`` are now
  silent no-ops — must migrate to ``ExecutionStrategy``. Test
  ``tests/test_integration.py:104+`` LegacyModule migrated to the new
  idiom (subclass ExecutionStrategy + register in on_attach).

- **P1-1 — `Controller.get_active_task_id` getter**: Pre-fix
  ``LlamAgent.get_active_task_id`` read ``self._controller.state.task_id``
  directly — 信使 principle violation (agent inspecting controller
  internals). New ``TaskModeController.get_active_task_id`` public
  method normalizes ``""`` → ``None`` so callers don't have to know
  the reset-quirk. agent.py routes through it.

- **P5-2 — `ToolsModule.build_isolated_for` + `JobModule.build_isolated_for`
  consolidate twin-factory clone**: Pre-fix
  ``modules/child_agent/module.py`` had two private helpers that
  imported ToolsModule / ScratchService / ProjectSyncService /
  ToolRegistry / JobModule / JobService internals (4+ layer-poking
  imports) AND duplicated ~60 LOC of JobModule.on_attach's
  register_tool block as ``_register_isolated_job_tools`` — a v3.5
  twin-factory mistake. Now: the construction logic is on
  ToolsModule / JobModule public class methods; child_agent calls
  them through one public surface each. ``JobModule._register_job_tools_on``
  is the single source of truth for the four job-tool registrations
  (used by both on_attach and build_isolated_for). 60+ LOC clone gone,
  net -113 LOC in child_agent/module.py.

- **P6-4 — `child_agent_*` config field audit (RESOLVED-NO-OP)**:
  v3.7.8 P1-P6 audit estimated "10 child_agent_* fields without
  external consumers". v3.8.2 grep'd all 18 fields against actual
  consumers (recorded in private docs/llamagent-v3.8.2-audit-child-agent-config.md).
  Result: 18/18 fields LIVE — no fields removed. The docs commit is
  the only deliverable for P6-4.

> Migrating to v3.8.2 (集成开发者注意事项):
>   - **API break**: ``Module.on_execute`` removed. Modules that
>     overrode it must register an ``ExecutionStrategy`` via
>     ``agent.set_execution_strategy(...)`` in ``on_attach`` (see
>     PlanningModule for the canonical pattern, or the migrated
>     ``tests/test_integration.py`` LegacyModule example).
>   - **Import path moved (warning only)**:
>     ``llamagent.modules.tools.snapshot`` is a 2-version deprecation
>     shim re-exporting from ``llamagent.core.snapshot``. Update
>     imports; shim removed in v3.10.
>   - ``LlamAgent.get_active_task_id`` is the public API for
>     querying the active task id; if you were reading
>     ``agent._controller.state.task_id`` directly, switch to the
>     accessor (also normalizes the empty-string reset).

v3.8.1 highlights (round-7 static audit cleanup pack — 13 commits, 30
findings closed, 1 builtin skill added):

- **R7-#2 MCP cross-loop fix**: ``MCPClient`` now owns a persistent
  background event loop on a daemon thread. All coroutines (connect,
  call, disconnect) run on that loop via ``run_coroutine_threadsafe``.
  Pre-fix per-call ``asyncio.run`` left sessions on destroyed loops;
  FastAPI / Gradio / Jupyter deployments hit ``RuntimeError: bound
  to a different event loop`` on first MCP tool invocation. New
  ``modules/mcp/_loop.py:PersistentEventLoop`` helper.
- **R7-#4 Web UI per-session state**: ``current_agent`` / ``runner_state``
  closure-globals replaced with ``gr.State`` + lambda factory. Each
  Gradio session gets its own dict instance — multi-user / share-link
  / LAN deployments no longer leak chat history, persona, runner
  state cross-client. 11 callbacks signature + body + ``.click()``
  inputs all carry the new ``session`` parameter.
- **R7-#5 + cross §9.5/§9.6 — API session pool RLock + snapshot-then-
  shutdown-outside-lock**: single ``_session_lock`` (RLock) guards
  ``agent_sessions / rate_limit_store / runner_sessions``. Eviction
  pops inside the lock and runs ``agent.shutdown()`` OUTSIDE so a
  slow on_shutdown (e.g. 1s memory consolidation join) doesn't
  block other concurrent ``_get_agent`` calls.
- **R7-#11 Skill pack revoke + persistence reload reconcile**:
  ``SkillModule`` now tracks per-skill packs in ``_skill_added_packs``
  and revokes packs of skills no longer activated each turn. New
  ``pin_packs: bool = False`` ``SkillMeta`` field for skills that
  legitimately need their packs persistent. ``on_attach`` reconciles
  the tracker from ``agent._active_packs`` after persistence reload.
  Replacement (not setdefault+update) so shrinking pack sets revoke
  correctly. Behavior change: existing user skills without
  ``pin_packs: true`` will see their packs auto-revoked on next
  deactivation (was: persist forever).
- **R7-#12 frontmatter migrated to python-frontmatter (industry
  standard) + builtin migrate-skills skill**: ``fs_store/parser.py``
  drops the 50-line custom line-parser in favor of
  ``python-frontmatter`` (wrapper around ``yaml.safe_load``; same
  library MkDocs/Pelican use). 3-layer architecture: strict parse
  first, lenient legacy fallback on failure, ``[skill-migrate]``
  warning so the new ``migrate-skills`` builtin skill can offer
  to repair files in-place. Hermes-style nested metadata that the
  custom parser couldn't handle now works. Fallback preserved
  indefinitely (no time deadline) so legacy v3.7.x permissive files
  keep loading.
- **R7-#22 SafetyModule logger handler tracking**:
  ``SafetyGuard._setup_logger`` records own handlers in
  ``self._own_handlers``; ``on_shutdown`` removes only those.
  Pre-fix walked ``logger.handlers[:]`` removing every handler on the
  process-shared ``safety_audit`` logger — agent A's shutdown
  silenced agent B's audit log.
- **R7-#23 Memory consolidation background thread + cancel-then-join**:
  ``MemoryModule.on_input`` now spawns a daemon thread for
  ``_consolidate``, with re-entry guard. ``on_shutdown`` does
  cancel-then-join with 1s bound (NOT 5s) so api_server session
  eviction's ``shutdown`` (running OUTSIDE the session lock per
  §9.6) doesn't lock-block other sessions for 5s. ``__deepcopy__``
  returns self — threading primitives aren't pickle-safe.
- **R7-#1/#20/#21 JobModule security pack**: rm-rf regex extended
  (``-rfv``, split flags, long-form ``--recursive --force``);
  absolute ``cwd`` validated against ``playground_dir``/``write_root``
  (raises PermissionError outside boundaries; ``realpath('')``
  fallback edge case fixed in round-4); artifact list capped to
  top-level when cwd == project_dir/write_root + 50-entry cap to
  avoid context blow-up.
- **R7-#3 / #6 / #7 / #8 API hardening**: ClientSession ``async with``
  paired enter/exit; rate limiter reads ``X-Forwarded-For`` for
  reverse-proxy deployments; rate_limit_store periodic prune of
  empty entries; WebSocket ``?session_id=<sid>`` query string for
  per-session agent isolation (industry standard pattern).
- **R7-#13 / #14 / #28 storage**: fs_store parser quote-aware list
  split (``["a, b", "c"]`` no longer breaks); PersonaManager._save
  atomic via tmp+os.replace (mid-write crash no longer loses all
  personas); FSStore filename defense-in-depth realpath-bounded
  to base_dir.
- **R7-#16 / #17 / #18 / #19 RAG correctness**: missing-distance
  sentinel (None instead of falsy 1.0); SQLite shared connection
  user-space lock around index/search/delete/clear/close; vector/
  lexical eventual-consistency contract documented + warning log;
  reranker dedup via ``dict.fromkeys``.
- **NIT pack (R7-#9/#10/#15/#24/#25/#27/#29/#30)**: cooldown floor
  60s; ``Retry-After`` HTTP-date support; prepare_trace_message
  defensive copy; apply_presets uses code-level defaults (bypasses
  user YAML re-load); ContinuousRunner inject TOCTOU lock; YAML
  bool int-coercion; SkillIndex case-insensitive alias collision
  check; MCPClient.max_retries doc clarification.
- **Q6 ChildContract-6**: child agent factory shallow-copies parent's
  ``_persisted_files`` LRU so child read of parent-persisted tool
  result paths don't trigger redundant re-persistence.

> Migrating to v3.8.1 (集成开发者注意事项):
>   - **Behavior change**: skills without ``pin_packs: true`` lose
>     their packs each turn (was: persist forever).
>   - **Behavior change**: ``start_job(cwd=<absolute path outside
>     playground/write_root>)`` raises PermissionError (was:
>     silently accepted).
>   - **New dependency**: ``python-frontmatter>=1.0`` (PyYAML wrapper,
>     0 transitive new deps).
>   - **Warning** emit: legacy v3.7.x permissive frontmatter files
>     emit ``[skill-migrate]`` log warnings on load. Files still
>     load via fallback parser. Run the ``migrate-skills`` builtin
>     skill (auto-triggers on natural-language queries about
>     "fix / migrate / repair / yaml") to upgrade in place.

v3.8 highlights (architectural root-cause fix: ``project_dir`` lifted
to Config — eliminates the entire init-ordering bug class):
- **`Config.project_dir` / `Config.playground_dir` fields**: two new
  None-default fields on ``Config``. ``LlamAgent.__init__`` now reads
  them first, falling back to ``os.getcwd()`` when unset. Setting
  them BEFORE constructing the agent makes ``__init__``-time work
  (auto_approve scope seeding, ``write_root`` derivation, snapshot
  capture) anchor to the right paths from instruction one. Not in
  ``_YAML_MAP`` — these are runtime-determined paths, not user YAML
  configuration.
- **Child-agent factory step 1 wires config**: the dir computation
  that used to live in step 4 (post-construct ``child.project_dir =
  child_root``) is moved up to step 1 (pre-construct
  ``config.project_dir = child_root``). ``LlamAgent(config)`` then
  runs through ``__init__`` on the right paths, so the seeded
  auto_approve scope, computed write_root, and captured snapshot are
  all rooted correctly. Step 4 reduced to a comment marker; step 6's
  isolated-branch ``_seed_auto_approve_scope`` re-call removed
  (redundant — ``__init__`` already seeded with the right path).
- **`agent_runner.py` (process-runner subprocess) wires config first**:
  same pattern. The subprocess now constructs ``Config`` with
  ``project_dir`` set, then ``LlamAgent(config)`` — instead of
  post-construct attribute mutation that pre-v3.8 left the
  subprocess child running its first instructions on the wrong paths.
- **`ApprovalScope.source` typed Literal**: the ``source`` metadata
  field is now ``ScopeSource``, a typed Literal listing all 8 emitter
  sites (``contract``, ``seed``, ``interactive``, ``session_authorize``,
  ``default``, ``trusted``, ``external``, plus the new ``auto_approve``
  emitted by ``_seed_auto_approve_scope``). Audit logs now distinguish
  the CI safety-net seed from contract / seed / import_scopes
  sources. Pre-v3.8 the field was a free-form string with a stale
  comment listing only 3 of the 6 then-emitted values.
- **`init_agent` test fixture migrated**: sets ``config.project_dir
  = tmp_path`` BEFORE ``LlamAgent(config)`` (was post-construct attr
  mutation). ``test_init_agent_runs_full_init`` upgraded from weak
  ``len(scopes) >= 1`` to path-precise: seeded scope's
  ``source == "auto_approve"`` and ``path_prefixes[0]`` equals
  ``tmp_path``. Absorbs v3.8.1's Q7 (demo test tightening).
- **What v3.8 closes (init-ordering bug class)**: every previous
  ``__init__``-time work that depended on ``self.project_dir`` (scope
  seed, ``write_root``, ``_compute_snapshot_session_id``) was, pre-v3.8,
  vulnerable to factory ``post_construct`` overwrite — the same
  pattern that produced the v3.7.7 "stale auto_approve scope" issue
  documented in v3.7.8's ABANDONED plan history. v3.8 closes the
  whole class architecturally instead of patching individual
  symptoms (the rejected "set then clean" approach).
- **`write_root` lazy re-derive branch retained for tests**: the
  ``write_root`` property's lazy re-derive (kicks in when
  ``agent.project_dir`` differs from cached) used to be the only
  defense against stale state. Post-v3.8 production never triggers
  it — the only callers are bypass-init test fixtures that set
  ``project_dir`` after ``LlamAgent.__new__()``. Branch documented
  as test-only contract; do NOT remove.

v3.7.7 highlights — known-issue block updated by v3.8: the previously
listed "stale auto_approve scope in isolated children" issue was an
audit-hygiene concern, not a security breach (correct production
deployments never combine auto_approve+isolated-child + write outside
child_root). v3.8 eliminates the underlying init-ordering bug class
architecturally; v3.7.7 deployments are functionally correct, the
cleanup lands in v3.8.

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

> **Known issue (downgraded by v3.8)**: v3.7.7 with ``auto_approve=True``
> AND ``share_parent_project_dir=False`` had an audit-hygiene issue —
> the isolated child's ``session_scopes`` lists carried both the parent's
> path and the child's, redundant but not a security breach (the
> contract still rejected writes outside ``write_root`` so no parent
> escape was possible). v3.8 eliminates the underlying init-ordering
> bug class architecturally by lifting ``project_dir`` to a ``Config``
> field; v3.7.7 deployments are functionally correct, the cleanup
> lands in v3.8. (The interim v3.7.8 plan that proposed a "set then
> clean" patch was abandoned in favor of the v3.8 root-cause fix.)

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

__version__ = "3.8.3"

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
