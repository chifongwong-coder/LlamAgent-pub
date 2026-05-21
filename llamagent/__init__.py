"""
LlamAgent — A modular AI Agent framework.

Core design:
- core/ provides a standalone base Agent (conversation, LLM calls,
  authorization engine, write-boundary primitives, persistence
  round-trip contracts).
- modules/ provides 14 pluggable Module subclasses (resilience,
  safety, compression, persistence, sandbox, tools, job, retrieval,
  memory, skill, reflection, reasoning, mcp, child_agent) plus two
  shared infrastructure packages (``rag``: vector+lexical+rerank
  pipeline used by retrieval / memory / reflection; ``fs_store``:
  atomic file-backed key/value used by skill / memory / reflection).
  Loading a module is one line; modules are loosely coupled (graceful
  degradation when peers are absent). Toolsmith lives as a pack inside
  the tools module, not a separate module. The original v1.x
  ``multi_agent`` module was deleted in v2.4.1; lightweight delegate
  collaboration is now ``ChildAgentModule._spawn_child(role="delegate")``.
- interfaces/ provides three interaction surfaces (CLI, Web UI, API)
  with shared module presets.

A bare LlamAgent is a fully functional conversational Agent. Each
module loaded grants a new capability.

v3.8.8 highlights (wheel packaging + module-level docs for the last
two undocumented modules — 2 commits):

- **Hero assets moved into the package** (commit 1): the v3.8.4 Web
  UI hero image lived at repo-root ``images/`` so wheel installs
  shipped without it; the UI fell through its ``Path.is_file()``
  guard to a text-only header. Two deferral questions from the
  v3.8.4 plan were answered (yes, wheel users exist; assets are
  self-drawn, no license issue). Moved ``images/`` to
  ``llamagent/assets/images/`` (preserved via ``git mv``), added
  ``[tool.setuptools.package-data] llamagent = ["assets/images/*.png"]``
  to ``pyproject.toml``, and updated the Web UI path resolution
  from ``parent.parent.parent / "images"`` to ``parent.parent /
  "assets" / "images"`` so both editable and wheel installs find
  the PNGs. ``Path.is_file()`` guard kept as defense.

- **Persistence + Resilience module-level docs** (commit 2): the
  two modules without their own ``docs/modules/<name>/`` directory
  are now described in ``docs/llamagent-architecture.md`` §6
  alongside the recommended-load-order diagram. They don't get
  their own subdirectory because (a) their user-facing API is
  100% Config-field driven (no methods to call), (b) their runtime
  behaviour is concentrated enough to fit a single architecture
  subsection, and (c) the gitignored-doc maintenance cost of a
  separate dir isn't justified for that surface area. Compression
  keeps its own subdirectory (added in v3.8.8 prep work) because
  the persistence-marker protocol it shares with ToolsModule is a
  cross-module contract worth isolating.

No behaviour change in the framework code; only the asset relocation
+ docs. Public 202/202 still pass; wheel build verified to include
both PNGs under ``llamagent/assets/images/``.

v3.8.7 highlights (close v3.8.5 backlog — 3 commits, ~120 LOC):

- **P5 cleanup completed** (commit 1):
  ``LlamAgent.import_authorization_scopes(scopes, *, source="trusted")``
  added as the public counterpart to v3.8.5's
  ``export_authorization_scopes``. The inline/thread-runner site at
  ``child_agent/module.py:1370`` migrated from
  ``parent._authorization_engine.export_scopes()`` +
  ``child._authorization_engine.import_scopes(...)`` to the public
  wrappers. After this commit, zero ``_authorization_engine.``
  reads remain in ``llamagent/modules/`` or ``llamagent/interfaces/``
  — the P5 hygiene work begun in v3.8.5 is fully complete.

- **Chroma client + SQLite FTS shutdown** (commit 2):
  ``ChromaVectorBackend.close()`` + ``RetrievalPipeline.close()`` +
  shutdown wiring in ``MemoryModule.on_shutdown`` and new
  ``RetrievalModule.on_shutdown``. Until v3.8.7, the ChromaDB
  persistent client (file handles + memory-mapped HNSW + SQLite
  metadata connection) was only released by GC. In multi-agent
  api_server deployments with LRU eviction this accumulated
  handles between GC cycles. v3.8.7 closes them deterministically
  on agent shutdown. ``hasattr`` guard keeps the change safe across
  the supported chromadb version range (>=0.6.1). Reranker is in
  the close loop too via ``hasattr`` so future cross-encoder
  rerankers with sessions clean up automatically.

- **Highlights trim** (commit 3): v3.7.x → v3.3 release notes
  (about 400 lines) removed from this docstring; pointer to
  ``docs/VERSION_CHANGELOG.md`` retained. The CHANGELOG is the
  authoritative history; only the most recent v3.8.x highlights
  stay inline for at-import-time discoverability.

No behaviour change; no new deps.

v3.8.6 highlights (post-v3.8.5 fresh-eyes audit — 3 commits, log quality
+ honored deprecation promise):

- **Log quality** (commit 1): six sites where exceptions were caught
  and either logged without a stack trace or swallowed entirely.
  Behaviour preserved (fail-closed / fail-soft as appropriate);
  observability restored. Touches confirm_handler exception path,
  ContinuousRunner task-failure log, CLI interactive-setup
  shutdown, agent_runner subprocess child setup, MemoryStore
  access-tracking, and SafetyModule injection log (was double-
  logged in v3.8.5).

- **v3.7.3 underscore-prefixed aliases removed** (commit 2):
  ``LlamAgent._ask_confirmation``,
  ``AuthorizationEngine._switch_policy``, and
  ``AuthorizationEngine._clear_all_scopes`` were renamed in v3.7.3
  and wrapped with DeprecationWarning in v3.7.8, with a stated
  removal target of v3.8.1. v3.8.1 through v3.8.5 shipped without
  enforcing the deadline. v3.8.6 honors it. Zero internal callers;
  3 deprecation-warning tests removed alongside.

  > Migrating to v3.8.6: third-party plugins calling
  > ``agent._ask_confirmation`` / ``engine._switch_policy`` /
  > ``engine._clear_all_scopes`` must rename to the underscore-free
  > public methods. The aliases have emitted DeprecationWarning
  > since v3.7.8.

No new dependencies; no other behaviour changes.

v3.8.5 highlights (post-v3.8.4 audit cleanup — 7 commits, security
hardening + P5 principle wrapper + test code modernization):

- **Safety audit logger → process-lifetime singleton** (closes a real
  v3.8.1 R7-#22 hole). Pre-fix the line-121 ``if not handlers`` gate
  meant only the first SafetyGuard instance owned a handler; when
  that instance's ``on_shutdown`` removed the handler, every still-
  running agent silently lost its audit log. ``safety/guard.py`` now
  installs the handler once per process via ``_ensure_audit_logger``
  and never removes it on instance shutdown; ``logging.shutdown()``
  flushes at process exit. Mismatched ``log_path`` requests get a
  warning. A test-only ``_reset_audit_logger_for_tests`` + conftest
  autouse fixture prevent cross-test pollution.

- **CORS allowlist env-driven + wildcard rejected** (api_server.py).
  Pre-fix ``allow_origins=["*"]`` + ``allow_credentials=True`` was
  the credential-reflection misconfiguration. Now reads
  ``CORS_ALLOW_ORIGINS`` env (comma-separated); default is dev
  localhost on 7860/8000; ``*`` in the env raises ValueError at
  startup. **Behaviour change**: cross-origin prod deployments must
  set the env var (one-line recovery). Bearer-token auth path is
  unaffected (Authorization header isn't a CORS credential).

- **Upload tempfile suffix sanitized** (api_server.py). POST /upload
  fed ``file.filename`` verbatim into ``NamedTemporaryFile(suffix=...)``;
  a filename like ``../var/log/foo`` composed a path outside /tmp.
  Sanitization is layered: ``os.path.basename`` strips POSIX, a
  ``\\``-to-underscore replace catches Windows backslashes (basename
  on Linux doesn't split them), and a control-char regex strips the
  rest. Endpoint requires auth, so threat was "authenticated user
  drops files outside /tmp", not RCE — defence-in-depth nonetheless.

- **`LlamAgent.export_authorization_scopes()` public accessor**
  (P5 wrapper). The child-agent process runner was reaching into
  ``parent._authorization_engine.export_scopes()`` — a single-
  underscore private attribute. New public method mirrors
  ``get_active_task_id``'s style (placed adjacent in the
  Execution-strategy section) and returns ``list[dict]`` (matching
  the engine's ``asdict()`` serialization, used for the process spec
  JSON). Two further inline/thread-runner sites in child_agent
  module retain the private access — out of scope for this hotfix,
  follow-up needed for full P5 alignment.

- **Controller empty-actions filter**. ``controller.py`` indexed
  ``s.actions[0]`` for every normalized scope; today all internal
  callers pass non-empty action lists, but external import_scopes
  could surface empty ones, IndexError'ing mid-contract. Filter at
  line 247 removes empty-actions scopes defensively. No display-side
  fallback (would be dead code after the filter).

- **Test fixture cwd → ``REPO_ROOT``**. Eleven test sites
  (3 conftest fixtures + 8 inline) bound ``agent.project_dir`` to
  ``os.path.realpath(os.getcwd())``, making pytest behavior depend
  on invocation directory. Migrated all 11 to a ``REPO_ROOT``
  constant anchored at the repo root (via ``Path(__file__).resolve``).
  Test-only change; framework code untouched.

- **Test dead-code cleanup**. Five public + twelve internal lines of
  ``agent._in_hook = False`` deleted — v3.7.3 moved the field to a
  class-level ``threading.local``; per-instance assignment has been a
  no-op for five releases. Also removed
  ``tests_internal/test_multi_agent_mock.py.archived`` (residual file
  from v2.4.1 multi_agent removal) and the skipped
  ``TestMultiAgentPack`` class that still imported the deleted
  ``MultiAgentModule``.

> Migrating to v3.8.5 (only one user-visible behaviour change):
>
>   - **CORS_ALLOW_ORIGINS env var**: prod deployments serving
>     cross-origin web clients must now explicitly list allowed
>     origins. Local dev (web_ui port 7860, api_server port 8000)
>     keeps working with default. ``CORS_ALLOW_ORIGINS=*`` is no
>     longer accepted — list specific origins.
>
> All other changes are internal hardening or principle hygiene.

v3.8.4 highlights (post-v3.8.3 audit cleanup + small CLI/Web feature):

- **FSStore defense-in-depth completion**: ``FSStore.delete_file`` and
  ``FSStore.file_exists`` now run the same realpath-bounded
  ``_validate_filename`` check that v3.8.1 R7-#28 added to
  ``write_file`` / ``read_file``. Both methods fail-soft on validation
  failure (delete silently returns, file_exists returns False) — the
  shape mirrors ``read_file``'s None-on-bad-path behaviour. Filenames
  flowing into FSStore come from internal code paths, so practical
  exploitability was low; the fix closes a documentation-vs-behaviour
  gap rather than an exploitable hole.

- **SkillIndex alias-collision diagnostic**: when an alias collides
  with an existing skill name or another already-registered alias,
  the new alias is dropped (first-write-wins, behaviour unchanged) but
  now also emits a ``logger.warning`` naming the offending skill,
  alias, and existing claimant. Side fix: the lowercase skill-name
  set used to be rebuilt inside the per-alias loop; lifted out.

- **api_server session_id length cap**: 5 Pydantic request models
  (ChatRequest, ChatStreamRequest, ModeRequest, RunnerStartRequest,
  InjectRequest) gain ``Field(..., max_length=128)`` on their
  ``session_id``. The ``/ws/chat`` WebSocket endpoint enforces the
  same 128-char ceiling manually because query-string params bypass
  Pydantic; over-length closes the socket with code 1008 (policy
  violation). 128 chars fits UUID + custom prefix; no charset pattern
  is enforced because i18n / business IDs may use non-ASCII.

  **Behaviour change**: HTTP requests with session_id over 128 chars
  now get 422; WebSocket gets a 1008 close. Real-world break is
  minimal (default ``"default"`` and UUIDs all fit), but document
  this if you wrap the API server.

- **CLI ASCII llama banner**: ``llamagent.interfaces.cli.BANNER`` now
  shows a chibi llama mascot holding a wrench (left) and a
  screwdriver (right) — same character as the project logo at
  ``images/llamagent.png``. Three style variants are checked in
  (active + two commented alternatives). The BANNER is built via
  string concatenation rather than ``.format()`` so any literal ``{``
  ``}`` in the art can't trigger KeyError at import time.

- **Web UI hero image**: ``create_web_ui`` shows the project logo
  (``images/llamagent.png``) next to the page title in a two-column
  ``gr.Row``. Falls back to text-only title when the asset is missing
  (e.g. wheel installs that don't ship ``images/``) via
  ``Path.is_file()`` guard, so the UI never breaks on missing art.
  Wheel packaging of the asset is intentionally deferred to a future
  release.

- **docs/modules/tools/api.md ScratchService rewrite**: the v3.4
  ``WorkspaceService`` → ``ScratchService`` rename and the v3.3
  removal of ``project:`` path prefix were never reflected in the
  module API doc; v3.8.4 brings the doc in line with current code.
  Also flags ``WORKSPACE_GUIDE`` → ``FILE_TOOL_GUIDE`` and the
  removed ``sync_workspace_to_project`` tool. Private doc, gitignored.

> Migrating to v3.8.4: only the ``session_id`` length cap is a public
> behaviour change. Everything else is internal hardening, diagnostic,
> or visual polish. No new dependencies.

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

For release notes prior to v3.8 (v3.7.x, v3.6, v3.5, v3.4, v3.3,
and earlier), see ``docs/VERSION_CHANGELOG.md``. The CHANGELOG is
the authoritative history; this docstring mirrors only the most
recent v3.8.x highlights (kept inline for at-import-time
discoverability).

Usage:
    from llamagent import LlamAgent, Config, Module
    agent = LlamAgent(Config())
    reply = agent.chat("Hello")
"""

__version__ = "3.9.0"

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
