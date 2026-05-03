"""
Public tests for the child agent control module.

Flow-oriented tests covering integration, budget enforcement, tool filtering,
task lifecycle, safety inheritance, and backward compatibility.

Mock strategy: only mock litellm.completion(); all framework-internal methods run real logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llamagent.core.agent import LlamAgent, Module
from llamagent.modules.child_agent.budget import (
    Budget,
    BudgetedLLM,
    BudgetExceededError,
    BudgetTracker,
)
from llamagent.modules.child_agent.policy import (
    AgentExecutionPolicy,
    ChildAgentSpec,
    ROLE_POLICIES,
)
from llamagent.modules.child_agent.module import ChildAgentModule

from conftest import make_llm_response


class TestModuleIntegrationAndBudget:
    """Module integration (register, spawn, list, collect) + budget enforcement + budgeted LLM."""

    def test_module_integration_and_budget(self, bare_agent, mock_llm_client):
        """Full integration: register module -> spawn child -> list -> collect.
        Then budget enforcement: zero-budget child, direct BudgetExceededError,
        and usage recording."""
        # --- Module integration ---
        mock_llm_client.set_responses([
            make_llm_response("research findings: AI is great"),
        ])

        module = ChildAgentModule()
        bare_agent.register_module(module)

        # Verify tools are registered
        assert "spawn_child" in bare_agent._tools
        assert "list_children" in bare_agent._tools
        assert "collect_results" in bare_agent._tools

        # Spawn a child via the tool
        result = bare_agent.call_tool("spawn_child", {
            "task": "research AI trends",
            "role": "worker",
        })
        assert isinstance(result, str)
        assert len(result) > 0

        # List children shows the spawned child
        list_result = bare_agent.call_tool("list_children", {})
        assert "worker" in list_result
        assert "research AI" in list_result

        # Collect results returns the completed result
        collect_result = bare_agent.call_tool("collect_results", {})
        assert "worker" in collect_result

        # --- Budget enforcement: zero-budget child ---
        policy = AgentExecutionPolicy(
            budget=Budget(max_llm_calls=0),
            can_spawn_children=False,
        )
        spec = ChildAgentSpec(
            task="impossible task",
            role="worker",
            policy=policy,
            parent_task_id=module._parent_id,
        )
        task_id = module.controller.spawn_child(spec, module._create_child_agent)
        record = module.controller.wait_child(task_id)
        # v3.5 + v3.5.2: BudgetExceededError propagates from BudgetedLLM
        # through agent.chat() (scoped exception policy lets framework
        # signaling exceptions through) to the runner's outer
        # ``except BudgetExceededError`` clause, which produces a v3.5-shape
        # fallback report. record.status flips to "failed" and result
        # carries the structured Status/Summary/Artifacts shape.
        assert record.status == "failed"
        assert "Status: failed" in record.result
        assert "budget exceeded" in record.result.lower()

        # --- BudgetedLLM raises directly when budget exhausted ---
        mock_llm = MagicMock()
        mock_llm.model = "mock"
        tracker = BudgetTracker(Budget(max_llm_calls=1))
        budgeted = BudgetedLLM(mock_llm, tracker)
        tracker.llm_calls = 1  # Already at limit
        with pytest.raises(BudgetExceededError, match="LLM call budget exceeded"):
            budgeted.chat([{"role": "user", "content": "hi"}])

        # --- BudgetedLLM records usage ---
        mock_llm2 = MagicMock()
        mock_llm2.model = "mock"
        mock_llm2.ask.return_value = "short answer"
        tracker2 = BudgetTracker(Budget(max_llm_calls=5))
        budgeted2 = BudgetedLLM(mock_llm2, tracker2)
        budgeted2.ask("question 1")
        budgeted2.ask("question 2")
        assert tracker2.llm_calls == 2
        assert tracker2.tokens_used > 0


class TestRolePoliciesAndTaskBoard:
    """Role-based tool filtering (coder, researcher) + task board lifecycle (spawn, list, collect)."""

    def test_role_policies_and_task_board(self, bare_agent, mock_llm_client):
        """Coder role gets only allowed tools; researcher role gets web-oriented tools.
        Then spawn multiple children, list them, and collect completed results."""
        # --- Coder role filtering ---
        bare_agent.register_tool("read_files", lambda paths: "data", "Read files")
        bare_agent.register_tool("write_files", lambda files: "ok", "Write files")
        bare_agent.register_tool("apply_patch", lambda t, e: "patched", "Patch")
        bare_agent.register_tool("start_job", lambda cmd: "out", "Job")
        bare_agent.register_tool("glob_files", lambda p: "files", "Glob")
        bare_agent.register_tool("search_text", lambda q: "found", "Search text")
        bare_agent.register_tool("web_search", lambda q: "results", "Search")
        bare_agent.register_tool("web_fetch", lambda u: "page", "Fetch")
        bare_agent.register_tool("search_knowledge", lambda q: "kb", "KB")
        bare_agent.register_tool("delete_database", lambda: "gone", "Delete")

        module = ChildAgentModule()
        bare_agent.register_module(module)

        # Coder allowlist
        coder_policy = ROLE_POLICIES["coder"]
        coder_spec = ChildAgentSpec(task="write code", role="coder", policy=coder_policy)
        coder_child = module._create_child_agent(coder_spec)

        assert "read_files" in coder_child._tools
        assert "write_files" in coder_child._tools
        assert "apply_patch" in coder_child._tools
        assert "start_job" in coder_child._tools
        assert "web_search" not in coder_child._tools
        assert "delete_database" not in coder_child._tools
        assert "spawn_child" not in coder_child._tools

        # --- Researcher role filtering ---
        researcher_policy = ROLE_POLICIES["researcher"]
        researcher_spec = ChildAgentSpec(task="research", role="researcher", policy=researcher_policy)
        researcher_child = module._create_child_agent(researcher_spec)

        assert "web_search" in researcher_child._tools
        assert "web_fetch" in researcher_child._tools
        assert "search_knowledge" in researcher_child._tools
        assert "search_text" in researcher_child._tools
        assert "read_files" in researcher_child._tools
        assert "start_job" not in researcher_child._tools

        # --- Task board lifecycle: spawn multiple, list, collect ---
        mock_llm_client.set_responses([
            make_llm_response("result from child 1"),
            make_llm_response("result from child 2"),
        ])

        result1 = module._spawn_child(task="task alpha", role="worker")
        result2 = module._spawn_child(task="task beta", role="analyst")

        listing = module._list_children()
        assert "worker" in listing
        assert "analyst" in listing
        assert "task alpha" in listing
        assert "task beta" in listing

        collected = module._collect_results()
        assert "worker" in collected or "analyst" in collected
        assert len(collected) > 0


class TestZoneInheritanceAndBackwardCompat:
    """Child inherits parent zone settings + backward compat without module."""

    def test_zone_inheritance_and_backward_compat(self, bare_agent, mock_llm_client):
        """Child inherits project_dir, playground_dir, confirm_handler, mode, tool_executor.
        Without ChildAgentModule, agent works normally as v1.1."""
        from llamagent.core.zone import ConfirmRequest, ConfirmResponse

        # --- playground_dir and project_dir inheritance ---
        bare_agent.project_dir = "/custom/project"
        bare_agent.playground_dir = "/custom/project/llama_playground"

        module = ChildAgentModule()
        bare_agent.register_module(module)

        spec = ChildAgentSpec(task="test task", role="worker")
        child = module._create_child_agent(spec)
        assert child.project_dir == "/custom/project"
        assert child.playground_dir == "/custom/project/llama_playground"

        # --- confirm_handler and mode inheritance ---
        handler = lambda req: ConfirmResponse(allow=True)
        bare_agent.confirm_handler = handler
        bare_agent.mode = "interactive"

        child2 = module._create_child_agent(spec)
        assert child2.confirm_handler is handler
        assert child2.mode == "interactive"

        # If parent has no handler, child also has none
        bare_agent.confirm_handler = None
        child_no_handler = module._create_child_agent(spec)
        assert child_no_handler.confirm_handler is None

        # --- tool_executor inheritance ---
        mock_executor = MagicMock()
        bare_agent.tool_executor = mock_executor
        spec_coder = ChildAgentSpec(task="sandbox task", role="coder")
        child_executor = module._create_child_agent(spec_coder)
        assert child_executor.tool_executor is mock_executor

        # --- Backward compat: no ChildAgentModule ---
        # Create a fresh agent without the module
        bare_agent.tool_executor = None  # reset
        bare_agent2 = bare_agent  # reuse; remove module reference for test
        # A separate bare_agent would be ideal but we can check the properties
        # before module was loaded were valid
        mock_llm_client.set_responses([
            make_llm_response("hello, I am your assistant"),
        ])

        # Create a truly bare agent for backward compat test
        from llamagent.core.config import Config
        from llamagent.core.agent import SimpleReAct
        from llamagent.core.authorization import AuthorizationEngine
        import os

        config2 = Config.__new__(Config)
        for attr in vars(bare_agent.config):
            setattr(config2, attr, getattr(bare_agent.config, attr))

        agent2 = LlamAgent.__new__(LlamAgent)
        agent2.config = config2
        agent2.persona = None
        agent2.llm = mock_llm_client
        agent2._llm_cache = {config2.model: mock_llm_client}
        agent2.modules = {}
        agent2.history = []
        agent2.summary = None
        agent2.conversation = agent2.history
        agent2._execution_strategy = SimpleReAct()
        agent2.confirm_handler = None
        agent2.interaction_handler = None
        agent2._confirm_wait_time = 0.0
        agent2.project_dir = os.path.realpath(os.getcwd())
        agent2.playground_dir = os.path.realpath(os.path.join(agent2.project_dir, "llama_playground"))
        agent2.tool_executor = None
        agent2._tools = {}
        agent2._active_packs = set()
        agent2._tools_version = 0
        agent2._hooks = {}
        agent2._session_started = False
        agent2._in_hook = False
        agent2.mode = "interactive"
        agent2._controller = None
        agent2._current_task_id = None
        agent2._abort = False
        agent2._open_questions_buffer = []
        agent2._interactive_config = {k: getattr(agent2.config, k) for k in LlamAgent._MODE_KEYS}
        agent2._authorization_engine = AuthorizationEngine(agent2)

        assert not agent2.has_module("child_agent")
        assert "spawn_child" not in agent2._tools

        response = agent2.chat("hi")
        assert isinstance(response, str)
        assert len(response) > 0

        agent2.register_tool("greet", lambda name: f"hello {name}", "Greet")
        assert "greet" in agent2._tools
        assert agent2.has_module("child_agent") is False


class TestSecurityFixes:
    """Security fixes: deep copy isolation, max children limit, permission enforcement, cleanup."""

    def test_security_fixes(self, bare_agent, mock_llm_client):
        """Child tool mutation does not bleed to parent; max_children enforced;
        child inherits zone settings; runner results cleaned after sync."""
        # --- Deep copy isolation ---
        mock_llm_client.set_responses([make_llm_response("done")] * 10)
        bare_agent.register_tool(
            "shared_tool", lambda: "ok", "test",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        )

        module = ChildAgentModule()
        bare_agent.register_module(module)

        spec = ChildAgentSpec(task="test")
        child = module._create_child_agent(spec)
        child._tools["shared_tool"]["parameters"]["properties"]["injected"] = {"type": "int"}
        assert "injected" not in bare_agent._tools["shared_tool"]["parameters"]["properties"]

        # --- Max children limit ---
        module.controller.max_children = 2
        r1 = module._spawn_child(task="t1")
        r2 = module._spawn_child(task="t2")
        assert "Cannot spawn" not in r1
        assert "Cannot spawn" not in r2
        r3 = module._spawn_child(task="overflow")
        assert "Max children limit" in r3

        # --- Child inherits zone settings (permission level via playground_dir) ---
        bare_agent.register_tool("dangerous", lambda: "boom", "high risk", safety_level=3)
        spec2 = ChildAgentSpec(task="test")
        child2 = module._create_child_agent(spec2)
        assert child2.playground_dir == bare_agent.playground_dir

        # --- Task board records completed results ---
        # Reset module for clean state
        module2 = ChildAgentModule()
        bare_agent.modules.pop("child_agent", None)
        bare_agent.register_module(module2)
        mock_llm_client.set_responses([make_llm_response("done")])
        module2._spawn_child(task="test")
        assert len(module2.controller.list_children(module2._parent_id)) == 1


class TestShareableModulesV37:
    """v3.7: parent-child memory sharing via share_parent_modules.

    Children spawned with ``share_parent_modules=["memory"]`` re-register a
    fresh MemoryModule on themselves and swap the parent's data-layer
    handles in via ``inherit_storage_from``. The framework forces
    read-only on the child (``memory_mode="off"``), so write tools
    (``save_memory`` / ``consolidate_memory``) are never registered.
    """

    def test_share_parent_memory_child_recalls_parent_save(
        self, bare_agent, mock_llm_client, tmp_path
    ):
        """Parent saves a fact via FS backend; child spawned with
        share_parent_modules=["memory"] reads the same fact through
        its inherited store handle."""
        from llamagent.modules.memory.module import MemoryModule
        from llamagent.modules.memory.fact import MemoryFact

        # Parent runs FS backend with a tmp dir + a fixed persona so
        # the test is hermetic.
        bare_agent.config.memory_backend = "fs"
        bare_agent.config.memory_mode = "autonomous"
        bare_agent.config.memory_recall_mode = "tool"
        bare_agent.config.memory_fs_dir = str(tmp_path / "memory")
        bare_agent.persona = type("P", (), {"persona_id": "alice"})()

        parent_mem = MemoryModule()
        bare_agent.register_module(parent_mem)
        assert parent_mem.store is not None
        assert parent_mem.shareable is True

        fact = MemoryFact(
            fact_id="f-001",
            kind="preference",
            subject="user",
            attribute="favorite_drink",
            value="orange juice",
            source_text="user said: i like orange juice",
        )
        parent_mem.store.save_fact(fact)

        # Spawn share=True child via the factory directly (not the
        # spawn_child tool, which adds LLM mock noise).
        module = ChildAgentModule()
        bare_agent.register_module(module)

        policy = AgentExecutionPolicy(share_parent_modules=["memory"])
        spec = ChildAgentSpec(task="recall", role="worker", policy=policy)
        child = module._create_child_agent(spec)

        # Child has its own MemoryModule instance, but the store is
        # physically the parent's.
        assert "memory" in child.modules
        assert child.modules["memory"] is not parent_mem
        assert child.modules["memory"].store is parent_mem.store

        # Child's _tools contains the read tool but NOT the write tools.
        # Read-only contract is enforced at on_attach, not denylist.
        assert "recall_memory" in child._tools or "list_memories" in child._tools
        assert "save_memory" not in child._tools
        assert "consolidate_memory" not in child._tools

        # Child reads the fact through its inherited store.
        all_facts = child.modules["memory"].store.list_all_active_facts()
        assert any(f.get("value") == "orange juice" for f in all_facts)

    def test_share_parent_memory_disabled_by_default(
        self, bare_agent, mock_llm_client, tmp_path
    ):
        """Without share_parent_modules, child has no memory tools at all
        (preserves pre-v3.7 contract that the legacy memory_mode="off"
        hardcode at the factory enforced)."""
        from llamagent.modules.memory.module import MemoryModule

        bare_agent.config.memory_backend = "fs"
        bare_agent.config.memory_mode = "autonomous"
        bare_agent.config.memory_recall_mode = "tool"
        bare_agent.config.memory_fs_dir = str(tmp_path / "memory")

        bare_agent.register_module(MemoryModule())

        module = ChildAgentModule()
        bare_agent.register_module(module)

        # No policy override -> share_parent_modules defaults to None
        spec = ChildAgentSpec(task="any", role="worker")
        child = module._create_child_agent(spec)

        # Child has no MemoryModule registered at all. The helper's
        # default branch only forces memory_mode/recall_mode to "off"
        # on the child config — it doesn't register the module.
        assert "memory" not in child.modules
        # And no memory tools landed in the child's tool table.
        # v3.7.1: iterate via class attrs so a future tool addition
        # is automatically covered (the helper's clear list + this
        # assertion both pull from MemoryModule._WRITE_TOOL_NAMES /
        # _READ_TOOL_NAMES).
        from llamagent.modules.memory.module import MemoryModule
        for write_tool in MemoryModule._WRITE_TOOL_NAMES:
            assert write_tool not in child._tools
        for read_tool in MemoryModule._READ_TOOL_NAMES:
            assert read_tool not in child._tools

    def test_share_parent_modules_parent_missing_raises(
        self, bare_agent, mock_llm_client
    ):
        """share_parent_modules=["memory"] on a parent that has no
        MemoryModule loaded raises ValueError at factory time
        (loud failure, typically a typo or wrong load order)."""
        module = ChildAgentModule()
        bare_agent.register_module(module)

        policy = AgentExecutionPolicy(share_parent_modules=["memory"])
        spec = ChildAgentSpec(task="any", role="worker", policy=policy)

        with pytest.raises(ValueError, match="parent has no such module"):
            module._create_child_agent(spec)

    def test_share_does_not_replace_child_llm_or_agent(
        self, bare_agent, mock_llm_client, tmp_path
    ):
        """v3.7.1: pin the original BLOCKER from plan §0.1 along the
        PRODUCTION wire-order.

        v3.7 commit-9 added this test but called ``_create_child_agent``
        directly with an empty ``runlog_path`` — that bypassed the
        production path's ``_attach_runlog`` step which wraps
        ``child.llm`` with ``LoggingLLM``. After that wrap,
        ``child_mem.llm`` (BudgetedLLM, set before wrap) is no longer
        ``is`` ``child.llm`` (LoggingLLM(BudgetedLLM)) -- so the
        commit-9 ``is`` assertion was passing only on a non-production
        code path.

        v3.7.1 fix: set ``spec.runlog_path`` explicitly so
        ``_attach_runlog`` runs. Use ``.tracker`` identity (which
        proxies through ``LoggingLLM.__getattr__``) instead of LLM
        identity -- the BudgetTracker is the actual invariant to pin.
        """
        from llamagent.modules.memory.module import MemoryModule
        from llamagent.core.logging_llm import LoggingLLM

        bare_agent.config.memory_backend = "fs"
        bare_agent.config.memory_mode = "autonomous"
        bare_agent.config.memory_recall_mode = "tool"
        bare_agent.config.memory_fs_dir = str(tmp_path / "memory")
        parent_mem = MemoryModule()
        bare_agent.register_module(parent_mem)
        # Sanity: parent module's LLM is parent's LLM.
        assert parent_mem.llm is bare_agent.llm

        module = ChildAgentModule()
        bare_agent.register_module(module)

        # Budget on policy -> factory wraps child.llm in BudgetedLLM.
        policy = AgentExecutionPolicy(
            share_parent_modules=["memory"],
            budget=Budget(max_llm_calls=5),
        )
        # v3.7.1: set runlog_path so _attach_runlog runs (production
        # path always sets this in _spawn_child via _runlog_path_for).
        spec = ChildAgentSpec(
            task="recall", role="worker", policy=policy,
            runlog_path=str(tmp_path / "child.log.jsonl"),
        )
        child = module._create_child_agent(spec)
        child_mem = child.modules["memory"]

        # Fresh module instance on the child.
        assert child_mem is not parent_mem
        # Production wire-order verified: _attach_runlog ran AFTER
        # _apply_shared_modules, so child.llm is the LoggingLLM wrap.
        assert isinstance(child.llm, LoggingLLM), (
            "child.llm should be LoggingLLM-wrapped in production path"
        )
        # The MemoryModule's llm was bound BEFORE the LoggingLLM wrap
        # (during register_module, which runs inside the helper),
        # so it stays at the BudgetedLLM layer.
        assert isinstance(child_mem.llm, BudgetedLLM)
        # The actual budget invariant: same BudgetTracker drives both
        # the memory module's calls and the child's main-loop calls.
        # tracker resolves through LoggingLLM.__getattr__ -> BudgetedLLM.
        assert child_mem.llm.tracker is child.llm.tracker
        # Neither path points at parent's plain LLM (the BLOCKER).
        assert child_mem.llm is not parent_mem.llm
        assert child_mem.llm is not bare_agent.llm
        # Agent identity: child's memory module points at child, not parent.
        assert child_mem.agent is child
        # Data layer IS shared (the v3.7 contract).
        assert child_mem.store is parent_mem.store

    def test_spawn_child_pre_validates_share_parent_modules(
        self, bare_agent, mock_llm_client
    ):
        """v3.7.1: ``_spawn_child`` validates ``share_parent_modules``
        BEFORE ``controller.spawn_child``. Without the pre-check, the
        runners' ``except Exception`` would wrap the helper's
        ValueError into TaskRecord(status="failed") and the model would
        see a misleading "Spawned child agent.\\n- task_id: ...\\nResult:
        ...execution error: ValueError..." (a false success header).

        With the pre-check, the model sees a clean
        ``Cannot spawn child agent: <reason>`` string."""
        module = ChildAgentModule()
        bare_agent.register_module(module)

        # Parent has no MemoryModule loaded; share=["memory"] in policy.
        out = module._spawn_child(
            task="recall",
            role="worker",
            context="",
        )
        # Sanity: default path with no policy works (no share check
        # triggers).
        assert "Cannot spawn" not in out

        # Now exercise the failing path through the real spawn tool.
        # We call _spawn_child via Python (the model's call path), but
        # supply a custom policy by injecting a ROLE_POLICIES override.
        # v3.7.1 commit-16: also snapshot the task_board so we can
        # assert no record was created -- the string-only check below
        # would still pass if a future regression ran controller.
        # spawn_child and then post-formatted the runner's swallowed
        # error.
        n_before = len(module.controller.list_children(module._parent_id))
        policy = AgentExecutionPolicy(share_parent_modules=["memory"])
        ROLE_POLICIES["v371_share_test"] = policy
        try:
            out = module._spawn_child(
                task="recall",
                role="v371_share_test",
            )
        finally:
            ROLE_POLICIES.pop("v371_share_test", None)

        # Pre-check produced the clean error string, NOT the runner-
        # swallowed false-success header.
        assert "Cannot spawn child agent:" in out
        assert "parent has no such module" in out
        assert "task_id:" not in out  # NOT a spawn success
        # State invariant: pre-check returned BEFORE controller.spawn_child,
        # so the task_board got no new entry from the failing call.
        assert len(module.controller.list_children(module._parent_id)) == n_before

    def test_spawn_child_refuses_process_runner_with_share(
        self, bare_agent, mock_llm_client, tmp_path
    ):
        """v3.7.1: process runner cannot alias an in-process store
        handle. Spawn tool returns a clean error instead of silently
        ignoring the share intent (the pre-v3.7.1 behavior)."""
        from llamagent.modules.memory.module import MemoryModule

        bare_agent.config.memory_backend = "fs"
        bare_agent.config.memory_mode = "autonomous"
        bare_agent.config.memory_recall_mode = "tool"
        bare_agent.config.memory_fs_dir = str(tmp_path / "memory")
        bare_agent.register_module(MemoryModule())

        module = ChildAgentModule()
        bare_agent.register_module(module)
        # Force the runner_name to "process" — the spawn tool checks
        # this directly. The runner backend itself isn't exercised
        # because we never reach controller.spawn_child.
        module._runner_name = "process"

        policy = AgentExecutionPolicy(share_parent_modules=["memory"])
        ROLE_POLICIES["v371_proc_test"] = policy
        try:
            out = module._spawn_child(
                task="recall",
                role="v371_proc_test",
            )
        finally:
            ROLE_POLICIES.pop("v371_proc_test", None)

        assert "Cannot spawn child agent:" in out
        assert "not supported on the process runner" in out
        assert "task_id:" not in out

    def test_continuous_factory_share_inherits_storage(
        self, bare_agent, mock_llm_client, tmp_path
    ):
        """v3.7 commit-9: cover the second factory call site.

        All other share tests exercise _create_short_child_agent. The
        continuous factory at _create_continuous_child_agent invokes
        the same _apply_shared_modules helper but with different
        ordering (set_mode runs BEFORE the helper). Pin the contract
        explicitly so a future set_mode change can't silently clobber
        the helper's memory_mode/recall_mode writes.

        v3.7.1 commit-16: pin BudgetedLLM-tracker invariant under the
        production wire-order (Budget on policy + runlog_path set).
        Mirrors the fix commit-13 applied to
        ``test_share_does_not_replace_child_llm_or_agent``. Without
        this, the prior version's ``child.modules["memory"].llm is
        child.llm`` assertion passed only because ``runlog_path=""``
        skipped ``_attach_runlog`` and ``budget=None`` skipped
        ``BudgetedLLM`` -- both wraps were absent so identity held by
        coincidence on a non-production code path."""
        from llamagent.modules.memory.module import MemoryModule
        from llamagent.core.logging_llm import LoggingLLM

        bare_agent.config.memory_backend = "fs"
        bare_agent.config.memory_mode = "autonomous"
        bare_agent.config.memory_recall_mode = "tool"
        bare_agent.config.memory_fs_dir = str(tmp_path / "memory")
        parent_mem = MemoryModule()
        bare_agent.register_module(parent_mem)

        module = ChildAgentModule()
        bare_agent.register_module(module)

        # Budget on policy -> factory wraps child.llm in BudgetedLLM.
        # runlog_path set -> factory then wraps in LoggingLLM(BudgetedLLM).
        # Production _spawn_continuous_child always sets runlog_path
        # via _runlog_path_for(spec.task_id), so this matches the
        # real wire-order.
        policy = AgentExecutionPolicy(
            share_parent_modules=["memory"],
            budget=Budget(max_llm_calls=5),
        )
        spec = ChildAgentSpec(
            task="watch",
            role="watcher",
            policy=policy,
            continuous=True,
            trigger_type="timer",
            trigger_interval=60,
            runlog_path=str(tmp_path / "cont.log.jsonl"),
        )
        # _create_child_agent dispatches to _create_continuous_child_agent
        # when spec.continuous is True.
        child = module._create_child_agent(spec)

        # Same invariants as the short-factory share test.
        assert "memory" in child.modules
        assert child.modules["memory"] is not parent_mem
        assert child.modules["memory"].store is parent_mem.store
        assert "save_memory" not in child._tools
        assert "consolidate_memory" not in child._tools
        # Helper runs AFTER set_mode("continuous"); the memory_mode
        # override survives that ordering.
        assert child.config.memory_mode == "off"
        assert child.modules["memory"].agent is child
        # Production wire-order: child.llm is LoggingLLM(BudgetedLLM(...))
        # while child_mem.llm is BudgetedLLM (set during register_module
        # inside _apply_shared_modules, BEFORE _attach_runlog wraps).
        # The actual budget invariant: same BudgetTracker drives both
        # paths. tracker resolves via LoggingLLM.__getattr__ proxy.
        assert isinstance(child.llm, LoggingLLM)
        assert isinstance(child.modules["memory"].llm, BudgetedLLM)
        assert child.modules["memory"].llm.tracker is child.llm.tracker
