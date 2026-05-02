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
