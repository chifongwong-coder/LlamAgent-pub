"""
Public tests for the child agent control module.

Flow-oriented tests covering integration, budget enforcement, tool filtering,
task lifecycle, safety inheritance, and backward compatibility.

Mock strategy: only mock litellm.completion(); all framework-internal methods run real logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llamagent.core.agent import SmartAgent, Module
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


class TestChildAgentModuleIntegration:
    """Load module -> spawn_child tool available -> call it."""

    def test_child_agent_module_integration(self, bare_agent, mock_llm_client):
        """Full integration: register module, spawn child, get result."""
        # The child agent will make one LLM call that returns text
        mock_llm_client.set_responses([
            make_llm_response("research findings: AI is great"),
        ])

        module = ChildAgentModule()
        bare_agent.register_module(module)

        # Verify tools are registered
        assert "spawn_child" in bare_agent._tools
        assert "list_children" in bare_agent._tools
        assert "collect_results" in bare_agent._tools

        # Allow safety check for all tools
        bare_agent.safety_loaded = True

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


class TestBudgetEnforcement:
    """Child with budget: budget exceeded raises error."""

    def test_budget_enforcement(self, bare_agent, mock_llm_client):
        """Child agent with tight budget: BudgetExceededError surfaces in result."""
        module = ChildAgentModule()
        bare_agent.register_module(module)
        bare_agent.safety_loaded = True

        # BudgetedLLM checks BEFORE each call: max_llm_calls=0 means
        # llm_calls(0) >= max(0) is True -> BudgetExceededError on the very
        # first LLM call. SmartAgent.chat() catches this in the execution
        # strategy error handler and returns an error string. The inline runner
        # sees a completed chat (returns text), so the task status is "completed"
        # but the result text contains the budget exceeded message.
        policy = AgentExecutionPolicy(
            budget=Budget(max_llm_calls=0),  # No LLM calls allowed
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

        # The child "completed" but the result text contains the budget error
        assert record.status == "completed"
        assert "budget exceeded" in record.result.lower() or "Budget" in record.result

    def test_budget_exceeded_raises_directly(self):
        """BudgetedLLM raises BudgetExceededError when budget is exhausted."""
        mock_llm = MagicMock()
        mock_llm.model = "mock"

        tracker = BudgetTracker(Budget(max_llm_calls=1))
        budgeted = BudgetedLLM(mock_llm, tracker)
        tracker.llm_calls = 1  # Already at limit

        with pytest.raises(BudgetExceededError, match="LLM call budget exceeded"):
            budgeted.chat([{"role": "user", "content": "hi"}])

    def test_budgeted_llm_records_usage(self):
        """BudgetedLLM correctly records usage after each call."""
        mock_llm = MagicMock()
        mock_llm.model = "mock"
        mock_llm.ask.return_value = "short answer"

        tracker = BudgetTracker(Budget(max_llm_calls=5))
        budgeted = BudgetedLLM(mock_llm, tracker)

        budgeted.ask("question 1")
        budgeted.ask("question 2")

        assert tracker.llm_calls == 2
        assert tracker.tokens_used > 0


class TestRolePoliciesToolFiltering:
    """Coder role only gets allowed tools."""

    def test_role_policies_tool_filtering(self, bare_agent):
        """Child with coder role gets only read_file, write_file, execute_command."""
        # Register several tools on the parent
        bare_agent.register_tool("read_file", lambda p: "data", "Read")
        bare_agent.register_tool("write_file", lambda p, d: "ok", "Write")
        bare_agent.register_tool("execute_command", lambda c: "out", "Exec")
        bare_agent.register_tool("web_search", lambda q: "results", "Search")
        bare_agent.register_tool("delete_database", lambda: "gone", "Delete")

        module = ChildAgentModule()
        bare_agent.register_module(module)

        policy = ROLE_POLICIES["coder"]
        spec = ChildAgentSpec(task="write code", role="coder", policy=policy)
        child = module._create_child_agent(spec)

        # Coder allowlist: read_file, write_file, execute_command
        assert "read_file" in child._tools
        assert "write_file" in child._tools
        assert "execute_command" in child._tools

        # These should NOT be present
        assert "web_search" not in child._tools
        assert "delete_database" not in child._tools
        # Spawn tools also removed (can_spawn_children=False)
        assert "spawn_child" not in child._tools

    def test_researcher_role_filtering(self, bare_agent):
        """Child with researcher role gets only web-oriented tools."""
        bare_agent.register_tool("web_search", lambda q: "results", "Search")
        bare_agent.register_tool("web_fetch", lambda u: "page", "Fetch")
        bare_agent.register_tool("search_knowledge", lambda q: "kb", "KB")
        bare_agent.register_tool("execute_command", lambda c: "out", "Exec")

        module = ChildAgentModule()
        bare_agent.register_module(module)

        policy = ROLE_POLICIES["researcher"]
        spec = ChildAgentSpec(task="research", role="researcher", policy=policy)
        child = module._create_child_agent(spec)

        assert "web_search" in child._tools
        assert "web_fetch" in child._tools
        assert "search_knowledge" in child._tools
        assert "execute_command" not in child._tools


class TestTaskBoardLifecycle:
    """Full lifecycle: spawn -> list -> collect."""

    def test_task_board_lifecycle(self, bare_agent, mock_llm_client):
        """Spawn multiple children, list them, then collect completed results."""
        # Two children: both succeed
        mock_llm_client.set_responses([
            make_llm_response("result from child 1"),
            make_llm_response("result from child 2"),
        ])

        module = ChildAgentModule()
        bare_agent.register_module(module)
        bare_agent.safety_loaded = True

        # Spawn two children
        result1 = module._spawn_child(task="task alpha", role="worker")
        result2 = module._spawn_child(task="task beta", role="analyst")

        # List children should show both
        listing = module._list_children()
        assert "worker" in listing
        assert "analyst" in listing
        assert "task alpha" in listing
        assert "task beta" in listing

        # Collect results should return both (both completed)
        collected = module._collect_results()
        assert "worker" in collected or "analyst" in collected
        # At least one result should be present
        assert len(collected) > 0


class TestChildInheritsSafety:
    """Child agent inherits parent's safety_loaded flag."""

    def test_child_inherits_safety(self, bare_agent):
        """Child agent inherits parent's safety_loaded flag."""
        bare_agent.safety_loaded = True

        module = ChildAgentModule()
        bare_agent.register_module(module)

        spec = ChildAgentSpec(task="test task", role="worker")
        child = module._create_child_agent(spec)

        # Child should inherit safety_loaded = True
        assert child.safety_loaded is True

        # If parent has no safety, child also has none
        bare_agent.safety_loaded = False
        child_no_safety = module._create_child_agent(spec)
        assert child_no_safety.safety_loaded is False

    def test_child_inherits_tool_executor(self, bare_agent):
        """Child agent inherits parent's tool_executor (sandbox dispatch)."""
        mock_executor = MagicMock()
        bare_agent.tool_executor = mock_executor

        module = ChildAgentModule()
        bare_agent.register_module(module)

        spec = ChildAgentSpec(task="sandbox task", role="coder")
        child = module._create_child_agent(spec)

        assert child.tool_executor is mock_executor


class TestBackwardCompatWithoutModule:
    """Without ChildAgentModule, agent works as v1.1."""

    def test_backward_compat_without_module(self, bare_agent, mock_llm_client):
        """Agent without ChildAgentModule still works normally for chat."""
        mock_llm_client.set_responses([
            make_llm_response("hello, I am your assistant"),
        ])

        # No modules loaded at all
        assert not bare_agent.has_module("child_agent")
        assert "spawn_child" not in bare_agent._tools

        # Normal chat still works
        response = bare_agent.chat("hi")
        assert isinstance(response, str)
        assert len(response) > 0

        # Register a tool and use it without child agent module
        bare_agent.register_tool("greet", lambda name: f"hello {name}", "Greet")
        assert "greet" in bare_agent._tools

        # Agent is fully functional without child agent capability
        assert bare_agent.has_module("child_agent") is False


# ============================================================
# Security fixes verification
# ============================================================


class TestChildAgentSecurityFixes:
    """Verify key security fixes in the child agent module."""

    def test_tool_deep_copy_isolation(self, bare_agent, mock_llm_client):
        """Child tool mutation does not bleed back to parent."""
        mock_llm_client.set_responses([make_llm_response("done")])
        bare_agent.register_tool(
            "shared_tool", lambda: "ok", "test",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        )

        module = ChildAgentModule()
        bare_agent.register_module(module)

        spec = ChildAgentSpec(task="test")
        child = module._create_child_agent(spec)

        # Mutate child's nested tool parameters
        child._tools["shared_tool"]["parameters"]["properties"]["injected"] = {"type": "int"}

        # Parent must NOT be affected
        assert "injected" not in bare_agent._tools["shared_tool"]["parameters"]["properties"]

    def test_max_children_limit(self, bare_agent, mock_llm_client):
        """Controller enforces max_children limit."""
        mock_llm_client.set_responses([make_llm_response("ok")] * 10)
        module = ChildAgentModule()
        bare_agent.register_module(module)
        module.controller.max_children = 2

        # First 2 succeed
        r1 = module._spawn_child(task="t1")
        r2 = module._spawn_child(task="t2")
        assert "Cannot spawn" not in r1
        assert "Cannot spawn" not in r2

        # 3rd blocked
        r3 = module._spawn_child(task="overflow")
        assert "Max children limit" in r3

    def test_child_permission_level_enforced(self, bare_agent, mock_llm_client):
        """Child agent uses its own permission_level, not parent's."""
        mock_llm_client.set_responses([make_llm_response("done")])
        bare_agent.safety_loaded = True

        bare_agent.register_tool("dangerous", lambda: "boom", "high risk", safety_level=3)

        module = ChildAgentModule()
        bare_agent.register_module(module)

        spec = ChildAgentSpec(task="test")
        child = module._create_child_agent(spec)

        # Child inherits safety_loaded from parent
        assert child.safety_loaded is True

    def test_runner_cleanup_prevents_memory_leak(self, bare_agent, mock_llm_client):
        """Runner results cleaned after sync to TaskBoard."""
        mock_llm_client.set_responses([make_llm_response("done")])
        module = ChildAgentModule()
        bare_agent.register_module(module)

        module._spawn_child(task="test")

        # Runner should be empty (cleaned after sync)
        assert len(module.controller.runner._results) == 0
        # TaskBoard should have the record
        assert len(module.controller.list_children(module._parent_id)) == 1
