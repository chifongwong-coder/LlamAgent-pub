"""
PlanReAct strategy tests: complexity routing, plan generation, step execution,
deadlock detection, replan, and quality evaluation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llamagent.core.agent import ReactResult, Module
from llamagent.modules.reasoning.module import PlanReAct, REPLAN_SCHEMA
from llamagent.modules.reasoning.planner import Step, TaskPlanner, validate_plan
from conftest import make_llm_response, make_tool_call


# ============================================================
# Helpers
# ============================================================

def _make_planner(mock_llm):
    return TaskPlanner(mock_llm)

def _make_plan_react(mock_llm, config=None, reflection_engine=None):
    planner = _make_planner(mock_llm)
    return PlanReAct(planner=planner, reflection_engine=reflection_engine, config=config)

def _make_config(**overrides):
    defaults = dict(
        max_plan_adjustments=7,
        max_react_steps=10,
        react_timeout=210.0,
        reflection_enabled=False,
        reflection_score_threshold=7.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ============================================================
# Complexity routing
# ============================================================

class TestComplexityRouting:
    """PlanReAct routes simple tasks to ReAct, complex tasks to planning."""

    def test_simple_task_uses_react(self, bare_agent, mock_llm_client):
        """LLM judges task as simple -> direct ReAct path."""
        strategy = _make_plan_react(mock_llm_client, _make_config())
        mock_llm_client.set_responses([
            make_llm_response('{"complex": false, "reason": "simple question"}'),
            make_llm_response("simple answer"),
        ])
        result = strategy.execute("hello", "", bare_agent)
        assert result == "simple answer"

    def test_complex_task_uses_planning(self, bare_agent, mock_llm_client):
        """LLM judges task as complex -> plan + execute + summarize."""
        strategy = _make_plan_react(mock_llm_client, _make_config())
        mock_llm_client.set_responses([
            make_llm_response('{"complex": true}'),
            make_llm_response(json.dumps({
                "steps": [
                    {"step_id": "s1", "order": 1, "action": "research",
                     "expected_output": "data", "depends_on": []},
                ]
            })),
            make_llm_response("step completed"),
            make_llm_response("final summary"),
        ])
        result = strategy.execute("complex research task", "", bare_agent)
        assert "summary" in result or "step completed" in result

    def test_judgment_failure_defaults_simple(self, bare_agent, mock_llm_client):
        """When complexity judgment fails (invalid JSON), defaults to simple path."""
        strategy = _make_plan_react(mock_llm_client, _make_config())
        mock_llm_client.set_responses([
            make_llm_response("not valid JSON"),
            make_llm_response("default answer"),
        ])
        result = strategy.execute("test", "", bare_agent)
        assert result == "default answer"


# ============================================================
# Plan generation + DAG validation
# ============================================================

class TestPlanGeneration:
    """TaskPlanner generates validated execution plans."""

    def test_valid_plan(self, mock_llm_client):
        """Valid linear plan passes DAG validation."""
        planner = _make_planner(mock_llm_client)
        mock_llm_client.set_responses([
            make_llm_response(json.dumps({
                "steps": [
                    {"step_id": "s1", "order": 1, "action": "gather",
                     "expected_output": "data", "depends_on": []},
                    {"step_id": "s2", "order": 2, "action": "analyze",
                     "expected_output": "report", "depends_on": ["s1"]},
                ]
            })),
        ])
        result = planner.plan("analyze something")
        assert len(result["steps"]) == 2
        assert isinstance(result["steps"][0], Step)

    def test_circular_dependency_raises(self, mock_llm_client):
        """Circular dependency in plan raises ValueError."""
        planner = _make_planner(mock_llm_client)
        mock_llm_client.set_responses([
            make_llm_response(json.dumps({
                "steps": [
                    {"step_id": "s1", "order": 1, "action": "a",
                     "expected_output": "o", "depends_on": ["s2"]},
                    {"step_id": "s2", "order": 2, "action": "b",
                     "expected_output": "o", "depends_on": ["s1"]},
                ]
            })),
        ])
        with pytest.raises(ValueError, match="Plan validation failed"):
            planner.plan("task")


# ============================================================
# Deadlock detection + replan
# ============================================================

class TestDeadlockAndReplan:
    """Step scheduling, deadlock detection, and replan mechanisms."""

    def test_ready_step_selection(self):
        """Among ready steps, the one with smallest order is selected."""
        steps = [
            Step(step_id="s1", order=3, action="late", expected_output=""),
            Step(step_id="s2", order=1, action="first", expected_output=""),
            Step(step_id="s3", order=2, action="mid", expected_output=""),
        ]
        ready = [s for s in steps if PlanReAct._deps_ready(s, steps)]
        assert len(ready) == 3
        selected = min(ready, key=lambda s: s.order)
        assert selected.step_id == "s2"

    def test_failed_dep_skips_step(self, bare_agent, mock_llm_client):
        """When a dependency fails, dependent steps are skipped."""
        config = _make_config()
        strategy = _make_plan_react(mock_llm_client, config)
        mock_llm_client.set_responses([
            make_llm_response('{"complex": true}'),
            make_llm_response(json.dumps({
                "steps": [
                    {"step_id": "s1", "order": 1, "action": "search",
                     "expected_output": "data", "depends_on": []},
                    {"step_id": "s2", "order": 2, "action": "analyze",
                     "expected_output": "result", "depends_on": ["s1"]},
                ]
            })),
            RuntimeError("LLM error"),  # s1 fails
        ])
        result = strategy.execute("test", "", bare_agent)
        assert "failed" in result.lower()

    def test_deadlock_triggers_replan(self, bare_agent, mock_llm_client):
        """Mutual dependency deadlock triggers automatic replan."""
        config = _make_config(max_plan_adjustments=2)
        strategy = _make_plan_react(mock_llm_client, config)
        mock_llm_client.set_responses([
            make_llm_response('{"complex": true}'),
            make_llm_response(json.dumps({
                "steps": [
                    {"step_id": "s0", "order": 1, "action": "prepare",
                     "expected_output": "ok", "depends_on": []},
                    {"step_id": "s1", "order": 2, "action": "A",
                     "expected_output": "a", "depends_on": ["s0", "s2"]},
                    {"step_id": "s2", "order": 3, "action": "B",
                     "expected_output": "b", "depends_on": ["s0", "s1"]},
                ]
            })),
            make_llm_response("s0 done"),  # s0 executes
            # deadlock -> replan
            make_llm_response(json.dumps({
                "adjusted_steps": [
                    {"step_id": "s0", "order": 1, "action": "prepare",
                     "expected_output": "ok", "depends_on": [], "status": "completed"},
                    {"step_id": "s3", "order": 2, "action": "merged",
                     "expected_output": "done", "depends_on": ["s0"]},
                ],
                "summary": "deadlock resolved"
            })),
            make_llm_response("merge done"),  # s3 executes
            make_llm_response("final result"),  # summarize
        ])
        # Patch validate_plan to let the circular plan pass validation,
        # so we reach the runtime deadlock detection code
        with patch('llamagent.modules.reasoning.planner.validate_plan', return_value=(True, "")), \
             patch('llamagent.modules.reasoning.module.validate_plan', return_value=(True, "")):
            result = strategy.execute("deadlock test", "", bare_agent)
        assert result is not None

    def test_replan_closure_interrupt(self, bare_agent, mock_llm_client):
        """Model calls replan tool during step execution -> interrupt + replan."""
        config = _make_config(max_plan_adjustments=3)
        strategy = _make_plan_react(mock_llm_client, config)
        mock_llm_client.set_responses([
            make_llm_response('{"complex": true}'),
            make_llm_response(json.dumps({
                "steps": [
                    {"step_id": "s1", "order": 1, "action": "try",
                     "expected_output": "result", "depends_on": []},
                ]
            })),
            # During s1 execution, model calls replan tool
            make_llm_response("", tool_calls=[
                make_tool_call("adjust_plan", {"feedback": "need different approach"}, "c1"),
            ]),
            # replan() returns new plan
            make_llm_response(json.dumps({
                "adjusted_steps": [
                    {"step_id": "s2", "order": 1, "action": "better approach",
                     "expected_output": "result", "depends_on": []},
                ],
                "summary": "plan adjusted"
            })),
            # s2 execution
            make_llm_response("success"),
            # summarize
            make_llm_response("task completed successfully"),
        ])
        result = strategy.execute("test", "", bare_agent)
        assert result is not None


# ============================================================
# Quality evaluation
# ============================================================

class TestQualityEvaluation:
    """Quality-driven replan via reflection engine."""

    def test_quality_check_pass(self, bare_agent, mock_llm_client):
        """Score above threshold -> no replan."""
        mock_reflection = MagicMock()
        mock_reflection.evaluate_result.return_value = {
            "score": 9.0,
            "strengths": ["thorough"],
            "weaknesses": [],
        }
        config = _make_config(reflection_enabled=True, reflection_score_threshold=7.0)
        strategy = _make_plan_react(mock_llm_client, config,
                                     reflection_engine=mock_reflection)
        mock_llm_client.set_responses([
            make_llm_response('{"complex": true}'),
            make_llm_response(json.dumps({
                "steps": [
                    {"step_id": "s1", "order": 1, "action": "do",
                     "expected_output": "done", "depends_on": []},
                ]
            })),
            make_llm_response("step done"),
            make_llm_response("great result"),  # summarize
        ])
        result = strategy.execute("test", "", bare_agent)
        assert result is not None
        mock_reflection.evaluate_result.assert_called_once()

    def test_quality_check_fail_triggers_replan(self, bare_agent, mock_llm_client):
        """Score below threshold -> triggers quality-driven replan."""
        mock_reflection = MagicMock()
        mock_reflection.evaluate_result.return_value = {
            "score": 3.0, "weaknesses": ["not detailed enough"],
        }
        config = _make_config(
            reflection_enabled=True,
            reflection_score_threshold=7.0,
            max_plan_adjustments=3,
        )
        strategy = _make_plan_react(mock_llm_client, config,
                                     reflection_engine=mock_reflection)
        mock_llm_client.set_responses([
            make_llm_response('{"complex": true}'),
            make_llm_response(json.dumps({
                "steps": [
                    {"step_id": "s1", "order": 1, "action": "do",
                     "expected_output": "done", "depends_on": []},
                ]
            })),
            make_llm_response("step completed"),
            make_llm_response("initial summary"),
            # Quality not met -> replan
            make_llm_response(json.dumps({
                "adjusted_steps": [
                    {"step_id": "s1", "order": 1, "action": "do",
                     "expected_output": "o", "depends_on": [], "status": "completed"},
                    {"step_id": "s4", "order": 2, "action": "add details",
                     "expected_output": "detailed", "depends_on": ["s1"]},
                ]
            })),
            make_llm_response("details added"),
            make_llm_response("detailed summary"),
        ])
        result = strategy.execute("test", "", bare_agent)
        assert result is not None
