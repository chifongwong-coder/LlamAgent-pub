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


class TestComplexityRoutingAndPlanGeneration:
    """Complexity routing (simple/complex/fallback) + plan generation and DAG validation."""

    def test_complexity_routing_and_plan_generation(self, bare_agent, mock_llm_client):
        """Simple task uses ReAct; complex task uses planning; judgment failure defaults
        to simple. Valid linear plan passes DAG validation; circular dependency raises."""
        # --- Simple task -> ReAct ---
        strategy = _make_plan_react(mock_llm_client, _make_config())
        mock_llm_client.set_responses([
            make_llm_response('{"complex": false, "reason": "simple question"}'),
            make_llm_response("simple answer"),
        ])
        result = strategy.execute("hello", "", bare_agent)
        assert result == "simple answer"

        # --- Complex task -> planning ---
        strategy2 = _make_plan_react(mock_llm_client, _make_config())
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
        result2 = strategy2.execute("complex research task", "", bare_agent)
        assert "summary" in result2 or "step completed" in result2

        # --- Judgment failure defaults to simple ---
        strategy3 = _make_plan_react(mock_llm_client, _make_config())
        mock_llm_client.set_responses([
            make_llm_response("not valid JSON"),
            make_llm_response("default answer"),
        ])
        result3 = strategy3.execute("test", "", bare_agent)
        assert result3 == "default answer"

        # --- Valid plan with DAG validation ---
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
        plan_result = planner.plan("analyze something")
        assert len(plan_result["steps"]) == 2
        assert isinstance(plan_result["steps"][0], Step)

        # --- Circular dependency raises ValueError ---
        planner2 = _make_planner(mock_llm_client)
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
            planner2.plan("task")


class TestDeadlockAndReplan:
    """Step scheduling, deadlock detection, and replan mechanisms."""

    def test_deadlock_and_replan(self, bare_agent, mock_llm_client):
        """Ready step selection by order; failed dep skips step; mutual dependency
        deadlock triggers replan; model replan tool call interrupts and replans."""
        # --- Ready step selection ---
        steps = [
            Step(step_id="s1", order=3, action="late", expected_output=""),
            Step(step_id="s2", order=1, action="first", expected_output=""),
            Step(step_id="s3", order=2, action="mid", expected_output=""),
        ]
        ready = [s for s in steps if PlanReAct._deps_ready(s, steps)]
        assert len(ready) == 3
        selected = min(ready, key=lambda s: s.order)
        assert selected.step_id == "s2"

        # --- Failed dep skips step ---
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
            RuntimeError("LLM error"),
        ])
        result = strategy.execute("test", "", bare_agent)
        assert "failed" in result.lower()

        # --- Deadlock triggers replan ---
        config2 = _make_config(max_plan_adjustments=2)
        strategy2 = _make_plan_react(mock_llm_client, config2)
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
            make_llm_response("s0 done"),
            make_llm_response(json.dumps({
                "adjusted_steps": [
                    {"step_id": "s0", "order": 1, "action": "prepare",
                     "expected_output": "ok", "depends_on": [], "status": "completed"},
                    {"step_id": "s3", "order": 2, "action": "merged",
                     "expected_output": "done", "depends_on": ["s0"]},
                ],
                "summary": "deadlock resolved"
            })),
            make_llm_response("merge done"),
            make_llm_response("final result"),
        ])
        with patch('llamagent.modules.reasoning.planner.validate_plan', return_value=(True, "")), \
             patch('llamagent.modules.reasoning.module.validate_plan', return_value=(True, "")):
            result2 = strategy2.execute("deadlock test", "", bare_agent)
        assert result2 is not None

        # --- Replan closure interrupt ---
        config3 = _make_config(max_plan_adjustments=3)
        strategy3 = _make_plan_react(mock_llm_client, config3)
        mock_llm_client.set_responses([
            make_llm_response('{"complex": true}'),
            make_llm_response(json.dumps({
                "steps": [
                    {"step_id": "s1", "order": 1, "action": "try",
                     "expected_output": "result", "depends_on": []},
                ]
            })),
            make_llm_response("", tool_calls=[
                make_tool_call("adjust_plan", {"feedback": "need different approach"}, "c1"),
            ]),
            make_llm_response(json.dumps({
                "adjusted_steps": [
                    {"step_id": "s2", "order": 1, "action": "better approach",
                     "expected_output": "result", "depends_on": []},
                ],
                "summary": "plan adjusted"
            })),
            make_llm_response("success"),
            make_llm_response("task completed successfully"),
        ])
        result3 = strategy3.execute("test", "", bare_agent)
        assert result3 is not None


class TestQualityEvaluation:
    """Quality-driven replan via reflection engine."""

    def test_quality_evaluation(self, bare_agent, mock_llm_client):
        """Score above threshold -> no replan. Score below threshold -> triggers
        quality-driven replan with additional steps."""
        # --- Quality check pass ---
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
            make_llm_response("great result"),
        ])
        result = strategy.execute("test", "", bare_agent)
        assert result is not None
        mock_reflection.evaluate_result.assert_called_once()

        # --- Quality check fail triggers replan ---
        mock_reflection2 = MagicMock()
        mock_reflection2.evaluate_result.return_value = {
            "score": 3.0, "weaknesses": ["not detailed enough"],
        }
        config2 = _make_config(
            reflection_enabled=True,
            reflection_score_threshold=7.0,
            max_plan_adjustments=3,
        )
        strategy2 = _make_plan_react(mock_llm_client, config2,
                                      reflection_engine=mock_reflection2)
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
        result2 = strategy2.execute("test", "", bare_agent)
        assert result2 is not None
