"""
PlanningModule: provides task planning capability for SmartAgent via PlanReAct execution strategy.

Architecture:
- PlanReAct (ExecutionStrategy subclass): Complex task execution strategy
  - Judge complexity -> simple tasks use SimpleReAct, complex tasks use plan + step-by-step execution
  - Three replan paths share the max_plan_adjustments counter
  - Optional quality evaluation (requires ReflectionEngine)
- TaskPlanner: Task decomposition and replanning
- PlanningModule (Module subclass): Creates strategy and injects into agent on on_attach

v1.1 changes:
- execute() uses ready-step + deadlock logic instead of simple while loop
- _execute_step() returns ReactResult, supports interrupted state
- Uses agent.run_react() instead of deprecated run_react_loop
- Uses agent.build_messages() instead of self-built messages
- Uses agent.call_tool(name, args) instead of _make_tool_dispatch
- Step uses step_id + order instead of the old step field
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llamagent.core.agent import Module, ExecutionStrategy, ReactResult
from llamagent.modules.reasoning.planner import TaskPlanner, Step, validate_plan

if TYPE_CHECKING:
    from llamagent.core.agent import SmartAgent
    from llamagent.modules.reflection.engine import ReflectionEngine

logger = logging.getLogger(__name__)


# ============================================================
# Replan tool schema (injected into step-level function calling)
# ============================================================

REPLAN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "replan",
        "description": "Use when the current plan needs adjustment. Provide the reason for adjustment, and the system will replan the remaining steps.",
        "parameters": {
            "type": "object",
            "properties": {
                "feedback": {
                    "type": "string",
                    "description": "Reason for adjustment and suggestions",
                }
            },
            "required": ["feedback"],
        },
    },
}


# ============================================================
# PlanReAct execution strategy
# ============================================================

class PlanReAct(ExecutionStrategy):
    """
    Planning execution strategy: first judges task complexity -- simple tasks go through
    direct conversation / ReAct, complex tasks are decomposed into a plan then executed
    step by step via ReAct.

    Three replan paths share the max_plan_adjustments counter:
    1. Model-initiated replan (calls the replan tool during step execution)
    2. Failure auto-replan (automatically triggered after step failure)
    3. Quality-driven replan (quality evaluation falls below threshold after all steps complete)
    """

    def __init__(
        self,
        planner: TaskPlanner,
        reflection_engine: "ReflectionEngine | None" = None,
        config=None,
    ):
        """
        Args:
            planner:           Task planner
            reflection_engine: Reflection engine; enables quality evaluation when not None and reflection_enabled=True
            config:            Configuration object
        """
        self.planner = planner
        self.reflection_engine = reflection_engine

        # Read parameters from config, with reasonable defaults
        if config is not None:
            self.max_plan_adjustments: int = getattr(config, "max_plan_adjustments", 7)
            self.max_react_steps: int = getattr(config, "max_react_steps", 10)
            self.react_timeout: float = getattr(config, "react_timeout", 210.0)
            self.reflection_enabled: bool = getattr(config, "reflection_enabled", False)
            self.reflection_score_threshold: float = getattr(
                config, "reflection_score_threshold", 7.0
            )
        else:
            self.max_plan_adjustments = 7
            self.max_react_steps = 10
            self.react_timeout = 210.0
            self.reflection_enabled = False
            self.reflection_score_threshold = 7.0

    def execute(self, query: str, context: str, agent: "SmartAgent") -> str:
        """
        PlanReAct main entry point.

        Flow:
        1. Complexity judgment
        2. Simple -> go directly through ReAct / conversation
        3. Complex -> generate plan -> execute step by step -> summarize results -> optional quality evaluation

        Args:
            query:   User input
            context: Context information
            agent:   Agent instance

        Returns:
            Final response text
        """
        # -- 1. Complexity judgment --
        tools_schema = self._get_tools_schema(agent)
        is_complex = self._judge_complexity(query, tools_schema, agent)

        if not is_complex:
            # Simple task: go directly through ReAct (consistent with SimpleReAct behavior)
            return self._execute_simple(query, context, agent, tools_schema)

        # -- Set task_id for workspace isolation (v1.5) --
        import uuid as _uuid
        task_id = _uuid.uuid4().hex
        agent._current_task_id = task_id
        try:
            return self._execute_complex(query, context, agent, tools_schema)
        finally:
            agent._current_task_id = None

    def _execute_complex(self, query: str, context: str, agent, tools_schema: list) -> str:
        """Execute a complex task with planning. Separated for task_id try/finally wrapper."""
        # -- 2. Generate plan --
        plan_result = self.planner.plan(query, tools_schema)
        steps = plan_result["steps"]
        valid, msg = validate_plan(steps)
        if not valid:
            logger.warning("Initial plan validation failed: %s, falling back to simple execution", msg)
            return self._execute_simple(query, context, agent, tools_schema)
        logger.info("Generated plan (%d steps):\n%s", len(steps), self.planner.format_plan(steps))

        # -- 3. Initialize --
        adjustment_count = 0
        interrupt_flag = False
        should_continue = lambda: "replanned" if interrupt_flag else None

        # Create replan closure
        def replan_closure(feedback: str) -> str:
            nonlocal adjustment_count, interrupt_flag
            if adjustment_count >= self.max_plan_adjustments:
                return "Maximum adjustment count reached, cannot adjust further"
            adjustment_count += 1
            completed = [s for s in steps if s.status == "completed"]
            replan_result = self.planner.replan(steps, completed, feedback)
            new_steps = replan_result["steps"]
            try:
                validate_plan(new_steps)
            except ValueError as e:
                return f"Replan failed validation: {e}"
            steps[:] = new_steps  # In-place replacement, visible to while loop
            interrupt_flag = True  # Notify run_react to stop
            return "Plan adjusted, remaining steps updated"

        # -- 4. Step-by-step execution (ready-step + deadlock logic) --
        while any(s.status == "pending" for s in steps):
            # 1. Skip steps whose deps have failed/skipped
            step_map = {s.step_id: s for s in steps}
            for s in steps:
                if s.status == "pending" and s.depends_on:
                    dep_statuses = {
                        step_map[d].status for d in s.depends_on if d in step_map
                    }
                    if dep_statuses & {"failed", "skipped"}:
                        s.status = "skipped"

            # 2. Find ready steps (pending + all deps completed)
            ready_steps = [
                s for s in steps
                if s.status == "pending" and self._deps_ready(s, steps)
            ]

            # 3. Deadlock detection
            if not ready_steps:
                if any(s.status == "pending" for s in steps):
                    # deadlock: pending but no ready
                    if adjustment_count < self.max_plan_adjustments:
                        adjustment_count += 1
                        completed = [s for s in steps if s.status == "completed"]
                        replan_result = self.planner.replan(
                            steps, completed, "Plan deadlock, the following steps cannot be executed"
                        )
                        new_steps = replan_result["steps"]
                        validate_plan(new_steps)
                        steps[:] = new_steps
                        # Refresh context after deadlock replan
                        context = ""
                        for mod in agent.modules.values():
                            context = mod.on_context(query, context)
                        continue
                    else:
                        break
                else:
                    break  # no pending steps

            # 4. Select step by order
            step_i = min(ready_steps, key=lambda s: s.order)
            step_i.status = "running"
            interrupt_flag = False  # reset before each step

            logger.info("Executing step %s (order=%d): %s", step_i.step_id, step_i.order, step_i.action)

            # 5. Execute step
            result = self._execute_step(
                step_i, query, context, agent, steps, self.planner, replan_closure, should_continue
            )

            # 6. Handle ReactResult
            if result.status == "interrupted":
                # After replan, steps[:] has been replaced; step_i points to old object
                current = next(
                    (s for s in steps if s.step_id == step_i.step_id), None
                )
                if current:
                    current.status = "failed"
                    current.result = f"Step interrupted (reason: {result.reason})"
                # Refresh context
                context = ""
                for mod in agent.modules.values():
                    context = mod.on_context(query, context)
                continue
            elif result.status == "completed":
                step_i.status = "completed"
                step_i.result = result.text
                logger.info("Step %s completed", step_i.step_id)
            else:
                step_i.status = "failed"
                step_i.result = result.error or result.text
                logger.warning("Step %s failed: %s", step_i.step_id, step_i.result)

                if adjustment_count < self.max_plan_adjustments:
                    adjustment_count += 1
                    completed = [s for s in steps if s.status == "completed"]
                    replan_result = self.planner.replan(steps, completed, step_i.result)
                    new_steps = replan_result["steps"]
                    validate_plan(new_steps)
                    steps[:] = new_steps
                    logger.info("Plan after failure replan:\n%s", self.planner.format_plan(steps))
                    # Refresh context
                    context = ""
                    for mod in agent.modules.values():
                        context = mod.on_context(query, context)
                    continue
                else:
                    logger.warning("Maximum adjustment count reached, aborting execution")
                    break

        # -- 5. Summarize results --
        summary = self._summarize_results(query, steps, agent)

        # -- 6. Quality evaluation (optional, requires ReflectionEngine and enabled) --
        if self.reflection_engine is None or not self.reflection_enabled:
            return summary

        try:
            evaluation = self.reflection_engine.evaluate_result(query, summary)
        except Exception as e:
            logger.warning("Quality evaluation failed, skipping: %s", e)
            return summary

        if evaluation.get("score", 0) >= self.reflection_score_threshold:
            return summary

        # Quality below threshold -> quality-driven replan
        if adjustment_count < self.max_plan_adjustments:
            adjustment_count += 1
            completed = [s for s in steps if s.status == "completed"]
            quality_feedback = "; ".join(evaluation.get("weaknesses", ["Quality below threshold"]))
            replan_result = self.planner.replan(steps, completed, quality_feedback)
            new_steps = replan_result["steps"]
            validate_plan(new_steps)
            steps[:] = new_steps
            logger.info("After quality-driven replan:\n%s", self.planner.format_plan(steps))

            # Execute newly added pending steps (reuse ready-step + deadlock logic)
            while any(s.status == "pending" for s in steps):
                step_map = {s.step_id: s for s in steps}
                for s in steps:
                    if s.status == "pending" and s.depends_on:
                        dep_statuses = {
                            step_map[d].status for d in s.depends_on if d in step_map
                        }
                        if dep_statuses & {"failed", "skipped"}:
                            s.status = "skipped"

                ready_steps = [
                    s for s in steps
                    if s.status == "pending" and self._deps_ready(s, steps)
                ]

                if not ready_steps:
                    break

                step_i = min(ready_steps, key=lambda s: s.order)
                step_i.status = "running"
                interrupt_flag = False

                result = self._execute_step(
                    step_i, query, context, agent, steps, self.planner, replan_closure, should_continue
                )

                if result.status == "completed":
                    step_i.status = "completed"
                    step_i.result = result.text
                else:
                    step_i.status = "failed"
                    step_i.result = result.error or result.text
                    break

            # Re-summarize
            summary = self._summarize_results(query, steps, agent)

        return summary

    # ----------------------------------------------------------
    # Complexity judgment
    # ----------------------------------------------------------

    def _judge_complexity(
        self, query: str, tools_schema: list[dict], agent: "SmartAgent"
    ) -> bool:
        """
        Call LLM to judge whether the task is complex.

        Simple tasks (single-step, casual chat, simple Q&A) return False,
        complex tasks (multi-step, requiring multi-tool collaboration) return True.
        """
        tools_desc = ""
        if tools_schema:
            names = []
            for t in tools_schema:
                if "function" in t:
                    names.append(t["function"].get("name", ""))
                else:
                    names.append(t.get("name", ""))
            tools_desc = ", ".join(n for n in names if n)

        prompt = f"""Determine whether the following task requires multi-step planning to complete.

Task: {query}
Available tools: {tools_desc or 'None'}

Please return JSON:
{{
    "complex": true or false,
    "reason": "Reason for judgment"
}}

Criteria:
- Simple task (complex=false): Casual chat, simple Q&A, can be completed with a single tool call
- Complex task (complex=true): Requires multiple steps, multi-tool collaboration, information integration"""

        try:
            result = agent.llm.ask_json(prompt, temperature=0.1)
            is_complex = result.get("complex", False)
            reason = result.get("reason", "")
            logger.info("Complexity judgment: complex=%s, reason=%s", is_complex, reason)
            return bool(is_complex)
        except Exception as e:
            logger.warning("Complexity judgment failed, defaulting to simple task: %s", e)
            return False

    # ----------------------------------------------------------
    # Simple task execution
    # ----------------------------------------------------------

    def _execute_simple(
        self,
        query: str,
        context: str,
        agent: "SmartAgent",
        tools_schema: list[dict],
    ) -> str:
        """Simple task: go directly through ReAct loop (consistent with SimpleReAct behavior)."""
        messages = agent.build_messages(query, context)
        tool_dispatch = agent.call_tool

        result = agent.run_react(messages, tools_schema, tool_dispatch)
        return result.text

    # ----------------------------------------------------------
    # Step execution
    # ----------------------------------------------------------

    def _execute_step(
        self,
        step: Step,
        original_query: str,
        context: str,
        agent: "SmartAgent",
        steps: list[Step],
        planner: TaskPlanner,
        replan_closure,
        should_continue,
    ) -> ReactResult:
        """
        Execute a single step: build step context + tool set, run one ReAct loop.

        Returns ReactResult, supports interrupted state (when replan is triggered).

        Anti-recursion mechanism: replan_closure only modifies data, does not trigger execution.
        """
        tools_schema = self._get_tools_schema(agent)
        all_tools = tools_schema + [REPLAN_SCHEMA]

        def tool_dispatch(name: str, args: dict) -> str:
            if name == "replan":
                return replan_closure(args.get("feedback", "Plan adjustment needed"))
            return agent.call_tool(name, args)

        step_prompt = (
            planner.format_plan(steps)
            + f"\nCurrently executing step: {step.step_id} - {step.action}"
        )
        step_messages = agent.build_messages(
            query=original_query,
            context=context,
            include_history=False,
            extra_system=step_prompt,
        )

        return agent.run_react(
            step_messages, all_tools, tool_dispatch,
            should_continue=should_continue,
        )

    # ----------------------------------------------------------
    # Result summarization
    # ----------------------------------------------------------

    def _summarize_results(
        self, query: str, steps: list[Step], agent: "SmartAgent"
    ) -> str:
        """Summarize all step results into a coherent response."""
        completed = [s for s in steps if s.status == "completed"]
        failed = [s for s in steps if s.status in ("failed", "skipped")]

        # All failed: return failure summary directly, no LLM call needed
        if not completed:
            lines = ["Unfortunately, all steps failed to complete successfully:"]
            for s in failed:
                lines.append(f"- Step {s.order}[{s.step_id}] ({s.action}): {s.result or 'Not executed'}")
            return "\n".join(lines)

        # Build results text
        results_text = []
        for s in completed:
            preview = (s.result or "")[:500]
            results_text.append(f"Step {s.order}[{s.step_id}] ({s.action}): {preview}")

        if failed:
            results_text.append("\nIncomplete steps:")
            for s in failed:
                results_text.append(f"- Step {s.order}[{s.step_id}] ({s.action}): {s.status}")

        prompt = f"""Task: {query}

Results from each step:
{chr(10).join(results_text)}

Please integrate the above results and provide a complete, coherent final answer."""

        try:
            return agent.llm.ask(prompt)
        except Exception as e:
            logger.warning("Result summarization LLM call failed: %s", e)
            return "\n".join(results_text)

    # ----------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------

    @staticmethod
    def _deps_ready(step: Step, all_steps: list[Step]) -> bool:
        """Check if all dependencies are completed."""
        if not step.depends_on:
            return True
        step_map = {s.step_id: s for s in all_steps}
        return all(
            step_map.get(dep_id) is not None and step_map[dep_id].status == "completed"
            for dep_id in step.depends_on
        )

    @staticmethod
    def _get_tools_schema(agent: "SmartAgent") -> list[dict]:
        """
        Get all available tool schemas from the agent.

        Prefers agent.get_all_tool_schemas() (target architecture),
        falls back to getting from the tools module (current implementation).
        """
        # Target architecture interface
        if hasattr(agent, "get_all_tool_schemas"):
            return agent.get_all_tool_schemas()

        # Current transition: get from tools module
        if not agent.has_module("tools"):
            return []

        tools_mod = agent.get_module("tools")

        schemas = []
        # Get common tool schemas
        common_reg = getattr(tools_mod, "common_registry", None)
        if common_reg:
            schemas.extend(common_reg.get_openai_schema(tiers=("common",)))

        # Get instance tool schemas
        agent_reg = getattr(tools_mod, "agent_registry", None)
        if agent_reg:
            is_admin = getattr(tools_mod, "_is_admin", False)
            tiers = ("default", "admin", "agent") if is_admin else ("default", "agent")
            schemas.extend(agent_reg.get_openai_schema(tiers=tiers))

        return schemas


# ============================================================
# PlanningModule (Module subclass)
# ============================================================

class PlanningModule(Module):
    """
    Planning module: creates PlanReAct execution strategy and injects into Agent via on_attach.

    In the target architecture, injection is done via agent.set_execution_strategy();
    the current transitional implementation uses the on_execute callback to intercept execution.
    """

    name: str = "planning"
    description: str = "Task planning: automatically decomposes complex tasks into multi-step execution"

    def __init__(self):
        self.strategy: PlanReAct | None = None

    def on_attach(self, agent: "SmartAgent"):
        """
        Create PlanReAct strategy and inject it.

        If the Reflection module is already loaded, obtain its engine for quality evaluation.
        """
        super().on_attach(agent)

        planner = TaskPlanner(agent.llm)

        # Check if Reflection module is available
        reflection_engine = None
        if agent.has_module("reflection"):
            reflection_mod = agent.get_module("reflection")
            reflection_engine = getattr(reflection_mod, "engine", None)

        self.strategy = PlanReAct(
            planner=planner,
            reflection_engine=reflection_engine,
            config=agent.config,
        )

        # Target architecture: agent.set_execution_strategy(self.strategy)
        # Current transition: intercept via on_execute callback
        if hasattr(agent, "set_execution_strategy"):
            agent.set_execution_strategy(self.strategy)

    def on_execute(self, query: str, context: str) -> str | None:
        """
        [Deprecated] Execution interception: delegates the request to the PlanReAct strategy.

        Once the ExecutionStrategy interface in core is ready, this callback will no longer be needed.
        This method is retained only for backward compatibility.
        """
        if self.strategy is None:
            return None

        try:
            return self.strategy.execute(query, context, self.agent)
        except Exception as e:
            logger.error("PlanReAct execution error: %s", e, exc_info=True)
            # Fall back to default conversation on error, do not block the user
            return None
