"""
Task planner: decomposes complex tasks into an executable list of steps with dynamic adjustment support.

Core components:
- Step: Step data structure (dataclass)
- validate_plan: Plan validity check (DAG validation)
- TaskPlanner: Task planner
  - plan(task, available_tools) -> dict  ({"steps": list[Step]})
  - replan(steps, completed, feedback) -> dict  ({"steps": list[Step]})
  - format_plan(steps) -> str
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from llamagent.core.llm import LLMClient

logger = logging.getLogger(__name__)


# ============================================================
# Step data structure
# ============================================================

@dataclass
class Step:
    """
    A single step in an execution plan.

    Field descriptions:
    - step_id:         Unique identifier (e.g. "s1", "s2"), cannot be modified during replan
    - order:           Display order (starting from 1), can be reordered during replan
    - action:          What specifically to do
    - tool:            Tool name to use (None if no tool is needed)
    - expected_output: Description of expected output
    - depends_on:      List of step_ids of prerequisite steps
    - status:          Current status ("pending" / "running" / "completed" / "failed" / "skipped")
    - result:          Execution result
    """
    step_id: str
    order: int
    action: str
    expected_output: str
    tool: str | None = None
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"
    result: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_id": self.step_id,
            "order": self.order,
            "action": self.action,
            "tool": self.tool,
            "expected_output": self.expected_output,
            "depends_on": self.depends_on,
            "status": self.status,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        """Create a Step instance from a dictionary, compatible with various LLM output formats."""
        # Backward compat: if "step" exists but not "step_id", auto-generate
        if "step_id" not in d and "step" in d:
            step_num = d["step"]
            step_id = f"s{step_num}"
            order = step_num
        else:
            step_id = d.get("step_id", "s1")
            order = d.get("order", 1)
        return cls(
            step_id=step_id,
            order=order,
            action=d.get("action", ""),
            tool=d.get("tool"),
            expected_output=d.get("expected_output", ""),
            depends_on=d.get("depends_on", []),
            status=d.get("status", "pending"),
            result=d.get("result"),
        )


# ============================================================
# Plan validity check
# ============================================================

def validate_plan(steps: list[Step]) -> tuple[bool, str]:
    """
    Plan validity check (DAG validation). Automatically called after plan() and replan() return.

    Validation rules:
    1. step_ids referenced in depends_on must exist
    2. Self-dependency is not allowed
    3. Must be a DAG (no circular dependencies) -- detected via topological sort
    4. At least one entry step must exist (empty depends_on)
    5. Duplicate step_ids are not allowed

    Returns:
        (True, "") means validation passed; (False, "error description") means validation failed.
    """
    # Rule 5: Duplicate step_ids are not allowed
    seen_ids: set[str] = set()
    for s in steps:
        if s.step_id in seen_ids:
            return False, f"Duplicate step_id found: {s.step_id}"
        seen_ids.add(s.step_id)

    all_ids = seen_ids

    # Rule 1: step_ids referenced in depends_on must exist
    for s in steps:
        for dep in s.depends_on:
            if dep not in all_ids:
                return False, f"Step {s.step_id} depends on non-existent step_id: {dep}"

    # Rule 2: Self-dependency is not allowed
    for s in steps:
        if s.step_id in s.depends_on:
            return False, f"Step {s.step_id} has a self-dependency"

    # Rule 4: At least one entry step must exist
    has_entry = any(len(s.depends_on) == 0 for s in steps)
    if not has_entry:
        return False, "No entry step exists (all steps have dependencies)"

    # Rule 3: DAG detection (Kahn's topological sort)
    in_degree: dict[str, int] = {s.step_id: 0 for s in steps}
    adjacency: dict[str, list[str]] = {s.step_id: [] for s in steps}
    for s in steps:
        for dep in s.depends_on:
            adjacency[dep].append(s.step_id)
            in_degree[s.step_id] += 1

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(steps):
        return False, "Circular dependency detected in the plan"

    return True, ""


# ============================================================
# Task planner
# ============================================================

class TaskPlanner:
    """
    Task planner: decomposes complex tasks into 3~8 executable steps.

    Uses llm.ask_json() to call the LLM for task decomposition and dynamic replanning.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, task: str, available_tools: list[dict] | None = None) -> dict:
        """
        Decompose a complex task into executable steps.

        Args:
            task:            Task description
            available_tools: List of available tool schemas

        Returns:
            {"steps": list[Step]}
        """
        tools_desc = self._format_tools(available_tools)

        prompt = f"""Decompose the following task into executable steps:

Task: {task}
Available tools: {tools_desc}

Please return JSON in the following format:
{{
    "steps": [
        {{
            "step_id": "s1",
            "order": 1,
            "action": "What specifically to do",
            "tool": "Tool name to use (null if not needed)",
            "expected_output": "Expected output",
            "depends_on": []
        }}
    ]
}}

Requirements:
- 3~8 steps, each should be independently executable
- step_id is a unique identifier (e.g. "s1", "s2"), order is the display order
- depends_on references step_ids of other steps
- Clearly annotate dependencies between steps
- Prefer tools from the available tools list
- The last step should integrate and output the final result"""

        try:
            result = self.llm.ask_json(prompt)
        except Exception as e:
            logger.warning("Task planning LLM call failed, using fallback single-step plan: %s", e)
            return {"steps": [Step(step_id="s1", order=1, action=task, expected_output=task)]}

        # ask_json may return a fallback dict containing "error"
        if isinstance(result, dict) and "error" in result:
            logger.warning("Task planning JSON parsing failed, using fallback single-step plan")
            return {"steps": [Step(step_id="s1", order=1, action=task, expected_output=task)]}

        raw_steps = self._extract_steps(result)
        steps = [Step.from_dict(s) for s in raw_steps]

        # Ensure at least one step exists
        if not steps:
            steps = [Step(step_id="s1", order=1, action=task, expected_output=task)]

        # DAG validity check
        valid, err = validate_plan(steps)
        if not valid:
            raise ValueError(f"Plan validation failed: {err}")

        return {"steps": steps}

    def replan(
        self,
        steps: list[Step],
        completed: list[Step],
        feedback: str,
    ) -> dict:
        """
        Adjust the plan based on completed steps and feedback.

        Constraint: do not modify completed steps; may modify/delete/add pending steps.

        Args:
            steps:     Current complete plan
            completed: Completed steps
            feedback:  Reason for adjustment (failure info / quality evaluation feedback)

        Returns:
            {"steps": list[Step]}  Adjusted complete step list (including completed steps)
        """
        steps_data = [s.to_dict() for s in steps]
        completed_data = [s.to_dict() for s in completed]

        prompt = f"""Adjust the plan based on execution status:

Current plan:
{json.dumps(steps_data, ensure_ascii=False, indent=2)}

Completed steps:
{json.dumps(completed_data, ensure_ascii=False, indent=2)}

Reason for adjustment: {feedback}

Please return the adjusted complete step list (JSON) in the following format:
{{
    "adjusted_steps": [
        {{
            "step_id": "s1",
            "order": 1,
            "action": "What to do",
            "tool": "Tool name or null",
            "expected_output": "Expected output",
            "depends_on": [],
            "status": "completed or pending"
        }}
    ],
    "summary": "One sentence describing the adjustment"
}}

Requirements:
- Do not modify the step_id of completed steps
- You may modify, delete, or add pending steps
- depends_on references step_ids of other steps
- Maintain correctness of inter-step dependencies"""

        try:
            result = self.llm.ask_json(prompt)
        except Exception as e:
            logger.warning("Replanning LLM call failed, keeping original plan: %s", e)
            return {"steps": steps}

        # ask_json may return a fallback dict containing "error"
        if isinstance(result, dict) and "error" in result:
            logger.warning("Replanning JSON parsing failed, keeping original plan")
            return {"steps": steps}

        raw_steps = self._extract_steps(result)
        adjusted = [Step.from_dict(s) for s in raw_steps]

        # If parsing result is empty, keep the original plan
        if not adjusted:
            return {"steps": steps}

        # DAG validity check
        valid, err = validate_plan(adjusted)
        if not valid:
            raise ValueError(f"Replan validation failed: {err}")

        summary = result.get("summary", "Plan adjusted") if isinstance(result, dict) else "Plan adjusted"
        logger.info("[Plan adjustment] %s", summary)

        return {"steps": adjusted}

    @staticmethod
    def format_plan(steps: list[Step]) -> str:
        """Format the plan as readable text."""
        icons = {
            "pending": "[Pending]",
            "running": "[Running]",
            "completed": "[Completed]",
            "failed": "[Failed]",
            "skipped": "[Skipped]",
        }

        lines = ["Execution Plan:"]
        for s in steps:
            icon = icons.get(s.status, "[?]")
            deps = f" (depends on: {s.depends_on})" if s.depends_on else ""
            tool = f" [tool: {s.tool}]" if s.tool else ""
            lines.append(f"  {icon} [{s.step_id}] Step {s.order}: {s.action}{tool}{deps}")

            if s.result:
                preview = str(s.result)[:80]
                lines.append(f"        Result: {preview}")

        done = sum(1 for s in steps if s.status == "completed")
        lines.append(f"\n  Progress: {done}/{len(steps)} steps completed")
        return "\n".join(lines)

    # ----------------------------------------------------------
    # Internal methods
    # ----------------------------------------------------------

    @staticmethod
    def _format_tools(tools: list[dict] | None) -> str:
        """Format the tool list as a text description."""
        if not tools:
            return "No available tools"
        lines = []
        for t in tools:
            # Compatible with OpenAI schema format and simple dict format
            if "function" in t:
                name = t["function"].get("name", "?")
                desc = t["function"].get("description", "")
            else:
                name = t.get("name", "?")
                desc = t.get("description", "")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines) if lines else "No available tools"

    @staticmethod
    def _extract_steps(result) -> list[dict]:
        """Extract the step list from LLM-returned JSON (compatible with multiple formats)."""
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for key in ("steps", "adjusted_steps", "plan"):
                if key in result and isinstance(result[key], list):
                    return result[key]
            return [result]
        return []
