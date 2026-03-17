"""
Multi-agent collaboration: role definition + orchestrator.

AgentRole: Data structure for collaboration roles
MultiAgentOrchestrator: Manages role registration and task delegation

Design notes:
- In the target architecture, orchestration logic is handled autonomously by the model via the delegate tool
- orchestrate() is retained as a programmatic API, but is no longer the primary entry point
- delegate uses lightweight llm.ask() calls only, no ReAct, no tools
"""

from dataclasses import dataclass, field

from llamagent.core.llm import LLMClient


@dataclass
class AgentRole:
    """
    Agent collaboration role definition.

    Attributes:
        name: Unique role identifier (e.g., "writer", "coder")
        description: Role capability description, used to inform the model about the role's position
        system_prompt: Role's system prompt, used when delegating tasks
        is_dynamic: Whether this is a dynamically created temporary role (valid for current session only)
    """

    name: str
    description: str
    system_prompt: str
    is_dynamic: bool = False


class MultiAgentOrchestrator:
    """
    Multi-agent orchestrator: manages role registration and task delegation.

    Responsibilities:
    - Maintain role registry (preset roles + dynamic roles)
    - Provide delegate() to delegate subtasks to specified roles for execution
    - Provide orchestrate() for automatic orchestration (LLM planning -> dispatch -> summarize)
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._roles: dict[str, AgentRole] = {}

    # ============================================================
    # Role Management
    # ============================================================

    def add_role(self, role: AgentRole) -> None:
        """Register a collaboration role."""
        self._roles[role.name] = role

    def remove_role(self, name: str) -> bool:
        """Remove a collaboration role. Returns True on success, False if not found."""
        if name in self._roles:
            del self._roles[name]
            return True
        return False

    def get_role(self, name: str) -> AgentRole | None:
        """Look up a role by name. Returns None if not found."""
        return self._roles.get(name)

    def list_roles(self) -> list[AgentRole]:
        """List all registered roles."""
        return list(self._roles.values())

    # Legacy interface compatibility
    def register_agent(self, agent: AgentRole) -> None:
        """Register an Agent role (legacy interface, prefer add_role)."""
        self.add_role(agent)

    # ============================================================
    # Task Delegation
    # ============================================================

    def delegate(self, task: str, role_name: str) -> str:
        """
        Delegate a task to a specified role for execution.

        Uses lightweight llm.ask() call: no ReAct, no tools, pure text Q&A.

        Args:
            task: Task description to execute
            role_name: Target role name

        Returns:
            Role execution result text, or error message
        """
        role = self._roles.get(role_name)
        if role is None:
            return f"Error: unknown role '{role_name}', available roles: {', '.join(self._roles.keys())}"

        try:
            return self.llm.ask(task, system=role.system_prompt)
        except Exception as e:
            return f"Error: failed to delegate task to '{role_name}' - {e}"

    # ============================================================
    # Automatic Orchestration (programmatic API)
    # ============================================================

    def orchestrate(self, task: str) -> str | None:
        """
        Automatic orchestration: analyze the task and determine which roles should collaborate.

        Flow: LLM plans execution -> dispatch according to plan -> summarize results.
        In the target architecture, this logic is handled autonomously by the model via the delegate tool.

        Returns:
            Orchestration result string, or None if no roles are available.
        """
        if not self._roles:
            return None

        # Step 1: Have LLM plan the execution
        roles_desc = "\n".join(
            f"- {role.name}: {role.description}"
            for role in self._roles.values()
        )

        try:
            plan = self.llm.ask_json(
                f"User task: {task}\n\n"
                f"Available roles:\n{roles_desc}\n\n"
                f"Please output an execution plan in JSON format:\n"
                f'{{"steps": [{{"agent": "role_name", "subtask": "subtask_description"}}, ...]}}\n'
                f"Only select roles that are truly needed; you don't have to use all of them.",
                system=(
                    "You are a task orchestrator responsible for breaking down complex tasks into subtasks "
                    "and assigning them to appropriate roles. "
                    "Please allocate tasks based on each role's expertise and output an execution plan in JSON format."
                ),
            )
        except Exception:
            # Fallback when parsing fails: let researcher handle the entire task
            plan = {"steps": [{"agent": "researcher", "subtask": task}]}

        # Defensive check
        if (
            not isinstance(plan, dict)
            or "error" in plan
            or not isinstance(plan.get("steps"), list)
        ):
            plan = {"steps": [{"agent": "researcher", "subtask": task}]}

        # Step 2: Execute subtasks according to the plan
        results = []
        for step in plan.get("steps", []):
            role_name = step.get("agent", "")
            subtask = step.get("subtask", "")
            if role_name in self._roles:
                result = self.delegate(subtask, role_name)
                results.append(f"[{role_name}] {subtask}:\n{result}")

        # Step 3: Summarize results
        if not results:
            return None

        try:
            return self.llm.ask(
                f"Task: {task}\n\nExecution results from each role:\n\n"
                + "\n\n".join(results)
                + "\n\nPlease integrate the results from all roles above and provide a complete, coherent final response."
            )
        except Exception as e:
            # On summarization failure, directly concatenate raw results
            return f"Role execution results (summarization failed: {e}):\n\n" + "\n\n".join(results)
