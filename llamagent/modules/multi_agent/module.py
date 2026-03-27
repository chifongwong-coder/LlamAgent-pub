"""
MultiAgentModule: multi-agent collaboration module.

Registers collaboration tools so the model autonomously decides when to delegate subtasks:
- list_agents:    View available collaboration roles
- create_agent:   Create a temporary collaboration role (valid for current session only)
- delegate:       Delegate a subtask to a specified role for execution

Tools are registered via agent.register_tool() (tier=default), independent of on_execute or keyword detection.
"""

from llamagent.core.agent import Module
from llamagent.modules.multi_agent.orchestrator import AgentRole, MultiAgentOrchestrator


class MultiAgentModule(Module):
    """Multi-agent collaboration module: registers collaboration tools for autonomous task delegation by the model."""

    name = "multi_agent"
    description = "Multi-agent collaboration: can delegate subtasks to expert roles (writer/coder/analyst/researcher)"

    def __init__(self):
        self.orchestrator: MultiAgentOrchestrator | None = None

    def on_attach(self, agent):
        """Initialize orchestrator + register preset roles + register collaboration tools."""
        super().on_attach(agent)
        self.orchestrator = MultiAgentOrchestrator(llm=agent.llm)

        # Register preset roles
        self._register_preset_roles()

        # Register collaboration tools
        self._register_tools()

    # ============================================================
    # Preset Roles
    # ============================================================

    def _register_preset_roles(self):
        """Register preset collaboration roles (permanently available)."""
        preset_roles = [
            AgentRole(
                name="writer",
                description="Content creation and copywriting",
                system_prompt=(
                    "You are a professional content creator skilled at writing clear, well-organized articles and copy. "
                    "You excel at organizing information, understanding reader psychology, and expressing complex concepts in vivid language."
                ),
            ),
            AgentRole(
                name="coder",
                description="Programming and architecture design",
                system_prompt=(
                    "You are a senior software engineer skilled in Python, JavaScript, and other languages for coding and architecture design. "
                    "You focus on code quality, performance optimization, and best practices, and excel at translating complex requirements into elegant code."
                ),
            ),
            AgentRole(
                name="analyst",
                description="Data analysis and problem diagnosis",
                system_prompt=(
                    "You are a data analysis expert skilled at extracting insights from complex information with rigorous logic. "
                    "You excel at quantitative analysis, trend identification, and problem diagnosis, providing data-backed recommendations."
                ),
            ),
            AgentRole(
                name="researcher",
                description="Information gathering and research compilation",
                system_prompt=(
                    "You are a professional researcher skilled at systematically collecting and organizing information. "
                    "You can quickly structure knowledge systems and deliver comprehensive, well-organized research reports."
                ),
            ),
        ]
        for role in preset_roles:
            self.orchestrator.add_role(role)

    # ============================================================
    # Tool Registration
    # ============================================================

    def _register_tools(self):
        """Register collaboration tools to the Agent (tier=default)."""
        self.agent.register_tool(
            name="list_agents",
            func=self._tool_list_agents,
            description="View the list of available collaboration roles and understand each role's position and capabilities.",
            parameters={
                "type": "object",
                "properties": {},
            },
            tier="default",
            safety_level=1,
            pack="multi-agent",
        )

        self.agent.register_tool(
            name="create_agent",
            func=self._tool_create_agent,
            description=(
                "Create a temporary collaboration role (valid for current session only). "
                "Use when existing roles don't cover the required expertise, e.g., legal consultant, UI designer, etc."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Role name (English, unique identifier), e.g., 'lawyer', 'designer'",
                    },
                    "role_description": {
                        "type": "string",
                        "description": "Role's capability description and position, e.g., 'Legal consultation and compliance review'",
                    },
                },
                "required": ["name", "role_description"],
            },
            tier="default",
            safety_level=2,
            pack="multi-agent",
        )

        self.agent.register_tool(
            name="delegate",
            func=self._tool_delegate,
            description=(
                "Delegate a subtask to a specified role for execution. "
                "The role will process the task from its professional perspective and return the result. "
                "Use list_agents first to see available roles."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Description of the subtask to delegate; be as specific and clear as possible",
                    },
                    "agent_name": {
                        "type": "string",
                        "description": "Target role name (e.g., 'writer', 'coder')",
                    },
                },
                "required": ["task", "agent_name"],
            },
            tier="default",
            safety_level=2,
            pack="multi-agent",
        )

    # ============================================================
    # Tool Implementations
    # ============================================================

    def _tool_list_agents(self) -> str:
        """List all available collaboration roles (preset + dynamic)."""
        roles = self.orchestrator.list_roles()
        if not roles:
            return "No collaboration roles are currently available."

        lines = ["Available collaboration roles:"]
        for role in roles:
            tag = " (temporary)" if role.is_dynamic else ""
            lines.append(f"- {role.name}: {role.description}{tag}")
        return "\n".join(lines)

    def _tool_create_agent(self, name: str, role_description: str) -> str:
        """Create a temporary collaboration role (valid for current session only)."""
        # Check if the name already exists
        if self.orchestrator.get_role(name) is not None:
            return f"Error: role '{name}' already exists, please use a different name."

        # Auto-generate system prompt based on description
        system_prompt = (
            f"You are an expert in {role_description}. "
            f"Please use your professional knowledge and experience to handle the tasks assigned to you. "
            f"Provide professional, accurate, and insightful responses."
        )

        role = AgentRole(
            name=name,
            description=role_description,
            system_prompt=system_prompt,
            is_dynamic=True,
        )
        self.orchestrator.add_role(role)

        return f"Temporary role '{name}' ({role_description}) created. You can delegate tasks to it via delegate."

    def _tool_delegate(self, task: str, agent_name: str) -> str:
        """Delegate a subtask to a specified role for execution."""
        return self.orchestrator.delegate(task, agent_name)

    # ============================================================
    # External Interface (programmatic API)
    # ============================================================

    def add_role(self, role: AgentRole) -> None:
        """Programmatic API: add a custom role."""
        if self.orchestrator:
            self.orchestrator.add_role(role)

    def remove_role(self, name: str) -> bool:
        """Programmatic API: remove a role."""
        if self.orchestrator:
            return self.orchestrator.remove_role(name)
        return False

    def delegate(self, task: str, role_name: str) -> str:
        """Programmatic API: directly delegate a task."""
        if self.orchestrator:
            return self.orchestrator.delegate(task, role_name)
        return "Error: orchestrator not initialized."

    def orchestrate(self, task: str) -> str | None:
        """Programmatic API: automatically orchestrate a task."""
        if self.orchestrator:
            return self.orchestrator.orchestrate(task)
        return None
