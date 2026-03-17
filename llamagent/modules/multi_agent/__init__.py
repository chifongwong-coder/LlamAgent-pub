"""Multi-agent collaboration module: enables the Agent to delegate subtasks to expert roles."""

from llamagent.modules.multi_agent.orchestrator import AgentRole, MultiAgentOrchestrator
from llamagent.modules.multi_agent.module import MultiAgentModule

__all__ = ["AgentRole", "MultiAgentOrchestrator", "MultiAgentModule"]
