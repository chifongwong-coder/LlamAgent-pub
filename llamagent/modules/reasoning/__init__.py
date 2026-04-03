"""Planning module: provides task planning and PlanReAct execution strategy for LlamAgent."""

from llamagent.modules.reasoning.module import PlanningModule

# Backward compatibility: main.py still references the old name ReasoningModule
ReasoningModule = PlanningModule

__all__ = ["PlanningModule", "ReasoningModule"]
