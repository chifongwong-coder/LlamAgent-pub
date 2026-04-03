"""Core module: LlamAgent's foundational capabilities, standalone and runnable."""

from llamagent.core.agent import SmartAgent, Module, ExecutionStrategy, SimpleReAct
from llamagent.core.authorization import AuthorizationEngine, AuthorizationResult
from llamagent.core.config import Config
from llamagent.core.hooks import (
    HookCallback,
    HookContext,
    HookEvent,
    HookHandler,
    HookMatcher,
    HookResult,
)
from llamagent.core.llm import LLMClient
from llamagent.core.contract import TaskContract, TaskModeState
from llamagent.core.zone import (
    ApprovalScope,
    ConfirmRequest,
    ConfirmResponse,
    RequestedScope,
    ZoneDecisionItem,
    ZoneEvaluation,
    ZoneVerdict,
)
from llamagent.core.persona import Persona, PersonaManager

__all__ = [
    "SmartAgent",
    "Module",
    "ExecutionStrategy",
    "SimpleReAct",
    "AuthorizationEngine",
    "AuthorizationResult",
    "Config",
    "HookCallback",
    "HookContext",
    "HookEvent",
    "HookHandler",
    "HookMatcher",
    "HookResult",
    "TaskContract",
    "TaskModeState",
    "RequestedScope",
    "ApprovalScope",
    "ConfirmRequest",
    "ConfirmResponse",
    "ZoneDecisionItem",
    "ZoneEvaluation",
    "ZoneVerdict",
    "LLMClient",
    "Persona",
    "PersonaManager",
]
