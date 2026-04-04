"""Core module: LlamAgent's foundational capabilities, standalone and runnable."""

from llamagent.core.agent import LlamAgent, Module, ExecutionStrategy, SimpleReAct
from llamagent.core.authorization import (
    ApprovalScope,
    AuthorizationEngine,
    AuthorizationResult,
    AuthorizationUpdateResult,
)
from llamagent.core.config import Config
from llamagent.core.contract import (
    AuthorizationUpdate,
    PipelineOutcome,
    TaskContract,
    TaskModeState,
)
from llamagent.core.controller import ModeAction, ModeController, TaskModeController
from llamagent.core.hooks import (
    HookCallback,
    HookContext,
    HookEvent,
    HookHandler,
    HookMatcher,
    HookResult,
)
from llamagent.core.llm import LLMClient
from llamagent.core.zone import (
    ConfirmRequest,
    ConfirmResponse,
    RequestedScope,
    ZoneDecisionItem,
    ZoneEvaluation,
    ZoneVerdict,
)
from llamagent.core.persona import Persona, PersonaManager

__all__ = [
    "LlamAgent",
    "Module",
    "ExecutionStrategy",
    "SimpleReAct",
    "AuthorizationEngine",
    "AuthorizationResult",
    "AuthorizationUpdate",
    "AuthorizationUpdateResult",
    "ApprovalScope",
    "Config",
    "HookCallback",
    "HookContext",
    "HookEvent",
    "HookHandler",
    "HookMatcher",
    "HookResult",
    "ModeAction",
    "ModeController",
    "TaskModeController",
    "PipelineOutcome",
    "TaskContract",
    "TaskModeState",
    "RequestedScope",
    "ConfirmRequest",
    "ConfirmResponse",
    "ZoneDecisionItem",
    "ZoneEvaluation",
    "ZoneVerdict",
    "LLMClient",
    "Persona",
    "PersonaManager",
]
