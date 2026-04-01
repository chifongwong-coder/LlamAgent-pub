"""Core module: LlamAgent's foundational capabilities, standalone and runnable."""

from llamagent.core.agent import SmartAgent, Module, ExecutionStrategy, SimpleReAct
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
from llamagent.core.persona import Persona, PersonaManager

__all__ = [
    "SmartAgent",
    "Module",
    "ExecutionStrategy",
    "SimpleReAct",
    "Config",
    "HookCallback",
    "HookContext",
    "HookEvent",
    "HookHandler",
    "HookMatcher",
    "HookResult",
    "LLMClient",
    "Persona",
    "PersonaManager",
]
