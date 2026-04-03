"""
Event Hook system: tool-level and lifecycle-level hooks.

Complements the existing pipeline callbacks (on_input/on_context/on_output) which are
turn-level. Event hooks fire per tool call and at lifecycle boundaries.

Core types:
- HookEvent:        Enum of all hook event types
- HookContext:      Data passed to hook handlers
- HookResult:       Handler return value (CONTINUE / SKIP / MODIFY)
- HookMatcher:      Filter conditions for selective hook firing
- HookHandler:      Abstract base for handler implementations
- CallableHandler:  Python callable handler
- ShellHandler:     Shell command handler (subprocess via env vars)
- HookRegistration: Internal record of a registered hook
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from llamagent.core.agent import LlamAgent

logger = logging.getLogger(__name__)


# ======================================================================
# Hook event types
# ======================================================================


class HookEvent(str, Enum):
    """All supported hook event types."""

    # Tool-level — fires per tool call
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    TOOL_ERROR = "tool_error"

    # Turn-level — fires per chat() call
    PRE_CHAT = "pre_chat"
    POST_CHAT = "post_chat"

    # Lifecycle — fires per session
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # Planning-level — reserved, not emitted yet
    PLAN_CREATED = "plan_created"
    STEP_START = "step_start"
    STEP_END = "step_end"
    REPLAN = "replan"

    # Authorization-level — v1.9.4
    SCOPE_ISSUED = "scope_issued"
    SCOPE_USED = "scope_used"
    SCOPE_DENIED = "scope_denied"
    SCOPE_REVOKED = "scope_revoked"


# Events that support SKIP (blocking the operation)
_SKIPPABLE_EVENTS = frozenset({HookEvent.PRE_TOOL_USE})


# ======================================================================
# Hook data structures
# ======================================================================


class HookResult(str, Enum):
    """Return value from hook handlers."""

    CONTINUE = "continue"   # Allow the operation to proceed
    SKIP = "skip"           # Block the operation (only effective for PRE_TOOL_USE)
    MODIFY = "modify"       # Reserved: ctx.data has been modified, continue


@dataclass
class HookMatcher:
    """Filter conditions for selective hook firing. All conditions use AND logic."""

    tool_name: str | None = None
    tool_names: list[str] | None = None
    pack: str | None = None
    safety_level: int | None = None

    def matches(self, data: dict) -> bool:
        """Check if the event data matches all conditions."""
        if self.tool_name is not None:
            if data.get("tool_name") != self.tool_name:
                return False

        if self.tool_names is not None:
            if data.get("tool_name") not in self.tool_names:
                return False

        if self.pack is not None:
            tool_info = data.get("tool_info")
            if tool_info is None or tool_info.get("pack") != self.pack:
                return False

        if self.safety_level is not None:
            tool_info = data.get("tool_info")
            if tool_info is None or tool_info.get("safety_level") != self.safety_level:
                return False

        return True


@dataclass
class HookContext:
    """Data passed to hook handlers."""

    agent: LlamAgent
    event: HookEvent
    data: dict
    matcher: HookMatcher | None = None


# Convenience type for registering hooks with plain callables
HookCallback = Callable[[HookContext], HookResult | None]


# ======================================================================
# Hook handlers (Strategy pattern)
# ======================================================================


class HookHandler(ABC):
    """
    Abstract base for hook handler implementations.

    v1.8 implements CallableHandler and ShellHandler.
    Future versions may add HttpHandler, AgentHandler, etc.
    """

    @abstractmethod
    def execute(self, ctx: HookContext) -> HookResult:
        """Execute the hook and return a result."""
        ...


class CallableHandler(HookHandler):
    """Python callable handler."""

    def __init__(self, func: HookCallback):
        self.func = func

    def execute(self, ctx: HookContext) -> HookResult:
        try:
            result = self.func(ctx)
            if result is None:
                return HookResult.CONTINUE
            return result
        except Exception as e:
            logger.warning("Callable hook error: %s", e)
            return HookResult.CONTINUE  # Hook failure does not block main flow


class ShellHandler(HookHandler):
    """
    Shell command handler. Passes event data via environment variables.

    Exit code semantics (only effective for PRE_TOOL_USE):
    - 0 = CONTINUE (allow)
    - non-0 = SKIP (block; stdout becomes the rejection reason)
    """

    def __init__(self, command: str, timeout: float = 30.0):
        self.command = command
        self.timeout = timeout

    def execute(self, ctx: HookContext) -> HookResult:
        return _execute_shell_hook(
            self.command,
            ctx.event,
            ctx.data,
            cwd=ctx.agent.project_dir,
            timeout=self.timeout,
        )


# Keys to exclude from shell environment variables (internal objects, not serializable)
_SHELL_EXCLUDE_KEYS = frozenset({"tool_info"})


def _serialize_env_value(value: Any) -> str:
    """Serialize a value for use as a shell environment variable."""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _execute_shell_hook(
    shell_cmd: str,
    event: HookEvent,
    data: dict,
    *,
    cwd: str | None = None,
    timeout: float = 30.0,
) -> HookResult:
    """
    Run a shell command with event data passed as environment variables.

    Environment variable naming: HOOK_{KEY_UPPER} for each key in data.
    Special handling:
    - $HOOK_EVENT is always set from the event type
    - dict/list values are JSON-serialized (not Python repr)
    - Internal keys (tool_info) are excluded
    - Values are truncated to 2000 characters
    """
    env = os.environ.copy()
    env["HOOK_EVENT"] = event.value

    for key, value in data.items():
        if key in _SHELL_EXCLUDE_KEYS:
            continue
        env_key = f"HOOK_{key.upper()}"
        env[env_key] = _serialize_env_value(value)[:2000]

    try:
        result = subprocess.run(
            shell_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        if result.returncode == 0:
            return HookResult.CONTINUE
        else:
            return HookResult.SKIP  # Non-0 = block (caller checks skippable)
    except subprocess.TimeoutExpired:
        logger.warning("Shell hook timed out (%.0fs): %s", timeout, shell_cmd)
        return HookResult.CONTINUE
    except Exception as e:
        logger.warning("Shell hook failed: %s", e)
        return HookResult.CONTINUE  # Hook failure does not block main flow


# ======================================================================
# Hook registration record
# ======================================================================


@dataclass
class HookRegistration:
    """Internal record of a registered hook."""

    event: HookEvent
    handler: HookHandler
    matcher: HookMatcher | None
    priority: int       # Lower = earlier execution. Code default=100, YAML default=200
    source: str         # "code" / "yaml" / future: "plugin" / "policy"
