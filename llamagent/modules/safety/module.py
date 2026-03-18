"""
SafetyModule: safety module providing three-layer security mechanism.

1. Input filtering (on_input): intercept injection attacks and harmful content
2. Tool permission checking (pre_call_check): compare safety_level with permission_level
3. Output sanitization (on_output): redact API keys, credentials, personal information, and other sensitive content

Design principles:
- Interface layers (CLI / Web / API) automatically load the safety module
- Even when permission_level is set to maximum, input filtering and output sanitization still apply
- When on_input returns an empty string, agent.chat() short-circuits with a rejection response
"""

from pathlib import Path

from llamagent.core.agent import Module
from llamagent.modules.safety.guard import SafetyGuard


class SafetyModule(Module):
    """Safety module: input filtering + permission checking + output sanitization."""

    name = "safety"
    description = "Safety guardrails: injection defense, permission control, sensitive information redaction"

    def __init__(self):
        self.guard: SafetyGuard | None = None

    def on_attach(self, agent) -> None:
        """
        Initialization when module is attached to Agent.

        1. Create SafetyGuard instance
        2. Inject pre_call_check callback into agent (reserved; agent.call_tool not yet implemented)
        """
        super().on_attach(agent)

        # Create audit log directory
        output_dir = Path(agent.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.guard = SafetyGuard(
            max_input_length=10000,
            enable_filter=True,
            log_path=str(output_dir / "safety_audit.log"),
        )

        # Inject permission check callback (reserved: takes effect after agent.call_tool is implemented)
        if hasattr(agent, "pre_call_check"):
            agent.pre_call_check = self._check_permission


    # ------------------------------------------------------------------
    # Pipeline Hooks
    # ------------------------------------------------------------------

    def on_input(self, user_input: str) -> str:
        """
        Input filtering hook. Applies regardless of permission_level.

        Return values:
        - Safe input -> original text
        - Overly long input -> truncated input (continue processing)
        - Injection attack / dangerous content -> empty string (agent.chat short-circuits with rejection)
        """
        if self.guard is None:
            return user_input

        result = self.guard.check_input(user_input)
        if not result["safe"]:
            print(f"  [Safety] {result['reason']}")
            if not result["sanitized_input"]:
                # Injection attack or dangerous content: return empty string, agent.chat() will short-circuit
                print(f"[Safety] Unsafe input intercepted: {result['reason']}")
                return ""
            # Overly long input: return truncated text, continue processing
            return result["sanitized_input"]
        return user_input

    def on_output(self, response: str) -> str:
        """
        Output sanitization hook. Applies regardless of permission_level.

        Checks and redacts sensitive information in output such as API keys, credentials, phone numbers, ID numbers, bank card numbers.
        """
        if self.guard is None:
            return response

        result = self.guard.check_output(response)
        if not result["safe"]:
            print(f"  [Safety] {result['reason']}")
            return result["sanitized_output"]
        return response

    def on_shutdown(self) -> None:
        """Close audit log file handler to prevent resource leaks."""
        if self.guard and hasattr(self.guard, '_logger'):
            for handler in self.guard._logger.handlers[:]:
                handler.close()
                self.guard._logger.removeHandler(handler)

    # ------------------------------------------------------------------
    # Command Checking (for internal tool use)
    # ------------------------------------------------------------------

    def check_command(self, cmd: str) -> str | None:
        """
        Command blacklist check, for internal use by high-risk tools like execute_command.

        Usage:
            safety = agent.get_module("safety")
            if safety:
                rejection = safety.check_command(cmd)
                if rejection:
                    return rejection

        Args:
            cmd: Shell command to check

        Returns:
            None means passed (safe), str is the rejection reason
        """
        if self.guard is None:
            return None
        return self.guard.check_command(cmd)

    # ------------------------------------------------------------------
    # Permission Checking (injected into agent.pre_call_check)
    # ------------------------------------------------------------------

    def _check_permission(self, tool, kwargs) -> str | None:
        """
        Permission check function, injected into the agent.pre_call_check slot.

        Compares tool's safety_level with role's permission_level:
        - permission_level >= safety_level -> allow (return None)
        - permission_level <  safety_level -> reject (return reason string)

        Permission source priority:
        1. agent.persona.permission_level (if persona exists and has this attribute)
        2. agent.config.permission_level (fallback default)
        """
        if self.guard is None:
            return None

        # Get tool's safety_level (tool is a dict from agent._tools)
        if isinstance(tool, dict):
            tool_safety_level = tool.get("safety_level", 1)
            tool_name = tool.get("name", "unknown")
        else:
            tool_safety_level = getattr(tool, "safety_level", 1)
            tool_name = getattr(tool, "name", "unknown")

        # Get role's permission_level
        perm = self._get_permission_level()

        result = self.guard.check_permission(tool_safety_level, perm)
        if result:
            return f"Insufficient permissions: tool '{tool_name}' requires permission level {tool_safety_level}, current level is {perm}"

        return None

    def _get_permission_level(self) -> int:
        """
        Get the current role's permission level.

        Priority: persona.permission_level > config.permission_level > default value 1
        """
        # Try to get from persona
        if hasattr(self, "agent") and self.agent:
            if self.agent.persona and hasattr(self.agent.persona, "permission_level"):
                perm = self.agent.persona.permission_level
                if perm is not None:
                    return perm
            # Get from config
            if hasattr(self.agent.config, "permission_level"):
                return self.agent.config.permission_level

        # Fallback default value
        return 1
