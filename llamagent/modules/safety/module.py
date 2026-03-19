"""
SafetyModule: safety module providing two-layer security mechanism.

1. Input filtering (on_input): intercept injection attacks and harmful content
2. Output sanitization (on_output): redact API keys, credentials, personal information, and other sensitive content

Tool execution safety is handled by the core zone system (v1.3), not by this module.
SafetyModule is an optional enhancement that provides on_input and on_output hooks.

Design principles:
- Interface layers (CLI / Web / API) automatically load the safety module
- Input filtering and output sanitization always apply regardless of role
- When on_input returns an empty string, agent.chat() short-circuits with a rejection response
"""

from pathlib import Path

from llamagent.core.agent import Module
from llamagent.modules.safety.guard import SafetyGuard


class SafetyModule(Module):
    """Safety module: input filtering + output sanitization."""

    name = "safety"
    description = "Safety guardrails: injection defense, sensitive information redaction"

    def __init__(self):
        self.guard: SafetyGuard | None = None

    def on_attach(self, agent) -> None:
        """
        Initialization when module is attached to Agent.

        Creates SafetyGuard instance for input filtering and output sanitization.
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


    # ------------------------------------------------------------------
    # Pipeline Hooks
    # ------------------------------------------------------------------

    def on_input(self, user_input: str) -> str:
        """
        Input filtering hook.

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
        Output sanitization hook.

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
