"""
Safety guardrails -- perform security checks on Agent inputs and outputs.

Capabilities:
1. Input filtering (check_input): injection attack detection, harmful content filtering, input length limits
2. Output sanitization (check_output): redact API keys / credentials / phone numbers / ID numbers / bank card numbers
3. Command checking (check_command): blacklist check for execute_command tool
4. Code scanning (scan_code): scan custom tool code and return suggested safety_level
"""

import os
import re
import logging
import threading
from datetime import datetime


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Module-level audit-logger singleton (v3.8.5)
# ----------------------------------------------------------------------
#
# v3.8.1 R7-#22 attempted to fix multi-agent audit log silencing by
# tracking which handlers each SafetyGuard instance added. But the
# original gate ``if not self._logger.handlers:`` meant only the FIRST
# instance ever added a handler; subsequent instances saw existing
# handlers and skipped, leaving their ``_own_handlers`` empty. When the
# first instance's ``on_shutdown`` later removed the only handler, all
# other still-running agents lost their audit log silently.
#
# v3.8.5 switches to a process-lifetime singleton: one handler installed
# at first SafetyGuard construction, never removed by ``on_shutdown``,
# automatically flushed/closed by ``logging.shutdown()`` at process exit.
# This matches the operator expectation of a single ``safety_audit.log``
# file per process and removes the multi-agent silencing class entirely.
_AUDIT_LOGGER_SETUP_LOCK = threading.Lock()
_AUDIT_LOGGER_INITIALIZED = False
_AUDIT_LOGGER_PATH: str | None = None


def _ensure_audit_logger(log_path: str) -> logging.Logger:
    """Install the ``safety_audit`` handler exactly once per process.

    First caller wins on ``log_path``; subsequent callers with a
    different path get a warning so the misconfig is observable
    (vs the silent first-wins behaviour v3.8.1 line-121 produced).
    """
    global _AUDIT_LOGGER_INITIALIZED, _AUDIT_LOGGER_PATH
    audit = logging.getLogger("safety_audit")
    audit.setLevel(logging.INFO)
    with _AUDIT_LOGGER_SETUP_LOCK:
        if _AUDIT_LOGGER_INITIALIZED:
            try:
                same = os.path.abspath(_AUDIT_LOGGER_PATH or "") == os.path.abspath(log_path)
            except (TypeError, ValueError):
                same = (_AUDIT_LOGGER_PATH == log_path)
            if not same and _AUDIT_LOGGER_PATH is not None:
                logger.warning(
                    "Safety audit logger already initialized at %r; "
                    "ignoring requested path %r. Multi-tenant deployments "
                    "share a single audit handler per process.",
                    _AUDIT_LOGGER_PATH, log_path,
                )
            return audit
        try:
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            audit.addHandler(handler)
            _AUDIT_LOGGER_PATH = log_path
        except IOError:
            audit.addHandler(logging.StreamHandler())
            _AUDIT_LOGGER_PATH = "<stderr>"
        _AUDIT_LOGGER_INITIALIZED = True
        return audit


def _reset_audit_logger_for_tests() -> None:
    """Test-only: clear module-level singleton state.

    NOT a public API. Tests that exercise SafetyGuard's logger setup
    must reset this between cases via the conftest autouse fixture;
    production code never calls it.
    """
    global _AUDIT_LOGGER_INITIALIZED, _AUDIT_LOGGER_PATH
    with _AUDIT_LOGGER_SETUP_LOCK:
        audit = logging.getLogger("safety_audit")
        for h in list(audit.handlers):
            try:
                h.close()
            except Exception:
                pass
            audit.removeHandler(h)
        _AUDIT_LOGGER_INITIALIZED = False
        _AUDIT_LOGGER_PATH = None


class SafetyGuard:
    """
    Safety guardrails: perform security checks on Agent inputs and outputs.

    Like airport security -- all content going in and out must be screened
    to ensure no "contraband" (harmful content, injection attacks, etc.).
    """

    def __init__(
        self,
        max_input_length: int = 10000,
        enable_filter: bool = True,
        log_path: str = "safety_audit.log",
    ):
        self.max_input_length = max_input_length
        self.enable_filter = enable_filter
        self._setup_logger(log_path)

        # Sensitive keywords (harmful content requests)
        self._blocked_patterns = [
            r"(?i)(how\s+to|teach\s+me).*(make|create|build).*(bomb|drugs|weapon)",
            r"(?i)(how\s+to).*(hack|attack|crack).*(system|website|server)",
            r"(?i)ignore\s+.*(?:previous|above).*(?:instructions?|rules?|restrictions?)",
            r"(?i)you\s+are\s+now.*(?:no\s+longer|don'?t\s+need\s+to).*(?:follow|obey)",
        ]

        # Injection attack detection patterns
        self._injection_patterns = [
            r"(?i)ignore\s+(all\s+)?previous\s+instructions?",
            r"(?i)disregard\s+(all\s+)?prior\s+",
            r"(?i)your\s+new\s+role\s+is",
            r"(?i)system\s*:\s*you\s+are\s+now",
            r"(?i)\[INST\]|\[/INST\]|<\|system\|>|<\|user\|>",
        ]

        # Command blacklist: high-risk system commands.
        # v3.8.1 R7-#1: extended rm-rf patterns to cover variants pre-fix
        # silently allowed: -rfv, -fr, --recursive --force long form, and
        # split flags like ``rm -r -f /``. Regex still single-token best-
        # effort; users wanting bulletproof shell parsing should run their
        # agent under sandbox with execute_command + execution_policy
        # restricting working directory (see SandboxModule).
        self._blocked_commands = [
            r"\brm\s+(-\w*\s+)*-rf?\w*\b",   # rm -rf, -rfv, etc.
            r"\brm\s+(-\w*\s+)*-fr?\w*\b",   # rm -fr, -frv, etc.
            r"\brm\s+(-\w*\s+)*-r\b.*-f\b",  # rm -r ... -f (split flags)
            r"\brm\s+(-\w*\s+)*-f\b.*-r\b",  # rm -f ... -r (split flags)
            r"\brm\s+--recursive\b.*--force\b",   # long form
            r"\brm\s+--force\b.*--recursive\b",   # long form, swapped
            r"\bmkfs\b",                      # format filesystem
            r"\bdd\s+",                       # dd disk operation
            r"\bshutdown\b",                  # shutdown
            r"\breboot\b",                    # reboot
            r"\binit\s+0\b",                  # shutdown
            r"\bhalt\b",                      # halt
            r"\bpoweroff\b",                  # power off
            r">\s*/dev/sd[a-z]",              # write to disk device
            r"\bchmod\s+(-\w+\s+)*777\b",    # full permissions
            r"\bchown\s+.*root\b",           # change ownership to root
            r":(){ :\|:& };:",               # fork bomb
            r"\bkillall\b",                   # kill all processes
            r"\bpkill\s+-9\b",               # force kill process
        ]

        # High-risk code patterns: used by scan_code() for detection
        self._high_risk_code_patterns = [
            r"\bos\.system\b",
            r"\bsubprocess\b",
            r"\bexec\s*\(",
            r"\beval\s*\(",
            r"\b__import__\s*\(",
            r"\bcompile\s*\(",
        ]

        self._medium_risk_code_patterns = [
            r"\bopen\s*\(",
            r"\.write\s*\(",
            r"\.writelines\s*\(",
            r"\bos\.remove\b",
            r"\bos\.unlink\b",
            r"\bos\.rmdir\b",
            r"\bshutil\b",
            r"\brequests\b",
            r"\burllib\b",
        ]

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------

    def _setup_logger(self, log_path: str) -> None:
        """Bind to the process-wide audit logger singleton.

        v3.8.5: the per-instance ``_own_handlers`` tracking from v3.8.1
        R7-#22 didn't actually solve multi-agent log silencing — only
        the first SafetyGuard ever owned a handler (line-121 gate), so
        its shutdown still removed the only handler. v3.8.5 moves to a
        module-level singleton handler installed once per process and
        never removed by ``on_shutdown`` (Python's ``logging.shutdown``
        flushes at process exit). ``_own_handlers`` is kept as an empty
        list for back-compat introspection.
        """
        self._logger = _ensure_audit_logger(log_path)
        # Back-compat: kept empty so external code that introspected the
        # attribute (e.g., test mocks) doesn't AttributeError. New
        # handlers are owned by the module singleton, not the instance.
        self._own_handlers: list[logging.Handler] = []

    # ------------------------------------------------------------------
    # Input Checking
    # ------------------------------------------------------------------

    def check_input(self, user_input: str) -> dict:
        """
        Check whether user input is safe.

        Check order: input length exceeded -> dangerous content -> injection attack.

        Returns:
            {
                "safe": bool,
                "reason": str,           # Reason for being unsafe (empty when safe)
                "sanitized_input": str    # Sanitized input
            }
        """
        # Check 1: Input length
        if len(user_input) > self.max_input_length:
            self._log_violation("input_too_long", user_input[:100])
            return {
                "safe": False,
                "reason": f"Input too long ({len(user_input)} characters, limit {self.max_input_length}), truncated",
                "sanitized_input": user_input[:self.max_input_length],
            }

        if not self.enable_filter:
            return {"safe": True, "reason": "", "sanitized_input": user_input}

        # Check 2: Sensitive content
        for pattern in self._blocked_patterns:
            if re.search(pattern, user_input):
                self._log_violation("blocked_content", user_input[:200])
                return {
                    "safe": False,
                    "reason": "Input contains disallowed content",
                    "sanitized_input": "",
                }

        # Check 3: Injection attack
        for pattern in self._injection_patterns:
            if re.search(pattern, user_input):
                self._log_violation("injection_attempt", user_input[:200])
                return {
                    "safe": False,
                    "reason": "Possible prompt injection attack detected",
                    "sanitized_input": "",
                }

        return {"safe": True, "reason": "", "sanitized_input": user_input}

    # ------------------------------------------------------------------
    # Output Checking (Sanitization)
    # ------------------------------------------------------------------

    def check_output(self, output: str) -> dict:
        """
        Check whether Agent output is safe (prevent sensitive information leakage).

        Sanitized content includes:
        - API Keys (sk-xxxx format)
        - Credentials (key/token/secret/password = xxx)
        - Phone numbers (11-digit numbers starting with 1)
        - ID numbers (18 digits)
        - Bank card numbers (16-19 digit numbers)

        Returns:
            {"safe": bool, "reason": str, "sanitized_output": str}
        """
        if not self.enable_filter:
            return {"safe": True, "reason": "", "sanitized_output": output}

        sanitized = output
        found_sensitive = False
        reasons = []

        # --- Credential sanitization ---
        credential_patterns = [
            # API Key (e.g., sk-xxxxx)
            (r"(sk-[a-zA-Z0-9]{20,})", "API Key"),
            # Quoted credentials: key="value" or key='value'
            (
                r"(?i)(?:key|token|secret|password)\s*[:=]\s*['\"]([^'\"]{10,})['\"]",
                "Credentials",
            ),
            # Unquoted credentials: key=value (at least 20 chars to avoid false matches)
            (
                r"(?i)(?:key|token|secret|password)\s*[:=]\s*([a-zA-Z0-9_\-]{20,})",
                "Credentials",
            ),
        ]

        for pattern, desc in credential_patterns:
            if re.search(pattern, sanitized):
                found_sensitive = True
                reasons.append(desc)
                sanitized = re.sub(
                    pattern,
                    lambda m: m.group(0).replace(m.group(1), "[REDACTED]") if m.lastindex and m.group(1) else "[REDACTED]",
                    sanitized,
                )
                self._log_violation(f"sensitive_output_{desc}", output[:200])

        # --- Personal information sanitization ---
        pii_patterns = [
            # Phone number: 11-digit number starting with 1 (non-digit boundary)
            (r"(?<!\d)(1[3-9]\d{9})(?!\d)", "Phone number",
             lambda m: m.group(0)[:3] + "****" + m.group(0)[-4:]),
            # ID number: 18 digits (last digit can be X)
            (r"(?<!\d)(\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])(?!\d)",
             "ID number",
             lambda m: m.group(0)[:6] + "********" + m.group(0)[-4:]),
            # Bank card number: 16-19 digit pure number (non-digit boundary)
            (r"(?<!\d)(\d{16,19})(?!\d)", "Bank card number",
             lambda m: m.group(0)[:4] + " **** **** " + m.group(0)[-4:]),
        ]

        for pattern, desc, replacer in pii_patterns:
            if re.search(pattern, sanitized):
                found_sensitive = True
                reasons.append(desc)
                sanitized = re.sub(pattern, replacer, sanitized)
                self._log_violation(f"sensitive_output_{desc}", output[:200])

        if found_sensitive:
            reason_text = ", ".join(sorted(set(reasons)))
            return {
                "safe": False,
                "reason": f"Output contains sensitive information ({reason_text}), sanitized",
                "sanitized_output": sanitized,
            }

        return {"safe": True, "reason": "", "sanitized_output": output}

    # ------------------------------------------------------------------
    # Command Blacklist Checking
    # ------------------------------------------------------------------

    def check_command(self, cmd: str) -> str | None:
        """
        Check whether a shell command is on the blacklist.

        For internal use by high-risk tools like execute_command.

        Args:
            cmd: Shell command to check

        Returns:
            None means passed (safe), str is the rejection reason
        """
        for pattern in self._blocked_commands:
            if re.search(pattern, cmd):
                self._log_violation("blocked_command", cmd[:200])
                return f"Command rejected by security policy: high-risk operation detected. Commands matching pattern '{pattern}' are prohibited."

        return None

    # ------------------------------------------------------------------
    # Permission Checking
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Code Scanning
    # ------------------------------------------------------------------

    def scan_code(self, code: str) -> int:
        """
        Scan custom tool code and return suggested safety_level.

        Detection rules:
        - Detected os.system / subprocess / exec() / eval() etc. -> 3 (high risk)
        - Detected open() / file write / network requests etc. -> 2 (has side effects)
        - Pure computation functions with no dangerous calls -> 1 (read-only)

        Args:
            code: Python function code string

        Returns:
            Suggested safety_level (1 / 2 / 3)
        """
        # High risk detection
        for pattern in self._high_risk_code_patterns:
            if re.search(pattern, code):
                self._log_violation("high_risk_code", code[:200])
                return 3

        # Medium risk detection
        for pattern in self._medium_risk_code_patterns:
            if re.search(pattern, code):
                return 2

        # Safe pure computation code
        return 1

    # ------------------------------------------------------------------
    # Audit Log
    # ------------------------------------------------------------------

    def _log_violation(self, violation_type: str, content: str) -> None:
        """Record security violation event to audit log."""
        self._logger.warning(
            f"VIOLATION | type={violation_type} | content={content[:200]}"
        )
