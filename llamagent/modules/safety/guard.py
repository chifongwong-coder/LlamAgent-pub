"""
Safety guardrails -- perform security checks on Agent inputs and outputs.

Capabilities:
1. Input filtering (check_input): injection attack detection, harmful content filtering, input length limits
2. Output sanitization (check_output): redact API keys / credentials / phone numbers / ID numbers / bank card numbers
3. Command checking (check_command): blacklist check for execute_command tool
4. Code scanning (scan_code): scan custom tool code and return suggested safety_level
"""

import re
import logging
from datetime import datetime


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

        # Command blacklist: high-risk system commands
        self._blocked_commands = [
            r"\brm\s+(-\w*\s+)*-rf\b",      # rm -rf
            r"\brm\s+(-\w*\s+)*-fr\b",      # rm -fr
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
        """Configure audit logger."""
        self._logger = logging.getLogger("safety_audit")
        self._logger.setLevel(logging.INFO)

        if not self._logger.handlers:
            try:
                handler = logging.FileHandler(log_path, encoding="utf-8")
                formatter = logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(message)s"
                )
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
            except IOError:
                # Cannot write to log file, use console output
                handler = logging.StreamHandler()
                self._logger.addHandler(handler)

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
