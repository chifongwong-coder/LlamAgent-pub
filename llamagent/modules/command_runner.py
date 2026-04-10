"""
CommandRunner: shared low-level command execution infrastructure.

Provides subprocess execution with timeout, output capture, and environment
isolation. Used by sandbox LocalProcessSession and (future) ProcessRunner.

This is NOT a pluggable Module — it does not participate in the callback
pipeline. It is a shared utility, similar in positioning to fs_store/.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass


@dataclass
class CommandResult:
    """Structured result of a command execution."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: float = 0
    timed_out: bool = False

    @property
    def success(self) -> bool:
        """True if the command completed normally with exit code 0."""
        return self.exit_code == 0 and not self.timed_out


class CommandRunner:
    """Low-level command execution: subprocess with timeout, output capture, env isolation.

    All methods are static — this class holds no state.
    """

    @staticmethod
    def run(
        cmd: list[str],
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        inherit_env: bool = False,
    ) -> CommandResult:
        """Run a command synchronously and return a CommandResult.

        Args:
            cmd: Command and arguments as a list of strings.
            cwd: Working directory for the subprocess.
            timeout: Maximum execution time in seconds (None = no limit).
            env: Environment variables for the subprocess. Behavior depends
                on *inherit_env*.
            inherit_env: If False (default), *env* is passed directly to
                subprocess (caller should use build_safe_env() for isolation).
                If True, start from the current process environment and overlay
                *env* on top.

        Returns:
            A CommandResult with stdout, stderr, exit_code, timing, and
            timeout information.
        """
        if inherit_env:
            effective_env = dict(os.environ)
            if env:
                effective_env.update(env)
        else:
            # env=None would cause subprocess to inherit full host environment,
            # so default to safe env for isolation
            effective_env = env if env is not None else CommandRunner.build_safe_env()

        start_time = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=effective_env,
            )
            duration_ms = (time.monotonic() - start_time) * 1000

            return CommandResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                duration_ms=duration_ms,
            )

        except subprocess.TimeoutExpired as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            return CommandResult(
                stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
                stderr=exc.stderr or "" if isinstance(exc.stderr, str) else "",
                exit_code=-1,
                duration_ms=duration_ms,
                timed_out=True,
            )

        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            return CommandResult(
                stderr=f"Subprocess execution failed: {exc}",
                exit_code=-1,
                duration_ms=duration_ms,
            )

    @staticmethod
    def start(
        cmd: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        inherit_env: bool = False,
    ) -> subprocess.Popen:
        """Start a command without waiting. Returns a Popen handle.

        The caller is responsible for communicating with and reaping the process.
        Environment handling is identical to run(): *inherit_env* controls whether
        the current process environment is inherited.

        Args:
            cmd: Command and arguments as a list of strings.
            cwd: Working directory for the subprocess.
            env: Environment variables for the subprocess.
            inherit_env: If True, start from the current process environment
                and overlay *env* on top. If False, use *env* directly (or
                build_safe_env() when *env* is None).

        Returns:
            A subprocess.Popen instance with stdout=PIPE, stderr=PIPE, text=True.
        """
        if inherit_env:
            effective_env = dict(os.environ)
            if env:
                effective_env.update(env)
        else:
            effective_env = env if env is not None else CommandRunner.build_safe_env()

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=effective_env,
        )

    @staticmethod
    def build_safe_env(extra: dict[str, str] | None = None) -> dict[str, str]:
        """Build a minimal safe environment for subprocess isolation.

        Includes only PATH, HOME, LANG, and TERM from the host environment
        (with sensible fallbacks). Never leaks API keys, credentials, or
        other sensitive environment variables.

        Args:
            extra: Additional variables to merge on top of the base set.

        Returns:
            A dict suitable for passing as *env* to CommandRunner.run().
        """
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "TERM": os.environ.get("TERM", "xterm"),
        }
        if extra:
            env.update(extra)
        return env
