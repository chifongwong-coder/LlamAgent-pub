"""
ExecutionPolicy: backend-agnostic execution requirements for tools.

An ExecutionPolicy declares what isolation and resource constraints a tool
execution needs, without specifying *how* those constraints are enforced.
The BackendResolver maps policies to concrete backends at runtime.

Fields:
    runtime          Language runtime: "python" | "shell" | "wasm" (reserved) | "native"
    isolation         Isolation level: "none" | "process" | "container" | "microvm"
    filesystem        FS access model: "host" | "read_only" | "task_workspace" | "isolated_rw"
    network           Network access: "none" | "allowlist" | "full"
    session_mode      Session lifetime: "one_shot" | "task_session"
    timeout_seconds   Max wall-clock seconds (None = no limit)
    max_memory_mb     Memory cap in MB (None = no limit)
    max_cpu_cores     CPU core limit (None = no limit)
    max_artifact_bytes  Max total artifact size in bytes (None = no limit)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutionPolicy:
    """Backend-agnostic execution requirements for a single tool invocation."""

    runtime: str = "python"
    isolation: str = "none"
    filesystem: str = "host"
    network: str = "full"
    session_mode: str = "one_shot"
    timeout_seconds: float | None = None
    max_memory_mb: int | None = None
    max_cpu_cores: float | None = None
    max_artifact_bytes: int | None = None


# ---------------------------------------------------------------------------
# Preset constants
# ---------------------------------------------------------------------------

# Default: same as v1.1 — direct host execution, no isolation.
POLICY_HOST = ExecutionPolicy()

# Read-only sandbox: process isolation, read-only FS, no network.
POLICY_READONLY = ExecutionPolicy(
    isolation="process",
    filesystem="read_only",
    network="none",
    timeout_seconds=30,
)

# Untrusted code execution: process isolation, task workspace, no network.
POLICY_UNTRUSTED_CODE = ExecutionPolicy(
    isolation="process",
    filesystem="task_workspace",
    network="none",
    timeout_seconds=60,
    max_memory_mb=512,
)

# Limited shell execution: shell runtime, process isolation, task workspace.
POLICY_SHELL_LIMITED = ExecutionPolicy(
    runtime="shell",
    isolation="process",
    filesystem="task_workspace",
    network="none",
    timeout_seconds=30,
    max_memory_mb=256,
)

# Sandboxed coder: persistent session, allowlisted network, generous limits.
# Requires a container/microVM backend that supports network isolation.
POLICY_SANDBOXED_CODER = ExecutionPolicy(
    runtime="python",
    isolation="process",
    filesystem="task_workspace",
    network="allowlist",
    session_mode="task_session",
    timeout_seconds=300,
    max_memory_mb=1024,
)

# Local process sandbox: runs in subprocess with timeout but no real isolation.
# This is the only preset compatible with LocalProcessBackend (the default).
# For real isolation, use the stricter presets with Docker/gVisor backends.
POLICY_LOCAL_SUBPROCESS = ExecutionPolicy(
    runtime="shell",
    isolation="none",
    filesystem="host",
    network="full",
    timeout_seconds=30,
)
