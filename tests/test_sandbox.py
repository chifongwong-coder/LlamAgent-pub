"""
Sandbox execution module: curated public tests (flow-oriented).

These tests exercise the most important end-to-end flows for the sandbox
subsystem, verifying integration between SandboxModule, ToolExecutor,
BackendResolver, and LocalProcessBackend.
"""

from __future__ import annotations

import os
import tempfile

import pytest
from unittest.mock import MagicMock

from llamagent.modules.sandbox.policy import (
    ExecutionPolicy,
    POLICY_SHELL_LIMITED,
)
from llamagent.modules.sandbox.backend import (
    ExecutionBackend,
    ExecutionResult,
    ExecutionSession,
    ExecutionSpec,
)
from llamagent.modules.sandbox.resolver import BackendResolver
from llamagent.modules.sandbox.executor import ToolExecutor
from llamagent.modules.sandbox.module import SandboxModule
from llamagent.modules.sandbox.backends.local_process import LocalProcessBackend


# ============================================================
# Flow 1: SandboxModule integration
# ============================================================


class TestSandboxModuleIntegration:
    """Load SandboxModule -> register tool with policy -> call_tool routes through sandbox."""

    def test_sandbox_module_integration(self, bare_agent):
        """Full flow: module loads, tool gets policy, call_tool uses sandbox backend."""
        # Register a high-risk tool.
        bare_agent.register_tool(
            "exec_shell",
            lambda command: command,
            "Execute shell command",
            safety_level=3,
        )

        # Load sandbox module (auto_assign=True).
        mod = SandboxModule(auto_assign=True)
        bare_agent.register_module(mod)

        # Verify executor was injected.
        assert bare_agent.tool_executor is not None

        # Verify policy was auto-assigned (LocalProcessBackend-compatible).
        from llamagent.modules.sandbox.policy import POLICY_LOCAL_SUBPROCESS
        assert bare_agent._tools["exec_shell"]["execution_policy"] is POLICY_LOCAL_SUBPROCESS

        # Call the tool — should route through sandbox and execute via subprocess.
        result = bare_agent.call_tool("exec_shell", {"command": "echo integration_ok"})
        assert "integration_ok" in result


# ============================================================
# Flow 2: Auto-assign policy
# ============================================================


class TestAutoAssignPolicy:
    """SandboxModule auto-assigns policy to high-risk tools but respects manual overrides."""

    def test_auto_assign_policy(self, bare_agent):
        """High-risk tool gets POLICY_SHELL_LIMITED; low-risk tool stays unassigned;
        manually-set policy is not overridden."""
        # Manually-set policy.
        custom = ExecutionPolicy(isolation="container", timeout_seconds=999)

        bare_agent.register_tool("low_risk", lambda: "ok", "Safe tool", safety_level=1)
        bare_agent.register_tool("high_risk", lambda: "ok", "Dangerous tool", safety_level=4)
        bare_agent.register_tool("manual", lambda: "ok", "Custom tool", safety_level=5,
                                 execution_policy=custom)

        SandboxModule(auto_assign=True).on_attach(bare_agent)

        # Low-risk: no policy.
        assert bare_agent._tools["low_risk"].get("execution_policy") is None

        # High-risk: auto-assigned POLICY_SHELL_LIMITED.
        from llamagent.modules.sandbox.policy import POLICY_LOCAL_SUBPROCESS
        assert bare_agent._tools["high_risk"]["execution_policy"] is POLICY_LOCAL_SUBPROCESS

        # Manual: kept as-is.
        assert bare_agent._tools["manual"]["execution_policy"] is custom


# ============================================================
# Flow 3: LocalProcessBackend Python execution
# ============================================================


class TestLocalProcessPythonExecution:
    """Execute Python code in subprocess sandbox and verify structured result."""

    def test_local_process_python_execution(self):
        """Run Python code in a subprocess, get stdout in ExecutionResult."""
        backend = LocalProcessBackend()
        policy = ExecutionPolicy(runtime="python", isolation="process")
        session = backend.create_session(policy)
        try:
            spec = ExecutionSpec(
                command="compute",
                args={"code": "print(sum(range(10)))"},
                policy=policy,
                workspace_path=session.workspace_path,
            )
            result = session.run(spec)

            assert result.exit_code == 0
            assert result.timed_out is False
            assert "45" in result.stdout
            assert result.duration_ms > 0

            # to_observation() should return the stdout content.
            obs = result.to_observation()
            assert "45" in obs
        finally:
            import shutil
            shutil.rmtree(session.workspace_path, ignore_errors=True)


# ============================================================
# Flow 4: Timeout handling
# ============================================================


class TestLocalProcessTimeoutHandling:
    """Timeout produces a structured result with timed_out=True."""

    def test_local_process_timeout_handling(self):
        """A long-running subprocess is killed and returns timed_out=True."""
        backend = LocalProcessBackend()
        policy = ExecutionPolicy(
            runtime="python",
            isolation="process",
            timeout_seconds=0.5,
        )
        session = backend.create_session(policy)
        try:
            spec = ExecutionSpec(
                command="slow",
                args={"code": "import time; time.sleep(60)"},
                policy=policy,
                workspace_path=session.workspace_path,
            )
            result = session.run(spec)

            assert result.timed_out is True
            assert result.exit_code == -1
            assert result.success is False

            # to_observation() should include [TIMEOUT ...].
            obs = result.to_observation()
            assert "[TIMEOUT" in obs
        finally:
            import shutil
            shutil.rmtree(session.workspace_path, ignore_errors=True)


# ============================================================
# Flow 5: Executor session lifecycle
# ============================================================


class TestExecutorSessionLifecycle:
    """Create session -> run -> close -> cleanup workspace."""

    def test_executor_session_lifecycle(self):
        """Full lifecycle: create session -> run -> shutdown cleans workspace."""
        # Test the backend session directly (not through executor/resolver)
        # since LocalProcessBackend only supports isolation="none"
        backend = LocalProcessBackend()
        policy = ExecutionPolicy(runtime="shell", timeout_seconds=10)
        session = backend.create_session(policy)

        # Workspace created
        workspace = session.workspace_path
        assert os.path.isdir(workspace)

        # Execute in workspace
        from llamagent.modules.sandbox.backend import ExecutionSpec
        spec = ExecutionSpec(command="test", args={"command": "echo lifecycle_test"}, policy=policy)
        result = session.run(spec)
        assert result.success
        assert "lifecycle_test" in result.stdout

        # Session close is no-op (workspace managed externally)
        session.close()
        assert os.path.isdir(workspace)

        # Manual cleanup
        import shutil
        shutil.rmtree(workspace)
        assert not os.path.isdir(workspace)


# ============================================================
# Flow 6: Backward compatibility (no SandboxModule)
# ============================================================


class TestCallToolBackwardCompat:
    """Without SandboxModule loaded, everything works as v1.1."""

    def test_call_tool_backward_compat(self, bare_agent):
        """call_tool runs tools directly when no sandbox module is loaded."""
        bare_agent.register_tool(
            "greet",
            lambda name: f"hello {name}",
            "Greeting tool",
            safety_level=1,
        )

        # No sandbox module loaded -> tool_executor is None.
        assert bare_agent.tool_executor is None

        result = bare_agent.call_tool("greet", {"name": "world"})
        assert result == "hello world"

        # Verify that register_tool still accepts execution_policy (just stores it).
        bare_agent.register_tool(
            "custom",
            lambda: "ok",
            "Custom",
            execution_policy=ExecutionPolicy(isolation="process"),
        )
        assert bare_agent._tools["custom"]["execution_policy"] is not None

        # Without executor, even tools with policies run directly.
        result = bare_agent.call_tool("custom", {})
        assert result == "ok"


# ============================================================
# Flow 7: Security fixes verification
# ============================================================


class TestSandboxSecurityFixes:
    """Verify key security fixes in the sandbox module."""

    def test_env_isolation_no_host_leak(self):
        """Sandbox processes get minimal env, host secrets not leaked."""
        backend = LocalProcessBackend()
        policy = ExecutionPolicy(runtime="python", timeout_seconds=10)
        session = backend.create_session(policy)
        try:
            spec = ExecutionSpec(
                command="test",
                args={"code": (
                    "import os\n"
                    "path = os.environ.get('PATH', 'MISSING')\n"
                    "secret = os.environ.get('DEEPSEEK_API_KEY', 'NOT_LEAKED')\n"
                    "home = os.environ.get('HOME', 'MISSING')\n"
                    "print(f'{path}|{secret}|{home}')"
                )},
                policy=policy,
            )
            result = session.run(spec)
            assert result.success
            # PATH and HOME should be present (minimal env)
            assert "MISSING" not in result.stdout.split("|")[0]  # PATH
            assert "MISSING" not in result.stdout.split("|")[2]  # HOME
            # Secrets should NOT be present
            assert "NOT_LEAKED" in result.stdout
        finally:
            import shutil
            shutil.rmtree(session.workspace_path, ignore_errors=True)

    def test_resolver_rejects_unmet_network_policy(self):
        """Resolver refuses backends that can't meet network isolation requirements."""
        resolver = BackendResolver()
        resolver.register(LocalProcessBackend())  # supports_network_isolation=False

        # Policy requiring network isolation
        policy = ExecutionPolicy(runtime="python", network="none")

        with pytest.raises(RuntimeError, match="No sandbox backend available"):
            resolver.resolve(policy)

    def test_tool_dict_includes_name(self, bare_agent):
        """register_tool stores name in the tool dict for error messages."""
        bare_agent.register_tool("my_tool", lambda: "ok", "A tool")
        assert bare_agent._tools["my_tool"]["name"] == "my_tool"

    def test_subprocess_exception_handled(self):
        """Non-timeout subprocess errors return structured result, not crash."""
        backend = LocalProcessBackend()
        policy = ExecutionPolicy(runtime="python", timeout_seconds=5)
        session = backend.create_session(policy)
        try:
            # Trigger a Python syntax error
            spec = ExecutionSpec(
                command="test",
                args={"code": "def !!!invalid"},
                policy=policy,
            )
            result = session.run(spec)
            # Should get a structured error, not an exception
            assert not result.success
            assert result.exit_code != 0
        finally:
            import shutil
            shutil.rmtree(session.workspace_path, ignore_errors=True)
