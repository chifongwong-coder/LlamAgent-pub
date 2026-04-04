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


class TestExecutorRunAndSession:
    """ToolExecutor.run_command() (basic, cwd, timeout, error) + session lifecycle."""

    def test_executor_run_and_session(self):
        """run_command basic execution, cwd routing, timeout, error handling,
        and full session lifecycle (create -> run -> close -> cleanup)."""
        resolver = BackendResolver()
        resolver.register(LocalProcessBackend())
        executor = ToolExecutor(resolver)

        # --- Basic run_command ---
        with tempfile.TemporaryDirectory() as cwd:
            result = executor.run_command("echo run_command_test", cwd, timeout=10)
            assert "run_command_test" in result

        # --- run_command uses cwd ---
        with tempfile.TemporaryDirectory() as cwd:
            open(os.path.join(cwd, "marker.txt"), "w").close()
            result = executor.run_command("ls marker.txt", cwd, timeout=10)
            assert "marker.txt" in result

        # --- run_command timeout ---
        with tempfile.TemporaryDirectory() as cwd:
            result = executor.run_command("sleep 60", cwd, timeout=0.5)
            assert "TIMEOUT" in result

        # --- run_command error ---
        with tempfile.TemporaryDirectory() as cwd:
            result = executor.run_command("exit 1", cwd, timeout=10)
            assert "exit_code" in result.lower() or result == "" or "1" in result

        # --- Session lifecycle: create -> run -> close -> cleanup ---
        backend = LocalProcessBackend()
        policy = ExecutionPolicy(runtime="shell", timeout_seconds=10)
        session = backend.create_session(policy)

        workspace = session.workspace_path
        assert os.path.isdir(workspace)

        spec = ExecutionSpec(command="test", args={"command": "echo lifecycle_test"}, policy=policy)
        run_result = session.run(spec)
        assert run_result.success
        assert "lifecycle_test" in run_result.stdout

        session.close()
        assert os.path.isdir(workspace)

        import shutil
        shutil.rmtree(workspace)
        assert not os.path.isdir(workspace)


class TestSandboxSecurity:
    """Security: env isolation, resolver rejects unmet policy, tool dict name, subprocess error."""

    def test_sandbox_security(self, bare_agent):
        """Sandbox processes get minimal env (no host secrets); resolver rejects
        unmet network policy; register_tool stores name; subprocess errors return
        structured result."""
        # --- Env isolation ---
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
            assert "MISSING" not in result.stdout.split("|")[0]  # PATH
            assert "MISSING" not in result.stdout.split("|")[2]  # HOME
            assert "NOT_LEAKED" in result.stdout
        finally:
            import shutil
            shutil.rmtree(session.workspace_path, ignore_errors=True)

        # --- Resolver rejects unmet network policy ---
        resolver = BackendResolver()
        resolver.register(LocalProcessBackend())
        net_policy = ExecutionPolicy(runtime="python", network="none")
        with pytest.raises(RuntimeError, match="No sandbox backend available"):
            resolver.resolve(net_policy)

        # --- Tool dict includes name ---
        bare_agent.register_tool("my_tool", lambda: "ok", "A tool")
        assert bare_agent._tools["my_tool"]["name"] == "my_tool"

        # --- Subprocess exception handled ---
        backend2 = LocalProcessBackend()
        policy2 = ExecutionPolicy(runtime="python", timeout_seconds=5)
        session2 = backend2.create_session(policy2)
        try:
            spec2 = ExecutionSpec(
                command="test",
                args={"code": "def !!!invalid"},
                policy=policy2,
            )
            result2 = session2.run(spec2)
            assert not result2.success
            assert result2.exit_code != 0
        finally:
            import shutil
            shutil.rmtree(session2.workspace_path, ignore_errors=True)


class TestSandboxIntegrationAndCompat:
    """SandboxModule integration + call_tool backward compat + auto-assign policy."""

    def test_sandbox_integration_and_compat(self, bare_agent):
        """Full flow: module loads, tool gets policy, call_tool uses sandbox backend.
        Without SandboxModule, call_tool runs tools directly. Auto-assign respects
        manual overrides."""
        # --- Module integration ---
        bare_agent.register_tool(
            "exec_shell", lambda command: command,
            "Execute shell command", safety_level=3,
        )
        mod = SandboxModule(auto_assign=True)
        bare_agent.register_module(mod)

        assert bare_agent.tool_executor is not None
        from llamagent.modules.sandbox.policy import POLICY_LOCAL_SUBPROCESS
        assert bare_agent._tools["exec_shell"]["execution_policy"] is POLICY_LOCAL_SUBPROCESS

        result = bare_agent.call_tool("exec_shell", {"command": "echo integration_ok"})
        assert "integration_ok" in result

        # --- Auto-assign policy ---
        # Use a fresh agent for auto-assign test
        from llamagent.core.agent import LlamAgent, SimpleReAct
        from llamagent.core.config import Config
        from llamagent.core.authorization import AuthorizationEngine
        import os

        config2 = Config.__new__(Config)
        for attr in vars(bare_agent.config):
            setattr(config2, attr, getattr(bare_agent.config, attr))

        agent2 = LlamAgent.__new__(LlamAgent)
        agent2.config = config2
        agent2.persona = None
        agent2.llm = bare_agent.llm
        agent2.modules = {}
        agent2.history = []
        agent2.summary = None
        agent2.conversation = agent2.history
        agent2._execution_strategy = SimpleReAct()
        agent2.confirm_handler = None
        agent2.interaction_handler = None
        agent2._confirm_wait_time = 0.0
        agent2.project_dir = os.path.realpath(os.getcwd())
        agent2.playground_dir = os.path.realpath(os.path.join(agent2.project_dir, "llama_playground"))
        agent2.tool_executor = None
        agent2._tools = {}
        agent2._active_packs = set()
        agent2._tools_version = 0
        agent2._hooks = {}
        agent2._session_started = False
        agent2._in_hook = False
        agent2.mode = "interactive"
        agent2._controller = None
        agent2._authorization_engine = AuthorizationEngine(agent2)

        custom = ExecutionPolicy(isolation="container", timeout_seconds=999)
        agent2.register_tool("low_risk", lambda: "ok", "Safe tool", safety_level=1)
        agent2.register_tool("high_risk", lambda: "ok", "Dangerous tool", safety_level=4)
        agent2.register_tool("manual", lambda: "ok", "Custom tool", safety_level=5,
                             execution_policy=custom)

        SandboxModule(auto_assign=True).on_attach(agent2)

        assert agent2._tools["low_risk"].get("execution_policy") is None
        assert agent2._tools["high_risk"]["execution_policy"] is POLICY_LOCAL_SUBPROCESS
        assert agent2._tools["manual"]["execution_policy"] is custom

        # --- Backward compat: no sandbox module ---
        agent3 = LlamAgent.__new__(LlamAgent)
        agent3.config = config2
        agent3.persona = None
        agent3.llm = bare_agent.llm
        agent3.modules = {}
        agent3.history = []
        agent3.summary = None
        agent3.conversation = agent3.history
        agent3._execution_strategy = SimpleReAct()
        agent3.confirm_handler = None
        agent3.interaction_handler = None
        agent3._confirm_wait_time = 0.0
        agent3.project_dir = os.path.realpath(os.getcwd())
        agent3.playground_dir = os.path.realpath(os.path.join(agent3.project_dir, "llama_playground"))
        agent3.tool_executor = None
        agent3._tools = {}
        agent3._active_packs = set()
        agent3._tools_version = 0
        agent3._hooks = {}
        agent3._session_started = False
        agent3._in_hook = False
        agent3.mode = "interactive"
        agent3._controller = None
        agent3._authorization_engine = AuthorizationEngine(agent3)

        agent3.register_tool("greet", lambda name: f"hello {name}", "Greeting tool", safety_level=1)
        assert agent3.tool_executor is None
        greet_result = agent3.call_tool("greet", {"name": "world"})
        assert greet_result == "hello world"

        agent3.register_tool("custom", lambda: "ok", "Custom",
                             execution_policy=ExecutionPolicy(isolation="process"))
        assert agent3._tools["custom"]["execution_policy"] is not None
        custom_result = agent3.call_tool("custom", {})
        assert custom_result == "ok"


class TestLocalProcessExecution:
    """LocalProcessBackend: Python execution + timeout handling."""

    def test_local_process_execution(self):
        """Run Python code in subprocess, verify structured result.
        Long-running subprocess killed with timed_out=True."""
        # --- Python execution ---
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

            obs = result.to_observation()
            assert "45" in obs
        finally:
            import shutil
            shutil.rmtree(session.workspace_path, ignore_errors=True)

        # --- Timeout handling ---
        backend2 = LocalProcessBackend()
        timeout_policy = ExecutionPolicy(
            runtime="python", isolation="process", timeout_seconds=0.5,
        )
        session2 = backend2.create_session(timeout_policy)
        try:
            spec2 = ExecutionSpec(
                command="slow",
                args={"code": "import time; time.sleep(60)"},
                policy=timeout_policy,
                workspace_path=session2.workspace_path,
            )
            result2 = session2.run(spec2)

            assert result2.timed_out is True
            assert result2.exit_code == -1
            assert result2.success is False

            obs2 = result2.to_observation()
            assert "[TIMEOUT" in obs2
        finally:
            import shutil
            shutil.rmtree(session2.workspace_path, ignore_errors=True)
