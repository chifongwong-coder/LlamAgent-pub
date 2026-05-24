"""Public regression tests for the 6 framework changes that shipped with
the cli-transfer branch (round-17 review HIGH-1).

These deltas are framework-public API additions / behaviour changes
that need their own isolation tests so future refactors don't silently
break them. Round-12 to round-16 reviewer rounds were focused on the
interface layer; the framework deltas had only inline scripts that
ran at commit time and didn't sink into the suite.

Covered (one test class per area):
- TestDisabledTools  — Config.disabled_tools + YAML mapping +
  agent.register_tool guard.
- TestRunnerCallback — ContinuousRunner.on_task_complete fires after
  trigger-driven _run_task, NOT after inject-driven _drain_queue.
- TestTimerFireImmediately — TimerTrigger.fire_immediately flag controls
  whether first poll fires.
- TestScratchExpanduser — _resolve_path expands leading ``~`` /
  ``~user`` while leaving absolute / relative / mid-string ``~``
  alone.
- TestSystemInfo — system_info builtin returns the expected JSON
  fields and is registered to the global registry at tier=default.
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Config.disabled_tools + agent.register_tool guard (commit 1b4d062)
# ---------------------------------------------------------------------------


class TestDisabledTools:
    def test_default_empty_list(self):
        from llamagent.core import Config
        assert Config().disabled_tools == []

    def test_yaml_mapping_tools_disabled(self):
        from llamagent.core import Config
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("tools:\n  disabled: [ask_user, web_search]\n")
            p = f.name
        try:
            c = Config(config_path=p)
            assert c.disabled_tools == ["ask_user", "web_search"]
        finally:
            os.unlink(p)

    def test_register_tool_skips_disabled(self):
        """A tool whose name is in disabled_tools must not enter
        agent._tools — the model never sees it in the schema."""
        from llamagent.core import Config, LlamAgent

        c = Config()
        c.disabled_tools = ["ask_user"]
        agent = LlamAgent(c)
        try:
            agent.register_tool(
                name="ask_user",
                func=lambda question: "x",
                description="t",
                parameters={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
            )
            assert "ask_user" not in agent._tools, "ask_user must be filtered"
        finally:
            agent.shutdown()

    def test_register_tool_keeps_non_disabled(self):
        """Sanity: tools NOT in the list register normally."""
        from llamagent.core import Config, LlamAgent

        c = Config()
        c.disabled_tools = ["ask_user"]
        agent = LlamAgent(c)
        try:
            agent.register_tool(
                name="foo",
                func=lambda: "y",
                description="t",
                parameters={"type": "object", "properties": {}},
            )
            assert "foo" in agent._tools
        finally:
            agent.shutdown()


# ---------------------------------------------------------------------------
# ContinuousRunner.on_task_complete callback (commit 5033e80)
# ---------------------------------------------------------------------------


class TestRunnerCallback:
    def _runner(self, *, callback=None):
        from llamagent.core.runner import ContinuousRunner, TimerTrigger
        agent = MagicMock()
        agent.chat.return_value = "agent reply"
        agent.abort.return_value = None
        return ContinuousRunner(
            agent,
            [TimerTrigger(60, "x")],
            on_task_complete=callback,
        ), agent

    def test_default_is_none(self):
        runner, _ = self._runner()
        assert runner.on_task_complete is None

    def test_callback_fires_on_run_task(self):
        from llamagent.core.runner import TaskLogEntry, TimerTrigger

        fired: list[TaskLogEntry] = []
        runner, agent = self._runner(callback=fired.append)
        runner._run_task("hello", TimerTrigger(60, "x"))

        assert len(fired) == 1
        entry = fired[0]
        assert isinstance(entry, TaskLogEntry)
        assert entry.input == "hello"
        assert entry.output == "agent reply"
        assert entry.status == "completed"
        assert entry.trigger_type == "TimerTrigger"
        assert agent.chat.called

    def test_callback_exception_does_not_crash_runner(self):
        from llamagent.core.runner import TimerTrigger

        def bad_callback(_entry):
            raise RuntimeError("simulated")

        runner, _ = self._runner(callback=bad_callback)
        # Must not raise — runner.py catches + logger.exception.
        runner._run_task("x", TimerTrigger(60, "x"))

    def test_callback_fires_on_error(self):
        """When agent.chat raises, callback still fires with status=error."""
        from llamagent.core.runner import ContinuousRunner, TimerTrigger

        agent = MagicMock()
        agent.chat.side_effect = RuntimeError("simulated chat failure")
        agent.abort.return_value = None
        fired = []
        runner = ContinuousRunner(
            agent, [TimerTrigger(60, "x")], on_task_complete=fired.append,
        )
        runner._run_task("hello", TimerTrigger(60, "x"))
        assert len(fired) == 1
        assert fired[0].status == "error"
        assert "simulated chat failure" in fired[0].error

    def test_inject_path_does_not_fire_callback(self):
        """Inject-driven tasks deliver responses via event.wait() to the
        caller; the trigger callback should NOT fire (it's reserved for
        trigger-driven tasks where there's no other observer)."""
        import threading
        runner, _ = self._runner(callback=lambda e: pytest.fail("should not fire"))
        runner._running = True
        evt = threading.Event()
        result: list = [None]
        runner._normal_queue.put(("hi", evt, result))
        runner._drain_queue(runner._normal_queue)
        assert evt.is_set()
        assert result[0] == "agent reply"


# ---------------------------------------------------------------------------
# TimerTrigger.fire_immediately (commit 51519cd)
# ---------------------------------------------------------------------------


class TestTimerFireImmediately:
    def test_default_first_poll_returns_none(self):
        from llamagent.core.runner import TimerTrigger
        t = TimerTrigger(interval=60, message="hi")
        assert t.fire_immediately is False
        assert t.poll() is None

    def test_fire_immediately_first_poll_returns_message(self):
        from llamagent.core.runner import TimerTrigger
        t = TimerTrigger(interval=60, message="hi", fire_immediately=True)
        assert t.fire_immediately is True
        assert t.poll() == "hi"

    def test_subsequent_polls_within_interval_return_none(self):
        from llamagent.core.runner import TimerTrigger
        t = TimerTrigger(interval=60, message="x", fire_immediately=True)
        assert t.poll() == "x"   # initial fire
        assert t.poll() is None  # within interval

    def test_periodic_continues_after_immediate(self):
        """After the immediate first fire, the timer should still fire
        every interval. Use a small interval to keep the test fast."""
        import time
        from llamagent.core.runner import TimerTrigger
        t = TimerTrigger(interval=0.1, message="ping", fire_immediately=True)
        assert t.poll() == "ping"
        time.sleep(0.15)
        assert t.poll() == "ping"


# ---------------------------------------------------------------------------
# scratch._resolve_path expanduser (commit d4ac7fe)
# ---------------------------------------------------------------------------


class TestScratchExpanduser:
    def test_tilde_expands_to_home(self):
        from llamagent.modules.tools.scratch import _resolve_path
        r = _resolve_path("~/.llamagent/cli_tui.log", base="/tmp")
        assert r == os.path.realpath(
            os.path.expanduser("~/.llamagent/cli_tui.log")
        )

    def test_bare_tilde_expands_to_home(self):
        from llamagent.modules.tools.scratch import _resolve_path
        r = _resolve_path("~", base="/tmp")
        assert r == os.path.realpath(os.path.expanduser("~"))

    def test_absolute_path_unchanged(self):
        from llamagent.modules.tools.scratch import _resolve_path
        r = _resolve_path("/etc/hosts", base="/tmp")
        assert r == os.path.realpath("/etc/hosts")

    def test_relative_path_joined_to_base(self):
        from llamagent.modules.tools.scratch import _resolve_path
        r = _resolve_path("foo/bar", base="/tmp")
        assert r == os.path.realpath("/tmp/foo/bar")

    def test_mid_string_tilde_not_expanded(self):
        """Only leading-``~`` is a home shortcut; mid-string ``~`` is
        kept literal so a directory actually named ``~`` doesn't break."""
        from llamagent.modules.tools.scratch import _resolve_path
        r = _resolve_path("foo/~/bar", base="/tmp")
        assert r == os.path.realpath("/tmp/foo/~/bar")


# ---------------------------------------------------------------------------
# system_info builtin tool (commit 8ee5b70)
# ---------------------------------------------------------------------------


class TestSystemInfo:
    def test_returns_json_with_expected_keys(self):
        from llamagent.modules.tools.builtin import system_info
        data = json.loads(system_info())
        for key in (
            "now_local", "now_utc", "timezone", "weekday",
            "os", "os_release", "platform", "hostname",
            "python_version", "cpu_count", "cwd", "user",
        ):
            assert key in data, f"system_info missing key: {key}"

    def test_registered_at_default_tier(self):
        """system_info auto-loads with every agent because it's
        tier=default in the global registry."""
        # Import to trigger @tool decorator registration
        from llamagent.modules.tools import builtin  # noqa: F401
        from llamagent.modules.tools.registry import global_registry
        spec = global_registry.get("system_info")
        assert spec is not None, "system_info not registered"
        assert spec.tier == "default"
        assert spec.safety_level == 1
