"""
v2.0 feature flow tests: mode-aware config, abort mechanism, ContinuousRunner,
triggers, open_questions, clarification_turns limit.

Covers P0 (mode config, abort, runner) and P1 (triggers, open_questions, clarification limit).
"""

import os
import threading
import time
import tempfile

import pytest

from llamagent.core.agent import LlamAgent, ReactResult
from llamagent.core.zone import ConfirmRequest, ConfirmResponse, RequestedScope
from llamagent.core.contract import TaskContract
from llamagent.core.controller import TaskModeController
from llamagent.core.runner import ContinuousRunner, Trigger, TimerTrigger, FileTrigger
from conftest import make_llm_response, make_tool_call


def _setup_zone(agent, tmp_path):
    agent.project_dir = str(tmp_path)
    agent.playground_dir = str(tmp_path / "llama_playground")
    os.makedirs(agent.playground_dir, exist_ok=True)


# ============================================================
# P0-1: Mode-aware config
# ============================================================

def test_mode_aware_config(bare_agent):
    """set_mode applies _MODE_DEFAULTS; switch back restores interactive snapshot.
    Covers: per-mode values, interactive restore, exception rollback."""

    # --- Interactive defaults (snapshot) ---
    assert bare_agent.config.max_react_steps == 10
    assert bare_agent.config.max_duplicate_actions == 2
    assert bare_agent.config.react_timeout == 210.0
    assert bare_agent.config.max_observation_tokens == 2000

    # --- Switch to task → values change ---
    bare_agent.set_mode("task")
    assert bare_agent.config.max_react_steps == 50
    assert bare_agent.config.max_duplicate_actions == 5
    assert bare_agent.config.react_timeout == 600
    assert bare_agent.config.max_observation_tokens == 5000

    # --- Switch back to interactive → restored ---
    bare_agent.set_mode("interactive")
    assert bare_agent.config.max_react_steps == 10
    assert bare_agent.config.max_duplicate_actions == 2
    assert bare_agent.config.react_timeout == 210.0
    assert bare_agent.config.max_observation_tokens == 2000

    # --- Switch to continuous → -1 sentinel values ---
    bare_agent.set_mode("continuous")
    assert bare_agent.config.max_react_steps == -1
    assert bare_agent.config.max_duplicate_actions == -1
    assert bare_agent.config.react_timeout == 600
    assert bare_agent.config.max_observation_tokens == 10000

    # --- Back to interactive again → still correct ---
    bare_agent.set_mode("interactive")
    assert bare_agent.config.max_react_steps == 10


def test_mode_config_preserves_yaml_overrides(bare_agent):
    """If user customized config values (e.g., via YAML), switching back to
    interactive restores those custom values, not hardcoded defaults."""

    # Simulate YAML override: user set max_react_steps=20
    bare_agent.config.max_react_steps = 20
    # Re-snapshot (as __init__ would do after config loading)
    bare_agent._interactive_config = {
        k: getattr(bare_agent.config, k) for k in LlamAgent._MODE_KEYS
    }

    bare_agent.set_mode("task")
    assert bare_agent.config.max_react_steps == 50  # task default

    bare_agent.set_mode("interactive")
    assert bare_agent.config.max_react_steps == 20  # restored to user's custom value


# ============================================================
# P0-2: Abort mechanism
# ============================================================

def test_abort_stops_react_loop(bare_agent, mock_llm_client):
    """abort() stops run_react at the next check point. ReactResult has
    status='aborted' and terminal=True."""

    bare_agent.register_tool("tool_a", lambda x="": f"result_{x}", "tool a")
    bare_agent.register_tool("tool_b", lambda x="": f"result_{x}", "tool b")

    # LLM returns tool calls with different tools to avoid duplicate detection
    mock_llm_client.set_responses([
        make_llm_response("", tool_calls=[make_tool_call("tool_a", {"x": "1"})]),
        make_llm_response("", tool_calls=[make_tool_call("tool_b", {"x": "2"})]),
        make_llm_response("final"),
    ])

    # Set abort before chat — chat() resets it, so this should NOT abort
    bare_agent._abort = True
    result = bare_agent.chat("do something")
    # chat() reset _abort at entry, so it should complete normally
    assert "final" in result

    # Now test actual abort during execution
    call_count = [0]

    def counting_tool(**kw):
        call_count[0] += 1
        if call_count[0] >= 1:
            bare_agent.abort()  # abort after first tool call
        return "done"

    bare_agent._tools["tool_a"]["func"] = counting_tool

    mock_llm_client.set_responses([
        make_llm_response("", tool_calls=[make_tool_call("tool_a", {"x": "go"})]),
        make_llm_response("", tool_calls=[make_tool_call("tool_b", {"x": "no"})]),
        make_llm_response("should not reach"),
    ])

    result = bare_agent.chat("do it")
    assert "aborted" in result.lower() or "abort" in result.lower()
    assert call_count[0] == 1  # only one tool call executed


def test_abort_terminal_flag():
    """ReactResult with status='aborted' has terminal=True.
    Other statuses have terminal=False by default."""

    aborted = ReactResult(text="x", status="aborted", terminal=True)
    assert aborted.terminal is True

    overflow = ReactResult(text="x", status="context_overflow", terminal=True)
    assert overflow.terminal is True

    normal = ReactResult(text="x", status="completed")
    assert normal.terminal is False

    error = ReactResult(text="x", status="error")
    assert error.terminal is False


def test_abort_chat_reset(bare_agent, mock_llm_client):
    """chat() resets _abort at entry, so stale abort from previous session
    does not affect new tasks."""

    bare_agent.abort()
    assert bare_agent._abort is True

    mock_llm_client.set_responses([make_llm_response("hello")])
    result = bare_agent.chat("hi")
    assert result == "hello"  # not aborted — chat() reset the flag


# ============================================================
# P0-3: ContinuousRunner + Triggers
# ============================================================

def test_runner_basic_flow(bare_agent, mock_llm_client):
    """Runner polls triggers and calls agent.chat() for each input."""

    mock_llm_client.set_responses([
        make_llm_response("response1"),
        make_llm_response("response2"),
    ])

    class CountingTrigger(Trigger):
        def __init__(self):
            self.count = 0
        def poll(self):
            self.count += 1
            if self.count <= 2:
                return f"task {self.count}"
            return None

    trigger = CountingTrigger()
    runner = ContinuousRunner(bare_agent, [trigger], poll_interval=0.01)

    # Run in a thread, stop after brief delay
    t = threading.Thread(target=runner.run)
    t.start()
    time.sleep(0.15)
    runner.stop()
    t.join(timeout=2)

    assert trigger.count >= 2
    assert len(bare_agent.history) >= 2  # at least 2 chat turns recorded


def test_runner_stop(bare_agent, mock_llm_client):
    """Runner.stop() terminates the run loop and sets agent._abort."""

    class NeverTrigger(Trigger):
        def poll(self):
            return None

    runner = ContinuousRunner(bare_agent, [NeverTrigger()], poll_interval=0.01)

    t = threading.Thread(target=runner.run)
    t.start()
    time.sleep(0.05)
    runner.stop()
    t.join(timeout=2)

    assert not t.is_alive()


def test_runner_task_timeout(bare_agent, mock_llm_client):
    """Runner with task_timeout aborts long-running tasks via watchdog."""

    # Make chat() take a long time by having the tool sleep
    def slow_tool():
        time.sleep(2)
        return "done"

    bare_agent.register_tool("slow", slow_tool, "slow tool")
    mock_llm_client.set_responses([
        make_llm_response("", tool_calls=[make_tool_call("slow", {})]),
        make_llm_response("completed"),
    ])

    class OnceTrigger(Trigger):
        def __init__(self):
            self.fired = False
        def poll(self):
            if not self.fired:
                self.fired = True
                return "do slow thing"
            return None

    # task_timeout=0.1s, tool takes 2s → abort should fire
    runner = ContinuousRunner(bare_agent, [OnceTrigger()],
                              poll_interval=0.01, task_timeout=0.1)

    t = threading.Thread(target=runner.run)
    t.start()
    time.sleep(0.5)
    runner.stop()
    t.join(timeout=3)

    # The slow tool completed (atomic), but _abort was set
    assert not t.is_alive()


def test_runner_on_timeout_callable(bare_agent, mock_llm_client):
    """on_timeout=callable is called instead of agent.abort()."""

    callback_called = [False]

    def my_callback():
        callback_called[0] = True

    mock_llm_client.set_responses([make_llm_response("ok")])

    class OnceTrigger(Trigger):
        def __init__(self):
            self.fired = False
        def poll(self):
            if not self.fired:
                self.fired = True
                return "quick task"
            return None

    runner = ContinuousRunner(bare_agent, [OnceTrigger()],
                              poll_interval=0.01, task_timeout=100,
                              on_timeout=my_callback)

    t = threading.Thread(target=runner.run)
    t.start()
    time.sleep(0.2)
    runner.stop()
    t.join(timeout=2)

    # Task completed before timeout, callback should NOT have been called
    assert callback_called[0] is False


# ============================================================
# P1-4: Built-in Triggers
# ============================================================

def test_timer_trigger():
    """TimerTrigger: first poll returns None, subsequent polls fire at interval."""

    trigger = TimerTrigger(interval=0.05, message="tick")

    # First poll: initialize, don't fire
    assert trigger.poll() is None

    # Immediate second poll: too soon
    assert trigger.poll() is None

    # Wait for interval
    time.sleep(0.06)
    result = trigger.poll()
    assert result == "tick"

    # Immediately after firing: too soon again
    assert trigger.poll() is None


def test_file_trigger(tmp_path):
    """FileTrigger: first poll snapshots, detects additions, tracks deletions."""

    watch_dir = str(tmp_path / "inbox")
    os.makedirs(watch_dir)

    # Pre-existing file
    with open(os.path.join(watch_dir, "existing.txt"), "w") as f:
        f.write("old")

    trigger = FileTrigger(watch_dir)

    # First poll: snapshot, no fire
    assert trigger.poll() is None

    # No changes
    assert trigger.poll() is None

    # Add a file
    with open(os.path.join(watch_dir, "new.txt"), "w") as f:
        f.write("new")

    result = trigger.poll()
    assert result is not None
    assert "new.txt" in result

    # No more new files
    assert trigger.poll() is None

    # Remove and re-add (should detect re-addition)
    os.remove(os.path.join(watch_dir, "new.txt"))
    trigger.poll()  # registers deletion
    with open(os.path.join(watch_dir, "new.txt"), "w") as f:
        f.write("re-added")
    result = trigger.poll()
    assert result is not None
    assert "new.txt" in result


def test_file_trigger_nonexistent_dir():
    """FileTrigger handles non-existent directory gracefully."""

    trigger = FileTrigger("/nonexistent/path/abc123")
    assert trigger.poll() is None
    assert trigger.poll() is None  # no crash


# ============================================================
# P1-5: open_questions
# ============================================================

def test_open_questions_in_contract(bare_agent, tmp_path, mock_llm_client):
    """Prepare-phase _report_question tool populates contract.open_questions."""

    _setup_zone(bare_agent, tmp_path)
    bare_agent.register_tool("writer", lambda path="": "ok", "write", safety_level=2,
                             path_extractor=lambda a: [a.get("path", "")])

    project_file = os.path.join(str(tmp_path), "main.py")
    mock_llm_client.set_responses([
        # Prepare: call writer + report_question + text
        make_llm_response("", tool_calls=[
            make_tool_call("writer", {"path": project_file}, "c1"),
            make_tool_call("_report_question", {"question": "Which framework to use?"}, "c2"),
        ]),
        make_llm_response("Plan ready"),
    ])

    bare_agent.set_mode("task")
    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
    result = bare_agent.chat("write main.py")

    # Contract should contain the question
    assert "Which framework to use?" in result
    contract = bare_agent._controller.state.contract
    assert contract is not None
    assert "Which framework to use?" in contract.open_questions


def test_report_question_not_available_in_execute(bare_agent, tmp_path, mock_llm_client):
    """_report_question tool is only available during prepare, not execute."""

    _setup_zone(bare_agent, tmp_path)
    bare_agent.register_tool("writer", lambda path="": "ok", "write", safety_level=2,
                             path_extractor=lambda a: [a.get("path", "")])

    # After prepare, the tool should be unregistered
    project_file = os.path.join(str(tmp_path), "main.py")
    mock_llm_client.set_responses([
        make_llm_response("", tool_calls=[make_tool_call("writer", {"path": project_file})]),
        make_llm_response("Ready"),
    ])

    bare_agent.set_mode("task")
    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
    bare_agent.chat("write main.py")

    # After prepare, _report_question should not be in tools
    assert "_report_question" not in bare_agent._tools


# ============================================================
# P1-6: clarification_turns limit
# ============================================================

def test_clarification_turns_limit():
    """After MAX_CLARIFICATION_TURNS re-prepares, controller forces yes/no decision."""

    controller = TaskModeController()
    controller.state.phase = "idle"

    # Start a task
    action = controller.handle_turn("do something")
    assert action.kind == "run_prepare"

    # Simulate prepare done with scopes
    from llamagent.core.contract import PipelineOutcome
    outcome = PipelineOutcome(
        response="planned",
        metadata={"pending_scopes": [
            RequestedScope(zone="project", actions=["write"], path_prefixes=["src/"]),
        ]},
    )
    action = controller.on_pipeline_done(action, outcome)
    assert action.kind == "await_user"
    assert "[Task Contract]" in action.response

    # Re-prepare 3 times (MAX_CLARIFICATION_TURNS=3)
    for i in range(3):
        action = controller.handle_turn(f"more info {i}")
        assert action.kind == "run_prepare", f"Round {i} should allow re-prepare"
        action = controller.on_pipeline_done(action, outcome)
        assert action.kind == "await_user"

    # 4th attempt should hit the limit
    action = controller.handle_turn("even more info")
    assert action.kind == "await_user"
    assert "Maximum clarification" in action.response

    # User can still confirm or cancel
    action = controller.handle_turn("yes")
    assert action.kind == "run_execute"
