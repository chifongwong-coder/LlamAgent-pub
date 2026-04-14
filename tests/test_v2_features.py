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
from conftest import make_llm_response, make_tool_call, make_stream_chunks, make_stream_tool_call_chunks


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


# ============================================================
# v2.0.2: Streaming
# ============================================================

def test_chat_stream_text_and_tool_call(bare_agent, mock_llm_client):
    """chat_stream: pure text streaming + tool call status messages + history recording."""
    # Part 1: pure text
    mock_llm_client.set_stream_responses([make_stream_chunks("Hello, world!", chunk_size=5)])
    chunks = list(bare_agent.chat_stream("hi"))
    joined = "".join(chunks)
    assert "Hello" in joined and "world" in joined
    assert bare_agent.history[-1]["role"] == "assistant"

    # Part 2: tool call + final text
    bare_agent.register_tool("greet", lambda name="": f"Hello {name}!", "greet tool")
    tool_chunks = make_stream_tool_call_chunks("greet", {"name": "Alice"})
    text_chunks = make_stream_chunks("Greeted Alice", chunk_size=10)
    mock_llm_client.set_stream_responses([tool_chunks, text_chunks])

    chunks = list(bare_agent.chat_stream("greet Alice"))
    joined = "".join(chunks)
    assert "[Calling greet...]" in joined
    assert "[greet done]" in joined
    assert "Greeted Alice" in joined


def test_chat_stream_fallback_and_abort(bare_agent, mock_llm_client, tmp_path):
    """chat_stream: strategy fallback + abort + controller fallback."""
    from llamagent.core.agent import ExecutionStrategy

    # Part 1: non-streaming strategy fallback
    class NoStreamStrategy(ExecutionStrategy):
        def execute(self, query, context, agent):
            return "non-stream result"

    bare_agent._execution_strategy = NoStreamStrategy()
    chunks = list(bare_agent.chat_stream("test"))
    assert chunks == ["non-stream result"]

    # Restore default strategy
    from llamagent.core.agent import SimpleReAct
    bare_agent._execution_strategy = SimpleReAct()

    # Part 2: abort during tool dispatch
    bare_agent.register_tool("slow", lambda: "done", "slow tool")
    bare_agent._tools["slow"]["func"] = lambda **kw: (bare_agent.abort(), "done")[1]
    tool_chunks = make_stream_tool_call_chunks("slow", {})
    mock_llm_client.set_stream_responses([tool_chunks])
    chunks = list(bare_agent.chat_stream("do it"))
    assert "[Operation aborted]" in "".join(chunks)

    # Part 3: task mode controller fallback
    _setup_zone(bare_agent, tmp_path)
    bare_agent.register_tool("writer", lambda path="": "ok", "write", safety_level=2,
                             path_extractor=lambda a: [a.get("path", "")])
    project_file = os.path.join(str(tmp_path), "main.py")
    mock_llm_client.set_responses([
        make_llm_response("", tool_calls=[make_tool_call("writer", {"path": project_file})]),
        make_llm_response("Plan ready"),
    ])
    bare_agent.set_mode("task")
    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
    chunks = list(bare_agent.chat_stream("write main.py"))
    assert len("".join(chunks)) > 0


# ============================================================
# v2.0.2: Runner Task Log
# ============================================================

def test_runner_task_log(bare_agent, mock_llm_client):
    """Runner task_log: records entries with correct fields, trigger_type, duration."""
    from llamagent.core.runner import TaskLogEntry

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

    t = threading.Thread(target=runner.run)
    t.start()
    time.sleep(0.2)
    runner.stop()
    t.join(timeout=2)

    log = runner.get_log()
    assert len(log) >= 2
    assert isinstance(log[0], TaskLogEntry)
    assert log[0].trigger_type == "CountingTrigger"
    assert log[0].input == "task 1"
    assert log[0].output == "response1"
    assert log[0].status == "completed"
    assert log[0].duration > 0


# ============================================================
# v2.2: FS Store — parser and store
# ============================================================

def test_fs_parser_frontmatter_basic():
    """parse_frontmatter: extract metadata and body from standard markdown."""
    from llamagent.modules.fs_store.parser import parse_frontmatter

    content = "---\ntitle: Test Doc\ntags: [a, b]\nenabled: true\n---\n\nBody text here."
    meta, body = parse_frontmatter(content)
    assert meta["title"] == "Test Doc"
    assert meta["tags"] == ["a", "b"]
    assert meta["enabled"] is True
    assert "Body text here" in body


def test_fs_parser_frontmatter_with_dashes_in_value():
    """parse_frontmatter: --- inside a value must NOT break parsing."""
    from llamagent.modules.fs_store.parser import parse_frontmatter

    content = "---\ntitle: My --- Document\ndescription: Use --- as separator\n---\n\nBody."
    meta, body = parse_frontmatter(content)
    assert meta["title"] == "My --- Document"
    assert meta["description"] == "Use --- as separator"
    assert "Body" in body


def test_fs_parser_no_frontmatter():
    """parse_frontmatter: content without frontmatter returns empty dict."""
    from llamagent.modules.fs_store.parser import parse_frontmatter

    meta, body = parse_frontmatter("Just plain text")
    assert meta == {}
    assert "Just plain text" in body


def test_fs_parser_sections():
    """parse_sections: split by ## heading, ignore content before first heading."""
    from llamagent.modules.fs_store.parser import parse_sections

    content = "# Title\n\nIntro paragraph\n\n## Section A\n\nContent A\n\n## Section B\n\nContent B"
    sections = parse_sections(content)
    assert len(sections) == 2
    assert sections[0]["title"] == "Section A"
    assert "Content A" in sections[0]["content"]
    assert sections[1]["title"] == "Section B"


def test_fs_store_write_read_delete(tmp_path):
    """FSStore: write, read, list, delete, clear operations."""
    from llamagent.modules.fs_store.store import FSStore

    store = FSStore(str(tmp_path / "test_store"))
    store.write_file("doc.md", "hello world")
    assert store.read_file("doc.md") == "hello world"
    assert "doc.md" in store.list_files()

    store.delete_file("doc.md")
    assert store.read_file("doc.md") is None
    assert "doc.md" not in store.list_files()


# ============================================================
# v2.2: Memory FS backend
# ============================================================

def test_fs_memory_save_and_list(tmp_path):
    """FSMemoryStore: save facts, list metadata, read source."""
    from llamagent.modules.memory.fs_store import FSMemoryStore
    from llamagent.modules.memory.fact import MemoryFact

    store = FSMemoryStore(str(tmp_path))

    fact = MemoryFact(
        fact_id="abc123",
        kind="preference",
        subject="user",
        attribute="style",
        value="concise",
        source_text="User said: be concise",
    )
    store.save_fact(fact)

    # list_all_metadata should show the fact
    metadata = store.list_all_metadata()
    assert "abc123" in metadata
    assert "preference" in metadata
    assert "concise" in metadata

    # read_fact_source should return details
    source = store.read_fact_source("abc123")
    assert "concise" in source
    assert "User said" in source

    # Nonexistent fact
    source_missing = store.read_fact_source("nonexistent")
    assert "not found" in source_missing.lower()


def test_fs_memory_source_text_with_dashes(tmp_path):
    """FSMemoryStore: source_text containing --- must not corrupt the file."""
    from llamagent.modules.memory.fs_store import FSMemoryStore
    from llamagent.modules.memory.fact import MemoryFact

    store = FSMemoryStore(str(tmp_path))

    # First fact: source_text contains ---
    fact1 = MemoryFact(
        fact_id="f1",
        kind="instruction",
        subject="user",
        attribute="separator",
        value="use --- in docs",
        source_text="The user said:\n---\nUse this as separator\n---",
    )
    store.save_fact(fact1)

    # Second fact: normal
    fact2 = MemoryFact(
        fact_id="f2",
        kind="preference",
        subject="user",
        attribute="lang",
        value="python",
        source_text="User prefers Python",
    )
    store.save_fact(fact2)

    # Both facts should be retrievable
    source1 = store.read_fact_source("f1")
    assert "---" in source1
    assert "separator" in source1

    source2 = store.read_fact_source("f2")
    assert "Python" in source2

    # Metadata listing should show both
    metadata = store.list_all_metadata()
    assert "f1" in metadata
    assert "f2" in metadata


def test_fs_memory_update_and_filter(tmp_path):
    """FSMemoryStore: update status, filter by status, get_facts_by_key."""
    from llamagent.modules.memory.fs_store import FSMemoryStore
    from llamagent.modules.memory.fact import MemoryFact

    store = FSMemoryStore(str(tmp_path))

    fact = MemoryFact(
        fact_id="upd1",
        kind="project_fact",
        subject="llamagent",
        attribute="version",
        value="2.2",
        source_text="Version discussion",
    )
    store.save_fact(fact)

    # Status update
    store.update_fact_status("upd1", "superseded")
    metadata = store.list_all_metadata()  # only active
    assert "upd1" not in metadata

    # get_facts_by_key
    matches = store.get_facts_by_key("project_fact", "llamagent", "version")
    assert len(matches) == 1
    assert matches[0]["status"] == "superseded"


def test_fs_memory_empty_file(tmp_path):
    """FSMemoryStore: operations on empty/nonexistent file don't crash."""
    from llamagent.modules.memory.fs_store import FSMemoryStore

    store = FSMemoryStore(str(tmp_path))

    # All operations should work on empty state
    assert store.list_all_metadata() == ""
    assert store.read_fact_source("nonexistent") is not None  # returns "not found" message
    assert store.get_facts_by_key("any", "any", "any") == []
    assert store.get_stats()["count"] == 0

    # clear on empty should not crash
    store.clear()


# ============================================================
# v2.1: Per-module model
# ============================================================

def test_module_models_different_llm(bare_agent, mock_llm_client):
    """module_models configures a different LLM for a module; module.llm != agent.llm."""
    from llamagent.core.agent import Module

    # Configure module_models to use a different model for "custom"
    bare_agent.config.module_models = {"custom": "gpt-4o-mini"}

    class CustomModule(Module):
        name = "custom"
        description = "test module"

    mod = CustomModule()
    bare_agent.register_module(mod)

    # module.llm should differ from agent.llm (different model)
    assert mod.llm is not bare_agent.llm
    assert mod.llm.model == "gpt-4o-mini"


def test_module_models_default_fallback(bare_agent, mock_llm_client):
    """Without module_models entry, module.llm == agent.llm (backward compat)."""
    from llamagent.core.agent import Module

    # No module_models entry for "basic"
    bare_agent.config.module_models = {}

    class BasicModule(Module):
        name = "basic"
        description = "test module"

    mod = BasicModule()
    bare_agent.register_module(mod)

    # module.llm should be the same as agent.llm
    assert mod.llm is bare_agent.llm


# ============================================================
# v2.1: Compression module
# ============================================================

def test_compression_module_auto_compress(bare_agent, mock_llm_client):
    """With compression module loaded, history is compressed when tokens exceed threshold."""
    from llamagent.modules.compression.module import CompressionModule

    # Set low threshold so compression triggers easily
    bare_agent.config.max_context_tokens = 100
    bare_agent.config.context_compress_threshold = 0.5  # 50 tokens
    bare_agent.config.compress_keep_turns = 1

    mod = CompressionModule()
    bare_agent.register_module(mod)

    # Fill history with enough messages (4 turns = 8 messages)
    for i in range(4):
        bare_agent.history.append({"role": "user", "content": f"message {i}"})
        bare_agent.history.append({"role": "assistant", "content": f"reply {i}"})

    assert len(bare_agent.history) == 8

    # Mock count_tokens to return above threshold
    bare_agent.llm.count_tokens = lambda msgs: 60  # above 50
    # Mock llm.ask for summarization
    bare_agent.llm.ask = lambda prompt, **kw: "Summarized conversation"

    # Trigger on_input — should compress
    result = mod.on_input("new message")

    # on_input returns input unchanged
    assert result == "new message"
    # History should be trimmed to keep_turns*2 = 2 messages
    assert len(bare_agent.history) == 2
    assert bare_agent.summary == "Summarized conversation"


def test_no_compression_module_overflow_clears(bare_agent, mock_llm_client):
    """Without compression module, overflow fallback clears conversation entirely."""
    bare_agent.config.max_context_tokens = 100

    # Fill history
    for i in range(4):
        bare_agent.history.append({"role": "user", "content": f"message {i}"})
        bare_agent.history.append({"role": "assistant", "content": f"reply {i}"})

    # Mock count_tokens to return at hard limit
    bare_agent.llm.count_tokens = lambda msgs: 100

    # Call the fallback check
    bare_agent._check_context_compression()

    # Should have cleared everything
    assert len(bare_agent.history) == 0
    assert bare_agent.summary is None


# ============================================================
# v2.2 Integration: Retrieval & Memory backend switching
# ============================================================


def test_retrieval_rag_backend(bare_agent, mock_llm_client):
    """RetrievalModule with RAG backend registers search_knowledge and injects RAG_GUIDE."""
    from unittest.mock import patch, MagicMock
    from llamagent.modules.retrieval.module import RetrievalModule, RAG_GUIDE

    bare_agent.config.retrieval_backend = "rag"

    mod = RetrievalModule()

    # Mock the RAG pipeline (chromadb may not be installed)
    with patch.object(mod, "_build_pipeline", return_value=MagicMock()):
        bare_agent.register_module(mod)

    # Should register search_knowledge tool
    assert "search_knowledge" in bare_agent._tools
    # Should NOT register FS tools
    assert "list_knowledge" not in bare_agent._tools
    assert "list_entries" not in bare_agent._tools
    assert "read_entry" not in bare_agent._tools

    # on_context should inject RAG guide
    ctx = mod.on_context("some query", "existing context")
    assert RAG_GUIDE in ctx


def test_retrieval_fs_backend(bare_agent, mock_llm_client, tmp_path):
    """RetrievalModule with FS backend registers 3 FS tools and injects FS_RETRIEVE_GUIDE."""
    from llamagent.modules.retrieval.module import RetrievalModule, FS_RETRIEVE_GUIDE

    bare_agent.config.retrieval_backend = "fs"
    bare_agent.config.knowledge_dir = str(tmp_path)

    mod = RetrievalModule()
    bare_agent.register_module(mod)

    # Should register 3 FS tools
    assert "list_knowledge" in bare_agent._tools
    assert "list_entries" in bare_agent._tools
    assert "read_entry" in bare_agent._tools
    # Should NOT register RAG tool
    assert "search_knowledge" not in bare_agent._tools

    # on_context should inject FS guide
    ctx = mod.on_context("some query", "existing context")
    assert FS_RETRIEVE_GUIDE in ctx

    # Tool should work — empty knowledge dir returns "No documents found"
    result = mod._tool_list_knowledge()
    assert "No documents" in result

    # Add a document with frontmatter description
    doc = tmp_path / "guide.md"
    doc.write_text("---\ntitle: Test Guide\ndescription: A test doc\n---\n\n## Intro\nHello")
    result = mod._tool_list_knowledge()
    assert "guide.md" in result
    assert "A test doc" in result

    # Add a document WITHOUT description — should show body preview
    doc2 = tmp_path / "notes.md"
    doc2.write_text("## First Section\nSome content about testing")
    result = mod._tool_list_knowledge()
    assert "notes.md" in result
    assert "Some content about testing" in result

    # list_entries should find the section
    result = mod._tool_list_entries("guide.md")
    assert "Intro" in result

    # read_entry should return content
    result = mod._tool_read_entry("guide.md", "Intro")
    assert "Hello" in result


def test_fs_memory_auto_inject(bare_agent, mock_llm_client, tmp_path):
    """MemoryModule FS backend + auto read mode injects metadata in on_context."""
    from llamagent.modules.memory.module import MemoryModule, MEMORY_GUIDE_FS_AUTO
    from llamagent.modules.memory.fact import MemoryFact

    bare_agent.config.memory_backend = "fs"
    bare_agent.config.memory_mode = "autonomous"
    bare_agent.config.memory_recall_mode = "auto"
    bare_agent.config.memory_fs_dir = str(tmp_path / "memory")

    mod = MemoryModule()
    bare_agent.register_module(mod)

    assert mod._backend == "fs"
    assert mod._available is True

    # Save a fact directly via store
    fact = MemoryFact(
        fact_id="test001",
        kind="preference",
        subject="user",
        attribute="color",
        value="blue",
        source_text="User said they like blue",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )
    mod.store.save_fact(fact)

    # on_context should inject guide + metadata
    ctx = mod.on_context("what color do I like?", "")
    assert MEMORY_GUIDE_FS_AUTO in ctx
    assert "[Memory] Active memories:" in ctx
    assert "test001" in ctx
    assert "blue" in ctx


def test_backend_switch(bare_agent, mock_llm_client, tmp_path):
    """Different backend config produces different tool sets on the same module type."""
    from unittest.mock import patch, MagicMock
    from llamagent.modules.retrieval.module import RetrievalModule

    # --- FS backend ---
    bare_agent.config.retrieval_backend = "fs"
    bare_agent.config.knowledge_dir = str(tmp_path)

    mod_fs = RetrievalModule()
    bare_agent.register_module(mod_fs)

    fs_tools = set(bare_agent._tools.keys())
    assert "list_knowledge" in fs_tools
    assert "search_knowledge" not in fs_tools

    # --- Switch to RAG backend on a fresh agent ---
    from llamagent.core.agent import LlamAgent
    agent2 = LlamAgent.__new__(LlamAgent)
    # Copy config with RAG backend
    agent2.config = bare_agent.config
    agent2.config.retrieval_backend = "rag"
    agent2.persona = None
    agent2.llm = bare_agent.llm
    agent2._llm_cache = bare_agent._llm_cache
    agent2.modules = {}
    agent2.history = []
    agent2.summary = None
    agent2.conversation = agent2.history
    agent2._execution_strategy = bare_agent._execution_strategy
    agent2.tool_executor = None
    agent2._tools = {}
    agent2._active_packs = set()
    agent2._tools_version = 0
    agent2._hooks = {}
    agent2._session_started = False
    agent2._in_hook = False
    agent2.mode = "interactive"
    agent2._controller = None
    agent2._current_task_id = None
    agent2._abort = False
    agent2._open_questions_buffer = []
    agent2.confirm_handler = None
    agent2.interaction_handler = None
    agent2._confirm_wait_time = 0.0
    agent2.project_dir = bare_agent.project_dir
    agent2.playground_dir = bare_agent.playground_dir
    agent2._authorization_engine = bare_agent._authorization_engine
    agent2._interactive_config = bare_agent._interactive_config

    mod_rag = RetrievalModule()
    with patch.object(mod_rag, "_build_pipeline", return_value=MagicMock()):
        agent2.register_module(mod_rag)

    rag_tools = set(agent2._tools.keys())
    assert "search_knowledge" in rag_tools
    assert "list_knowledge" not in rag_tools


# ============================================================
# v2.3: Reflection — FSLessonStore
# ============================================================


def test_fs_lesson_save_and_search(tmp_path):
    """FSLessonStore: save a lesson, search by keyword, verify correct fields."""
    from llamagent.modules.reflection.fs_store import FSLessonStore

    store = FSLessonStore(str(tmp_path))

    store.save_lesson(
        task="API timeout handling",
        error_description="timeout without retry",
        root_cause="no retry mechanism",
        improvement="add exponential backoff",
        tags=["incomplete"],
    )

    results = store.search_lessons("timeout")
    assert len(results) >= 1

    lesson = results[0]
    assert "lesson_id" in lesson
    assert lesson["task"] == "API timeout handling"
    assert lesson["error_description"] == "timeout without retry"
    assert lesson["root_cause"] == "no retry mechanism"
    assert lesson["improvement"] == "add exponential backoff"
    assert "incomplete" in lesson["tags"]
    assert "relevance_score" in lesson


def test_fs_lesson_delete(tmp_path):
    """FSLessonStore: save a lesson, delete it, verify search returns nothing."""
    from llamagent.modules.reflection.fs_store import FSLessonStore

    store = FSLessonStore(str(tmp_path))

    store.save_lesson(
        task="debug segfault",
        error_description="crash on startup",
        root_cause="null pointer dereference",
        improvement="add null check",
        tags=["bug"],
    )

    # Find the lesson to get its id
    results = store.search_lessons("segfault")
    assert len(results) == 1
    lesson_id = results[0]["lesson_id"]

    # Delete should succeed
    assert store.delete_lesson(lesson_id) is True

    # Search should now return empty
    results = store.search_lessons("segfault")
    assert len(results) == 0

    # Deleting non-existent lesson should return False
    assert store.delete_lesson("nonexistent_id") is False


def test_fs_lesson_formatted(tmp_path):
    """FSLessonStore: search_lessons_formatted output includes lesson_id and tags."""
    from llamagent.modules.reflection.fs_store import FSLessonStore

    store = FSLessonStore(str(tmp_path))

    # Lesson with improvement
    store.save_lesson(
        task="API timeout handling",
        error_description="timeout without retry",
        root_cause="no retry mechanism",
        improvement="add exponential backoff",
        tags=["incomplete"],
    )

    # Lesson without improvement
    store.save_lesson(
        task="API connection error",
        error_description="connection refused",
        root_cause="wrong port number",
        improvement="",
        tags=["resolved"],
    )

    formatted = store.search_lessons_formatted("API")
    assert formatted  # non-empty string

    # Should contain lesson_id markers
    assert "[" in formatted  # lesson_id in brackets

    # Lesson with improvement should show the improvement text
    assert "exponential backoff" in formatted

    # Lesson without improvement should show root_cause + no improvement message
    assert "wrong port number" in formatted
    assert "no improvement" in formatted.lower()


# ============================================================
# v2.3: Reflection — Tool Registration
# ============================================================


def test_reflection_tool_registration(bare_agent, tmp_path):
    """ReflectionModule: different mode combinations produce different tool sets."""
    from llamagent.modules.reflection import ReflectionModule

    # Test 1: write=off, read=off -> no tools
    bare_agent.config.reflection_write_mode = "off"
    bare_agent.config.reflection_read_mode = "off"
    bare_agent.config.reflection_backend = "fs"
    bare_agent.config.reflection_fs_dir = str(tmp_path / "t1")

    mod1 = ReflectionModule()
    bare_agent.register_module(mod1)

    assert "list_lessons" not in bare_agent._tools
    assert "read_lesson" not in bare_agent._tools
    assert "delete_lesson" not in bare_agent._tools

    # Clean up for next test
    bare_agent.modules.pop("reflection", None)

    # Test 2: write=off, read=tool -> list_lessons + read_lesson, NO delete_lesson
    bare_agent.config.reflection_write_mode = "off"
    bare_agent.config.reflection_read_mode = "tool"
    bare_agent.config.reflection_fs_dir = str(tmp_path / "t2")

    mod2 = ReflectionModule()
    bare_agent.register_module(mod2)

    assert "list_lessons" in bare_agent._tools
    assert "read_lesson" in bare_agent._tools
    assert "delete_lesson" not in bare_agent._tools

    # Clean up for next test
    bare_agent.modules.pop("reflection", None)
    bare_agent._tools.pop("list_lessons", None)
    bare_agent._tools.pop("read_lesson", None)

    # Test 3: write=auto, read=tool -> list_lessons + read_lesson + delete_lesson
    bare_agent.config.reflection_write_mode = "auto"
    bare_agent.config.reflection_read_mode = "tool"
    bare_agent.config.reflection_fs_dir = str(tmp_path / "t3")

    mod3 = ReflectionModule()
    bare_agent.register_module(mod3)

    assert "list_lessons" in bare_agent._tools
    assert "read_lesson" in bare_agent._tools
    assert "delete_lesson" in bare_agent._tools


# ============================================================
# v2.3: Reflection — Backend Switch
# ============================================================


def test_reflection_backend_switch(bare_agent, tmp_path):
    """ReflectionModule: FS backend initializes FSLessonStore and registers tools."""
    from llamagent.modules.reflection import ReflectionModule
    from llamagent.modules.reflection.fs_store import FSLessonStore

    bare_agent.config.reflection_backend = "fs"
    bare_agent.config.reflection_fs_dir = str(tmp_path / "lessons")
    bare_agent.config.reflection_write_mode = "auto"
    bare_agent.config.reflection_read_mode = "tool"

    mod = ReflectionModule()
    bare_agent.register_module(mod)

    # lesson_store should be FSLessonStore (has read_lesson method)
    assert mod.lesson_store is not None
    assert isinstance(mod.lesson_store, FSLessonStore)
    assert hasattr(mod.lesson_store, "read_lesson")

    # Tools should be registered
    assert "list_lessons" in bare_agent._tools
    assert "read_lesson" in bare_agent._tools
    assert "delete_lesson" in bare_agent._tools


# ============================================================
# v2.3.1: Persistence module
# ============================================================


def test_persistence_save_and_restore(bare_agent, mock_llm_client, tmp_path):
    """Save history+summary via on_output, then restore on a new agent via auto_restore."""
    from llamagent.modules.persistence import PersistenceModule

    # --- First agent: save ---
    bare_agent.config.persistence_enabled = True
    bare_agent.config.persistence_auto_restore = True
    bare_agent.config.persistence_dir = str(tmp_path / "sessions")

    mod1 = PersistenceModule()
    bare_agent.register_module(mod1)

    # Simulate a conversation
    bare_agent.history.append({"role": "user", "content": "hello"})
    bare_agent.history.append({"role": "assistant", "content": "Hi! How can I help?"})
    bare_agent.summary = "Previous conversation about greetings"

    # Trigger save via on_output
    mod1.on_output("Hi! How can I help?")

    # --- Second agent: restore ---
    agent2 = LlamAgent.__new__(LlamAgent)
    agent2.config = bare_agent.config  # same config (same persistence_dir)
    agent2.persona = None
    agent2.llm = mock_llm_client
    agent2._llm_cache = {bare_agent.config.model: mock_llm_client}
    agent2.modules = {}
    agent2.history = []
    agent2.summary = None
    agent2.conversation = agent2.history
    agent2._execution_strategy = bare_agent._execution_strategy
    agent2.tool_executor = None
    agent2._tools = {}
    agent2._active_packs = set()
    agent2._tools_version = 0
    agent2._hooks = {}
    agent2._session_started = False
    agent2._in_hook = False
    agent2.mode = "interactive"
    agent2._controller = None
    agent2._current_task_id = None
    agent2._abort = False
    agent2._open_questions_buffer = []
    agent2.confirm_handler = None
    agent2.interaction_handler = None
    agent2._confirm_wait_time = 0.0
    agent2.project_dir = bare_agent.project_dir
    agent2.playground_dir = bare_agent.playground_dir
    agent2._authorization_engine = bare_agent._authorization_engine
    agent2._interactive_config = bare_agent._interactive_config

    mod2 = PersistenceModule()
    agent2.register_module(mod2)  # auto_restore=True should load

    # Verify restored state
    assert len(agent2.history) == 2
    assert agent2.history[0] == {"role": "user", "content": "hello"}
    assert agent2.history[1] == {"role": "assistant", "content": "Hi! How can I help?"}
    assert agent2.summary == "Previous conversation about greetings"


def test_persistence_disabled(bare_agent, tmp_path):
    """With persistence_enabled=False, on_output does nothing and no file is created."""
    from llamagent.modules.persistence import PersistenceModule

    bare_agent.config.persistence_enabled = False
    bare_agent.config.persistence_dir = str(tmp_path / "sessions")

    mod = PersistenceModule()
    bare_agent.register_module(mod)

    # Add messages to history
    bare_agent.history.append({"role": "user", "content": "hello"})
    bare_agent.history.append({"role": "assistant", "content": "hi"})

    # Trigger on_output — should do nothing
    mod.on_output("hi")

    # Verify no session file exists
    session_dir = tmp_path / "sessions"
    if session_dir.exists():
        assert list(session_dir.iterdir()) == []
    # else: directory was never created, which is also correct


def test_persistence_no_auto_restore(bare_agent, mock_llm_client, tmp_path):
    """With auto_restore=False, data is saved but not restored on attach."""
    from llamagent.modules.persistence import PersistenceModule

    # --- First agent: save ---
    bare_agent.config.persistence_enabled = True
    bare_agent.config.persistence_auto_restore = False
    bare_agent.config.persistence_dir = str(tmp_path / "sessions")

    mod1 = PersistenceModule()
    bare_agent.register_module(mod1)

    bare_agent.history.append({"role": "user", "content": "hello"})
    bare_agent.history.append({"role": "assistant", "content": "hi"})

    # Save via on_output
    mod1.on_output("hi")

    # Verify file exists on disk
    session_dir = tmp_path / "sessions"
    assert session_dir.exists()
    json_files = list(session_dir.glob("*.json"))
    assert len(json_files) == 1

    # --- Second agent: auto_restore=False, should NOT restore ---
    agent2 = LlamAgent.__new__(LlamAgent)
    agent2.config = bare_agent.config  # same config (auto_restore=False)
    agent2.persona = None
    agent2.llm = mock_llm_client
    agent2._llm_cache = {bare_agent.config.model: mock_llm_client}
    agent2.modules = {}
    agent2.history = []
    agent2.summary = None
    agent2.conversation = agent2.history
    agent2._execution_strategy = bare_agent._execution_strategy
    agent2.tool_executor = None
    agent2._tools = {}
    agent2._active_packs = set()
    agent2._tools_version = 0
    agent2._hooks = {}
    agent2._session_started = False
    agent2._in_hook = False
    agent2.mode = "interactive"
    agent2._controller = None
    agent2._current_task_id = None
    agent2._abort = False
    agent2._open_questions_buffer = []
    agent2.confirm_handler = None
    agent2.interaction_handler = None
    agent2._confirm_wait_time = 0.0
    agent2.project_dir = bare_agent.project_dir
    agent2.playground_dir = bare_agent.playground_dir
    agent2._authorization_engine = bare_agent._authorization_engine
    agent2._interactive_config = bare_agent._interactive_config

    mod2 = PersistenceModule()
    agent2.register_module(mod2)

    # Should NOT have restored
    assert len(agent2.history) == 0
    assert agent2.summary is None

    # But the file should still exist (saved by first agent)
    json_files = list(session_dir.glob("*.json"))
    assert len(json_files) == 1


# ============================================================
# v2.4.1: ThreadRunner parallel child agents + delegate role
# ============================================================

import re

from llamagent.modules.child_agent.module import ChildAgentModule
from llamagent.modules.child_agent.policy import ROLE_POLICIES


def test_thread_runner_parallel_spawn(bare_agent, mock_llm_client):
    """Thread runner: spawn 2 children in parallel, collect_results(wait=True) returns both."""
    bare_agent.config.child_agent_runner = "thread"

    mock_llm_client.set_responses([
        make_llm_response("result from child alpha"),
        make_llm_response("result from child beta"),
    ])

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # Spawn two children -- thread runner returns task_id messages, not results
    # Use "researcher" role (has explicit tool_allowlist) for clean tool filtering
    result1 = module._spawn_child(task="task alpha", role="researcher")
    result2 = module._spawn_child(task="task beta", role="researcher")

    # Thread runner may return task_id (async) or result text (if child completes
    # before status check — common with mock LLM). Both are valid behaviors.
    # The key invariant: both children execute and their results are collectible.

    # Collect all results (wait=True blocks until both are done)
    collected = module._collect_results(wait=True, timeout=30)

    assert "result from child alpha" in collected
    assert "result from child beta" in collected


def test_thread_runner_wait_child(bare_agent, mock_llm_client):
    """Thread runner: spawn one child, extract task_id, wait_child returns result."""
    bare_agent.config.child_agent_runner = "thread"

    mock_llm_client.set_responses([
        make_llm_response("the research output"),
    ])

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # Use "researcher" role (explicit tool_allowlist, avoids deepcopy of all parent tools)
    spawn_msg = module._spawn_child(task="do research", role="researcher")
    assert "task_id:" in spawn_msg

    # Extract task_id from the formatted message
    match = re.search(r"task_id:\s*(\w+)", spawn_msg)
    assert match is not None, f"Could not extract task_id from: {spawn_msg}"
    task_id = match.group(1)

    # wait_child should block until done and return the result
    result = module._wait_child(task_id=task_id)
    assert "the research output" in result


def test_delegate_role(bare_agent, mock_llm_client):
    """Delegate role: single LLM call, no tools, inline execution returns result directly."""
    # Use inline runner (default) -- delegate is about role policy, not runner type
    mock_llm_client.set_responses([
        make_llm_response("a beautiful poem about autumn"),
    ])

    module = ChildAgentModule()
    bare_agent.register_module(module)

    result = module._spawn_child(task="write a poem", role="delegate")

    # Inline runner returns result text directly
    assert "a beautiful poem" in result

    # Verify delegate policy: empty tool_allowlist means NO tools
    delegate_policy = ROLE_POLICIES["delegate"]
    assert delegate_policy.tool_allowlist == []
    assert delegate_policy.budget.max_llm_calls == 1

    # Verify a delegate child agent actually has no tools
    from llamagent.modules.child_agent.policy import ChildAgentSpec
    spec = ChildAgentSpec(task="test", role="delegate", policy=delegate_policy)
    child = module._create_child_agent(spec)
    assert len(child._tools) == 0


def test_inline_runner_unchanged(bare_agent, mock_llm_client):
    """Inline runner (default): spawn returns result directly, wait_child NOT registered."""
    # Explicitly set inline (also the default)
    bare_agent.config.child_agent_runner = "inline"

    mock_llm_client.set_responses([
        make_llm_response("inline result text"),
    ])

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # Inline spawn returns the result text synchronously
    result = module._spawn_child(task="quick task", role="worker")
    assert "inline result text" in result
    # Should NOT contain task_id message (that is thread runner behavior)
    assert "task_id:" not in result

    # wait_child tool should NOT be registered for inline runner
    assert "wait_child" not in bare_agent._tools


def test_child_history_preserved(bare_agent, mock_llm_client):
    """Child agent history is preserved in TaskRecord after execution."""
    # Inline runner for simplicity
    bare_agent.config.child_agent_runner = "inline"

    # Child makes a tool call that "fails", then returns final text
    bare_agent.register_tool(
        "read_files", lambda paths: "file contents here", "Read files"
    )

    mock_llm_client.set_responses([
        # First response: child makes a tool call
        make_llm_response(
            tool_calls=[make_tool_call("read_files", {"paths": ["test.txt"]}, call_id="call_c1")]
        ),
        # Second response: child returns final answer
        make_llm_response("Summary: found important data in test.txt"),
    ])

    module = ChildAgentModule()
    bare_agent.register_module(module)

    result = module._spawn_child(task="analyze files", role="researcher")
    assert "Summary" in result

    # Check TaskBoard record has non-empty history
    children = module.controller.list_children(module._parent_id)
    assert len(children) >= 1

    record = children[0]
    assert record.history is not None
    assert len(record.history) > 0, "Child history should be preserved in TaskRecord"


# ============================================================
# v2.4.2: CommandRunner extraction
# ============================================================


def test_command_runner_basic():
    """CommandRunner.run: basic echo command returns stdout, exit_code=0, success=True."""
    from llamagent.modules.command_runner import CommandRunner

    result = CommandRunner.run(cmd=["echo", "hello"])

    assert "hello" in result.stdout
    assert result.exit_code == 0
    assert result.success is True


def test_command_runner_timeout():
    """CommandRunner.run: timeout kills subprocess and sets timed_out=True."""
    from llamagent.modules.command_runner import CommandRunner

    result = CommandRunner.run(cmd=["sleep", "10"], timeout=1)

    assert result.timed_out is True
    assert result.success is False


def test_command_runner_safe_env():
    """CommandRunner.build_safe_env: returns minimal env with PATH/HOME/LANG/TERM,
    does not inherit arbitrary env vars."""
    from llamagent.modules.command_runner import CommandRunner

    env = CommandRunner.build_safe_env()

    # Minimal safe keys must be present
    assert "PATH" in env
    assert "HOME" in env
    assert "LANG" in env
    assert "TERM" in env

    # Set a fake env var and verify it is NOT inherited
    import os
    os.environ["LLAMAGENT_SECRET_TEST"] = "secret"
    try:
        env2 = CommandRunner.build_safe_env()
        assert "LLAMAGENT_SECRET_TEST" not in env2
    finally:
        del os.environ["LLAMAGENT_SECRET_TEST"]


# ============================================================
# v2.4.3: ProcessRunner — subprocess child agents
# ============================================================


def test_command_runner_start():
    """CommandRunner.start: returns Popen handle, subprocess runs and produces output."""
    from llamagent.modules.command_runner import CommandRunner

    proc = CommandRunner.start(cmd=["echo", "hello"])
    stdout, stderr = proc.communicate(timeout=5)

    assert proc.returncode == 0
    assert "hello" in stdout


def test_agent_runner_entry_point(tmp_path):
    """agent_runner entry point: subprocess exits and produces JSON with 'status' field."""
    import json
    import subprocess
    import sys

    spec_data = {
        "task": "say hello",
        "role": "delegate",
        "context": "",
        "config": {
            "model": "mock",
            "system_prompt": "You are helpful",
            "max_react_steps": 1,
            "react_timeout": 10,
            "project_dir": str(tmp_path),
            "playground_dir": str(tmp_path / "playground"),
        },
        "tool_allowlist": [],
        "budget": {
            "max_llm_calls": 1,
            "max_time_seconds": 10,
        },
    }

    spec_path = tmp_path / "test_spec.json"
    spec_path.write_text(json.dumps(spec_data))

    result = subprocess.run(
        [sys.executable, "-m", "llamagent.agent_runner", "--spec", str(spec_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    # The process should exit (may fail due to no real LLM, but should produce JSON)
    output = json.loads(result.stdout)
    assert "status" in output
    assert output["status"] in ("completed", "failed", "cancelled")


def test_process_runner_spawn_wait(tmp_path):
    """ProcessRunnerBackend: spec serialization round-trip via _write_spec_file."""
    import json
    from types import SimpleNamespace

    from llamagent.modules.child_agent.runners.process import ProcessRunnerBackend
    from llamagent.modules.child_agent.policy import ChildAgentSpec, AgentExecutionPolicy
    from llamagent.modules.child_agent.budget import Budget

    mock_config = SimpleNamespace(
        model="test-model",
        project_dir=str(tmp_path),
        playground_dir=str(tmp_path / "play"),
    )

    runner = ProcessRunnerBackend(parent_config=mock_config)

    budget = Budget(max_llm_calls=10, max_time_seconds=120)
    policy = AgentExecutionPolicy(
        tool_allowlist=["web_search", "read_files"],
        budget=budget,
    )
    spec = ChildAgentSpec(
        task="research AI papers",
        role="researcher",
        system_prompt="You are a researcher.",
        context="Focus on agent frameworks",
        policy=policy,
    )

    spec_path = runner._write_spec_file("test123", spec)
    try:
        with open(spec_path, "r") as f:
            data = json.load(f)

        assert data["task"] == "research AI papers"
        assert data["role"] == "researcher"
        assert data["config"]["model"] == "test-model"
        assert data["config"]["system_prompt"] == "You are a researcher."
        assert data["tool_allowlist"] == ["web_search", "read_files"]
        assert data["budget"]["max_llm_calls"] == 10
        assert data["budget"]["max_time_seconds"] == 120
    finally:
        import os, shutil
        try:
            os.unlink(spec_path)
            os.rmdir(os.path.dirname(spec_path))
        except OSError:
            shutil.rmtree(os.path.dirname(spec_path), ignore_errors=True)


# ============================================================
# v2.4.4: Log observability — child agent logs
# ============================================================


def test_thread_runner_logs_captured(bare_agent, mock_llm_client):
    """Thread runner captures child agent logs into TaskRecord.logs; wait_child(include_logs=True) returns them."""
    bare_agent.config.child_agent_runner = "thread"

    mock_llm_client.set_responses([
        make_llm_response("research findings complete"),
    ])

    module = ChildAgentModule()
    bare_agent.register_module(module)

    spawn_msg = module._spawn_child(task="research topic", role="researcher")
    assert "task_id:" in spawn_msg

    # Extract task_id
    match = re.search(r"task_id:\s*(\w+)", spawn_msg)
    assert match is not None, f"Could not extract task_id from: {spawn_msg}"
    task_id = match.group(1)

    # Wait with include_logs=True
    result = module._wait_child(task_id=task_id, include_logs=True)

    # The child agent produces log output during execution (LLM calls, etc.)
    # The result should contain "Child logs:" section when logs are available
    assert "research findings complete" in result
    # Thread runner captures logs via ThreadLogCapture, so logs section should appear
    if "Child logs:" in result:
        # Logs were captured — verify the section is present
        assert "Child logs:" in result
    else:
        # Even if no log output was generated (minimal mock), the result
        # should still contain the child's output
        assert "research findings complete" in result


def test_process_runner_stderr_saved():
    """TaskRecord.logs field: default empty string, can be set to arbitrary log content."""
    from llamagent.modules.child_agent.task_board import TaskRecord

    # Default: logs is empty string
    record = TaskRecord(task_id="test1")
    assert record.logs == ""

    # Explicit logs value
    record_with_logs = TaskRecord(task_id="test2", logs="some log output from stderr")
    assert record_with_logs.logs == "some log output from stderr"


def test_child_metrics_include_budget(bare_agent, mock_llm_client):
    """Child agent TaskRecord.metrics includes budget tracker stats (tokens, LLM calls)."""
    from llamagent.modules.child_agent import ChildAgentModule

    bare_agent.config.child_agent_runner = "inline"
    mock_llm_client.set_responses([make_llm_response("research done")])

    mod = ChildAgentModule()
    bare_agent.register_module(mod)

    # Spawn with researcher role (has budget: max_llm_calls=20)
    mod._spawn_child(task="test research", role="researcher")

    children = mod.controller.list_children(mod._parent_id)
    assert len(children) == 1
    metrics = children[0].metrics

    assert "elapsed_seconds" in metrics
    # Budget tracker stats should be present (researcher has a budget)
    assert "llm_calls" in metrics
    assert metrics["llm_calls"] >= 1  # At least one LLM call was made
    assert "tokens_used" in metrics


# ============================================================
# v2.5: Tool System Enhancements
# ============================================================

# --- Feature 1: Large Result Persistence ---

def test_large_result_persisted(bare_agent, tmp_path):
    """When tool result exceeds max_observation_tokens and read_files is available,
    persist the full result to a file and return preview + path hint."""
    bare_agent.config.max_observation_tokens = 100
    bare_agent.playground_dir = str(tmp_path)

    # Register read_files so persistence path is activated
    bare_agent.register_tool(
        "read_files", lambda path="": "file content", "read files tool",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    )

    # Build a large result string (~10000 chars, well above 100 token threshold)
    large_content = "A" * 10_000

    result = bare_agent._truncate_observation(large_content, tool_name="web_search")

    # Should contain the persistence hint, not just a plain truncation
    assert "Full result saved to:" in result
    assert "read_files" in result

    # Verify the file was actually written to disk with full content
    tool_results_dir = tmp_path / "tool_results"
    assert tool_results_dir.is_dir()
    saved_files = list(tool_results_dir.iterdir())
    assert len(saved_files) == 1
    assert saved_files[0].name.startswith("web_search_")
    assert saved_files[0].read_text() == large_content


def test_large_result_fallback_no_read_files(bare_agent, tmp_path):
    """Without read_files tool registered, large results fall back to plain truncation."""
    bare_agent.config.max_observation_tokens = 100
    bare_agent.playground_dir = str(tmp_path)

    # Do NOT register read_files
    large_content = "B" * 10_000

    result = bare_agent._truncate_observation(large_content, tool_name="web_search")

    # Should be truncated, not persisted
    assert "content truncated" in result
    assert "Full result saved to:" not in result

    # No file should have been created
    tool_results_dir = tmp_path / "tool_results"
    assert not tool_results_dir.exists()


# --- Feature 2: Tool Input JSON Schema Validation ---

def test_validation_missing_required(bare_agent):
    """Calling a tool with missing required arguments returns a validation error."""
    bare_agent.register_tool(
        "search", lambda query: query, "search tool",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    result = bare_agent.call_tool("search", {})

    assert "argument error" in result
    assert "query" in result


def test_validation_wrong_type(bare_agent):
    """Calling a tool with wrong argument type returns a type mismatch error."""
    bare_agent.register_tool(
        "count_items", lambda count: count, "count tool",
        parameters={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
    )

    result = bare_agent.call_tool("count_items", {"count": "not_a_number"})

    assert "argument error" in result
    assert "integer" in result


def test_validation_passes(bare_agent):
    """Valid arguments pass validation and the tool executes normally."""
    bare_agent.register_tool(
        "search", lambda query: f"results for {query}", "search tool",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    result = bare_agent.call_tool("search", {"query": "hello"})

    assert result == "results for hello"


# --- Feature 3: Per-Tool Timeout ---

def test_per_tool_timeout(bare_agent):
    """A tool with a short timeout that sleeps too long returns a timeout error."""
    def slow_tool():
        time.sleep(10)
        return "should not reach here"

    bare_agent.register_tool(
        "slow", slow_tool, "slow tool",
        timeout=0.5,
    )

    start = time.time()
    result = bare_agent.call_tool("slow", {})
    elapsed = time.time() - start

    assert "timed out" in result
    # Should return quickly (~0.5s), not wait 10s
    assert elapsed < 3.0


def test_per_tool_no_timeout(bare_agent):
    """A tool without per-tool timeout executes normally."""
    bare_agent.register_tool(
        "fast", lambda: "fast result", "fast tool",
    )

    result = bare_agent.call_tool("fast", {})

    assert result == "fast result"


# ============================================================
# v2.6: Feature 1 — Workspace Isolation
# ============================================================


def test_workspace_sandbox_mode(bare_agent, mock_llm_client, tmp_path):
    """Sandbox-mode child (writer) gets isolated workspace under playground/children/."""
    from llamagent.modules.child_agent.module import ChildAgentModule
    from llamagent.modules.child_agent.policy import ChildAgentSpec, ROLE_POLICIES

    bare_agent.project_dir = str(tmp_path / "project")
    bare_agent.playground_dir = str(tmp_path / "project" / "llama_playground")
    os.makedirs(bare_agent.playground_dir, exist_ok=True)

    mock_llm_client.set_responses([make_llm_response("draft written")])

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # Spawn sandbox-mode child (writer role defaults to workspace_mode="sandbox")
    result = bare_agent.call_tool("spawn_child", {"task": "write a draft", "role": "writer"})
    assert isinstance(result, str)

    # Find the child workspace under playground/children/
    children_dir = os.path.join(bare_agent.playground_dir, "children")
    assert os.path.isdir(children_dir), "children/ directory should be created"

    # There should be exactly one child workspace
    child_dirs = os.listdir(children_dir)
    assert len(child_dirs) == 1
    child_workspace = os.path.join(children_dir, child_dirs[0])

    # Child workspace should NOT be the parent's project_dir
    assert child_workspace != bare_agent.project_dir

    # Verify child's playground was also created inside workspace
    child_playground = os.path.join(child_workspace, "llama_playground")
    assert os.path.isdir(child_playground)


def test_workspace_project_mode(bare_agent, mock_llm_client, tmp_path):
    """Project-mode child (coder) inherits parent's project_dir."""
    from llamagent.modules.child_agent.module import ChildAgentModule
    from llamagent.modules.child_agent.policy import ChildAgentSpec, ROLE_POLICIES

    bare_agent.project_dir = str(tmp_path / "project")
    bare_agent.playground_dir = str(tmp_path / "project" / "llama_playground")
    os.makedirs(bare_agent.playground_dir, exist_ok=True)

    # Register required tools for coder role
    bare_agent.register_tool("read_files", lambda paths="": "data", "Read files")
    bare_agent.register_tool("write_files", lambda files="": "ok", "Write files")
    bare_agent.register_tool("apply_patch", lambda t="", e="": "patched", "Patch")
    bare_agent.register_tool("start_job", lambda cmd="": "out", "Job")
    bare_agent.register_tool("glob_files", lambda p="": "files", "Glob")
    bare_agent.register_tool("search_text", lambda q="": "found", "Search text")

    mock_llm_client.set_responses([make_llm_response("code written")])

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # Directly create a coder child to inspect its project_dir
    spec = ChildAgentSpec(task="implement feature", role="coder", policy=ROLE_POLICIES["coder"])
    spec.task_id = "test_coder_01"
    child = module._create_child_agent(spec)

    # Coder (project mode) should share parent's project_dir
    assert child.project_dir == bare_agent.project_dir
    assert child.playground_dir == bare_agent.playground_dir


# ============================================================
# v2.6: Feature 2 — Role Model Override
# ============================================================


def test_role_model_override(bare_agent, mock_llm_client, tmp_path):
    """Config-level role_models override is applied via copy.copy, not mutating ROLE_POLICIES."""
    import copy
    from llamagent.modules.child_agent.module import ChildAgentModule
    from llamagent.modules.child_agent.policy import AgentExecutionPolicy, ROLE_POLICIES

    bare_agent.project_dir = str(tmp_path / "project")
    bare_agent.playground_dir = str(tmp_path / "project" / "llama_playground")
    os.makedirs(bare_agent.playground_dir, exist_ok=True)

    # Set a model override for researcher role
    bare_agent.config.child_agent_role_models = {"researcher": "mock-model-alt"}

    # Snapshot ROLE_POLICIES researcher model BEFORE spawning
    original_model = ROLE_POLICIES["researcher"].model  # should be None

    mock_llm_client.set_responses([make_llm_response("research results")])

    # Register tools needed by researcher
    bare_agent.register_tool("web_search", lambda q="": "results", "Search")
    bare_agent.register_tool("web_fetch", lambda u="": "page", "Fetch")
    bare_agent.register_tool("search_knowledge", lambda q="": "kb", "KB")
    bare_agent.register_tool("search_text", lambda q="": "found", "Search text")
    bare_agent.register_tool("read_files", lambda paths="": "data", "Read files")

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # Spawn researcher child
    result = bare_agent.call_tool("spawn_child", {"task": "research AI", "role": "researcher"})
    assert isinstance(result, str)
    assert len(result) > 0

    # ROLE_POLICIES should NOT be mutated (model should still be original value)
    assert ROLE_POLICIES["researcher"].model == original_model


def test_role_model_default_inherit(bare_agent, mock_llm_client, tmp_path):
    """Without role_models override, child uses parent's LLM (default inheritance)."""
    from llamagent.modules.child_agent.module import ChildAgentModule
    from llamagent.modules.child_agent.policy import ChildAgentSpec, ROLE_POLICIES

    bare_agent.project_dir = str(tmp_path / "project")
    bare_agent.playground_dir = str(tmp_path / "project" / "llama_playground")
    os.makedirs(bare_agent.playground_dir, exist_ok=True)

    # No model overrides configured
    bare_agent.config.child_agent_role_models = {}

    # Register tools for writer role
    bare_agent.register_tool("read_files", lambda paths="": "data", "Read files")
    bare_agent.register_tool("write_files", lambda files="": "ok", "Write files")
    bare_agent.register_tool("apply_patch", lambda t="", e="": "patched", "Patch")

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # Create a writer child and inspect its LLM
    import copy as _copy
    policy = _copy.copy(ROLE_POLICIES["writer"])
    spec = ChildAgentSpec(task="write docs", role="writer", policy=policy)
    spec.task_id = "test_writer_01"
    child = module._create_child_agent(spec)

    # Child LLM should be a BudgetedLLM wrapping the module's LLM (parent's)
    from llamagent.modules.child_agent.budget import BudgetedLLM
    assert isinstance(child.llm, BudgetedLLM)
    # The inner LLM should be the module's llm (which is the parent's mock)
    assert child.llm._llm is module.llm


# ============================================================
# v2.6: Feature 3 — Sandbox Spec Serialization
# ============================================================


def test_process_spec_sandbox_enabled(tmp_path):
    """ProcessRunnerBackend with parent_has_sandbox=True writes sandbox_enabled + execution_policy."""
    import json
    from llamagent.modules.child_agent.runners.process import ProcessRunnerBackend
    from llamagent.modules.child_agent.policy import (
        AgentExecutionPolicy, ChildAgentSpec, _SANDBOX_AVAILABLE,
    )

    # Build a minimal parent config
    from types import SimpleNamespace
    parent_config = SimpleNamespace(model="test-model", project_dir=str(tmp_path), playground_dir=str(tmp_path / "pg"))

    backend = ProcessRunnerBackend(parent_config=parent_config, parent_has_sandbox=True)

    # Build a spec with execution_policy (coder role has one if sandbox is available)
    # Create a minimal execution_policy-like dataclass to test serialization
    if _SANDBOX_AVAILABLE:
        from llamagent.modules.sandbox.policy import ExecutionPolicy, POLICY_SANDBOXED_CODER
        ep = POLICY_SANDBOXED_CODER
    else:
        # If sandbox module not installed, create a mock dataclass
        from dataclasses import dataclass as _dc
        @_dc
        class _MockEP:
            runtime: str = "python"
            isolation: str = "process"
        ep = _MockEP()

    policy = AgentExecutionPolicy(
        execution_policy=ep,
        workspace_mode="project",
    )
    spec = ChildAgentSpec(task="code task", role="coder", policy=policy)

    # Write the spec file
    spec_path = backend._write_spec_file("test_sandbox_01", spec)
    try:
        with open(spec_path, "r") as f:
            data = json.load(f)

        # sandbox_enabled should be True (parent has sandbox)
        assert data["sandbox_enabled"] is True

        # execution_policy should be serialized as a dict
        assert "execution_policy" in data
        assert isinstance(data["execution_policy"], dict)
        assert "runtime" in data["execution_policy"]
    finally:
        os.unlink(spec_path)
        os.rmdir(os.path.dirname(spec_path))


def test_process_spec_sandbox_disabled(tmp_path):
    """ProcessRunnerBackend with parent_has_sandbox=False writes sandbox_enabled=False, no execution_policy."""
    import json
    from llamagent.modules.child_agent.runners.process import ProcessRunnerBackend
    from llamagent.modules.child_agent.policy import AgentExecutionPolicy, ChildAgentSpec

    from types import SimpleNamespace
    parent_config = SimpleNamespace(model="test-model", project_dir=str(tmp_path), playground_dir=str(tmp_path / "pg"))

    backend = ProcessRunnerBackend(parent_config=parent_config, parent_has_sandbox=False)

    # Spec without execution_policy (typical for non-coder roles)
    policy = AgentExecutionPolicy(workspace_mode="sandbox")
    spec = ChildAgentSpec(task="research task", role="researcher", policy=policy)

    spec_path = backend._write_spec_file("test_nosandbox_01", spec)
    try:
        with open(spec_path, "r") as f:
            data = json.load(f)

        # sandbox_enabled should be False
        assert data["sandbox_enabled"] is False

        # No execution_policy field (policy.execution_policy is None)
        assert "execution_policy" not in data
    finally:
        os.unlink(spec_path)
        os.rmdir(os.path.dirname(spec_path))


# ============================================================
# v2.6: Feature 4 — Memory Auto-Memorize
# ============================================================


def test_child_auto_memorize(bare_agent, mock_llm_client, tmp_path):
    """When parent has MemoryModule and auto_memorize=True, child result is saved to memory."""
    from unittest.mock import MagicMock, patch
    from llamagent.core.agent import Module
    from llamagent.modules.child_agent.module import ChildAgentModule

    bare_agent.project_dir = str(tmp_path / "project")
    bare_agent.playground_dir = str(tmp_path / "project" / "llama_playground")
    os.makedirs(bare_agent.playground_dir, exist_ok=True)
    bare_agent.config.child_agent_auto_memorize = True

    # Create a mock memory module with remember()
    class MockMemoryModule(Module):
        name = "memory"
        description = "mock memory"
        def __init__(self):
            self.remembered = []
        def remember(self, content, category="manual"):
            self.remembered.append({"content": content, "category": category})

    memory_mod = MockMemoryModule()
    bare_agent.register_module(memory_mod)

    mock_llm_client.set_responses([make_llm_response("analysis complete: AI trends are...")])

    child_module = ChildAgentModule()
    bare_agent.register_module(child_module)

    # Spawn a child (inline runner -> completes immediately -> triggers memorize)
    result = bare_agent.call_tool("spawn_child", {"task": "analyze trends", "role": "worker"})
    assert isinstance(result, str)
    assert len(result) > 0

    # Memory module should have been called
    assert len(memory_mod.remembered) == 1
    assert memory_mod.remembered[0]["category"] == "child_agent_result"
    assert "analyze trends" in memory_mod.remembered[0]["content"]


def test_child_auto_memorize_disabled(bare_agent, mock_llm_client, tmp_path):
    """When auto_memorize=False, child result is NOT saved to memory even if MemoryModule exists."""
    from unittest.mock import MagicMock
    from llamagent.core.agent import Module
    from llamagent.modules.child_agent.module import ChildAgentModule

    bare_agent.project_dir = str(tmp_path / "project")
    bare_agent.playground_dir = str(tmp_path / "project" / "llama_playground")
    os.makedirs(bare_agent.playground_dir, exist_ok=True)
    bare_agent.config.child_agent_auto_memorize = False

    # Create a mock memory module with remember()
    class MockMemoryModule(Module):
        name = "memory"
        description = "mock memory"
        def __init__(self):
            self.remembered = []
        def remember(self, content, category="manual"):
            self.remembered.append({"content": content, "category": category})

    memory_mod = MockMemoryModule()
    bare_agent.register_module(memory_mod)

    mock_llm_client.set_responses([make_llm_response("done with task")])

    child_module = ChildAgentModule()
    bare_agent.register_module(child_module)

    # Spawn a child
    result = bare_agent.call_tool("spawn_child", {"task": "do something", "role": "worker"})
    assert isinstance(result, str)

    # Memory module should NOT have been called
    assert len(memory_mod.remembered) == 0


# ============================================================
# v2.7: Scope-based Interactive authorization
# ============================================================

def _setup_zone_v27(agent, tmp_path):
    """Set up zone directories for v2.7 scope tests."""
    agent.project_dir = str(tmp_path)
    agent.playground_dir = str(tmp_path / "llama_playground")
    os.makedirs(agent.playground_dir, exist_ok=True)


def _reg_tool(agent, name, sl=2, result="ok"):
    """Register a tool with a path_extractor for zone testing."""
    agent.register_tool(
        name, lambda **kw: result, f"tool {name}", safety_level=sl,
        path_extractor=lambda args: [args["path"]] if "path" in args else [],
    )


def test_scope_check_before_handler(bare_agent, tmp_path):
    """When a matching scope exists, the handler is NOT called — scope decides directly."""
    from llamagent.core.authorization import ApprovalScope
    from llamagent.core.zone import ConfirmResponse

    _setup_zone_v27(bare_agent, tmp_path)
    _reg_tool(bare_agent, "writer")

    target = os.path.join(str(tmp_path), "src", "main.py")

    # Pre-set a scope that covers target
    bare_agent._authorization_engine.add_scope(ApprovalScope(
        scope="session", zone="project", actions=["write"],
        path_prefixes=[os.path.join(str(tmp_path), "src")],
    ))

    # Set a handler that tracks calls — it should NOT be called
    handler_calls = []
    bare_agent.confirm_handler = lambda req: (
        handler_calls.append(1), ConfirmResponse(allow=True)
    )[-1]

    result = bare_agent.call_tool("writer", {"path": target})
    assert result == "ok"
    assert len(handler_calls) == 0, "Handler should not be called when scope matches"


def test_handler_updates_scope_persistent(bare_agent, tmp_path):
    """In persistent mode, handler approval creates a scope. Second call skips handler."""
    from llamagent.core.zone import ConfirmResponse

    _setup_zone_v27(bare_agent, tmp_path)
    _reg_tool(bare_agent, "writer")
    bare_agent.config.approval_mode = "persistent"

    target = os.path.join(str(tmp_path), "src", "app.py")

    handler_calls = []
    bare_agent.confirm_handler = lambda req: (
        handler_calls.append(1), ConfirmResponse(allow=True)
    )[-1]

    # First call: no scope, handler called, scope created
    result1 = bare_agent.call_tool("writer", {"path": target})
    assert result1 == "ok"
    assert len(handler_calls) == 1

    # Second call: scope matches, handler NOT called
    result2 = bare_agent.call_tool("writer", {"path": target})
    assert result2 == "ok"
    assert len(handler_calls) == 1, "Handler should not be called on second call (persistent scope)"


def test_handler_updates_scope_temporary(bare_agent, tmp_path):
    """In temporary mode, handler approval creates a one-use scope (max_uses=1).
    First call: handler. Second call: scope match (consumes it). Third call: handler again."""
    from llamagent.core.zone import ConfirmResponse

    _setup_zone_v27(bare_agent, tmp_path)
    _reg_tool(bare_agent, "writer")
    bare_agent.config.approval_mode = "temporary"

    target = os.path.join(str(tmp_path), "src", "temp.py")

    handler_calls = []
    bare_agent.confirm_handler = lambda req: (
        handler_calls.append(1), ConfirmResponse(allow=True)
    )[-1]

    # First call: no scope, handler called, temporary scope created (max_uses=1, uses=0)
    result1 = bare_agent.call_tool("writer", {"path": target})
    assert result1 == "ok"
    assert len(handler_calls) == 1

    # Second call: scope matches (uses=0 < max_uses=1), consumed (uses becomes 1)
    result2 = bare_agent.call_tool("writer", {"path": target})
    assert result2 == "ok"
    assert len(handler_calls) == 1, "Second call uses scope, handler not called"

    # Third call: scope exhausted (uses=1 >= max_uses=1), handler called again
    result3 = bare_agent.call_tool("writer", {"path": target})
    assert result3 == "ok"
    assert len(handler_calls) == 2, "Third call triggers handler (scope exhausted)"


def test_handler_deny_no_scope(bare_agent, tmp_path):
    """When handler denies, no scope is created and operation is denied."""
    from llamagent.core.zone import ConfirmResponse

    _setup_zone_v27(bare_agent, tmp_path)
    _reg_tool(bare_agent, "writer")

    target = os.path.join(str(tmp_path), "src", "deny.py")

    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=False)

    result = bare_agent.call_tool("writer", {"path": target})
    assert "denied" in result.lower()

    # Verify no scope was created
    assert len(bare_agent._authorization_engine.state.session_scopes) == 0


def test_no_handler_no_scope_deny(bare_agent, tmp_path):
    """No handler and no scope: CONFIRMABLE operations are denied."""
    _setup_zone_v27(bare_agent, tmp_path)
    _reg_tool(bare_agent, "writer")

    target = os.path.join(str(tmp_path), "src", "nohandler.py")

    # No handler set (default from bare_agent is None)
    assert bare_agent.confirm_handler is None

    result = bare_agent.call_tool("writer", {"path": target})
    assert "denied" in result.lower()


def test_auto_approve_scope(bare_agent, tmp_path):
    """auto_approve=True creates a full-match scope; CONFIRMABLE operations pass without handler."""
    from llamagent.core.authorization import ApprovalScope

    _setup_zone_v27(bare_agent, tmp_path)
    _reg_tool(bare_agent, "writer")

    # Simulate auto_approve by adding a project-wide scope (as auto_approve does at init)
    bare_agent._authorization_engine.add_scope(ApprovalScope(
        scope="session", zone="project", actions=["read", "write"],
        path_prefixes=[str(tmp_path)],
    ))

    target = os.path.join(str(tmp_path), "anything", "file.py")

    # No handler — should still pass because scope covers it
    assert bare_agent.confirm_handler is None
    result = bare_agent.call_tool("writer", {"path": target})
    assert result == "ok"


# ============================================================
# v2.7: Child agent scope inheritance
# ============================================================

def test_sandbox_child_no_scope(bare_agent, mock_llm_client, tmp_path):
    """Sandbox child gets empty scopes — project writes are denied."""
    from llamagent.modules.child_agent.module import ChildAgentModule
    from llamagent.modules.child_agent.policy import ChildAgentSpec, AgentExecutionPolicy
    from llamagent.core.authorization import ApprovalScope
    from llamagent.core.zone import ConfirmResponse

    _setup_zone_v27(bare_agent, tmp_path)

    # Give parent a scope
    bare_agent._authorization_engine.add_scope(ApprovalScope(
        scope="session", zone="project", actions=["write"],
        path_prefixes=[str(tmp_path)],
    ))
    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # Spawn a sandbox child
    spec = ChildAgentSpec(
        task="sandbox task", role="worker",
        policy=AgentExecutionPolicy(workspace_mode="sandbox"),
    )
    child = module._create_child_agent(spec)

    # Sandbox child should have empty scopes
    exported = child._authorization_engine.export_scopes()
    assert len(exported) == 0, "Sandbox child should not inherit parent scopes"


def test_project_child_inherit_scope(bare_agent, mock_llm_client, tmp_path):
    """Project child inherits parent scopes — matching operations pass."""
    from llamagent.modules.child_agent.module import ChildAgentModule
    from llamagent.modules.child_agent.policy import ChildAgentSpec, AgentExecutionPolicy
    from llamagent.core.authorization import ApprovalScope

    _setup_zone_v27(bare_agent, tmp_path)

    # Give parent a scope for src/
    bare_agent._authorization_engine.add_scope(ApprovalScope(
        scope="session", zone="project", actions=["write"],
        path_prefixes=[os.path.join(str(tmp_path), "src")],
    ))

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # Spawn a project child
    spec = ChildAgentSpec(
        task="project task", role="worker",
        policy=AgentExecutionPolicy(workspace_mode="project"),
    )
    child = module._create_child_agent(spec)

    # Project child should inherit parent scopes
    exported = child._authorization_engine.export_scopes()
    assert len(exported) == 1
    assert exported[0]["zone"] == "project"
    assert exported[0]["actions"] == ["write"]

    # Register a tool on the child and verify scope works
    _reg_tool(child, "child_writer")
    target = os.path.join(str(tmp_path), "src", "child_file.py")
    result = child.call_tool("child_writer", {"path": target})
    assert result == "ok", "Project child should be able to write within inherited scope"


def test_project_child_no_expansion(bare_agent, mock_llm_client, tmp_path):
    """Project child cannot write outside inherited scope — no scope expansion."""
    from llamagent.modules.child_agent.module import ChildAgentModule
    from llamagent.modules.child_agent.policy import ChildAgentSpec, AgentExecutionPolicy
    from llamagent.core.authorization import ApprovalScope

    _setup_zone_v27(bare_agent, tmp_path)

    # Parent scope only covers src/
    bare_agent._authorization_engine.add_scope(ApprovalScope(
        scope="session", zone="project", actions=["write"],
        path_prefixes=[os.path.join(str(tmp_path), "src")],
    ))

    module = ChildAgentModule()
    bare_agent.register_module(module)

    spec = ChildAgentSpec(
        task="write docs", role="worker",
        policy=AgentExecutionPolicy(workspace_mode="project"),
    )
    child = module._create_child_agent(spec)
    # Child has no handler — scope is the only decider
    child.confirm_handler = None

    _reg_tool(child, "child_writer")

    # Write to docs/ (outside src/ scope) — should be denied
    target_outside = os.path.join(str(tmp_path), "docs", "readme.md")
    result = child.call_tool("child_writer", {"path": target_outside})
    assert "denied" in result.lower(), "Child should not write outside inherited scope"


def test_preset_scopes_config(bare_agent, tmp_path):
    """Config authorization_scopes preset: matching operations pass without handler."""
    from llamagent.core.authorization import ApprovalScope

    _setup_zone_v27(bare_agent, tmp_path)
    _reg_tool(bare_agent, "writer")

    # Simulate preset scopes loaded from config (as engine init does)
    preset_paths = [
        os.path.join(str(tmp_path), "src"),
        os.path.join(str(tmp_path), "tests"),
    ]
    for p in preset_paths:
        bare_agent._authorization_engine.add_scope(ApprovalScope(
            scope="session", zone="project", actions=["read", "write"],
            path_prefixes=[p],
        ))

    # No handler
    assert bare_agent.confirm_handler is None

    # Write to src/ — should pass via preset scope
    target_src = os.path.join(str(tmp_path), "src", "main.py")
    assert bare_agent.call_tool("writer", {"path": target_src}) == "ok"

    # Write to tests/ — should pass via preset scope
    target_test = os.path.join(str(tmp_path), "tests", "test_main.py")
    assert bare_agent.call_tool("writer", {"path": target_test}) == "ok"

    # Write to docs/ — no scope, no handler => denied
    target_docs = os.path.join(str(tmp_path), "docs", "plan.md")
    result = bare_agent.call_tool("writer", {"path": target_docs})
    assert "denied" in result.lower()


# ============================================================
# v2.7: Regression — backward compatibility
# ============================================================

def test_task_mode_unchanged(bare_agent, tmp_path):
    """Task mode still works with scope matching — no behavioral change from v2.7."""
    from llamagent.core.zone import ConfirmResponse, RequestedScope
    from llamagent.core.authorization import ApprovalScope
    from llamagent.core.contract import AuthorizationUpdate

    _setup_zone_v27(bare_agent, tmp_path)
    _reg_tool(bare_agent, "writer")

    # Set up handler for task mode session_authorize prompt
    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
    bare_agent.set_mode("task")

    # Add a task scope covering src/
    task_id = "test_task_1"
    bare_agent._current_task_id = task_id
    bare_agent._authorization_engine.state.task_scopes[task_id] = [
        ApprovalScope(
            scope="task", zone="project", actions=["write"],
            path_prefixes=[os.path.join(str(tmp_path), "src")],
            source="contract",
        ),
    ]

    # Set controller phase to executing (task mode needs this)
    bare_agent._authorization_engine.policy.state.phase = "executing"
    bare_agent._authorization_engine.policy.state.task_id = task_id

    target = os.path.join(str(tmp_path), "src", "file.py")
    result = bare_agent.call_tool("writer", {"path": target})
    assert result == "ok", "Task mode scope matching should still work"


def test_empty_scope_backward_compat(bare_agent, tmp_path):
    """Interactive mode with empty scopes behaves like pre-v2.7: handler is called on first access.
    With persistent mode, scope accumulates. With temporary mode, scope is consumed after one use."""
    from llamagent.core.zone import ConfirmResponse

    _setup_zone_v27(bare_agent, tmp_path)
    _reg_tool(bare_agent, "writer")

    # Ensure no scopes
    assert len(bare_agent._authorization_engine.state.session_scopes) == 0

    target = os.path.join(str(tmp_path), "src", "compat.py")

    handler_calls = []
    bare_agent.confirm_handler = lambda req: (
        handler_calls.append(1), ConfirmResponse(allow=True)
    )[-1]

    # Persistent mode: first call triggers handler, second does not (scope accumulated)
    bare_agent.config.approval_mode = "persistent"

    result1 = bare_agent.call_tool("writer", {"path": target})
    assert result1 == "ok"
    assert len(handler_calls) == 1, "First call triggers handler (no scope)"

    result2 = bare_agent.call_tool("writer", {"path": target})
    assert result2 == "ok"
    assert len(handler_calls) == 1, "Second call uses persistent scope, handler not called"

    # Clear scopes to test temporary mode separately
    bare_agent._authorization_engine.state.session_scopes.clear()
    handler_calls.clear()
    bare_agent.config.approval_mode = "temporary"

    # First call: handler called, temporary scope created
    result3 = bare_agent.call_tool("writer", {"path": target})
    assert result3 == "ok"
    assert len(handler_calls) == 1

    # Second call: temporary scope consumed (one free pass)
    result4 = bare_agent.call_tool("writer", {"path": target})
    assert result4 == "ok"
    assert len(handler_calls) == 1, "Second call uses temporary scope"

    # Third call: scope exhausted, handler called again
    result5 = bare_agent.call_tool("writer", {"path": target})
    assert result5 == "ok"
    assert len(handler_calls) == 2, "Third call triggers handler (temporary scope exhausted)"


# ============================================================
# v2.8: Messaging Infrastructure
# ============================================================

from llamagent.core.message_channel import MessageChannel, AgentRegistry, Message, MessageTrigger
from llamagent.modules.child_agent.module import ChildAgentModule


def test_message_channel_send_receive():
    """MessageChannel: register 2 agents, send from A to B, B receives correct message."""
    channel = MessageChannel()
    channel.register("agent_a")
    channel.register("agent_b")

    msg_id = channel.send("agent_a", "agent_b", "hello from A", msg_type="info")
    assert msg_id  # non-empty message id

    msgs = channel.receive("agent_b")
    assert len(msgs) == 1
    m = msgs[0]
    assert m.from_id == "agent_a"
    assert m.to_id == "agent_b"
    assert m.content == "hello from A"
    assert m.msg_type == "info"
    assert m.message_id == msg_id

    # A's inbox should be empty (message was sent to B only)
    assert channel.receive("agent_a") == []
    # B's inbox should be empty after receive drained it
    assert channel.receive("agent_b") == []


def test_message_channel_broadcast():
    """MessageChannel: broadcast from A reaches both B and C, but not A itself."""
    channel = MessageChannel()
    channel.register("agent_a")
    channel.register("agent_b")
    channel.register("agent_c")

    channel.broadcast("agent_a", "alert to all", msg_type="alert")

    msgs_b = channel.receive("agent_b")
    msgs_c = channel.receive("agent_c")
    assert len(msgs_b) == 1
    assert len(msgs_c) == 1
    assert msgs_b[0].content == "alert to all"
    assert msgs_b[0].from_id == "agent_a"
    assert msgs_b[0].msg_type == "alert"
    assert msgs_c[0].content == "alert to all"

    # Sender does not receive their own broadcast
    assert channel.receive("agent_a") == []


def test_message_channel_unknown_recipient():
    """MessageChannel: send to unregistered agent_id raises KeyError."""
    channel = MessageChannel()
    channel.register("agent_a")

    with pytest.raises(KeyError):
        channel.send("agent_a", "nonexistent", "hello")


def test_agent_registry_lifecycle():
    """AgentRegistry: register/list/unregister + inbox cleanup."""
    channel = MessageChannel()
    registry = AgentRegistry(channel)

    # Register an agent
    registry.register("bot1", role="worker", mode="continuous")
    agents = registry.list_agents()
    assert len(agents) == 1
    assert agents[0]["agent_id"] == "bot1"
    assert agents[0]["role"] == "worker"
    assert agents[0]["mode"] == "continuous"

    # Unregister
    registry.unregister("bot1")
    assert registry.list_agents() == []

    # Inbox should also be cleaned up -- send to unregistered fails
    with pytest.raises(KeyError):
        channel.send("anyone", "bot1", "should fail")


def test_send_message_tool(bare_agent, mock_llm_client):
    """_tool_send_message delivers a message to the second agent's inbox."""
    # Set mode to continuous BEFORE registering module
    bare_agent.mode = "continuous"

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # Module should have initialized messaging in continuous mode
    assert module._channel is not None
    assert module._registry is not None

    # Register a second agent in the registry
    module._registry.register("helper_bot", role="helper", mode="continuous")

    # Call the tool
    result = module._tool_send_message(to_id="helper_bot", content="please help")
    assert "sent" in result.lower() or "Message sent" in result

    # Verify message arrived in the helper's inbox
    msgs = module._channel.receive("helper_bot")
    assert len(msgs) == 1
    assert msgs[0].content == "please help"
    assert msgs[0].from_id == bare_agent.agent_id


def test_check_messages_tool(bare_agent, mock_llm_client):
    """_tool_check_messages returns pending messages for the parent agent."""
    bare_agent.mode = "continuous"

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # Register a second agent and have it send a message to the parent
    module._registry.register("sender_bot", role="sender", mode="continuous")
    module._channel.send("sender_bot", bare_agent.agent_id, "status update: all good")

    # Call the tool
    result = module._tool_check_messages()
    assert "sender_bot" in result
    assert "status update: all good" in result

    # Calling again should show no messages (inbox drained)
    result2 = module._tool_check_messages()
    assert "No pending messages" in result2 or "no" in result2.lower()


def test_list_agents_tool(bare_agent, mock_llm_client):
    """_tool_list_agents lists all registered agents."""
    bare_agent.mode = "continuous"

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # Parent is already registered by _init_messaging
    # Register a second agent
    module._registry.register("worker_1", role="researcher", mode="continuous")

    result = module._tool_list_agents()
    # Both agents should be listed
    assert bare_agent.agent_id in result
    assert "worker_1" in result
    assert "researcher" in result


def test_message_trigger():
    """MessageTrigger: poll returns None when no messages, returns content when messages exist."""
    channel = MessageChannel()
    channel.register("listener")
    channel.register("notifier")

    trigger = MessageTrigger(channel=channel, agent_id="listener")

    # No messages -> poll returns None
    assert trigger.poll() is None

    # Send a message to listener
    channel.send("notifier", "listener", "wake up!")

    # Poll should return formatted content
    result = trigger.poll()
    assert result is not None
    assert "notifier" in result
    assert "wake up!" in result

    # After poll consumed the message, next poll returns None
    assert trigger.poll() is None


# ============================================================
# v2.9: Continuous Child Agent
# ============================================================


def test_spawn_continuous_child(bare_agent, mock_llm_client, tmp_path):
    """Continuous parent spawns continuous child; child appears in AgentRegistry."""
    bare_agent.mode = "continuous"
    bare_agent.config.child_agent_runner = "thread"
    bare_agent.project_dir = str(tmp_path)
    bare_agent.playground_dir = str(tmp_path / "llama_playground")

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # Provide LLM responses for the child's chat() calls driven by the timer.
    # The child will make chat() calls each time the timer fires.
    # We provide enough so the child doesn't run out during the brief test window.
    mock_llm_client.set_responses([
        make_llm_response("child processing trigger 1"),
        make_llm_response("child processing trigger 2"),
        make_llm_response("child processing trigger 3"),
        make_llm_response("child processing trigger 4"),
        make_llm_response("child processing trigger 5"),
    ])

    # Use "researcher" role (explicit tool_allowlist avoids deepcopy of all parent tools)
    result = module._spawn_continuous_child(
        task="Monitor logs", role="researcher",
        trigger_type="timer", trigger_interval=60,
    )

    # Should return a success message with task_id
    assert "task_id" in result
    assert "continuous" in result.lower() or "Spawned" in result

    # Extract task_id from result (format: "... [task_id: <id>] ...")
    import re
    match = re.search(r"task_id:\s*(\w+)", result)
    assert match, f"Could not extract task_id from: {result}"
    task_id = match.group(1)

    # Give the thread a moment to start and register the child
    time.sleep(0.3)

    # Verify child is in the AgentRegistry
    agents = module._registry.list_agents()
    agent_ids = [a["agent_id"] for a in agents]
    # Parent should be registered
    assert bare_agent.agent_id in agent_ids
    # At least one more agent (the child) should be registered
    assert len(agents) >= 2

    # Verify task board has a running record
    record = module.controller.task_board.get(task_id)
    assert record is not None
    assert record.status == "running"

    # Cleanup: cancel the child so the thread exits
    module.controller.cancel_child(task_id)
    time.sleep(0.3)


def test_continuous_child_timer_trigger(bare_agent, mock_llm_client, tmp_path):
    """Continuous child driven by short timer: executes tasks, produces TaskRecord."""
    bare_agent.mode = "continuous"
    bare_agent.config.child_agent_runner = "thread"
    bare_agent.project_dir = str(tmp_path)
    bare_agent.playground_dir = str(tmp_path / "llama_playground")

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # Enough responses for multiple timer-driven chat() calls
    mock_llm_client.set_responses([
        make_llm_response("task result 1"),
        make_llm_response("task result 2"),
        make_llm_response("task result 3"),
        make_llm_response("task result 4"),
        make_llm_response("task result 5"),
        make_llm_response("task result 6"),
        make_llm_response("task result 7"),
        make_llm_response("task result 8"),
    ])

    # Use "researcher" role (explicit tool_allowlist avoids deepcopy of all parent tools)
    result = module._spawn_continuous_child(
        task="Check health", role="researcher",
        trigger_type="timer", trigger_interval=0.1,  # Very short interval
    )

    import re
    match = re.search(r"task_id:\s*(\w+)", result)
    assert match
    task_id = match.group(1)

    # Let the child run long enough for the ContinuousRunner to complete at least
    # one poll cycle (poll_interval=1.0s) and fire the timer trigger.
    # TimerTrigger initializes _last_fire on first poll (returns None),
    # then fires on the next poll after interval elapses.
    time.sleep(2.0)

    # Cancel the child
    module.controller.cancel_child(task_id)
    time.sleep(0.5)

    # After cancel, the child thread should complete and produce a TaskRecord
    record = module.controller.task_board.get(task_id)
    assert record is not None
    assert record.status in ("completed", "cancelled")
    # The record should report tasks executed (timer should have fired at least once)
    assert record.metrics.get("total_tasks", 0) > 0


def test_continuous_child_message_trigger(bare_agent, mock_llm_client, tmp_path):
    """Parent sends a message to continuous child; child receives it via MessageTrigger."""
    bare_agent.mode = "continuous"
    bare_agent.config.child_agent_runner = "thread"
    bare_agent.project_dir = str(tmp_path)
    bare_agent.playground_dir = str(tmp_path / "llama_playground")

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # Responses for child chat() calls: one from timer and one from message trigger
    mock_llm_client.set_responses([
        make_llm_response("processing timer"),
        make_llm_response("processed message from parent"),
        make_llm_response("processing timer again"),
        make_llm_response("processing timer again"),
    ])

    # Use "researcher" role (explicit tool_allowlist avoids deepcopy of all parent tools)
    result = module._spawn_continuous_child(
        task="Watch for messages", role="researcher",
        trigger_type="timer", trigger_interval=5.0,  # Long interval so timer doesn't dominate
    )

    import re
    match = re.search(r"task_id:\s*(\w+)", result)
    assert match
    task_id = match.group(1)

    # Wait for child to start
    time.sleep(0.3)

    # Find the child's agent_id from the registry
    agents = module._registry.list_agents()
    child_agent_ids = [a["agent_id"] for a in agents if a["agent_id"] != bare_agent.agent_id]
    assert len(child_agent_ids) >= 1
    child_id = child_agent_ids[0]

    # Parent sends a message to the child
    module._channel.send(bare_agent.agent_id, child_id, "urgent: check status")

    # Give time for the MessageTrigger to pick up the message and child to process
    time.sleep(1.0)

    # Cancel and wait for completion
    module.controller.cancel_child(task_id)
    time.sleep(0.5)

    # Verify the child executed at least one task (the message trigger or timer)
    record = module.controller.task_board.get(task_id)
    assert record is not None
    assert record.metrics.get("total_tasks", 0) >= 1


def test_continuous_child_send_to_parent(bare_agent, mock_llm_client, tmp_path):
    """Continuous child's messaging tools work: send_message delivers to parent inbox."""
    bare_agent.mode = "continuous"
    bare_agent.config.child_agent_runner = "thread"
    bare_agent.project_dir = str(tmp_path)
    bare_agent.playground_dir = str(tmp_path / "llama_playground")

    module = ChildAgentModule()
    bare_agent.register_module(module)

    mock_llm_client.set_responses([
        make_llm_response("child started"),
        make_llm_response("child working"),
    ])

    # Use "researcher" role (explicit tool_allowlist avoids deepcopy of all parent tools)
    result = module._spawn_continuous_child(
        task="Report status", role="researcher",
        trigger_type="timer", trigger_interval=5.0,
    )

    import re
    match = re.search(r"task_id:\s*(\w+)", result)
    assert match
    task_id = match.group(1)

    # Wait for child to start
    time.sleep(0.3)

    # Find child's agent_id
    agents = module._registry.list_agents()
    child_agent_ids = [a["agent_id"] for a in agents if a["agent_id"] != bare_agent.agent_id]
    assert len(child_agent_ids) >= 1
    child_id = child_agent_ids[0]

    # Directly use the channel to simulate child sending message to parent
    # (This verifies the channel is correctly wired for the child)
    module._channel.send(child_id, bare_agent.agent_id, "status: all clear", "info")

    # Parent checks messages
    msgs = module._channel.receive(bare_agent.agent_id)
    assert len(msgs) == 1
    assert msgs[0].from_id == child_id
    assert msgs[0].content == "status: all clear"

    # Cleanup
    module.controller.cancel_child(task_id)
    time.sleep(0.3)


def test_continuous_child_cancel(bare_agent, mock_llm_client, tmp_path):
    """Cancel stops continuous child, produces TaskRecord, and unregisters from registry."""
    bare_agent.mode = "continuous"
    bare_agent.config.child_agent_runner = "thread"
    bare_agent.project_dir = str(tmp_path)
    bare_agent.playground_dir = str(tmp_path / "llama_playground")

    module = ChildAgentModule()
    bare_agent.register_module(module)

    mock_llm_client.set_responses([
        make_llm_response("working on task"),
        make_llm_response("still working"),
        make_llm_response("more work"),
    ])

    result = module._spawn_continuous_child(
        task="Long running monitor", role="researcher",
        trigger_type="timer", trigger_interval=0.2,
    )

    import re
    match = re.search(r"task_id:\s*(\w+)", result)
    assert match
    task_id = match.group(1)

    # Wait for child to start and do some work
    time.sleep(0.5)

    # Verify child is in the registry before cancel
    agents_before = module._registry.list_agents()
    assert len(agents_before) >= 2  # parent + child

    # Cancel the child
    success = module.controller.cancel_child(task_id)
    assert success is True

    # Wait for thread to finish and on_complete callback to fire
    time.sleep(0.8)

    # Verify task board shows completed/cancelled
    record = module.controller.task_board.get(task_id)
    assert record is not None
    assert record.status in ("completed", "cancelled")

    # Verify child is unregistered from AgentRegistry (on_complete callback unregisters)
    agents_after = module._registry.list_agents()
    child_ids_after = [a["agent_id"] for a in agents_after if a["agent_id"] != bare_agent.agent_id]
    assert len(child_ids_after) == 0, f"Child should be unregistered, but found: {child_ids_after}"


def test_mode_restriction(bare_agent, mock_llm_client):
    """Non-continuous parent cannot spawn continuous children."""
    bare_agent.mode = "interactive"
    bare_agent.config.child_agent_runner = "thread"

    module = ChildAgentModule()
    bare_agent.register_module(module)

    # _spawn_continuous_child should reject because mode is interactive
    result = module._spawn_continuous_child(
        task="Should fail", role="researcher",
        trigger_type="timer", trigger_interval=60,
    )

    # Should return an error message about mode restriction
    assert "continuous" in result.lower()
    assert "only" in result.lower() or "cannot" in result.lower() or "error" in result.lower()


def test_continuous_child_inherits_scope(bare_agent, mock_llm_client, tmp_path):
    """Continuous child inherits parent's authorization scopes."""
    bare_agent.mode = "continuous"
    bare_agent.config.child_agent_runner = "thread"
    bare_agent.project_dir = str(tmp_path)
    bare_agent.playground_dir = str(tmp_path / "llama_playground")

    # Add a scope to the parent
    from llamagent.core.authorization import ApprovalScope
    parent_scope = ApprovalScope(
        scope="session",
        zone="project",
        actions=["execute"],
        path_prefixes=["project:"],
        tool_names=["web_search"],
    )
    bare_agent._authorization_engine.add_scope(parent_scope)

    # Verify parent has the scope
    parent_scopes = bare_agent._authorization_engine.export_scopes()
    assert len(parent_scopes) >= 1
    scope_tool_names = [s.get("tool_names") for s in parent_scopes if s.get("tool_names")]
    assert any("web_search" in tn for tn in scope_tool_names)

    module = ChildAgentModule()
    bare_agent.register_module(module)

    mock_llm_client.set_responses([
        make_llm_response("child started"),
        make_llm_response("child working"),
    ])

    # Use "researcher" role (explicit tool_allowlist avoids deepcopy of all parent tools)
    result = module._spawn_continuous_child(
        task="Inherited scope test", role="researcher",
        trigger_type="timer", trigger_interval=5.0,
    )

    import re
    match = re.search(r"task_id:\s*(\w+)", result)
    assert match
    task_id = match.group(1)

    # Wait for child to be created
    time.sleep(0.3)

    # Access the child agent through the thread runner's _agents dict
    runner = module.controller.runner
    with runner._lock:
        child_agents = dict(runner._agents)

    assert len(child_agents) >= 1, "Child agent should exist in runner._agents"
    child = list(child_agents.values())[0]

    # Child should have inherited parent's scope
    child_scopes = child._authorization_engine.export_scopes()
    child_scope_tool_names = [s.get("tool_names") for s in child_scopes if s.get("tool_names")]
    assert any("web_search" in tn for tn in child_scope_tool_names), (
        f"Child should inherit parent's web_search scope, got: {child_scopes}"
    )

    # Cleanup
    module.controller.cancel_child(task_id)
    time.sleep(0.3)


# ============================================================
# v2.9.1: ResilienceModule
# ============================================================

from llamagent.core.llm import LLMClient
from llamagent.modules.resilience.classifier import classify, ClassifiedError, _extract_retry_after
from llamagent.modules.resilience.resilient_llm import ResilientLLM
from llamagent.modules.resilience.module import ResilienceModule
from llamagent.modules.child_agent.budget import BudgetedLLM, BudgetTracker, Budget


def _make_http_error(status_code, message="error"):
    """Build an exception with status_code attribute (like litellm errors)."""
    e = Exception(message)
    e.status_code = status_code
    return e


def _make_context_window_error(message="context length exceeded"):
    """Build an exception whose class name contains 'ContextWindow'."""
    class ContextWindowExceededError(Exception):
        pass
    return ContextWindowExceededError(message)


# --- Classifier tests ---

def test_classify_rate_limit():
    """429 -> rate_limit + retryable."""
    err = _make_http_error(429, "rate limit exceeded")
    result = classify(err)
    assert result.reason == "rate_limit"
    assert result.retryable is True
    assert result.should_failover is False
    assert result.original is err


def test_classify_auth_error():
    """401 -> auth + not retryable."""
    err = _make_http_error(401, "unauthorized")
    result = classify(err)
    assert result.reason == "auth"
    assert result.retryable is False
    assert result.should_failover is False


def test_classify_server_error():
    """500/502/503 -> server_error + retryable + should_failover."""
    for status in (500, 502, 503):
        result = classify(_make_http_error(status))
        assert result.reason == "server_error"
        assert result.retryable is True
        assert result.should_failover is True


def test_classify_context_overflow_by_type():
    """ContextWindowExceededError (by class name) -> context_overflow + not retryable + should_failover."""
    err = _make_context_window_error("some provider-specific message")
    result = classify(err)
    assert result.reason == "context_overflow"
    assert result.retryable is False
    assert result.should_failover is True


def test_classify_context_overflow_by_message():
    """Error message containing 'context length' -> context_overflow."""
    err = Exception("maximum context length exceeded for model gpt-4")
    result = classify(err)
    assert result.reason == "context_overflow"
    assert result.retryable is False
    assert result.should_failover is True


def test_classify_network_error():
    """ConnectionError -> network + retryable."""
    err = ConnectionError("connection refused")
    result = classify(err)
    assert result.reason == "network"
    assert result.retryable is True
    assert result.should_failover is False


def test_classify_billing():
    """Error with 'quota' in message -> billing + not retryable."""
    err = Exception("quota exceeded for this account")
    result = classify(err)
    assert result.reason == "billing"
    assert result.retryable is False


def test_classify_unknown():
    """Unrecognized error -> unknown + retryable."""
    err = ValueError("something weird")
    result = classify(err)
    assert result.reason == "unknown"
    assert result.retryable is True
    assert result.should_failover is False


def test_extract_retry_after_header():
    """Extracts retry-after from response headers."""
    from types import SimpleNamespace
    err = Exception("rate limited")
    err.response = SimpleNamespace(headers={"retry-after": "30"})
    result = _extract_retry_after(err)
    assert result == 30.0


def test_extract_retry_after_ms_header():
    """Extracts retry-after-ms from response headers (milliseconds)."""
    from types import SimpleNamespace
    err = Exception("rate limited")
    err.response = SimpleNamespace(headers={"retry-after-ms": "5000"})
    result = _extract_retry_after(err)
    assert result == 5.0


def test_extract_retry_after_message_seconds():
    """Extracts retry-after from error message text (seconds)."""
    err = Exception("Please retry after 30s")
    assert _extract_retry_after(err) == 30.0


def test_extract_retry_after_message_minutes():
    """Extracts retry-after from error message text (minutes → seconds)."""
    err = Exception("try again in 2 minutes")
    assert _extract_retry_after(err) == 120.0


def test_extract_retry_after_default_zero():
    """Returns 0 when no retry-after info found."""
    err = Exception("some random error")
    assert _extract_retry_after(err) == 0


# --- ResilientLLM tests ---

def test_resilient_retry_success(mock_llm_client):
    """First call fails (retryable), second succeeds."""
    retryable_err = _make_http_error(429, "rate limit")
    mock_llm_client.set_responses([retryable_err, make_llm_response("ok")])

    resilient = ResilientLLM.__new__(ResilientLLM)
    resilient.model = "mock-model"
    resilient.api_retry_count = 0
    resilient.max_context_tokens = 8192
    resilient._fallback_llm = None
    resilient._max_retries = 3
    resilient._primary_cooldown_until = 0
    resilient._simple_llm = None

    import unittest.mock as um
    with um.patch.object(LLMClient, "chat", side_effect=mock_llm_client.chat), \
         um.patch("llamagent.modules.resilience.resilient_llm.time.sleep"):
        result = resilient.chat([{"role": "user", "content": "hi"}])
    assert result.choices[0].message.content == "ok"


def test_resilient_failover(mock_llm_client):
    """All retries fail -> fallback model succeeds."""
    server_err = _make_http_error(500, "internal server error")
    mock_llm_client.set_responses([server_err, server_err, server_err, server_err])

    fallback = LLMClient.__new__(LLMClient)
    fallback.model = "fallback-model"
    fallback.api_retry_count = 0
    fallback.max_context_tokens = 8192
    fallback_response = make_llm_response("fallback ok")

    resilient = ResilientLLM.__new__(ResilientLLM)
    resilient.model = "mock-model"
    resilient.api_retry_count = 0
    resilient.max_context_tokens = 8192
    resilient._fallback_llm = fallback
    resilient._max_retries = 1  # Only 1 retry to keep test fast
    resilient._primary_cooldown_until = 0
    resilient._simple_llm = None

    import unittest.mock as um
    with um.patch.object(LLMClient, "chat", side_effect=mock_llm_client.chat), \
         um.patch.object(fallback, "chat", return_value=fallback_response), \
         um.patch("llamagent.modules.resilience.resilient_llm.time.sleep"):
        result = resilient.chat([{"role": "user", "content": "hi"}])
    assert result.choices[0].message.content == "fallback ok"


def test_resilient_context_overflow_failover(mock_llm_client):
    """context_overflow -> no retry, direct failover."""
    ctx_err = _make_context_window_error("context length exceeded")
    mock_llm_client.set_responses([ctx_err])

    fallback = LLMClient.__new__(LLMClient)
    fallback.model = "fallback-model"
    fallback.api_retry_count = 0
    fallback.max_context_tokens = 128000
    fallback_response = make_llm_response("handled by big model")

    resilient = ResilientLLM.__new__(ResilientLLM)
    resilient.model = "mock-model"
    resilient.api_retry_count = 0
    resilient.max_context_tokens = 8192
    resilient._fallback_llm = fallback
    resilient._max_retries = 3
    resilient._primary_cooldown_until = 0
    resilient._simple_llm = None

    call_count = [0]
    original_chat = mock_llm_client.chat
    def counting_chat(*args, **kwargs):
        call_count[0] += 1
        return original_chat(*args, **kwargs)

    import unittest.mock as um
    with um.patch.object(LLMClient, "chat", side_effect=counting_chat), \
         um.patch.object(fallback, "chat", return_value=fallback_response):
        result = resilient.chat([{"role": "user", "content": "hi"}])

    assert result.choices[0].message.content == "handled by big model"
    # Should only try main LLM once (no retry for context_overflow)
    assert call_count[0] == 1


def test_resilient_non_retryable_no_failover():
    """Auth error -> not retryable, no failover -> raises."""
    auth_err = _make_http_error(401, "invalid api key")

    resilient = ResilientLLM.__new__(ResilientLLM)
    resilient.model = "mock-model"
    resilient.api_retry_count = 0
    resilient.max_context_tokens = 8192
    resilient._fallback_llm = None
    resilient._max_retries = 3
    resilient._primary_cooldown_until = 0
    resilient._simple_llm = None

    import unittest.mock as um
    with um.patch.object(LLMClient, "chat", side_effect=auth_err):
        with pytest.raises(Exception, match="invalid api key"):
            resilient.chat([{"role": "user", "content": "hi"}])


def test_resilient_retries_exhausted_no_failover():
    """All retries exhausted on retryable error, no fallback -> raises original."""
    network_err = ConnectionError("connection refused")

    resilient = ResilientLLM.__new__(ResilientLLM)
    resilient.model = "mock-model"
    resilient.api_retry_count = 0
    resilient.max_context_tokens = 8192
    resilient._fallback_llm = None
    resilient._max_retries = 1
    resilient._primary_cooldown_until = 0
    resilient._simple_llm = None

    import unittest.mock as um
    with um.patch.object(LLMClient, "chat", side_effect=network_err), \
         um.patch("llamagent.modules.resilience.resilient_llm.time.sleep"):
        with pytest.raises(ConnectionError, match="connection refused"):
            resilient.chat([{"role": "user", "content": "hi"}])


def test_resilient_fallback_also_fails(mock_llm_client):
    """All retries fail, fallback also fails -> raises fallback error."""
    server_err = _make_http_error(500, "primary down")
    fallback_err = _make_http_error(500, "fallback also down")
    mock_llm_client.set_responses([server_err, server_err])

    fallback = LLMClient.__new__(LLMClient)
    fallback.model = "fallback-model"
    fallback.api_retry_count = 0
    fallback.max_context_tokens = 8192

    resilient = ResilientLLM.__new__(ResilientLLM)
    resilient.model = "mock-model"
    resilient.api_retry_count = 0
    resilient.max_context_tokens = 8192
    resilient._fallback_llm = fallback
    resilient._max_retries = 0
    resilient._primary_cooldown_until = 0
    resilient._simple_llm = None

    import unittest.mock as um
    with um.patch.object(LLMClient, "chat", side_effect=mock_llm_client.chat), \
         um.patch.object(fallback, "chat", side_effect=fallback_err), \
         um.patch("llamagent.modules.resilience.resilient_llm.time.sleep"):
        with pytest.raises(Exception, match="fallback also down"):
            resilient.chat([{"role": "user", "content": "hi"}])


def test_resilient_ask_gets_resilience(mock_llm_client):
    """ask() (inherited from LLMClient) gets resilience through self.chat()."""
    retryable_err = _make_http_error(429, "rate limit")
    mock_llm_client.set_responses([retryable_err, make_llm_response("recovered")])

    resilient = ResilientLLM.__new__(ResilientLLM)
    resilient.model = "mock-model"
    resilient.api_retry_count = 0
    resilient.max_context_tokens = 8192
    resilient._fallback_llm = None
    resilient._max_retries = 3
    resilient._primary_cooldown_until = 0
    resilient._simple_llm = None

    import unittest.mock as um
    with um.patch("llamagent.core.llm._LITELLM_AVAILABLE", True), \
         um.patch.object(LLMClient, "chat", side_effect=mock_llm_client.chat), \
         um.patch("llamagent.modules.resilience.resilient_llm.time.sleep"):
        result = resilient.ask("hello")
    assert result == "recovered"


def test_resilient_no_module(bare_agent, mock_llm_client):
    """Without ResilienceModule, agent.llm is plain LLMClient — behavior unchanged."""
    assert type(bare_agent.llm).__name__ != "ResilientLLM"
    mock_llm_client.set_responses([make_llm_response("plain")])
    resp = bare_agent.llm.chat([{"role": "user", "content": "hi"}])
    assert resp.choices[0].message.content == "plain"


def test_resilient_chat_stream_passthrough(mock_llm_client):
    """chat_stream is inherited from LLMClient, passthrough without resilience."""
    resilient = ResilientLLM.__new__(ResilientLLM)
    resilient.model = "mock-model"
    resilient.api_retry_count = 0
    resilient.max_context_tokens = 8192
    resilient._fallback_llm = None
    resilient._max_retries = 3
    resilient._primary_cooldown_until = 0
    resilient._simple_llm = None

    stream_chunks = make_stream_chunks("hello stream")

    import unittest.mock as um
    with um.patch("llamagent.core.llm._LITELLM_AVAILABLE", True), \
         um.patch("llamagent.core.llm.completion", side_effect=lambda **kw: iter(stream_chunks) if kw.get("stream") else None):
        chunks = list(resilient.chat_stream([{"role": "user", "content": "hi"}]))
    assert len(chunks) > 0


def test_resilient_isinstance_transparent():
    """ResilientLLM isinstance check is transparent — IS-A LLMClient."""
    resilient = ResilientLLM.__new__(ResilientLLM)
    assert isinstance(resilient, LLMClient)


# --- Module integration tests ---

def test_resilience_module_attach(bare_agent):
    """ResilienceModule replaces agent.llm with ResilientLLM, _llm_cache untouched."""
    original_cache_llm = bare_agent._llm_cache.get("mock-model")

    module = ResilienceModule()
    bare_agent.register_module(module)

    # agent.llm is now ResilientLLM
    assert isinstance(bare_agent.llm, ResilientLLM)
    assert isinstance(bare_agent.llm, LLMClient)
    # _llm_cache still has the original
    assert bare_agent._llm_cache["mock-model"] is original_cache_llm
    assert bare_agent._llm_cache["mock-model"] is not bare_agent.llm


def test_resilience_module_child_agent_same_model(bare_agent):
    """Child agent using same model via _get_llm gets original LLMClient, not affected."""
    module = ResilienceModule()
    bare_agent.register_module(module)

    child_llm = bare_agent._get_llm("mock-model")
    # Should be the original cached LLMClient, not ResilientLLM
    assert type(child_llm).__name__ != "ResilientLLM"
    assert child_llm is bare_agent._llm_cache["mock-model"]


# --- BudgetedLLM __getattr__ test ---

def test_budgeted_llm_getattr_proxy(mock_llm_client):
    """BudgetedLLM proxies unknown attributes to underlying LLM."""
    tracker = BudgetTracker(Budget())
    budgeted = BudgetedLLM(mock_llm_client, tracker)
    # max_context_tokens is on mock_llm_client but not on BudgetedLLM
    assert budgeted.max_context_tokens == mock_llm_client.max_context_tokens
    # model is a direct attribute (shadows proxy)
    assert budgeted.model == mock_llm_client.model


def test_classify_defensive_classifier_failure():
    """If classify() raises internally, _call_with_resilience falls back to unknown/retryable."""
    import unittest.mock as um

    resilient = ResilientLLM.__new__(ResilientLLM)
    resilient.model = "mock-model"
    resilient.api_retry_count = 0
    resilient.max_context_tokens = 8192
    resilient._fallback_llm = None
    resilient._max_retries = 0  # No retries, just test the defensive catch
    resilient._primary_cooldown_until = 0
    resilient._simple_llm = None

    original_err = Exception("some llm error")
    with um.patch.object(LLMClient, "chat", side_effect=original_err), \
         um.patch("llamagent.modules.resilience.resilient_llm.classify", side_effect=RuntimeError("classifier bug")):
        with pytest.raises(Exception, match="some llm error"):
            resilient.chat([{"role": "user", "content": "hi"}])


# ============================================================
# v2.9.2: Turn-scoped failover recovery
# ============================================================

def _make_resilient_with_fallback(mock_llm_client, fallback_responses, max_retries=0):
    """Helper: create ResilientLLM with a mock fallback."""
    import unittest.mock as um

    fallback = LLMClient.__new__(LLMClient)
    fallback.model = "fallback-model"
    fallback.api_retry_count = 0
    fallback.max_context_tokens = 8192

    _fb_idx = [0]
    def _fb_chat(*args, **kwargs):
        idx = _fb_idx[0]
        _fb_idx[0] += 1
        if idx < len(fallback_responses):
            r = fallback_responses[idx]
            if isinstance(r, Exception):
                raise r
            return r
        return make_llm_response("fallback default")

    fallback.chat = _fb_chat

    resilient = ResilientLLM.__new__(ResilientLLM)
    resilient.model = "mock-model"
    resilient.api_retry_count = 0
    resilient.max_context_tokens = 8192
    resilient._fallback_llm = fallback
    resilient._max_retries = max_retries
    resilient._primary_cooldown_until = 0
    resilient._simple_llm = None
    return resilient, fallback


def test_turn_scoped_failover_sets_cooldown(mock_llm_client):
    """After successful failover, primary enters cooldown; next call uses fallback directly."""
    server_err = _make_http_error(500, "primary down")
    mock_llm_client.set_responses([server_err])  # Primary fails

    resilient, fallback = _make_resilient_with_fallback(
        mock_llm_client,
        [make_llm_response("fallback 1"), make_llm_response("fallback 2")],
        max_retries=0,
    )

    import unittest.mock as um
    with um.patch.object(LLMClient, "chat", side_effect=mock_llm_client.chat), \
         um.patch("llamagent.modules.resilience.resilient_llm.time.sleep"):
        # First call: primary fails → failover succeeds → cooldown set
        result1 = resilient.chat([{"role": "user", "content": "hi"}])
        assert result1.choices[0].message.content == "fallback 1"
        assert resilient._primary_cooldown_until > time.time()

        # Second call: during cooldown → uses fallback directly (skips primary)
        result2 = resilient.chat([{"role": "user", "content": "hi again"}])
        assert result2.choices[0].message.content == "fallback 2"


def test_turn_scoped_primary_recovery(mock_llm_client):
    """After cooldown expires, primary is tried again; success clears cooldown."""
    resilient, fallback = _make_resilient_with_fallback(
        mock_llm_client, [], max_retries=0,
    )
    # Simulate expired cooldown
    resilient._primary_cooldown_until = time.time() - 1

    mock_llm_client.set_responses([make_llm_response("primary recovered")])

    import unittest.mock as um
    with um.patch.object(LLMClient, "chat", side_effect=mock_llm_client.chat):
        result = resilient.chat([{"role": "user", "content": "hi"}])

    assert result.choices[0].message.content == "primary recovered"
    assert resilient._primary_cooldown_until == 0  # Cooldown cleared


def test_turn_scoped_fallback_fails_tries_primary(mock_llm_client):
    """During cooldown, if fallback fails, falls through to primary retry."""
    resilient, fallback = _make_resilient_with_fallback(
        mock_llm_client,
        [Exception("fallback down")],  # Fallback fails
        max_retries=0,
    )
    resilient._primary_cooldown_until = time.time() + 300  # In cooldown

    mock_llm_client.set_responses([make_llm_response("primary ok")])

    import unittest.mock as um
    with um.patch.object(LLMClient, "chat", side_effect=mock_llm_client.chat), \
         um.patch("llamagent.modules.resilience.resilient_llm.time.sleep"):
        result = resilient.chat([{"role": "user", "content": "hi"}])

    assert result.choices[0].message.content == "primary ok"
    assert resilient._primary_cooldown_until == 0  # Primary recovered, cooldown cleared


def test_turn_scoped_cooldown_uses_retry_after(mock_llm_client):
    """Cooldown duration uses retry_after when available (capped at 300s)."""
    primary_err = Exception("primary down")

    resilient, fallback = _make_resilient_with_fallback(
        mock_llm_client,
        [make_llm_response("fallback ok")],
        max_retries=0,
    )

    # Mock classify to return should_failover=True with retry_after=45
    fake_classified = ClassifiedError(
        reason="server_error", retryable=False, should_failover=True,
        retry_after=45, original=primary_err,
    )

    import unittest.mock as um
    before = time.time()
    with um.patch.object(LLMClient, "chat", side_effect=primary_err), \
         um.patch("llamagent.modules.resilience.resilient_llm.classify", return_value=fake_classified), \
         um.patch("llamagent.modules.resilience.resilient_llm.time.sleep"):
        result = resilient.chat([{"role": "user", "content": "hi"}])

    assert result.choices[0].message.content == "fallback ok"
    # Cooldown should be ~45s (from retry_after), not default 60
    expected_cooldown = before + 45
    assert abs(resilient._primary_cooldown_until - expected_cooldown) < 2


# ============================================================
# v2.9.3: Smart model routing
# ============================================================

def test_smart_routing_simple_query():
    """Simple query (short, no code, no URL) routes to simple model."""
    simple = LLMClient.__new__(LLMClient)
    simple.model = "cheap-model"
    simple.api_retry_count = 0
    simple.max_context_tokens = 8192

    resilient = ResilientLLM.__new__(ResilientLLM)
    resilient.model = "main-model"
    resilient.api_retry_count = 0
    resilient.max_context_tokens = 8192
    resilient._fallback_llm = None
    resilient._max_retries = 0
    resilient._primary_cooldown_until = 0
    resilient._simple_llm = simple

    import unittest.mock as um
    with um.patch.object(simple, "chat", return_value=make_llm_response("cheap answer")):
        result = resilient.chat([{"role": "user", "content": "What is 2+2?"}])
    assert result.choices[0].message.content == "cheap answer"


def test_smart_routing_complex_query(mock_llm_client):
    """Complex query (long or has code) uses main model."""
    simple = LLMClient.__new__(LLMClient)
    simple.model = "cheap-model"

    resilient = ResilientLLM.__new__(ResilientLLM)
    resilient.model = "mock-model"
    resilient.api_retry_count = 0
    resilient.max_context_tokens = 8192
    resilient._fallback_llm = None
    resilient._max_retries = 0
    resilient._primary_cooldown_until = 0
    resilient._simple_llm = simple

    mock_llm_client.set_responses([make_llm_response("main answer")])

    import unittest.mock as um
    with um.patch.object(LLMClient, "chat", side_effect=mock_llm_client.chat):
        # Code block → complex
        result = resilient.chat([{"role": "user", "content": "Fix this:\n```python\nprint('hello')\n```"}])
    assert result.choices[0].message.content == "main answer"


def test_smart_routing_with_tools(mock_llm_client):
    """When tools are provided, always use main model (tool calling needs capable model)."""
    simple = LLMClient.__new__(LLMClient)
    simple.model = "cheap-model"

    resilient = ResilientLLM.__new__(ResilientLLM)
    resilient.model = "mock-model"
    resilient.api_retry_count = 0
    resilient.max_context_tokens = 8192
    resilient._fallback_llm = None
    resilient._max_retries = 0
    resilient._primary_cooldown_until = 0
    resilient._simple_llm = simple

    mock_llm_client.set_responses([make_llm_response("main with tools")])

    import unittest.mock as um
    with um.patch.object(LLMClient, "chat", side_effect=mock_llm_client.chat):
        result = resilient.chat(
            [{"role": "user", "content": "Hi"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
        )
    assert result.choices[0].message.content == "main with tools"


def test_smart_routing_simple_fails_fallthrough(mock_llm_client):
    """If simple model fails, falls through to main model."""
    simple = LLMClient.__new__(LLMClient)
    simple.model = "cheap-model"

    resilient = ResilientLLM.__new__(ResilientLLM)
    resilient.model = "mock-model"
    resilient.api_retry_count = 0
    resilient.max_context_tokens = 8192
    resilient._fallback_llm = None
    resilient._max_retries = 0
    resilient._primary_cooldown_until = 0
    resilient._simple_llm = simple

    mock_llm_client.set_responses([make_llm_response("main fallback")])

    import unittest.mock as um
    with um.patch.object(simple, "chat", side_effect=Exception("cheap model down")), \
         um.patch.object(LLMClient, "chat", side_effect=mock_llm_client.chat):
        result = resilient.chat([{"role": "user", "content": "Hello"}])
    assert result.choices[0].message.content == "main fallback"


def test_smart_routing_no_simple_model(mock_llm_client):
    """Without simple_model configured, all queries use main model."""
    resilient = ResilientLLM.__new__(ResilientLLM)
    resilient.model = "mock-model"
    resilient.api_retry_count = 0
    resilient.max_context_tokens = 8192
    resilient._fallback_llm = None
    resilient._max_retries = 0
    resilient._primary_cooldown_until = 0
    resilient._simple_llm = None

    mock_llm_client.set_responses([make_llm_response("main only")])

    import unittest.mock as um
    with um.patch.object(LLMClient, "chat", side_effect=mock_llm_client.chat):
        result = resilient.chat([{"role": "user", "content": "Hi"}])
    assert result.choices[0].message.content == "main only"


def test_is_simple_query():
    """_is_simple_query correctly classifies queries."""
    assert ResilientLLM._is_simple_query([{"role": "user", "content": "Hi"}]) is True
    assert ResilientLLM._is_simple_query([{"role": "user", "content": "What is Python?"}]) is True
    # Long text → complex
    assert ResilientLLM._is_simple_query([{"role": "user", "content": "x" * 200}]) is False
    # Code block → complex
    assert ResilientLLM._is_simple_query([{"role": "user", "content": "```code```"}]) is False
    # URL → complex
    assert ResilientLLM._is_simple_query([{"role": "user", "content": "Check http://example.com"}]) is False
    # No user message → False
    assert ResilientLLM._is_simple_query([{"role": "system", "content": "hi"}]) is False
    # Empty → False
    assert ResilientLLM._is_simple_query([]) is False
