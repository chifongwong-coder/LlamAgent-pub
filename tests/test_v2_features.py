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

    assert "task_id:" in result1
    assert "task_id:" in result2
    # Should NOT contain the child's actual result text (that comes from collect)
    assert "result from child" not in result1
    assert "result from child" not in result2

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
