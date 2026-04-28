"""
Workspace and project sync flow tests: v1.6 tool system.

Tests cover ToolsModule registration, pack mechanism, workspace exploration,
project sync (apply_patch with preview, revert), workspace-only write restriction,
context injection (FILE_TOOL_GUIDE + capability hint block), and workspace lifecycle.
"""

import base64
import json
import os

from llamagent.modules.tools.module import ToolsModule, FILE_TOOL_GUIDE, CAPABILITY_HINT_BLOCK


# ============================================================
# Helpers
# ============================================================

def _make_agent_with_tools(bare_agent, tmp_path):
    """Set up bare_agent with project_dir + playground_dir (production
    layout: playground INSIDE project), then register ToolsModule.

    v3.3 routes file-tool paths via classify_write, which checks the
    playground prefix relative to project_dir. The pre-v3.3 sibling
    layout (playground at tmp_path/llama_playground) only worked while
    zone="playground" routed explicitly; with the kwarg gone, playground
    must physically live under project_dir for the model-supplied
    "llama_playground/..." path to land in the correct zone."""
    project_dir = os.path.realpath(os.path.join(str(tmp_path), "project"))
    playground_dir = os.path.realpath(os.path.join(project_dir, "llama_playground"))
    os.makedirs(playground_dir, exist_ok=True)

    bare_agent.project_dir = project_dir
    bare_agent.playground_dir = playground_dir

    mod = ToolsModule()
    bare_agent.register_module(mod)
    return mod


def _call_tool(agent, name, **kwargs):
    """Invoke a registered tool by name and return the raw result string."""
    func = agent._tools[name]["func"]
    return func(**kwargs)


def _call_tool_json(agent, name, **kwargs):
    """Invoke a registered tool by name and return parsed JSON."""
    raw = _call_tool(agent, name, **kwargs)
    return json.loads(raw)


# ============================================================
# Flow tests
# ============================================================

class TestToolRegistrationAndPackFiltering:
    """Default tools registered, pack tools hidden/visible, merged tools removed, toolsmith pack."""

    def test_tool_registration_and_pack_filtering(self, bare_agent, tmp_path):
        _make_agent_with_tools(bare_agent, tmp_path)

        # -- Default surface tools registered without pack (v3.3: 5 core tools) --
        default_tools = [
            "list_tree", "read_files", "write_files",
            "apply_patch", "revert_changes",
        ]
        for tool_name in default_tools:
            assert tool_name in bare_agent._tools, f"default tool '{tool_name}' not registered"
            assert bare_agent._tools[tool_name].get("pack") is None, \
                f"'{tool_name}' should not have a pack (default surface)"

        # -- v3.3: sync_workspace_to_project deleted --
        assert "sync_workspace_to_project" not in bare_agent._tools

        # -- path-fallback tools registered (auto-load when no shell tool).
        #    v3.3 includes search_text in this pack (B1 decision); shell
        #    `grep -rn pattern .` is the v3.3 preferred path. --
        pack_expectations = {
            "glob_files": "path-fallback",
            "search_text": "path-fallback",
            "stat_paths": "path-fallback",
            "create_temp_file": "path-fallback",
            "move_path": "path-fallback",
            "copy_path": "path-fallback",
            "delete_path": "path-fallback",
        }
        for tool_name, expected_pack in pack_expectations.items():
            assert tool_name in bare_agent._tools, f"pack tool '{tool_name}' not registered"
            assert bare_agent._tools[tool_name].get("pack") == expected_pack

        # -- Merged tools removed --
        removed = ["read_ranges", "preview_patch", "replace_block"]
        for tool_name in removed:
            assert tool_name not in bare_agent._tools, f"'{tool_name}' should have been removed"

        # -- Pack tools hidden from LLM by default --
        schemas = bare_agent.get_all_tool_schemas()
        schema_names = [s["function"]["name"] for s in schemas]

        assert "read_files" in schema_names
        assert "write_files" in schema_names
        assert "apply_patch" in schema_names

        # -- Toolsmith pack hidden by default --
        assert "create_tool" not in schema_names
        assert "list_my_tools" not in schema_names
        assert "delete_tool" not in schema_names

        # -- path-fallback tools auto-visible when no shell tool registered --
        # (bare_agent has no JobModule / SandboxModule, so path-fallback should
        # auto-activate via on_input/on_context.)
        bare_agent._active_packs.clear()
        mod = bare_agent.modules["tools"]
        mod.on_input("organize my files")
        mod.on_context("organize my files", "")
        assert "path-fallback" in bare_agent._active_packs, \
            "path-fallback should auto-activate when no shell tool is available"
        schemas = bare_agent.get_all_tool_schemas()
        schema_names = [s["function"]["name"] for s in schemas]
        assert "glob_files" in schema_names
        assert "move_path" in schema_names
        assert "stat_paths" in schema_names


class TestContextInjection:
    """on_context injects FILE_TOOL_GUIDE + CAPABILITY_HINT_BLOCK."""

    def test_context_injection(self, bare_agent, tmp_path):
        mod = _make_agent_with_tools(bare_agent, tmp_path)

        # -- Guide and hints injected into existing context --
        result = mod.on_context("test query", "existing context")
        assert FILE_TOOL_GUIDE in result
        assert CAPABILITY_HINT_BLOCK in result
        assert "existing context" in result

        # -- Guide and hints injected when context is empty --
        result = mod.on_context("test query", "")
        assert FILE_TOOL_GUIDE in result
        assert CAPABILITY_HINT_BLOCK in result


class TestWorkspaceExplorationAndFileTypes:
    """list_tree, read_files (zone parameter), ranges, text/binary detection."""

    def test_workspace_exploration_and_file_types(self, bare_agent, tmp_path):
        _make_agent_with_tools(bare_agent, tmp_path)

        # -- write_files creates files in project (v3.3 default) --
        result = _call_tool_json(bare_agent, "write_files", files={
            "src/main.py": "print('hello')",
            "README.md": "# Project",
        })
        assert result["status"] == "success"

        tree_result = _call_tool_json(bare_agent, "list_tree")
        assert "main.py" in tree_result["tree"]
        assert "README.md" in tree_result["tree"]

        # -- read_files default = project (v3.3, no prefix) --
        project_dir = bare_agent.project_dir
        with open(os.path.join(project_dir, "test.txt"), "w") as f:
            f.write("project file content")

        result = _call_tool_json(bare_agent, "read_files", paths=["test.txt"])
        assert result["status"] == "success"
        assert "project file content" in result["files"][0]["content"]

        # -- read_files with ranges reads specific line ranges --
        _call_tool_json(bare_agent, "write_files", files={
            "lines.txt": "\n".join(f"line{i}" for i in range(1, 21)),
        })

        result = _call_tool_json(bare_agent, "read_files",
            paths=["lines.txt"], ranges={"lines.txt": "5-8"})
        assert result["status"] == "success"
        content = result["files"][0]["content"]
        assert "line5" in content
        assert "line8" in content
        assert "line1\t" not in content

        # -- Text file (.py) returns content with line numbers in auto mode --
        _call_tool_json(bare_agent, "write_files", files={
            "hello.py": "print('hello')\nprint('world')\n",
        })

        result = _call_tool_json(bare_agent, "read_files", paths=["hello.py"])
        assert result["status"] == "success"
        file_info = result["files"][0]
        assert "content" in file_info
        assert "hello" in file_info["content"]
        assert file_info.get("binary") is not True

        # -- Binary file (.png) returns metadata only in auto mode --
        bin_path = os.path.join(project_dir, "image.png")
        with open(bin_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

        result = _call_tool_json(bare_agent, "read_files", paths=["image.png"])
        assert result["status"] == "success"
        file_info = result["files"][0]
        assert file_info["binary"] is True
        assert "size" in file_info
        assert "mime" in file_info
        assert "content" not in file_info

        # -- mode='binary' returns base64-encoded content --
        bin_path2 = os.path.join(project_dir, "data.bin")
        original_bytes = b"\x00\x01\x02\x03\xff\xfe\xfd"
        with open(bin_path2, "wb") as f:
            f.write(original_bytes)

        result = _call_tool_json(bare_agent, "read_files", paths=["data.bin"], mode="binary")
        assert result["status"] == "success"
        file_info = result["files"][0]
        assert file_info["binary"] is True
        assert "content_base64" in file_info
        decoded = base64.b64decode(file_info["content_base64"])
        assert decoded == original_bytes

        # -- File without extension but text content is detected as text --
        _call_tool_json(bare_agent, "write_files", files={
            "Makefile": "all:\n\techo hello\n",
        })

        result = _call_tool_json(bare_agent, "read_files", paths=["Makefile"])
        assert result["status"] == "success"
        file_info = result["files"][0]
        assert "content" in file_info
        assert "echo hello" in file_info["content"]

        # -- v3.3: writes/reads under llama_playground/ route to playground
        #    semantic via classify_write (write succeeds, no changeset). --
        result = _call_tool_json(bare_agent, "write_files",
            files={"llama_playground/scratch.txt": "scratch contents"})
        assert result["status"] == "success"
        scratch_path = os.path.join(bare_agent.playground_dir, "scratch.txt")
        assert os.path.isfile(scratch_path)
        assert open(scratch_path).read() == "scratch contents"

        result = _call_tool_json(bare_agent, "read_files",
            paths=["llama_playground/scratch.txt"])
        assert result["status"] == "success"
        assert "scratch contents" in result["files"][0]["content"]


class TestWriteFilesAndRestrictions:
    """v3.3: writes default to project (write_root); zone='playground' for scratch."""

    def test_write_files_and_restrictions(self, bare_agent, tmp_path):
        _make_agent_with_tools(bare_agent, tmp_path)
        project_dir = bare_agent.project_dir

        # -- mode='binary' writes base64-decoded bytes to project (default) --
        original_bytes = b"\x89PNG\r\n\x1a\n\x00\x01\x02\x03"
        b64_content = base64.b64encode(original_bytes).decode("ascii")

        result = _call_tool_json(bare_agent, "write_files",
            files={"output.png": b64_content}, mode="binary")
        assert result["status"] == "success"

        with open(os.path.join(project_dir, "output.png"), "rb") as f:
            assert f.read() == original_bytes

        # -- Default mode='text' writes UTF-8 string content to project --
        result = _call_tool_json(bare_agent, "write_files",
            files={"test.txt": "hello world"})
        assert result["status"] == "success"

        with open(os.path.join(project_dir, "test.txt"), "r") as f:
            assert f.read() == "hello world"

        # -- v3.3: 'project:' prefix is no longer special; literal path
        #    "project:hack.py" treated as a filename, fails containment
        #    check (the colon doesn't prevent containment but '..' would). --
        # -- write_files rejects path-escape attempts --
        result = _call_tool_json(bare_agent, "write_files", files={
            "../escape.py": "malicious content",
        })
        assert result["status"] == "partial"
        assert len(result["errors"]) == 1
        assert "escape" in result["errors"][0]["error"].lower()

        # -- write_files allows normal project-relative paths --
        result = _call_tool_json(bare_agent, "write_files", files={
            "normal.txt": "safe content",
        })
        assert result["status"] == "success"
        assert "normal.txt" in result["written"]

        # -- v3.3: paths under llama_playground/ route to playground (no
        #    changeset), not project. Same-named file at project root is
        #    a separate path. --
        result = _call_tool_json(bare_agent, "write_files",
            files={"llama_playground/scratch.json": "{\"x\": 1}"})
        assert result["status"] == "success"
        assert os.path.isfile(os.path.join(bare_agent.playground_dir, "scratch.json"))
        # Project root file with same basename was NOT created.
        assert not os.path.isfile(os.path.join(project_dir, "scratch.json"))


class TestPatchAndSyncLifecycle:
    """apply_patch atomic, preview, revert, sync to project, shutdown cleanup."""

    def test_patch_and_sync_lifecycle(self, bare_agent, tmp_path):
        mod = _make_agent_with_tools(bare_agent, tmp_path)
        project_dir = bare_agent.project_dir

        # -- apply_patch is atomic: all-or-nothing on multiple edits --
        target = os.path.join(project_dir, "code.py")
        original = "def hello():\n    return 'hello'\n\ndef world():\n    return 'world'\n"
        with open(target, "w") as f:
            f.write(original)

        result = _call_tool_json(bare_agent, "apply_patch", target="code.py", edits=[
            {"match": "'hello'", "replace": "'hi'"},
            {"match": "'world'", "replace": "'earth'"},
        ])
        assert result["status"] == "success"
        assert result["edits_applied"] == 2

        with open(target) as f:
            content = f.read()

        result2 = _call_tool_json(bare_agent, "apply_patch", target="code.py", edits=[
            {"match": "'hi'", "replace": "'hey'"},
            {"match": "NONEXISTENT", "replace": "x"},
        ])
        assert result2["status"] == "error"
        with open(target) as f:
            assert f.read() == content

        # -- apply_patch with preview=True validates without writing --
        target2 = os.path.join(project_dir, "preview.txt")
        original2 = "original content\n"
        with open(target2, "w") as f:
            f.write(original2)

        result = _call_tool_json(bare_agent, "apply_patch", target="preview.txt",
            edits=[{"match": "original", "replace": "modified"}], preview=True)
        assert result["status"] == "preview"
        assert result.get("edits_valid") is True

        with open(target2) as f:
            assert f.read() == original2

        # -- revert_changes restores pre-patch state --
        target3 = os.path.join(project_dir, "data.txt")
        original3 = "line one\nline two\nline three\n"
        with open(target3, "w") as f:
            f.write(original3)

        _call_tool_json(bare_agent, "apply_patch", target="data.txt", edits=[
            {"match": "line two", "replace": "LINE TWO MODIFIED"},
        ])

        revert_result = _call_tool_json(bare_agent, "revert_changes")
        assert revert_result["status"] == "success"

        with open(target3) as f:
            assert f.read() == original3

        # v3.3: sync_workspace_to_project removed; project writes go through
        # write_files / apply_patch directly. The lifecycle test ends here.

        # -- on_shutdown removes the scratch session directory --
        _call_tool_json(bare_agent, "write_files", files={"temp.txt": "temporary"})

        scratch_root = mod.scratch_service.scratch_root
        assert os.path.isdir(scratch_root)

        mod.on_shutdown()

        session_dir = os.path.join(
            bare_agent.playground_dir, "sessions",
            mod.scratch_service.scratch_id,
        )
        assert not os.path.isdir(session_dir)


# ============================================================
# v3.3: write_root primitive + write_files changeset tracking
# ============================================================

class TestWriteRootBoundary:
    """v3.3 D8: write_root is the single write boundary used by all
    typed write tools. Verify the soft-fallback semantics for invalid
    edit_root values and the boundary enforcement."""

    def test_write_root_defaults_to_project_dir(self, bare_agent, tmp_path):
        _make_agent_with_tools(bare_agent, tmp_path)
        assert bare_agent.write_root == os.path.realpath(bare_agent.project_dir)

    def test_edit_root_narrows_write_root(self, bare_agent, tmp_path):
        # Production layout: playground inside project (matches v3.3
        # classify_write expectations).
        project_dir = os.path.realpath(os.path.join(str(tmp_path), "project"))
        os.makedirs(os.path.join(project_dir, "src"), exist_ok=True)
        playground_dir = os.path.realpath(os.path.join(project_dir, "llama_playground"))
        os.makedirs(playground_dir, exist_ok=True)
        bare_agent.config.edit_root = "src"
        bare_agent.project_dir = project_dir
        bare_agent.playground_dir = playground_dir
        bare_agent.register_module(ToolsModule())

        # write_root is now project/src; writes outside src are rejected.
        assert bare_agent.write_root.endswith("/src")

        # v3.3: paths are relative to project_dir (not write_root). Write
        # 'src/main.py' to land inside the narrowed write_root.
        result = _call_tool_json(bare_agent, "write_files",
            files={"src/main.py": "x"})
        assert result["status"] == "success"
        assert os.path.isfile(os.path.join(project_dir, "src", "main.py"))

        # Writing a path outside write_root (project root, not under src/)
        # → rejected.
        result = _call_tool_json(bare_agent, "write_files",
            files={"README.md": "y"})
        assert result["status"] == "partial"
        assert len(result["errors"]) == 1
        assert "outside writable root" in result["errors"][0]["error"]

    def test_invalid_edit_root_falls_back_with_warning(self, bare_agent, tmp_path, caplog):
        project_dir = os.path.join(str(tmp_path), "project")
        os.makedirs(project_dir, exist_ok=True)
        bare_agent.config.edit_root = "../../../escape"
        bare_agent.project_dir = project_dir
        bare_agent.playground_dir = os.path.join(str(tmp_path), "llama_playground")
        os.makedirs(bare_agent.playground_dir, exist_ok=True)

        with caplog.at_level("WARNING"):
            bare_agent.register_module(ToolsModule())

        # Falls back to project_dir.
        assert bare_agent.write_root == os.path.realpath(project_dir)
        # Warning was emitted.
        assert any("escapes project_dir" in r.message for r in caplog.records)


class TestWriteFilesChangesetTracking:
    """v3.3 D6: write_files records a changeset so revert_changes can
    undo a write. Playground writes are NOT tracked."""

    def test_write_then_revert_restores_original(self, bare_agent, tmp_path):
        _make_agent_with_tools(bare_agent, tmp_path)
        # Seed an existing project file.
        target = os.path.join(bare_agent.project_dir, "config.txt")
        with open(target, "w") as f:
            f.write("version=1.0\n")

        # Overwrite via write_files.
        _call_tool_json(bare_agent, "write_files",
            files={"config.txt": "version=2.0\n"})
        assert open(target).read() == "version=2.0\n"

        # Revert restores the original.
        result = _call_tool_json(bare_agent, "revert_changes")
        assert result["status"] == "success"
        assert open(target).read() == "version=1.0\n"

    def test_write_new_file_then_revert_deletes(self, bare_agent, tmp_path):
        _make_agent_with_tools(bare_agent, tmp_path)
        target = os.path.join(bare_agent.project_dir, "new_file.txt")
        assert not os.path.exists(target)

        _call_tool_json(bare_agent, "write_files",
            files={"new_file.txt": "fresh content"})
        assert os.path.isfile(target)

        result = _call_tool_json(bare_agent, "revert_changes")
        assert result["status"] == "success"
        assert not os.path.exists(target)

    def test_playground_writes_are_not_tracked(self, bare_agent, tmp_path):
        _make_agent_with_tools(bare_agent, tmp_path)

        _call_tool_json(bare_agent, "write_files",
            files={"llama_playground/scratch.txt": "ephemeral"})

        # No changeset created → revert finds nothing.
        result = _call_tool_json(bare_agent, "revert_changes")
        assert result["status"] == "error"


class TestChangesetLRU:
    """v3.3 §五 changeset cap: LRU eviction + evicted ledger surfaces a
    precise error in revert_changes."""

    def test_lru_evicts_oldest_unreverted_when_count_exceeded(self, bare_agent, tmp_path):
        # Tighten the cap to make the test fast.
        bare_agent.config.changeset_max_count = 3
        bare_agent.config.changeset_max_total_bytes = 0  # disable byte cap
        _make_agent_with_tools(bare_agent, tmp_path)

        # Apply 5 patches → only the latest 3 changesets remain.
        for i in range(5):
            target = os.path.join(bare_agent.project_dir, f"f{i}.txt")
            with open(target, "w") as f:
                f.write(f"v{i}\n")
            _call_tool_json(bare_agent, "apply_patch",
                target=f"f{i}.txt", edits=[{"match": f"v{i}", "replace": "X"}])

        ps = bare_agent.modules["tools"].project_sync_service
        assert len(ps._changesets) == 3
        # The oldest 2 paths are in the evicted ledger.
        assert any("f0.txt" in p for p in ps._evicted_paths)
        assert any("f1.txt" in p for p in ps._evicted_paths)

    def test_command_tool_registered_and_runs(self, bare_agent, tmp_path):
        """v3.3 D4: SandboxModule registers the `command` tool and the
        engine's _evaluate_command classifies safe commands as ALLOW."""
        from llamagent.modules.sandbox.module import SandboxModule

        _make_agent_with_tools(bare_agent, tmp_path)
        bare_agent.register_module(SandboxModule(auto_assign=False))

        assert "command" in bare_agent._tools

        # Run a benign command — engine classifies as ALLOW (default).
        raw = bare_agent._tools["command"]["func"](cmd="echo hello")
        result = json.loads(raw)
        assert result["status"] == "success"
        assert "hello" in result["stdout"]

    def test_command_safety_hard_reject(self, bare_agent, tmp_path):
        """`rm -rf /` is auto-classified as hard_reject by the engine."""
        from llamagent.modules.sandbox.module import SandboxModule

        _make_agent_with_tools(bare_agent, tmp_path)
        bare_agent.register_module(SandboxModule(auto_assign=False))

        engine = bare_agent._authorization_engine
        tool = bare_agent._tools["command"]
        result = engine.evaluate(tool, {"cmd": "rm -rf /"})
        # Decision is non-None when blocked.
        assert result.decision is not None
        assert "block" in result.decision.lower() or "denied" in result.decision.lower() \
            or "hard" in str(result.events).lower() or "deny" in str(result.events).lower()

    def test_command_safety_ask_for_rm(self, bare_agent, tmp_path):
        """Plain `rm foo.txt` is classified as ASK (CONFIRMABLE)."""
        from llamagent.modules.sandbox.module import SandboxModule
        from llamagent.core.zone import ZoneVerdict

        _make_agent_with_tools(bare_agent, tmp_path)
        bare_agent.register_module(SandboxModule(auto_assign=False))

        engine = bare_agent._authorization_engine
        evaluation = engine._evaluate_command({"cmd": "rm somefile.txt"})
        assert evaluation.overall_verdict == ZoneVerdict.CONFIRMABLE

    def test_command_safety_default_allow(self, bare_agent, tmp_path):
        """Read-only commands (ls / cat / grep) are not flagged."""
        from llamagent.modules.sandbox.module import SandboxModule
        from llamagent.core.zone import ZoneVerdict

        _make_agent_with_tools(bare_agent, tmp_path)
        bare_agent.register_module(SandboxModule(auto_assign=False))

        engine = bare_agent._authorization_engine
        for cmd in ("ls -la", "cat README.md", "grep foo bar.txt"):
            evaluation = engine._evaluate_command({"cmd": cmd})
            assert evaluation.overall_verdict == ZoneVerdict.ALLOW, \
                f"Expected ALLOW for {cmd!r}, got {evaluation.overall_verdict}"

    def test_command_pattern_scope_short_circuits_ask(self, bare_agent, tmp_path):
        """ApprovalScope.command_patterns can pre-approve a pattern."""
        from llamagent.modules.sandbox.module import SandboxModule
        from llamagent.core.zone import ZoneVerdict
        from llamagent.core.authorization import ApprovalScope

        _make_agent_with_tools(bare_agent, tmp_path)
        bare_agent.register_module(SandboxModule(auto_assign=False))

        engine = bare_agent._authorization_engine
        # Without the scope: rm is CONFIRMABLE.
        ev1 = engine._evaluate_command({"cmd": "rm node_modules"})
        assert ev1.overall_verdict == ZoneVerdict.CONFIRMABLE

        # Add a session-scope command_pattern allowance.
        engine.add_scope(ApprovalScope(
            scope="session", zone="external",
            actions=["execute"], path_prefixes=[],
            tool_names=["command"],
            command_patterns=["rm node_modules*"],
        ))
        ev2 = engine._evaluate_command({"cmd": "rm node_modules"})
        assert ev2.overall_verdict == ZoneVerdict.ALLOW

    def test_snapshot_taken_before_first_project_write(self, bare_agent, tmp_path):
        """v3.3 D7 (eager init): snapshot is captured at agent.ensure_snapshot()
        which runs in LlamAgent.__init__. Tests that bypass __init__ (the
        bare_agent fixture pattern) trigger it explicitly after wiring."""
        _make_agent_with_tools(bare_agent, tmp_path)
        bare_agent.config.snapshot_enabled = True
        # Seed an existing project file we'll modify.
        original_path = os.path.join(bare_agent.project_dir, "main.py")
        with open(original_path, "w") as f:
            f.write("v1\n")

        # Tests that build agent via __new__ (bypassing __init__) must
        # explicitly trigger snapshot after wiring config + project_dir.
        bare_agent.ensure_snapshot()

        # Subsequent write does NOT re-trigger snapshot (it's idempotent
        # and was already captured above).
        _call_tool_json(bare_agent, "write_files",
            files={"main.py": "v2\n"})

        snap_dir = bare_agent._snapshot_service._snapshot_dir
        assert snap_dir is not None and os.path.isdir(snap_dir)
        # Manifest exists with write_root recorded.
        manifest_path = os.path.join(snap_dir, "MANIFEST.json")
        assert os.path.isfile(manifest_path)
        manifest = json.loads(open(manifest_path).read())
        assert manifest["write_root"] == os.path.realpath(bare_agent.project_dir)
        # Tree captured the original content.
        snap_main = os.path.join(snap_dir, "tree", "main.py")
        assert os.path.isfile(snap_main)
        assert open(snap_main).read() == "v1\n"

    def test_snapshot_disabled_by_default(self, bare_agent, tmp_path):
        """In interactive mode (auto_approve=False, snapshot_enabled=False),
        no snapshot is captured."""
        _make_agent_with_tools(bare_agent, tmp_path)
        # Default state — neither flag set.
        bare_agent.config.snapshot_enabled = False
        bare_agent.config.auto_approve = False

        _call_tool_json(bare_agent, "write_files",
            files={"x.txt": "1"})

        # ensure_snapshot was called but is_enabled returned False.
        assert getattr(bare_agent, "_snapshot_service", None) is None or \
            bare_agent._snapshot_service._snapshot_dir is None

    def test_write_files_refuses_text_overwrite_of_binary(self, bare_agent, tmp_path):
        """v3.3 fixup: text-mode write to a pre-existing binary file is
        rejected with a clear hint, NOT silently registered with
        pre_image=None (which would cause revert to delete the file)."""
        _make_agent_with_tools(bare_agent, tmp_path)
        # Seed a binary file in the project.
        target = os.path.join(bare_agent.project_dir, "model.bin")
        with open(target, "wb") as f:
            f.write(b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\xff\xfe\xfd")

        # Attempt to overwrite via text mode.
        result = _call_tool_json(bare_agent, "write_files",
            files={"model.bin": "this would clobber the binary"})
        assert result["status"] == "partial"
        assert len(result["errors"]) == 1
        err = result["errors"][0]["error"].lower()
        assert "binary" in err and "mode='binary'" in err

        # Original binary content preserved.
        with open(target, "rb") as f:
            assert f.read() == b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\xff\xfe\xfd"

        # No changeset registered for this rejected write.
        ps = bare_agent.modules["tools"].project_sync_service
        assert all(cs.target_path != os.path.realpath(target)
                   for cs in ps._changesets)

    def test_snapshot_force_enabled_under_auto_approve(self, bare_agent, tmp_path):
        """auto_approve=True force-enables snapshot even if user
        didn't set snapshot_enabled."""
        _make_agent_with_tools(bare_agent, tmp_path)
        bare_agent.config.auto_approve = True
        bare_agent.config.snapshot_enabled = False

        # Eager init mode — explicitly trigger after config wiring (the
        # bare_agent fixture bypasses __init__).
        bare_agent.ensure_snapshot()

        _call_tool_json(bare_agent, "write_files",
            files={"y.txt": "2"})

        assert bare_agent._snapshot_service._snapshot_dir is not None

    def test_snapshot_excludes_playground_subdir(self, bare_agent, tmp_path):
        """v3.3 commit-14: snapshot copies write_root but skips the
        playground subtree (ephemeral framework scratch). Without this
        exclusion, snapshots inflate with tool_results/ files that
        provide no recovery value."""
        _make_agent_with_tools(bare_agent, tmp_path)
        bare_agent.config.snapshot_enabled = True
        # Seed a project file (should snapshot) and a playground file
        # (should NOT snapshot).
        with open(os.path.join(bare_agent.project_dir, "main.py"), "w") as f:
            f.write("project content\n")
        with open(os.path.join(bare_agent.playground_dir, "scratch.txt"), "w") as f:
            f.write("ephemeral content\n")

        bare_agent.ensure_snapshot()
        snap_dir = bare_agent._snapshot_service._snapshot_dir
        assert snap_dir is not None and os.path.isdir(snap_dir)
        tree = os.path.join(snap_dir, "tree")
        # Project file is captured
        assert os.path.isfile(os.path.join(tree, "main.py"))
        # Playground subtree is NOT captured
        assert not os.path.exists(os.path.join(tree, "llama_playground"))

    def test_byte_cap_evicts_when_pre_image_total_too_large(self, bare_agent, tmp_path):
        """v3.3 §五: changeset_max_total_bytes triggers eviction even
        when count cap is loose. Verifies the byte-cap branch (was
        untested pre-fixup #4)."""
        # Disable count cap; cap pre_image bytes at 200.
        bare_agent.config.changeset_max_count = 0
        bare_agent.config.changeset_max_total_bytes = 200
        _make_agent_with_tools(bare_agent, tmp_path)

        # Each patch keeps a pre_image of ~80 bytes. After 3 patches we
        # should be over the 200-byte cap and the oldest changesets get
        # evicted.
        for i in range(4):
            target = os.path.join(bare_agent.project_dir, f"big{i}.txt")
            with open(target, "w") as f:
                f.write(f"v{i} " + "x" * 70 + "\n")
            _call_tool_json(bare_agent, "apply_patch", target=f"big{i}.txt",
                edits=[{"match": f"v{i}", "replace": "Y"}])

        ps = bare_agent.modules["tools"].project_sync_service
        total = sum(len(cs.pre_image) for cs in ps._changesets if cs.pre_image)
        assert total <= 200, f"byte cap not enforced; total={total}"
        # At least one path was evicted.
        assert ps._evicted_paths

    def test_pass1_drops_only_enough_tombstones(self, bare_agent, tmp_path):
        """v3.3 fixup: Pass 1 (reverted/tombstone eviction) re-checks
        _over() after each drop, instead of dropping every tombstone in
        a single pass. Prevents over-eviction when tombstones are many
        but cap is only slightly exceeded."""
        bare_agent.config.changeset_max_count = 3
        bare_agent.config.changeset_max_total_bytes = 0
        _make_agent_with_tools(bare_agent, tmp_path)

        # Apply 3 patches and revert all 3 (3 tombstones, count == cap).
        for i in range(3):
            target = os.path.join(bare_agent.project_dir, f"f{i}.txt")
            with open(target, "w") as f:
                f.write(f"v{i}\n")
            _call_tool_json(bare_agent, "apply_patch", target=f"f{i}.txt",
                edits=[{"match": f"v{i}", "replace": "Y"}])
        # Revert each (creates 3 tombstones, count still 3).
        for i in range(3):
            _call_tool_json(bare_agent, "revert_changes",
                targets=[f"f{i}.txt"])

        ps = bare_agent.modules["tools"].project_sync_service
        # All 3 tombstones still present — count == cap, not over.
        assert len(ps._changesets) == 3
        assert all(cs.reverted for cs in ps._changesets)

        # Add one more patch → over the cap → Pass 1 drops ONE tombstone
        # (the oldest), keeping the rest.
        target = os.path.join(bare_agent.project_dir, "fresh.txt")
        with open(target, "w") as f:
            f.write("z\n")
        _call_tool_json(bare_agent, "apply_patch", target="fresh.txt",
            edits=[{"match": "z", "replace": "X"}])

        # 3 left: 2 tombstones + 1 fresh. Over-evict bug would have
        # dropped all 3 tombstones, leaving only the fresh one.
        assert len(ps._changesets) == 3, \
            f"Pass 1 over-dropped tombstones; left {len(ps._changesets)}"
        live_count = sum(1 for cs in ps._changesets if not cs.reverted)
        assert live_count == 1
        tomb_count = sum(1 for cs in ps._changesets if cs.reverted)
        assert tomb_count == 2

    def test_command_pattern_scope_filters_by_tool_name(self, bare_agent, tmp_path):
        """v3.3 fixup: an ApprovalScope whose tool_names does NOT
        contain 'command' must not pre-approve a command pattern,
        even if it carries a command_patterns field."""
        from llamagent.modules.sandbox.module import SandboxModule
        from llamagent.core.zone import ZoneVerdict
        from llamagent.core.authorization import ApprovalScope

        _make_agent_with_tools(bare_agent, tmp_path)
        bare_agent.register_module(SandboxModule(auto_assign=False))
        engine = bare_agent._authorization_engine

        # Scope intended for some other tool, with command_patterns set
        # by mistake. Should NOT short-circuit the command engine.
        engine.add_scope(ApprovalScope(
            scope="session", zone="external",
            actions=["execute"], path_prefixes=[],
            tool_names=["some_other_tool"],   # ← not "command"
            command_patterns=["rm node_modules*"],
        ))

        # `rm node_modules` would be ASK without scope short-circuit.
        ev = engine._evaluate_command({"cmd": "rm node_modules"})
        assert ev.overall_verdict == ZoneVerdict.CONFIRMABLE, \
            "scope without 'command' in tool_names must NOT pre-approve"

    def test_revert_evicted_path_surfaces_precise_error(self, bare_agent, tmp_path):
        bare_agent.config.changeset_max_count = 2
        bare_agent.config.changeset_max_total_bytes = 0
        _make_agent_with_tools(bare_agent, tmp_path)

        # 3 writes → first changeset evicted.
        for i in range(3):
            target = os.path.join(bare_agent.project_dir, f"g{i}.txt")
            with open(target, "w") as f:
                f.write(f"v{i}\n")
            _call_tool_json(bare_agent, "apply_patch",
                target=f"g{i}.txt", edits=[{"match": f"v{i}", "replace": "X"}])

        # Try to revert the evicted (g0.txt) target — error mentions eviction.
        result = _call_tool_json(bare_agent, "revert_changes",
            targets=["g0.txt"])
        assert result["status"] == "error"
        # Either the top-level error or per-file error mentions eviction.
        msg_blob = json.dumps(result)
        assert "evicted" in msg_blob.lower()
