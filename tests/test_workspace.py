"""
Workspace and project sync flow tests: v1.6 tool system.

Tests cover ToolsModule registration, pack mechanism, workspace exploration,
project sync (apply_patch with preview, revert), workspace-only write restriction,
context injection (WORKSPACE_GUIDE + capability hint block), and workspace lifecycle.
"""

import base64
import json
import os

from llamagent.modules.tools.module import ToolsModule, WORKSPACE_GUIDE, CAPABILITY_HINT_BLOCK


# ============================================================
# Helpers
# ============================================================

def _make_agent_with_tools(bare_agent, tmp_path):
    """Set up bare_agent with playground_dir and project_dir, then register ToolsModule."""
    project_dir = os.path.join(str(tmp_path), "project")
    playground_dir = os.path.join(str(tmp_path), "llama_playground")
    os.makedirs(project_dir, exist_ok=True)
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
            "list_tree", "search_text", "read_files", "write_files",
            "apply_patch", "revert_changes",
        ]
        for tool_name in default_tools:
            assert tool_name in bare_agent._tools, f"default tool '{tool_name}' not registered"
            assert bare_agent._tools[tool_name].get("pack") is None, \
                f"'{tool_name}' should not have a pack (default surface)"

        # -- v3.3: sync_workspace_to_project deleted --
        assert "sync_workspace_to_project" not in bare_agent._tools

        # -- path-fallback tools registered (auto-load when no shell tool) --
        pack_expectations = {
            "glob_files": "path-fallback",
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
    """on_context injects WORKSPACE_GUIDE + CAPABILITY_HINT_BLOCK."""

    def test_context_injection(self, bare_agent, tmp_path):
        mod = _make_agent_with_tools(bare_agent, tmp_path)

        # -- Guide and hints injected into existing context --
        result = mod.on_context("test query", "existing context")
        assert WORKSPACE_GUIDE in result
        assert CAPABILITY_HINT_BLOCK in result
        assert "existing context" in result

        # -- Guide and hints injected when context is empty --
        result = mod.on_context("test query", "")
        assert WORKSPACE_GUIDE in result
        assert CAPABILITY_HINT_BLOCK in result


class TestWorkspaceExplorationAndFileTypes:
    """list_tree, read_files with project prefix, ranges, text/binary detection."""

    def test_workspace_exploration_and_file_types(self, bare_agent, tmp_path):
        _make_agent_with_tools(bare_agent, tmp_path)

        # -- write_files creates files, list_tree shows them --
        result = _call_tool_json(bare_agent, "write_files", files={
            "src/main.py": "print('hello')",
            "README.md": "# Project",
        })
        assert result["status"] == "success"

        tree_result = _call_tool_json(bare_agent, "list_tree")
        assert "main.py" in tree_result["tree"]
        assert "README.md" in tree_result["tree"]

        # -- read_files resolves project: prefix to project_dir --
        project_dir = bare_agent.project_dir
        with open(os.path.join(project_dir, "test.txt"), "w") as f:
            f.write("project file content")

        result = _call_tool_json(bare_agent, "read_files", paths=["project:test.txt"])
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
        ws_root = bare_agent.modules["tools"].workspace_service.workspace_root
        bin_path = os.path.join(ws_root, "image.png")
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
        bin_path2 = os.path.join(ws_root, "data.bin")
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


class TestWriteFilesAndRestrictions:
    """Binary write, text write, reject project prefix, allow workspace."""

    def test_write_files_and_restrictions(self, bare_agent, tmp_path):
        _make_agent_with_tools(bare_agent, tmp_path)

        # -- mode='binary' writes base64-decoded bytes --
        original_bytes = b"\x89PNG\r\n\x1a\n\x00\x01\x02\x03"
        b64_content = base64.b64encode(original_bytes).decode("ascii")

        result = _call_tool_json(bare_agent, "write_files",
            files={"output.png": b64_content}, mode="binary")
        assert result["status"] == "success"

        ws_root = bare_agent.modules["tools"].workspace_service.workspace_root
        with open(os.path.join(ws_root, "output.png"), "rb") as f:
            assert f.read() == original_bytes

        # -- Default mode='text' writes UTF-8 string content --
        result = _call_tool_json(bare_agent, "write_files",
            files={"test.txt": "hello world"})
        assert result["status"] == "success"

        with open(os.path.join(ws_root, "test.txt"), "r") as f:
            assert f.read() == "hello world"

        # -- write_files rejects project: prefix --
        result = _call_tool_json(bare_agent, "write_files", files={
            "project:hack.py": "malicious content",
        })
        assert result["status"] == "partial"
        assert len(result["errors"]) == 1
        assert "workspace" in result["errors"][0]["error"].lower()

        # -- write_files allows normal workspace-relative paths --
        result = _call_tool_json(bare_agent, "write_files", files={
            "normal.txt": "safe content",
        })
        assert result["status"] == "success"
        assert "normal.txt" in result["written"]


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

        # -- on_shutdown removes the workspace session directory --
        _call_tool_json(bare_agent, "write_files", files={"temp.txt": "temporary"})

        ws_root = mod.workspace_service.workspace_root
        assert os.path.isdir(ws_root)

        mod.on_shutdown()

        session_dir = os.path.join(
            bare_agent.playground_dir, "sessions",
            mod.workspace_service.workspace_id,
        )
        assert not os.path.isdir(session_dir)
