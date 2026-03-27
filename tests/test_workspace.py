"""
Workspace and project sync flow tests: v1.6 tool system.

Tests cover ToolsModule registration, pack mechanism, workspace exploration,
project sync (apply_patch with preview, revert), workspace-only write restriction,
context injection (WORKSPACE_GUIDE + capability hint block), and workspace lifecycle.
"""

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
# Tool registration
# ============================================================

class TestToolRegistration:
    """ToolsModule registers default surface + pack tools on attach."""

    def test_default_surface_tools_registered(self, bare_agent, tmp_path):
        """Default surface tools (no pack) appear in agent._tools."""
        _make_agent_with_tools(bare_agent, tmp_path)

        default_tools = [
            "list_tree", "search_text", "read_files", "write_files",
            "apply_patch", "sync_workspace_to_project", "revert_changes",
        ]
        for tool_name in default_tools:
            assert tool_name in bare_agent._tools, f"default tool '{tool_name}' not registered"
            assert bare_agent._tools[tool_name].get("pack") is None, \
                f"'{tool_name}' should not have a pack (default surface)"

    def test_pack_tools_registered_with_pack(self, bare_agent, tmp_path):
        """Pack tools are registered with correct pack names."""
        _make_agent_with_tools(bare_agent, tmp_path)

        pack_expectations = {
            "glob_files": "workspace-maintenance",
            "stat_paths": "workspace-maintenance",
            "create_temp_file": "workspace-maintenance",
            "move_path": "workspace-maintenance",
            "copy_path": "workspace-maintenance",
            "delete_path": "workspace-maintenance",
        }
        for tool_name, expected_pack in pack_expectations.items():
            assert tool_name in bare_agent._tools, f"pack tool '{tool_name}' not registered"
            assert bare_agent._tools[tool_name].get("pack") == expected_pack

    def test_merged_tools_removed(self, bare_agent, tmp_path):
        """Merged tools (read_ranges, preview_patch, replace_block) are no longer registered."""
        _make_agent_with_tools(bare_agent, tmp_path)

        removed = ["read_ranges", "preview_patch", "replace_block"]
        for tool_name in removed:
            assert tool_name not in bare_agent._tools, f"'{tool_name}' should have been removed"


# ============================================================
# Pack filtering in get_all_tool_schemas
# ============================================================

class TestPackFiltering:
    """Pack tools are hidden from LLM unless pack is active."""

    def test_pack_tools_hidden_by_default(self, bare_agent, tmp_path):
        """Tools with pack are not in get_all_tool_schemas when pack is inactive."""
        _make_agent_with_tools(bare_agent, tmp_path)

        schemas = bare_agent.get_all_tool_schemas()
        schema_names = [s["function"]["name"] for s in schemas]

        # Default surface should be visible
        assert "read_files" in schema_names
        assert "write_files" in schema_names
        assert "apply_patch" in schema_names

        # Pack tools should be hidden
        assert "glob_files" not in schema_names
        assert "move_path" not in schema_names
        assert "stat_paths" not in schema_names

    def test_pack_tools_visible_when_activated(self, bare_agent, tmp_path):
        """Tools with pack appear in schemas when their pack is activated."""
        _make_agent_with_tools(bare_agent, tmp_path)

        bare_agent._active_packs.add("workspace-maintenance")
        schemas = bare_agent.get_all_tool_schemas()
        schema_names = [s["function"]["name"] for s in schemas]

        assert "glob_files" in schema_names
        assert "move_path" in schema_names
        assert "stat_paths" in schema_names

    def test_toolsmith_pack_hidden_by_default(self, bare_agent, tmp_path):
        """Meta-tools (toolsmith pack) are hidden by default."""
        _make_agent_with_tools(bare_agent, tmp_path)

        schemas = bare_agent.get_all_tool_schemas()
        schema_names = [s["function"]["name"] for s in schemas]

        assert "create_tool" not in schema_names
        assert "list_my_tools" not in schema_names
        assert "delete_tool" not in schema_names


# ============================================================
# Workspace context injection
# ============================================================

class TestWorkspaceContextInjection:
    """on_context injects WORKSPACE_GUIDE + CAPABILITY_HINT_BLOCK."""

    def test_on_context_injects_guide_and_hints(self, bare_agent, tmp_path):
        """on_context appends both WORKSPACE_GUIDE and CAPABILITY_HINT_BLOCK."""
        mod = _make_agent_with_tools(bare_agent, tmp_path)

        result = mod.on_context("test query", "existing context")
        assert WORKSPACE_GUIDE in result
        assert CAPABILITY_HINT_BLOCK in result
        assert "existing context" in result

    def test_on_context_empty_context(self, bare_agent, tmp_path):
        """on_context returns guide + hints when existing context is empty."""
        mod = _make_agent_with_tools(bare_agent, tmp_path)

        result = mod.on_context("test query", "")
        assert WORKSPACE_GUIDE in result
        assert CAPABILITY_HINT_BLOCK in result


# ============================================================
# Workspace exploration tools
# ============================================================

class TestWorkspaceExploration:
    """Workspace file creation and listing flow."""

    def test_list_tree_in_workspace(self, bare_agent, tmp_path):
        """write_files creates files, list_tree shows them in the tree."""
        _make_agent_with_tools(bare_agent, tmp_path)

        result = _call_tool_json(bare_agent, "write_files", files={
            "src/main.py": "print('hello')",
            "README.md": "# Project",
        })
        assert result["status"] == "success"

        tree_result = _call_tool_json(bare_agent, "list_tree")
        assert "main.py" in tree_result["tree"]
        assert "README.md" in tree_result["tree"]

    def test_read_files_with_project_prefix(self, bare_agent, tmp_path):
        """read_files resolves 'project:' prefix to project_dir."""
        _make_agent_with_tools(bare_agent, tmp_path)

        project_dir = bare_agent.project_dir
        with open(os.path.join(project_dir, "test.txt"), "w") as f:
            f.write("project file content")

        result = _call_tool_json(bare_agent, "read_files", paths=["project:test.txt"])
        assert result["status"] == "success"
        assert "project file content" in result["files"][0]["content"]

    def test_read_files_with_ranges(self, bare_agent, tmp_path):
        """read_files with ranges parameter reads specific line ranges."""
        _make_agent_with_tools(bare_agent, tmp_path)

        _call_tool_json(bare_agent, "write_files", files={
            "lines.txt": "\n".join(f"line{i}" for i in range(1, 21)),
        })

        result = _call_tool_json(bare_agent, "read_files",
            paths=["lines.txt"], ranges={"lines.txt": "5-8"})
        assert result["status"] == "success"
        content = result["files"][0]["content"]
        assert "line5" in content
        assert "line8" in content
        assert "line1\t" not in content  # line 1 should not appear


# ============================================================
# File type detection (v1.6 Phase 5)
# ============================================================

class TestFileTypeDetection:
    """read_files auto-detects text vs binary files."""

    def test_read_text_file_auto(self, bare_agent, tmp_path):
        """Text file (.py) returns content with line numbers in auto mode."""
        _make_agent_with_tools(bare_agent, tmp_path)

        _call_tool_json(bare_agent, "write_files", files={
            "hello.py": "print('hello')\nprint('world')\n",
        })

        result = _call_tool_json(bare_agent, "read_files", paths=["hello.py"])
        assert result["status"] == "success"
        file_info = result["files"][0]
        assert "content" in file_info
        assert "hello" in file_info["content"]
        assert file_info.get("binary") is not True

    def test_read_binary_file_auto(self, bare_agent, tmp_path):
        """Binary file (.png) returns metadata only in auto mode."""
        import base64
        _make_agent_with_tools(bare_agent, tmp_path)

        # Write a binary file directly to workspace
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
        assert "content" not in file_info  # no content dump for binary in auto mode

    def test_read_binary_file_explicit_mode(self, bare_agent, tmp_path):
        """mode='binary' returns base64-encoded content."""
        _make_agent_with_tools(bare_agent, tmp_path)

        ws_root = bare_agent.modules["tools"].workspace_service.workspace_root
        bin_path = os.path.join(ws_root, "data.bin")
        original_bytes = b"\x00\x01\x02\x03\xff\xfe\xfd"
        with open(bin_path, "wb") as f:
            f.write(original_bytes)

        result = _call_tool_json(bare_agent, "read_files", paths=["data.bin"], mode="binary")
        assert result["status"] == "success"
        file_info = result["files"][0]
        assert file_info["binary"] is True
        assert "content_base64" in file_info
        # Verify round-trip
        import base64
        decoded = base64.b64decode(file_info["content_base64"])
        assert decoded == original_bytes

    def test_read_no_extension_text(self, bare_agent, tmp_path):
        """File without extension but text content is detected as text."""
        _make_agent_with_tools(bare_agent, tmp_path)

        _call_tool_json(bare_agent, "write_files", files={
            "Makefile": "all:\n\techo hello\n",
        })

        result = _call_tool_json(bare_agent, "read_files", paths=["Makefile"])
        assert result["status"] == "success"
        file_info = result["files"][0]
        assert "content" in file_info
        assert "echo hello" in file_info["content"]


# ============================================================
# write_files binary mode (v1.6 Phase 5)
# ============================================================

class TestWriteFilesBinaryMode:
    """write_files supports mode='binary' for base64-encoded content."""

    def test_write_binary_file(self, bare_agent, tmp_path):
        """mode='binary' writes base64-decoded bytes."""
        import base64
        _make_agent_with_tools(bare_agent, tmp_path)

        original_bytes = b"\x89PNG\r\n\x1a\n\x00\x01\x02\x03"
        b64_content = base64.b64encode(original_bytes).decode("ascii")

        result = _call_tool_json(bare_agent, "write_files",
            files={"output.png": b64_content}, mode="binary")
        assert result["status"] == "success"

        ws_root = bare_agent.modules["tools"].workspace_service.workspace_root
        with open(os.path.join(ws_root, "output.png"), "rb") as f:
            assert f.read() == original_bytes

    def test_write_text_file_default(self, bare_agent, tmp_path):
        """Default mode='text' writes UTF-8 string content (unchanged behavior)."""
        _make_agent_with_tools(bare_agent, tmp_path)

        result = _call_tool_json(bare_agent, "write_files",
            files={"test.txt": "hello world"})
        assert result["status"] == "success"

        ws_root = bare_agent.modules["tools"].workspace_service.workspace_root
        with open(os.path.join(ws_root, "test.txt"), "r") as f:
            assert f.read() == "hello world"


# ============================================================
# Workspace-only write restriction
# ============================================================

class TestWorkspaceOnlyRestriction:
    """Write operations reject project: prefix and out-of-workspace paths."""

    def test_write_files_rejects_project_prefix(self, bare_agent, tmp_path):
        """write_files returns error for project: prefix paths."""
        _make_agent_with_tools(bare_agent, tmp_path)

        result = _call_tool_json(bare_agent, "write_files", files={
            "project:hack.py": "malicious content",
        })
        assert result["status"] == "partial"
        assert len(result["errors"]) == 1
        assert "workspace" in result["errors"][0]["error"].lower()

    def test_write_files_allows_workspace_paths(self, bare_agent, tmp_path):
        """write_files succeeds for normal workspace-relative paths."""
        _make_agent_with_tools(bare_agent, tmp_path)

        result = _call_tool_json(bare_agent, "write_files", files={
            "normal.txt": "safe content",
        })
        assert result["status"] == "success"
        assert "normal.txt" in result["written"]


# ============================================================
# Project sync: apply_patch (with preview) + revert
# ============================================================

class TestApplyPatch:
    """apply_patch atomic editing, preview mode, and revert flow."""

    def test_apply_patch_atomic(self, bare_agent, tmp_path):
        """apply_patch is atomic: all-or-nothing on multiple edits."""
        _make_agent_with_tools(bare_agent, tmp_path)

        project_dir = bare_agent.project_dir
        target = os.path.join(project_dir, "code.py")
        original = "def hello():\n    return 'hello'\n\ndef world():\n    return 'world'\n"
        with open(target, "w") as f:
            f.write(original)

        # 2 valid edits -> both succeed
        result = _call_tool_json(bare_agent, "apply_patch", target="code.py", edits=[
            {"match": "'hello'", "replace": "'hi'"},
            {"match": "'world'", "replace": "'earth'"},
        ])
        assert result["status"] == "success"
        assert result["edits_applied"] == 2

        # 1 valid + 1 invalid -> file unchanged (atomicity)
        with open(target) as f:
            content = f.read()
        result2 = _call_tool_json(bare_agent, "apply_patch", target="code.py", edits=[
            {"match": "'hi'", "replace": "'hey'"},
            {"match": "NONEXISTENT", "replace": "x"},
        ])
        assert result2["status"] == "error"
        with open(target) as f:
            assert f.read() == content

    def test_apply_patch_preview(self, bare_agent, tmp_path):
        """apply_patch with preview=True validates edits without writing."""
        _make_agent_with_tools(bare_agent, tmp_path)

        project_dir = bare_agent.project_dir
        target = os.path.join(project_dir, "preview.txt")
        original = "original content\n"
        with open(target, "w") as f:
            f.write(original)

        result = _call_tool_json(bare_agent, "apply_patch", target="preview.txt",
            edits=[{"match": "original", "replace": "modified"}], preview=True)
        assert result["status"] == "preview"
        assert result.get("edits_valid") is True

        # File must be unchanged
        with open(target) as f:
            assert f.read() == original

    def test_revert_changes(self, bare_agent, tmp_path):
        """revert_changes restores the file to its pre-patch state."""
        _make_agent_with_tools(bare_agent, tmp_path)

        project_dir = bare_agent.project_dir
        target = os.path.join(project_dir, "data.txt")
        original = "line one\nline two\nline three\n"
        with open(target, "w") as f:
            f.write(original)

        _call_tool_json(bare_agent, "apply_patch", target="data.txt", edits=[
            {"match": "line two", "replace": "LINE TWO MODIFIED"},
        ])

        revert_result = _call_tool_json(bare_agent, "revert_changes")
        assert revert_result["status"] == "success"

        with open(target) as f:
            assert f.read() == original


# ============================================================
# Project sync: sync_workspace_to_project
# ============================================================

class TestSyncWorkspaceToProject:
    """sync_workspace_to_project copies workspace files to project directory."""

    def test_sync_workspace_to_project_auto(self, bare_agent, tmp_path):
        """Sync with mode='auto' copies new workspace files to project."""
        _make_agent_with_tools(bare_agent, tmp_path)

        _call_tool_json(bare_agent, "write_files", files={
            "new_file.txt": "workspace content",
        })

        result = _call_tool_json(bare_agent, "sync_workspace_to_project", mode="auto")
        assert result["status"] == "success"
        assert result["synced"] >= 1

        synced_path = os.path.join(bare_agent.project_dir, "new_file.txt")
        assert os.path.isfile(synced_path)
        with open(synced_path) as f:
            assert f.read() == "workspace content"


# ============================================================
# Workspace lifecycle
# ============================================================

class TestWorkspaceLifecycle:
    """on_shutdown cleans up the workspace session directory."""

    def test_on_shutdown_cleans_workspace(self, bare_agent, tmp_path):
        """on_shutdown removes the workspace session directory."""
        mod = _make_agent_with_tools(bare_agent, tmp_path)

        _call_tool_json(bare_agent, "write_files", files={"temp.txt": "temporary"})

        ws_root = mod.workspace_service.workspace_root
        assert os.path.isdir(ws_root)

        mod.on_shutdown()

        session_dir = os.path.join(
            bare_agent.playground_dir, "sessions",
            mod.workspace_service.workspace_id,
        )
        assert not os.path.isdir(session_dir)
