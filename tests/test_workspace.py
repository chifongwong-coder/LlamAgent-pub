"""
Workspace and project sync flow tests: v1.5 tool system end-to-end.

Tests cover ToolsModule registration, workspace exploration tools,
project sync (apply_patch, revert_changes, sync), and workspace lifecycle.
"""

import json
import os

from llamagent.modules.tools.module import ToolsModule, WORKSPACE_GUIDE


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
    """ToolsModule registers all workspace + project sync tools on attach."""

    def test_tools_module_registers_workspace_tools(self, bare_agent, tmp_path):
        """All 11 workspace tools + 5 project sync tools appear in agent._tools."""
        _make_agent_with_tools(bare_agent, tmp_path)

        workspace_tools = [
            "list_tree", "glob_files", "search_text", "read_files",
            "read_ranges", "stat_paths", "write_files", "create_temp_file",
            "move_path", "copy_path", "delete_path",
        ]
        project_sync_tools = [
            "apply_patch", "preview_patch", "replace_block",
            "sync_workspace_to_project", "revert_changes",
        ]

        for tool_name in workspace_tools:
            assert tool_name in bare_agent._tools, f"workspace tool '{tool_name}' not registered"

        for tool_name in project_sync_tools:
            assert tool_name in bare_agent._tools, f"project sync tool '{tool_name}' not registered"


# ============================================================
# Workspace context injection
# ============================================================

class TestWorkspaceContextInjection:
    """on_context injects WORKSPACE_GUIDE into the LLM context."""

    def test_workspace_on_context_injects_guide(self, bare_agent, tmp_path):
        """on_context appends WORKSPACE_GUIDE text to the context string."""
        mod = _make_agent_with_tools(bare_agent, tmp_path)

        result = mod.on_context("test query", "existing context")
        assert WORKSPACE_GUIDE in result
        assert "existing context" in result

    def test_workspace_on_context_empty_context(self, bare_agent, tmp_path):
        """on_context returns WORKSPACE_GUIDE alone when existing context is empty."""
        mod = _make_agent_with_tools(bare_agent, tmp_path)

        result = mod.on_context("test query", "")
        assert result == WORKSPACE_GUIDE


# ============================================================
# Workspace exploration tools
# ============================================================

class TestWorkspaceExploration:
    """Workspace file creation and listing flow."""

    def test_list_tree_in_workspace(self, bare_agent, tmp_path):
        """write_files creates files, list_tree shows them in the tree."""
        _make_agent_with_tools(bare_agent, tmp_path)

        # Write files into workspace
        result = _call_tool_json(bare_agent, "write_files", files={
            "src/main.py": "print('hello')",
            "src/utils.py": "# utils",
            "README.md": "# Project",
        })
        assert result["status"] == "success"
        assert len(result["written"]) == 3

        # List the workspace tree
        tree_result = _call_tool_json(bare_agent, "list_tree")
        assert tree_result["status"] == "success"
        tree_text = tree_result["tree"]
        assert "main.py" in tree_text
        assert "utils.py" in tree_text
        assert "README.md" in tree_text

    def test_read_files_with_project_prefix(self, bare_agent, tmp_path):
        """read_files resolves 'project:' prefix to project_dir."""
        _make_agent_with_tools(bare_agent, tmp_path)

        # Create a file directly in project_dir
        project_dir = bare_agent.project_dir
        test_file = os.path.join(project_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("project file content")

        # Read via project: prefix
        result = _call_tool_json(bare_agent, "read_files", paths=["project:test.txt"])
        assert result["status"] == "success"
        assert len(result["files"]) == 1
        assert "project file content" in result["files"][0]["content"]


# ============================================================
# Project sync: apply_patch + revert
# ============================================================

class TestApplyPatch:
    """apply_patch atomic editing and revert flow."""

    def test_apply_patch_atomic(self, bare_agent, tmp_path):
        """apply_patch is atomic: all-or-nothing on multiple edits."""
        _make_agent_with_tools(bare_agent, tmp_path)

        # Create a project file
        project_dir = bare_agent.project_dir
        target = os.path.join(project_dir, "code.py")
        original = "def hello():\n    return 'hello'\n\ndef world():\n    return 'world'\n"
        with open(target, "w") as f:
            f.write(original)

        # Apply 2 valid edits -> both should succeed
        result = _call_tool_json(bare_agent, "apply_patch", target="code.py", edits=[
            {"match": "'hello'", "replace": "'hi'"},
            {"match": "'world'", "replace": "'earth'"},
        ])
        assert result["status"] == "success"
        assert result["edits_applied"] == 2

        with open(target) as f:
            content = f.read()
        assert "'hi'" in content
        assert "'earth'" in content
        assert "'hello'" not in content
        assert "'world'" not in content

        # Now try 1 valid + 1 invalid edit -> file should remain unchanged
        result2 = _call_tool_json(bare_agent, "apply_patch", target="code.py", edits=[
            {"match": "'hi'", "replace": "'hey'"},
            {"match": "NONEXISTENT_STRING", "replace": "x"},
        ])
        assert result2["status"] == "error"

        # File must be unchanged (atomicity)
        with open(target) as f:
            unchanged = f.read()
        assert unchanged == content

    def test_revert_changes(self, bare_agent, tmp_path):
        """revert_changes restores the file to its pre-patch state."""
        _make_agent_with_tools(bare_agent, tmp_path)

        # Create a project file
        project_dir = bare_agent.project_dir
        target = os.path.join(project_dir, "data.txt")
        original = "line one\nline two\nline three\n"
        with open(target, "w") as f:
            f.write(original)

        # Apply a patch
        result = _call_tool_json(bare_agent, "apply_patch", target="data.txt", edits=[
            {"match": "line two", "replace": "LINE TWO MODIFIED"},
        ])
        assert result["status"] == "success"

        with open(target) as f:
            assert "LINE TWO MODIFIED" in f.read()

        # Revert
        revert_result = _call_tool_json(bare_agent, "revert_changes")
        assert revert_result["status"] == "success"

        with open(target) as f:
            restored = f.read()
        assert restored == original


# ============================================================
# Project sync: sync_workspace_to_project
# ============================================================

class TestSyncWorkspaceToProject:
    """sync_workspace_to_project copies workspace files to project directory."""

    def test_sync_workspace_to_project_auto(self, bare_agent, tmp_path):
        """Sync with mode='auto' copies new workspace files to project."""
        _make_agent_with_tools(bare_agent, tmp_path)

        # Write a file into the workspace
        _call_tool_json(bare_agent, "write_files", files={
            "new_file.txt": "workspace content",
        })

        # Sync to project
        result = _call_tool_json(bare_agent, "sync_workspace_to_project", mode="auto")
        assert result["status"] == "success"
        assert result["synced"] >= 1

        # Verify the file arrived in project_dir
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

        # Write files so the workspace session dir is created
        _call_tool_json(bare_agent, "write_files", files={
            "temp.txt": "temporary",
        })

        # Verify workspace session directory exists
        ws_root = mod.workspace_service.workspace_root
        assert os.path.isdir(ws_root)

        # Shutdown should remove the session directory
        mod.on_shutdown()

        session_dir = os.path.join(
            bare_agent.playground_dir,
            "sessions",
            mod.workspace_service.workspace_id,
        )
        assert not os.path.isdir(session_dir)
