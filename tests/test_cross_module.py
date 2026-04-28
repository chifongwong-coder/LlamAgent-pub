"""
Cross-module integration flow tests.

Tests verify that multiple modules work together correctly through the
shared retrieval layer, pack mechanism, and hook pipeline.
"""

import json
import os
import tempfile

from llamagent.modules.tools.module import ToolsModule
from llamagent.modules.skill.module import SkillModule
from llamagent.modules.job.module import JobModule


# ============================================================
# Helpers
# ============================================================

class _MockToolExecutor:
    """Minimal ToolExecutor for job tests."""
    def run_command(self, command, cwd, timeout=300):
        import subprocess
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=cwd,
        )
        return (result.stdout or "") + (result.stderr or "")


def _setup_tools_and_skill(bare_agent, tmp_path):
    """Set up ToolsModule + SkillModule for pack integration tests."""
    project_dir = os.path.join(str(tmp_path), "project")
    playground_dir = os.path.join(str(tmp_path), "playground")
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(playground_dir, exist_ok=True)

    bare_agent.project_dir = project_dir
    bare_agent.playground_dir = playground_dir

    # Register ToolsModule first (pack reset + state eval)
    tools_mod = ToolsModule()
    bare_agent.register_module(tools_mod)

    # Then SkillModule (skill-driven pack activation)
    bare_agent.config.skill_dirs = []
    bare_agent.config.skill_max_active = 2
    skill_mod = SkillModule()
    bare_agent.register_module(skill_mod)

    return tools_mod, skill_mod


def _call_tool_json(agent, name, **kwargs):
    """Invoke a registered tool by name and return parsed JSON."""
    func = agent._tools[name]["func"]
    raw = func(**kwargs)
    return json.loads(raw)


# ============================================================
# Tests
# ============================================================

class TestCrossModule:
    """Consolidated cross-module integration flow tests."""

    def test_pack_skill_integration(self, bare_agent, tmp_path):
        """Skill activation + state-driven pack auto-activation, on_input resets per turn."""
        tools_mod, skill_mod = _setup_tools_and_skill(bare_agent, tmp_path)

        # v3.3: bare_agent has no shell tool, so path-fallback auto-activates
        # when on_context runs (state-driven). Verify the cycle.
        bare_agent._active_packs.clear()
        tools_mod.on_input("move some files around")
        # After on_input alone, packs are cleared.
        assert "path-fallback" not in bare_agent._active_packs

        tools_mod.on_context("move some files around", "")
        # After on_context, path-fallback auto-activates because no
        # shell tool is registered on bare_agent.
        assert "path-fallback" in bare_agent._active_packs
        schemas = bare_agent.get_all_tool_schemas()
        schema_names = [s["function"]["name"] for s in schemas]
        assert "glob_files" in schema_names
        assert "move_path" in schema_names

        # Skill match still works (workspace-ops requires path-fallback).
        skill_mod.on_context("move some files around", "")
        # Whether the skill activated depends on tag matching; either
        # outcome is fine — the path-fallback pack is already active.

        # --- on_input clears packs ---
        tools_mod.on_input("hello")
        assert len(bare_agent._active_packs) == 0

    def test_pack_job_integration(self, bare_agent, tmp_path):
        """start_job activates followup pack, and state-driven eval re-activates it next turn."""
        tools_mod, _ = _setup_tools_and_skill(bare_agent, tmp_path)

        # Add job module with mock executor
        bare_agent.tool_executor = _MockToolExecutor()
        job_mod = JobModule()
        bare_agent.register_module(job_mod)

        # Before start_job: followup tools hidden
        schemas = bare_agent.get_all_tool_schemas()
        schema_names = [s["function"]["name"] for s in schemas]
        assert "inspect_job" not in schema_names

        # Execute start_job
        _call_tool_json(bare_agent, "start_job", command="echo test", wait=True)

        # After: job-followup pack active, tools visible
        assert "job-followup" in bare_agent._active_packs
        schemas = bare_agent.get_all_tool_schemas()
        schema_names = [s["function"]["name"] for s in schemas]
        assert "inspect_job" in schema_names
        assert "wait_job" in schema_names
        assert "cancel_job" in schema_names

        # --- State-driven persistence across turns ---
        # Start an async job (stays in service)
        result = _call_tool_json(bare_agent, "start_job", command="sleep 2", wait=False)
        job_id = result["job_id"]

        # New turn: on_input clears, on_context re-evaluates
        tools_mod.on_input("check my job")
        tools_mod.on_context("check my job", "")

        # State-driven: JobService has jobs -> pack re-activated
        assert "job-followup" in bare_agent._active_packs

        # Clean up
        bare_agent._tools["cancel_job"]["func"](job_id=job_id)

    def test_workspace_project_flow(self, bare_agent, tmp_path):
        """Patch+revert flow on project files (v3.3: sync_workspace_to_project deleted)."""
        _setup_tools_and_skill(bare_agent, tmp_path)

        # --- Patch project file then revert ---
        config_file = os.path.join(bare_agent.project_dir, "config.txt")
        original = "version = 1.0\n"
        with open(config_file, "w") as f:
            f.write(original)

        result = _call_tool_json(bare_agent, "apply_patch", target="config.txt", edits=[
            {"match": "version = 1.0", "replace": "version = 2.0"},
        ])
        assert result["status"] == "success"

        with open(config_file) as f:
            assert "version = 2.0" in f.read()

        result = _call_tool_json(bare_agent, "revert_changes")
        assert result["status"] == "success"

        with open(config_file) as f:
            assert f.read() == original

        # --- v3.3: write_files rejects path-escape attempts ---
        # 'project:' prefix no longer special; use '../' for the escape test.
        result = _call_tool_json(bare_agent, "write_files", files={
            "../escape.py": "bad content",
        })
        assert result["status"] == "partial"
        assert len(result["errors"]) == 1

    def test_hook_and_context_stacking(self, bare_agent, tmp_path):
        """Hook pipeline ordering and context injection stacking across modules."""
        tools_mod, skill_mod = _setup_tools_and_skill(bare_agent, tmp_path)

        # --- Hook pipeline ordering: tools on_input clears packs before skill on_context ---
        bare_agent._active_packs.add("toolsmith")

        processed = "create a tool for me"
        processed = tools_mod.on_input(processed)
        processed = skill_mod.on_input(processed)

        # After tools on_input: packs cleared
        assert "toolsmith" not in bare_agent._active_packs

        context = ""
        context = tools_mod.on_context(processed, context)
        context = skill_mod.on_context(processed, context)

        # --- Context injection stacking: both modules contribute ---
        # Force a skill activation for context stacking test
        skill_mod._forced_skill = "toolsmith"

        context2 = ""
        context2 = tools_mod.on_context("create a tool", context2)
        context2 = skill_mod.on_context("create a tool", context2)

        # v3.3 FILE_TOOL_GUIDE header was rewritten — assert a stable
        # substring from the new body instead of the old "[File Tool
        # Guidelines]" bracketed header.
        assert "All paths are relative" in context2
        assert "[Available Tool Packs]" in context2
        # If toolsmith skill was found, its playbook is also injected
        if "[Active Skill: toolsmith]" in context2:
            assert "[End Skill]" in context2
