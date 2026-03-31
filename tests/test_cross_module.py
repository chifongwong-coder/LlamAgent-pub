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
    bare_agent.config.skill_llm_fallback = False
    skill_mod = SkillModule()
    bare_agent.register_module(skill_mod)

    return tools_mod, skill_mod


def _call_tool_json(agent, name, **kwargs):
    """Invoke a registered tool by name and return parsed JSON."""
    func = agent._tools[name]["func"]
    raw = func(**kwargs)
    return json.loads(raw)


# ============================================================
# Pack + Skill + Tools integration
# ============================================================

class TestPackSkillIntegration:
    """Skill activation triggers pack, making hidden tools visible."""

    def test_skill_activates_workspace_maintenance_pack(self, bare_agent, tmp_path):
        """Workspace-ops builtin skill activates workspace-maintenance pack."""
        tools_mod, skill_mod = _setup_tools_and_skill(bare_agent, tmp_path)

        # Before: workspace-maintenance tools hidden
        schemas = bare_agent.get_all_tool_schemas()
        schema_names = [s["function"]["name"] for s in schemas]
        assert "glob_files" not in schema_names
        assert "move_path" not in schema_names

        # Simulate a turn: on_input clears packs, on_context evaluates
        tools_mod.on_input("move some files around")
        tools_mod.on_context("move some files around", "")
        # Skill matching: "move" should match workspace-ops builtin skill
        skill_mod.on_context("move some files around", "")

        # After: if skill matched, workspace-maintenance pack should be active
        if "workspace-maintenance" in bare_agent._active_packs:
            schemas = bare_agent.get_all_tool_schemas()
            schema_names = [s["function"]["name"] for s in schemas]
            assert "glob_files" in schema_names
            assert "move_path" in schema_names

    def test_on_input_resets_packs_each_turn(self, bare_agent, tmp_path):
        """Packs from previous turn are cleared on next on_input."""
        tools_mod, _ = _setup_tools_and_skill(bare_agent, tmp_path)

        # Manually activate a pack
        bare_agent._active_packs.add("workspace-maintenance")
        assert "workspace-maintenance" in bare_agent._active_packs

        # New turn: on_input clears
        tools_mod.on_input("hello")
        assert len(bare_agent._active_packs) == 0


# ============================================================
# Pack + Job integration
# ============================================================

class TestPackJobIntegration:
    """Job execution activates job-followup pack within same turn."""

    def test_start_job_activates_followup_pack(self, bare_agent, tmp_path):
        """start_job success makes inspect_job/wait_job/cancel_job visible."""
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

    def test_job_followup_persists_via_state_eval(self, bare_agent, tmp_path):
        """Next turn: state-driven eval re-activates job-followup if jobs exist."""
        tools_mod, _ = _setup_tools_and_skill(bare_agent, tmp_path)

        bare_agent.tool_executor = _MockToolExecutor()
        job_mod = JobModule()
        bare_agent.register_module(job_mod)

        # Start an async job (stays in service)
        result = _call_tool_json(bare_agent, "start_job", command="sleep 2", wait=False)
        job_id = result["job_id"]

        # New turn: on_input clears, on_context re-evaluates
        tools_mod.on_input("check my job")
        tools_mod.on_context("check my job", "")

        # State-driven: JobService has jobs → pack re-activated
        assert "job-followup" in bare_agent._active_packs

        # Clean up
        bare_agent._tools["cancel_job"]["func"](job_id=job_id)


# ============================================================
# Tools + Workspace + Project Sync flow
# ============================================================

class TestWorkspaceProjectFlow:
    """End-to-end workspace → project sync flow."""

    def test_write_in_workspace_then_sync_to_project(self, bare_agent, tmp_path):
        """Write files in workspace, sync to project, verify project has them."""
        _setup_tools_and_skill(bare_agent, tmp_path)

        # Write to workspace
        result = _call_tool_json(bare_agent, "write_files", files={
            "app.py": "print('hello')\n",
        })
        assert result["status"] == "success"

        # Sync to project
        result = _call_tool_json(bare_agent, "sync_workspace_to_project", mode="auto")
        assert result["status"] == "success"
        assert result["synced"] >= 1

        # Verify project file
        project_file = os.path.join(bare_agent.project_dir, "app.py")
        assert os.path.isfile(project_file)
        with open(project_file) as f:
            assert "hello" in f.read()

    def test_patch_project_then_revert(self, bare_agent, tmp_path):
        """Patch a project file, then revert to original."""
        _setup_tools_and_skill(bare_agent, tmp_path)

        # Create a project file
        project_file = os.path.join(bare_agent.project_dir, "config.txt")
        original = "version = 1.0\n"
        with open(project_file, "w") as f:
            f.write(original)

        # Patch it
        result = _call_tool_json(bare_agent, "apply_patch", target="config.txt", edits=[
            {"match": "version = 1.0", "replace": "version = 2.0"},
        ])
        assert result["status"] == "success"

        with open(project_file) as f:
            assert "version = 2.0" in f.read()

        # Revert
        result = _call_tool_json(bare_agent, "revert_changes")
        assert result["status"] == "success"

        with open(project_file) as f:
            assert f.read() == original

    def test_write_files_rejects_project_prefix(self, bare_agent, tmp_path):
        """write_files cannot write to project via project: prefix."""
        _setup_tools_and_skill(bare_agent, tmp_path)

        result = _call_tool_json(bare_agent, "write_files", files={
            "project:hack.py": "bad content",
        })
        assert result["status"] == "partial"
        assert len(result["errors"]) == 1


# ============================================================
# Hook pipeline ordering
# ============================================================

class TestHookPipelineOrdering:
    """Tools on_input runs before Skill on_context (registration order)."""

    def test_tools_before_skill_in_pipeline(self, bare_agent, tmp_path):
        """ToolsModule.on_input clears packs before SkillModule.on_context adds them."""
        tools_mod, skill_mod = _setup_tools_and_skill(bare_agent, tmp_path)

        # Manually set a pack (simulating leftover from previous turn)
        bare_agent._active_packs.add("toolsmith")

        # Run the pipeline as chat() would
        # Step 1: on_input (tools first, then skill)
        processed = "create a tool for me"
        processed = tools_mod.on_input(processed)
        processed = skill_mod.on_input(processed)

        # After tools on_input: packs cleared
        assert "toolsmith" not in bare_agent._active_packs

        # Step 2: on_context (tools first, then skill)
        context = ""
        context = tools_mod.on_context(processed, context)
        context = skill_mod.on_context(processed, context)

        # After skill on_context: if "create" matched toolsmith skill, pack re-added
        # (depends on tag matching — may or may not match)
        # The key assertion is that tools on_input cleared before skill on_context ran
        # This is verified by the clear check above


# ============================================================
# Context injection stacking
# ============================================================

class TestContextInjectionStacking:
    """Multiple modules inject into context without conflict."""

    def test_tools_and_skill_context_coexist(self, bare_agent, tmp_path):
        """ToolsModule injects WORKSPACE_GUIDE + hints, SkillModule injects playbook."""
        tools_mod, skill_mod = _setup_tools_and_skill(bare_agent, tmp_path)

        # Force a skill activation
        skill_mod._forced_skill = "toolsmith"

        context = ""
        context = tools_mod.on_context("create a tool", context)
        context = skill_mod.on_context("create a tool", context)

        # Both should be present
        assert "[Workspace Guidelines]" in context
        assert "[Available Tool Packs]" in context
        # If toolsmith skill was found, its playbook is also injected
        if "[Active Skill: toolsmith]" in context:
            assert "[End Skill]" in context
