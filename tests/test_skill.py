"""
Skill module flow tests: end-to-end skill scanning, matching, and injection.

Tests cover the full A/B/C activation chain, /skill command handling,
context injection format, and multi-module cooperation.
"""

import json
import os

import pytest

from llamagent.modules.skill.module import SkillModule
from conftest import make_llm_response


# ============================================================
# Helpers
# ============================================================

def _create_skill(base_dir, name, description, tags=None, aliases=None,
                  invocation="both", skill_content="## Goal\nTest skill.",
                  required_tool_packs=None):
    """Create a skill directory with config.yaml + SKILL.md."""
    skill_dir = os.path.join(base_dir, name)
    os.makedirs(skill_dir, exist_ok=True)

    config = f"name: {name}\ndescription: {description}\n"
    if tags:
        config += f"tags: [{', '.join(tags)}]\n"
    if aliases:
        config += f"aliases: [{', '.join(aliases)}]\n"
    if invocation != "both":
        config += f"invocation: {invocation}\n"
    if required_tool_packs:
        config += f"required_tool_packs: [{', '.join(required_tool_packs)}]\n"

    with open(os.path.join(skill_dir, "config.yaml"), "w") as f:
        f.write(config)
    with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
        f.write(skill_content)

    return skill_dir


def _make_agent_with_skills(bare_agent, tmp_path, skills_data):
    """Create skill directories and register SkillModule on bare_agent."""
    skills_dir = os.path.join(str(tmp_path), ".llamagent", "skills")
    os.makedirs(skills_dir, exist_ok=True)

    for sd in skills_data:
        _create_skill(
            skills_dir, sd["name"], sd["description"],
            tags=sd.get("tags"), aliases=sd.get("aliases"),
            invocation=sd.get("invocation", "both"),
            skill_content=sd.get("content", f"## Goal\n{sd['name']} playbook."),
            required_tool_packs=sd.get("required_tool_packs"),
        )

    bare_agent.project_dir = str(tmp_path)
    bare_agent.config.skill_dirs = []
    bare_agent.config.skill_max_active = 2
    bare_agent.config.skill_llm_fallback = False

    mod = SkillModule()
    bare_agent.register_module(mod)
    return mod


# ============================================================
# Flow 1: Tag matching (B-level) + LLM fallback (C-level)
# ============================================================

class TestTagMatchingAndLLMFallbackFlow:
    """B-level tag matching and C-level LLM fallback: single match, multi
    match + LLM disambiguate, LLM error, no match, max_active truncation,
    fallback disabled / enabled / empty."""

    def test_tag_matching_and_llm_fallback_flow(self, bare_agent, tmp_path, mock_llm_client):
        """Full tag matching and LLM fallback flow across multiple scenarios."""

        # --- Single tag match activates skill without LLM ---
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "db-migration", "description": "Database migration",
             "tags": ["migration", "alembic"], "content": "Migration playbook"},
            {"name": "code-review", "description": "Code review",
             "tags": ["review", "pr"], "content": "Review playbook"},
        ])

        result = mod.on_context("run the migration", "existing context")
        assert "existing context" in result
        assert "[Active Skill: db-migration]" in result
        assert "[Active Skill: code-review]" not in result

        # --- Multiple candidates: LLM disambiguates ---
        bare_agent.modules.clear()
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "db-migration", "description": "Database migration",
             "tags": ["database"], "content": "Migration playbook"},
            {"name": "db-backup", "description": "Database backup",
             "tags": ["database"], "content": "Backup playbook"},
        ])

        mock_llm_client.set_responses([
            make_llm_response(json.dumps({"selected": ["db-migration"]})),
        ])

        result = mod.on_context("database task", "")
        assert "[Active Skill: db-migration]" in result
        assert "[Active Skill: db-backup]" not in result

        # --- LLM error falls back to first candidate ---
        bare_agent.modules.clear()
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "skill-a", "description": "Skill A",
             "tags": ["shared"], "content": "A playbook"},
            {"name": "skill-b", "description": "Skill B",
             "tags": ["shared"], "content": "B playbook"},
        ])

        mock_llm_client.set_responses([Exception("LLM error")])
        result = mod.on_context("shared task", "")
        assert "[Active Skill: skill-a]" in result

        # --- No tag match returns original context ---
        bare_agent.modules.clear()
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "deploy", "description": "Deploy", "tags": ["deploy"]},
        ])

        result = mod.on_context("what is the weather today", "original context")
        assert "original context" in result
        # No skill should be activated (L2 miss), but L3 index may be appended
        assert "[Active Skill:" not in result

        # --- max_active truncation: only N skills injected ---
        bare_agent.modules.clear()
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "s1", "description": "Skill 1", "tags": ["common"], "content": "S1"},
            {"name": "s2", "description": "Skill 2", "tags": ["common"], "content": "S2"},
            {"name": "s3", "description": "Skill 3", "tags": ["common"], "content": "S3"},
        ])

        bare_agent.config.skill_max_active = 1
        mock_llm_client.set_responses([
            make_llm_response(json.dumps({"selected": ["s1", "s2", "s3"]})),
        ])

        result = mod.on_context("common task", "")
        assert result.count("[Active Skill:") == 1

        # --- Fallback disabled (C-level not triggered) ---
        bare_agent.modules.clear()
        bare_agent.config.skill_max_active = 2
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "hidden", "description": "Hidden skill",
             "tags": ["zzz_unique"], "content": "Hidden playbook"},
        ])

        result = mod.on_context("something completely different", "ctx")
        assert "ctx" in result
        # No skill should be activated
        assert "[Active Skill:" not in result

        # --- Fallback deprecated: skill_llm_fallback=True no longer triggers C-level ---
        # L3 (skill index + load_skill tool) replaces C-level fallback.
        # The skill appears in the index instead of being force-activated.
        bare_agent.modules.clear()
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "release-checklist", "description": "Pre-release checklist",
             "tags": ["release"], "content": "Release playbook"},
        ])

        bare_agent.config.skill_llm_fallback = True

        result = mod.on_context("what should I do before going live", "")
        # Not force-activated, but available in the L3 index
        assert "[Active Skill: release-checklist]" not in result
        assert "release-checklist" in result  # present in L3 index

        # --- No tag match: skill appears in index, not activated ---
        bare_agent.modules.clear()
        bare_agent.config.skill_llm_fallback = True
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "deploy", "description": "Deploy", "tags": ["deploy"]},
        ])

        result = mod.on_context("what is the weather", "ctx")
        assert "ctx" in result
        assert "[Active Skill:" not in result


# ============================================================
# Flow 2: Injection format and context preservation
# ============================================================

class TestInjectionFormatAndContext:
    """Skill content injection format, context ordering, and empty index."""

    def test_injection_format_and_context(self, bare_agent, tmp_path):
        """Injection block format, existing context preserved, empty index returns original."""

        # --- Skill block format: [Active Skill: name]...[End Skill] ---
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "my-skill", "description": "My skill",
             "tags": ["myskill"], "content": "Line 1\nLine 2"},
        ])

        mod._forced_skill = "my-skill"
        result = mod.on_context("test", "")
        assert "[Active Skill: my-skill]" in result
        assert "[End Skill]" in result
        assert "Line 1\nLine 2" in result

        # --- Existing context preserved, skill appended after ---
        bare_agent.modules.clear()
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "s1", "description": "S1", "tags": ["deployment"],
             "content": "S1 content"},
        ])

        result = mod.on_context("run the deployment", "[Memory] prior context")
        assert result.startswith("[Memory] prior context")
        assert "[Active Skill: s1]" in result
        assert result.index("[Memory]") < result.index("[Active Skill:")

        # --- No custom skills: context preserved, but L3 index may include builtins ---
        bare_agent.modules.clear()
        skills_dir = os.path.join(str(tmp_path), ".llamagent", "skills")
        os.makedirs(skills_dir, exist_ok=True)

        bare_agent.project_dir = str(tmp_path)
        bare_agent.config.skill_dirs = []
        bare_agent.config.skill_max_active = 2
        bare_agent.config.skill_llm_fallback = False

        mod = SkillModule()
        bare_agent.register_module(mod)
        result = mod.on_context("hello there", "original")
        assert "original" in result
        # No skill should be activated (no tag match)
        assert "[Active Skill:" not in result


# ============================================================
# Flow 3: Pack activation + /skill command
# ============================================================

class TestPackActivationAndSlashCommand:
    """Pack activation via required_tool_packs and /skill command flow."""

    def test_pack_activation_and_slash_command(self, bare_agent, tmp_path):
        """Tag match activates packs, no packs leaves _active_packs empty,
        /skill command activates packs, and full /skill command flow."""

        # --- Tag match activates required_tool_packs ---
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "my-toolsmith", "description": "Create tools",
             "tags": ["tool", "create"],
             "required_tool_packs": ["toolsmith"],
             "content": "Use create_tool to build helpers."},
        ])

        assert "toolsmith" not in bare_agent._active_packs

        result = mod.on_context("create a tool", "")
        assert "[Active Skill: my-toolsmith]" in result
        assert "toolsmith" in bare_agent._active_packs

        # --- Skill without packs does not modify _active_packs ---
        bare_agent.modules.clear()
        bare_agent._active_packs.clear()
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "deploy", "description": "Deploy workflow",
             "tags": ["deploy"],
             "content": "Deploy playbook."},
        ])

        mod.on_context("deploy the app", "")
        assert len(bare_agent._active_packs) == 0

        # --- /skill command also activates required_tool_packs ---
        bare_agent.modules.clear()
        bare_agent._active_packs.clear()
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "web-access", "description": "Fetch web pages",
             "tags": ["web"],
             "required_tool_packs": ["web"],
             "content": "Use web_fetch to get pages."},
        ])

        query = mod.on_input("/skill web-access fetch this page")
        result = mod.on_context(query, "")
        assert "web" in bare_agent._active_packs

        # --- Full /skill command flow: forced activation + injection ---
        bare_agent.modules.clear()
        bare_agent._active_packs.clear()
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "deploy", "description": "Deploy workflow",
             "tags": ["deploy"], "content": "## Steps\n1. Build\n2. Ship"},
        ])

        query = mod.on_input("/skill deploy run the deployment")
        assert mod._forced_skill == "deploy"
        assert query == "run the deployment"

        result = mod.on_context(query, "")
        assert "[Active Skill: deploy]" in result
        assert "## Steps" in result
        assert "[End Skill]" in result

        # --- /skill miss degrades to B-level tag match ---
        bare_agent.modules.clear()
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "deploy", "description": "Deploy workflow",
             "tags": ["deploy"], "content": "Deploy playbook"},
        ])

        mod.on_input("/skill nonexistent")
        result = mod.on_context("deploy the app", "")
        assert "[Active Skill: deploy]" in result

        # --- Normal input passes through without forced activation ---
        bare_agent.modules.clear()
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "deploy", "description": "Deploy", "tags": ["deploy"]},
        ])

        result = mod.on_input("just a normal question")
        assert mod._forced_skill is None
        assert result == "just a normal question"


# ============================================================
# Flow 4: Builtin skills
# ============================================================

class TestBuiltinSkills:
    """Builtin skills scanning and project-level override."""

    def test_builtin_skills(self, bare_agent, tmp_path):
        """Builtin skills are loaded from package directory, and project-level
        skills with the same name override builtins."""

        # --- Builtin skills loaded ---
        bare_agent.project_dir = str(tmp_path)
        bare_agent.config.skill_dirs = []
        bare_agent.config.skill_max_active = 2
        bare_agent.config.skill_llm_fallback = False

        mod = SkillModule()
        bare_agent.register_module(mod)

        assert mod.index.lookup("toolsmith") is not None
        assert mod.index.lookup("web-access") is not None
        assert mod.index.lookup("workspace-ops") is not None
        assert mod.index.lookup("lightweight-collab") is not None

        # --- Project skill overrides builtin ---
        bare_agent.modules.clear()
        skills_dir = os.path.join(str(tmp_path), ".llamagent", "skills")
        _create_skill(skills_dir, "toolsmith",
                      "Custom project toolsmith",
                      tags=["tool"], skill_content="Custom toolsmith playbook.")

        bare_agent.project_dir = str(tmp_path)
        bare_agent.config.skill_dirs = []
        bare_agent.config.skill_max_active = 2
        bare_agent.config.skill_llm_fallback = False

        mod = SkillModule()
        bare_agent.register_module(mod)

        meta = mod.index.lookup("toolsmith")
        assert meta is not None
        assert meta.description == "Custom project toolsmith"
        assert meta.priority == "project"
