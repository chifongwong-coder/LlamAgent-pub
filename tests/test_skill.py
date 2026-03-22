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
                  invocation="both", skill_content="## Goal\nTest skill."):
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
        )

    bare_agent.project_dir = str(tmp_path)
    bare_agent.config.skill_dirs = []
    bare_agent.config.skill_max_active = 2
    bare_agent.config.skill_llm_fallback = False

    mod = SkillModule()
    bare_agent.register_module(mod)
    return mod


# ============================================================
# A-level: /skill command flow
# ============================================================

class TestSkillCommandFlow:
    """/skill command end-to-end: on_input intercept -> on_context inject."""

    def test_slash_skill_activates_and_injects(self, bare_agent, tmp_path):
        """Full A-level flow: /skill command -> forced activation -> content injected."""
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "deploy", "description": "Deploy workflow",
             "tags": ["deploy"], "content": "## Steps\n1. Build\n2. Ship"},
        ])

        # on_input intercepts the command
        query = mod.on_input("/skill deploy run the deployment")
        assert mod._forced_skill == "deploy"
        assert query == "run the deployment"

        # on_context injects the skill
        result = mod.on_context(query, "")
        assert "[Active Skill: deploy]" in result
        assert "## Steps" in result
        assert "[End Skill]" in result

    def test_slash_skill_not_found_falls_to_tag_match(self, bare_agent, tmp_path):
        """A-level miss degrades to B-level tag match."""
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "deploy", "description": "Deploy workflow",
             "tags": ["deploy"], "content": "Deploy playbook"},
        ])

        mod.on_input("/skill nonexistent")
        result = mod.on_context("deploy the app", "")
        assert "[Active Skill: deploy]" in result

    def test_normal_input_does_not_force(self, bare_agent, tmp_path):
        """Normal input passes through without forced activation."""
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "deploy", "description": "Deploy", "tags": ["deploy"]},
        ])

        result = mod.on_input("just a normal question")
        assert mod._forced_skill is None
        assert result == "just a normal question"


# ============================================================
# B-level: tag matching flow
# ============================================================

class TestTagMatchingFlow:
    """B-level tag matching: single match, multi match + LLM, no match."""

    def test_single_tag_match_activates(self, bare_agent, tmp_path):
        """Single tag hit activates the skill without LLM call."""
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

    def test_multiple_candidates_llm_disambiguate(self, bare_agent, tmp_path, mock_llm_client):
        """2+ tag hits triggers LLM disambiguation."""
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

    def test_llm_error_falls_back_to_first_candidate(self, bare_agent, tmp_path, mock_llm_client):
        """LLM failure during disambiguation falls back to first candidate."""
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "skill-a", "description": "Skill A",
             "tags": ["shared"], "content": "A playbook"},
            {"name": "skill-b", "description": "Skill B",
             "tags": ["shared"], "content": "B playbook"},
        ])

        mock_llm_client.set_responses([Exception("LLM error")])
        result = mod.on_context("shared task", "")
        assert "[Active Skill: skill-a]" in result

    def test_no_tag_match_returns_original(self, bare_agent, tmp_path):
        """No tag match -> context unchanged."""
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "deploy", "description": "Deploy", "tags": ["deploy"]},
        ])

        result = mod.on_context("what is the weather today", "original context")
        assert result == "original context"

    def test_skill_max_active_truncation(self, bare_agent, tmp_path, mock_llm_client):
        """Only skill_max_active skills are injected."""
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


# ============================================================
# C-level: LLM fallback flow
# ============================================================

class TestLLMFallbackFlow:
    """C-level LLM full metadata scan (optional fallback)."""

    def test_fallback_disabled_by_default(self, bare_agent, tmp_path):
        """C-level not triggered when skill_llm_fallback=False."""
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "hidden", "description": "Hidden skill",
             "tags": ["zzz_unique"], "content": "Hidden playbook"},
        ])

        result = mod.on_context("something completely different", "ctx")
        assert result == "ctx"

    def test_fallback_activates_when_enabled(self, bare_agent, tmp_path, mock_llm_client):
        """C-level triggers when B-level has 0 candidates and fallback is on."""
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "release-checklist", "description": "Pre-release checklist",
             "tags": ["release"], "content": "Release playbook"},
        ])

        bare_agent.config.skill_llm_fallback = True
        mock_llm_client.set_responses([
            make_llm_response(json.dumps({"selected": ["release-checklist"]})),
        ])

        result = mod.on_context("what should I do before going live", "")
        assert "[Active Skill: release-checklist]" in result

    def test_fallback_llm_returns_empty(self, bare_agent, tmp_path, mock_llm_client):
        """C-level LLM returns no match -> no skill injected."""
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "deploy", "description": "Deploy", "tags": ["deploy"]},
        ])

        bare_agent.config.skill_llm_fallback = True
        mock_llm_client.set_responses([
            make_llm_response(json.dumps({"selected": []})),
        ])

        result = mod.on_context("what is the weather", "ctx")
        assert result == "ctx"


# ============================================================
# Injection format and context preservation
# ============================================================

class TestInjectionFormat:
    """Skill content injection format and context ordering."""

    def test_skill_block_format(self, bare_agent, tmp_path):
        """Injected skill has [Active Skill: name]...[End Skill] wrapper."""
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "my-skill", "description": "My skill",
             "tags": ["myskill"], "content": "Line 1\nLine 2"},
        ])

        mod._forced_skill = "my-skill"
        result = mod.on_context("test", "")
        assert result.startswith("[Active Skill: my-skill]")
        assert result.endswith("[End Skill]")
        assert "Line 1\nLine 2" in result

    def test_existing_context_preserved(self, bare_agent, tmp_path):
        """Existing context is preserved, skill appended after."""
        mod = _make_agent_with_skills(bare_agent, tmp_path, [
            {"name": "s1", "description": "S1", "tags": ["deployment"],
             "content": "S1 content"},
        ])

        result = mod.on_context("run the deployment", "[Memory] prior context")
        assert result.startswith("[Memory] prior context")
        assert "[Active Skill: s1]" in result
        assert result.index("[Memory]") < result.index("[Active Skill:")

    def test_empty_index_returns_original(self, bare_agent, tmp_path):
        """No skills indexed -> context unchanged."""
        skills_dir = os.path.join(str(tmp_path), ".llamagent", "skills")
        os.makedirs(skills_dir, exist_ok=True)

        bare_agent.project_dir = str(tmp_path)
        bare_agent.config.skill_dirs = []
        bare_agent.config.skill_max_active = 2
        bare_agent.config.skill_llm_fallback = False

        mod = SkillModule()
        bare_agent.register_module(mod)
        result = mod.on_context("any query", "original")
        assert result == "original"
