"""
SkillModule: task-level playbook injection via on_context.

Tool handles "what can be done", Skill handles "how things should be done".

Lifecycle:
- on_attach: scan skill directories, build metadata index (no content loaded)
- on_input:  intercept /skill commands, record forced activation
- on_context: match query -> activate skills -> inject SKILL.md content into context
"""

from __future__ import annotations

import logging
import os
import re

from llamagent.core.agent import Module, LlamAgent
from llamagent.modules.skill.index import SkillIndex, SkillMeta
from llamagent.modules.skill.matcher import (
    DISAMBIGUATE_SYSTEM,
    DISAMBIGUATE_USER_TEMPLATE,
    FALLBACK_SYSTEM,
    FALLBACK_USER_TEMPLATE,
    format_skill_list,
    tokenize_query,
)

logger = logging.getLogger(__name__)

_SKILL_CMD_RE = re.compile(r"^/skill\s+(\S+)\s*(.*)", re.DOTALL)


class SkillModule(Module):
    """
    Skill Module: inject task-level playbook content into LLM context.

    Skills are loaded from config.yaml + SKILL.md directory pairs.
    Matching is done via /skill command (A-level), tag matching (B-level),
    or optional LLM full-metadata scan (C-level fallback).
    """

    name: str = "skill"
    description: str = "Skill system: task-level playbook injection via on_context"

    def __init__(self):
        self.index: SkillIndex | None = None
        self._forced_skill: str | None = None

    # ============================================================
    # Lifecycle Callbacks
    # ============================================================

    def on_attach(self, agent: LlamAgent) -> None:
        """Scan skill directories and build metadata index."""
        super().on_attach(agent)

        # Build scan paths (highest priority first)
        paths = self._build_scan_paths(agent)

        self.index = SkillIndex()
        skills = self.index.scan(paths)

        if skills:
            names = [s.name for s in skills]
            logger.info("SkillModule loaded %d skill(s): %s", len(skills), names)

    def _build_scan_paths(self, agent: LlamAgent) -> list[tuple[str, str]]:
        """
        Build the ordered list of (directory_path, priority_label) tuples.

        Scan order (highest priority first):
        1. {project_dir}/.llamagent/skills/  (project)
        2. {project_dir}/.agents/skills/     (compat)
        3. ~/.llamagent/skills/              (user)
        4. ~/.agents/skills/                 (user-compat)
        5. config.skill_dirs entries         (custom)
        """
        project = agent.project_dir
        home = os.path.expanduser("~")

        paths = [
            (os.path.join(project, ".llamagent", "skills"), "project"),
            (os.path.join(project, ".agents", "skills"), "compat"),
            (os.path.join(home, ".llamagent", "skills"), "user"),
            (os.path.join(home, ".agents", "skills"), "user-compat"),
        ]

        # Append custom paths from config
        for custom_dir in agent.config.skill_dirs:
            resolved = os.path.realpath(custom_dir)
            paths.append((resolved, "custom"))

        # Builtin skills (lowest priority — project/user skills override these)
        builtin_dir = os.path.join(os.path.dirname(__file__), "builtin_skills")
        paths.append((builtin_dir, "builtin"))

        return paths

    # ============================================================
    # Pipeline Callbacks
    # ============================================================

    def on_input(self, user_input: str) -> str:
        """
        Intercept /skill <name> commands.

        Sets _forced_skill for on_context to consume. Strips the command
        prefix and returns the remaining text (or a default activation message).
        """
        m = _SKILL_CMD_RE.match(user_input)
        if m:
            self._forced_skill = m.group(1)
            rest = m.group(2).strip()
            return rest or f"Execute the {m.group(1)} skill workflow."

        self._forced_skill = None
        return user_input

    def on_context(self, query: str, context: str) -> str:
        """
        Match query against skills and inject activated playbook content.

        Matching flow:
        A-level: /skill command (forced activation via _forced_skill)
        B-level: tokenize query -> match tags -> LLM disambiguate if 2+ candidates
        C-level: LLM full metadata scan (optional, config.skill_llm_fallback)
        """
        if self.index is None or len(self.index) == 0:
            self._forced_skill = None
            return context

        activated: list[SkillMeta] = []

        # --- A-level: forced skill ---
        if self._forced_skill:
            meta = self.index.lookup(self._forced_skill)
            if meta:
                activated = [meta]
                logger.info("Skill '%s' activated via /skill command", meta.name)
            else:
                logger.warning(
                    "Skill '%s' not found, falling back to auto-match",
                    self._forced_skill,
                )
                # Fall through to B-level
            self._forced_skill = None

        # --- B-level: tag matching ---
        if not activated:
            tokens = tokenize_query(query)
            candidates = self.index.match_tags(tokens, query)

            if len(candidates) == 1:
                activated = candidates
                logger.debug("Skill '%s' activated via tag match", candidates[0].name)
            elif len(candidates) >= 2:
                activated = self._disambiguate(query, candidates)
            # candidates == 0 -> fall through to C-level check

        # --- C-level: LLM full metadata fallback (optional) ---
        if not activated and getattr(self.agent.config, "skill_llm_fallback", False):
            activated = self._llm_fallback(query)

        # --- Truncation ---
        max_active = getattr(self.agent.config, "skill_max_active", 2)
        if len(activated) > max_active:
            activated = activated[:max_active]

        # --- Injection ---
        if not activated:
            return context

        # Activate required tool packs for matched skills
        for meta in activated:
            if meta.required_tool_packs:
                for pack_name in meta.required_tool_packs:
                    self.agent._active_packs.add(pack_name)

        blocks = []
        for meta in activated:
            content = self.index.load_content(meta)
            if content:
                blocks.append(
                    f"[Active Skill: {meta.name}]\n{content}\n[End Skill]"
                )
                logger.info("Skill '%s' injected into context", meta.name)

        if not blocks:
            return context

        skill_text = "\n\n".join(blocks)
        return f"{context}\n\n{skill_text}" if context else skill_text

    # ============================================================
    # LLM disambiguation and fallback
    # ============================================================

    def _disambiguate(self, query: str, candidates: list[SkillMeta]) -> list[SkillMeta]:
        """
        B-level LLM disambiguation: select from 2+ tag-matched candidates.

        Sends only metadata (name + description) to the LLM.
        On any error, falls back to returning the first candidate.
        """
        try:
            prompt = DISAMBIGUATE_USER_TEMPLATE.format(
                query=query,
                candidates=format_skill_list(candidates),
            )
            result = self.llm.ask_json(
                prompt=prompt, system=DISAMBIGUATE_SYSTEM, temperature=0.3
            )
            selected_names = result.get("selected", [])
            if not isinstance(selected_names, list):
                selected_names = []

            # Map names back to SkillMeta
            name_set = {n.strip() for n in selected_names if isinstance(n, str)}
            selected = [c for c in candidates if c.name in name_set]

            if selected:
                logger.debug(
                    "LLM disambiguation selected: %s",
                    [s.name for s in selected],
                )
                return selected

        except Exception as e:
            logger.warning("LLM skill disambiguation failed: %s", e)

        # Fallback: return the first candidate
        logger.debug(
            "LLM disambiguation fallback: using first candidate '%s'",
            candidates[0].name,
        )
        return [candidates[0]]

    def _llm_fallback(self, query: str) -> list[SkillMeta]:
        """
        C-level LLM full metadata scan: send all skill metadata to LLM.

        Only called when B-level returns 0 candidates and config.skill_llm_fallback is True.
        On any error, returns empty list (no skill activated).
        """
        all_skills = self.index.all_skills()
        if not all_skills:
            return []

        # Filter to auto-triggerable skills only
        eligible = [s for s in all_skills if s.invocation in ("auto-trigger", "both")]
        if not eligible:
            return []

        try:
            prompt = FALLBACK_USER_TEMPLATE.format(
                query=query,
                skills=format_skill_list(eligible),
            )
            result = self.llm.ask_json(
                prompt=prompt, system=FALLBACK_SYSTEM, temperature=0.3
            )
            selected_names = result.get("selected", [])
            if not isinstance(selected_names, list):
                selected_names = []

            # Map names back to SkillMeta
            name_set = {n.strip() for n in selected_names if isinstance(n, str)}
            selected = [s for s in eligible if s.name in name_set]

            if selected:
                logger.info(
                    "LLM fallback activated skill(s): %s",
                    [s.name for s in selected],
                )
                return selected

        except Exception as e:
            logger.warning("LLM skill fallback failed: %s", e)

        return []

    # ============================================================
    # External API
    # ============================================================

    def list_skills(self) -> list[SkillMeta]:
        """List all indexed skills."""
        if self.index is None:
            return []
        return self.index.all_skills()

    def get_skill(self, name: str) -> SkillMeta | None:
        """Get a skill by name or alias."""
        if self.index is None:
            return None
        return self.index.lookup(name)

    def activate(self, name: str) -> str:
        """Manually activate a skill and return its SKILL.md content."""
        if self.index is None:
            return ""
        meta = self.index.lookup(name)
        if meta is None:
            return ""
        return self.index.load_content(meta)

    def reload(self) -> int:
        """Re-scan skill directories and refresh the index. Returns new skill count."""
        if self.index is None:
            return 0
        paths = self._build_scan_paths(self.agent)
        self.index.scan(paths)
        return len(self.index)
