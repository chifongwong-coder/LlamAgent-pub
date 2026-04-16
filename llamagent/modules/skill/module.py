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
    format_skill_list,
    tokenize_query,
)

logger = logging.getLogger(__name__)

_SKILL_CMD_RE = re.compile(r"^/skill\s+(\S+)\s*(.*)", re.DOTALL)

# Prompt injection detection patterns for security scan
# Note: "act as" removed — too broad, triggers on legitimate skill content
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?prior\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a\s+different", re.IGNORECASE),
    re.compile(r"^system\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
]


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
        self._current_activated: list = []

    # ============================================================
    # Lifecycle Callbacks
    # ============================================================

    def on_attach(self, agent: LlamAgent) -> None:
        """Scan skill directories, build metadata index, and register load_skill tool."""
        super().on_attach(agent)

        # Build scan paths (highest priority first)
        paths = self._build_scan_paths(agent)

        self.index = SkillIndex()
        skills = self.index.scan(paths)

        if skills:
            names = [s.name for s in skills]
            logger.info("SkillModule loaded %d skill(s): %s", len(skills), names)

        # Register load_skill tool (L3: LLM can self-serve skill content)
        agent.register_tool(
            name="load_skill",
            func=self._load_skill_handler,
            description=(
                "Load a skill's full content by name. Use this when you need detailed "
                "instructions from an available skill listed in the context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name from the available skills list",
                    },
                },
                "required": ["name"],
            },
            tier="default",
            safety_level=1,
        )

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
        self._current_activated = []

        m = _SKILL_CMD_RE.match(user_input)
        if m:
            self._forced_skill = m.group(1)
            rest = m.group(2).strip()
            return rest or f"Execute the {m.group(1)} skill workflow."

        self._forced_skill = None
        return user_input

    def on_context(self, query: str, context: str) -> str:
        """
        4-layer skill matching and injection.

        L1: /skill command (forced activation)
        L4: always=true skills (injected every turn, exempt from truncation)
        L2: tag matching (skipped if L1 fired)
        L3: skill index injection (name + description for LLM self-serve via load_skill)
        """
        if self.index is None or len(self.index) == 0:
            self._forced_skill = None
            return context

        activated: list[SkillMeta] = []
        activated_names: set[str] = set()
        l1_fired = False

        # --- L1: /skill command ---
        if self._forced_skill:
            meta = self.index.lookup(self._forced_skill)
            if meta:
                activated.append(meta)
                activated_names.add(meta.name)
                l1_fired = True
                logger.info("Skill '%s' activated via /skill command", meta.name)
            else:
                logger.warning(
                    "Skill '%s' not found, falling back to auto-match",
                    self._forced_skill,
                )
            self._forced_skill = None

        # --- L4: always=true skills (exempt from truncation) ---
        always_skills: list[SkillMeta] = []
        for skill in self.index.all_skills():
            if skill.always and skill.name not in activated_names:
                always_skills.append(skill)
                activated_names.add(skill.name)

        # --- L2: tag matching (skip if L1 fired) ---
        if not l1_fired:
            tokens = tokenize_query(query)
            candidates = self.index.match_tags(tokens, query)
            candidates = [c for c in candidates if c.name not in activated_names]

            if len(candidates) == 1:
                activated.append(candidates[0])
                activated_names.add(candidates[0].name)
                logger.debug("Skill '%s' activated via tag match", candidates[0].name)
            elif len(candidates) >= 2:
                selected = self._disambiguate(query, candidates)
                for s in selected:
                    activated.append(s)
                    activated_names.add(s.name)

        # --- Truncation (only L1+L2 skills; always skills are exempt) ---
        max_active = getattr(self.agent.config, "skill_max_active", 2)
        if len(activated) > max_active:
            activated = activated[:max_active]

        # Merge: always skills + L1/L2 activated
        activated = always_skills + activated
        activated_names = {s.name for s in activated}

        # Record activated skills for cross-module queries (e.g. ReflectionModule)
        self._current_activated = activated

        # --- Inject activated skill content + activate tool packs ---
        blocks: list[str] = []
        for meta in activated:
            content = self.index.load_content(meta)
            if content:
                blocks.append(
                    f"[Active Skill: {meta.name}]\n{content}\n[End Skill]"
                )
                logger.info("Skill '%s' injected into context", meta.name)
            if meta.required_tool_packs:
                for pack_name in meta.required_tool_packs:
                    self.agent._active_packs.add(pack_name)

        # --- L3: inject skill index (exclude already activated) ---
        skill_index = self._build_skill_index(activated_names)

        # --- Assemble ---
        parts: list[str] = []
        if context:
            parts.append(context)
        if blocks:
            parts.append("\n\n".join(blocks))
        if skill_index:
            parts.append(skill_index)

        return "\n\n".join(parts) if parts else ""

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

    # ============================================================
    # L3: Skill index and load_skill tool
    # ============================================================

    def _build_skill_index(self, exclude_names: set[str]) -> str:
        """Build lightweight skill index for system prompt (L3).

        Lists available skills by name + description so the LLM can decide
        whether to load them via the load_skill tool.

        Args:
            exclude_names: Names of already-activated skills to exclude.
        """
        all_skills = self.index.all_skills()
        available = [
            s for s in all_skills
            if s.name not in exclude_names and not s.always
        ]
        if not available:
            return ""

        lines = ["[Available Skills] Use load_skill tool to load detailed instructions:"]
        for s in available:
            lines.append(f"- {s.name}: {s.description}")
        return "\n".join(lines)

    def _load_skill_handler(self, name: str) -> str:
        """Tool handler for load_skill (L3).

        Returns the full skill content. If the skill has required_tool_packs,
        appends a note that /skill command is needed for full tool activation.
        """
        if self.index is None:
            return f"Skill '{name}' not found. No skills are loaded."

        meta = self.index.lookup(name)
        if meta is None:
            return f"Skill '{name}' not found. Check the available skills list in context."

        content = self.index.load_content(meta)
        if not content:
            return f"Skill '{name}' has no content."

        # L3 limitation: tool packs cannot be activated mid-ReAct
        note = ""
        if meta.required_tool_packs:
            note = (
                f"\n\n[Note: This skill works best with '/skill {name}' "
                f"for full tool activation]"
            )

        return content + note

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

    def get_active_skills(self) -> list:
        """Return skills activated in the current turn (set during on_context)."""
        return list(self._current_activated)

    def update_skill(self, name: str, new_content: str) -> dict:
        """Update a skill's content with safety mechanisms.

        Flow: security scan -> backup -> write -> validate -> rollback on failure.

        Returns:
            {"success": True/False, "error": "..." if failed}
        """
        meta = self.index.lookup(name)
        if meta is None:
            return {"success": False, "error": f"Skill '{name}' not found"}

        content_path = meta.content_path
        if not os.path.isfile(content_path):
            return {"success": False, "error": f"Skill file not found: {content_path}"}

        # 1. Prepare final content (prepend frontmatter for format B/C)
        final_content = new_content
        if meta.source_format in ("frontmatter", "plain_md"):
            from llamagent.modules.fs_store.parser import parse_frontmatter
            with open(content_path, "r", encoding="utf-8") as f:
                old_fm, _ = parse_frontmatter(f.read())
            if old_fm:
                fm_lines = ["---"]
                for k, v in old_fm.items():
                    if isinstance(v, list):
                        # Serialize lists as YAML-compatible format: [item1, item2]
                        fm_lines.append(f"{k}: [{', '.join(str(x) for x in v)}]")
                    elif isinstance(v, bool):
                        fm_lines.append(f"{k}: {'true' if v else 'false'}")
                    elif isinstance(v, str) and (":" in v or '"' in v or "'" in v or "\n" in v):
                        # Quote strings with special characters for YAML safety
                        escaped = v.replace('"', '\\"')
                        fm_lines.append(f'{k}: "{escaped}"')
                    else:
                        fm_lines.append(f"{k}: {v}")
                fm_lines.append("---\n")
                final_content = "\n".join(fm_lines) + new_content

        # 2. Security scan BEFORE writing (fail fast)
        scan_error = self._security_scan(final_content)
        if scan_error:
            return {"success": False, "error": f"Security scan failed: {scan_error}"}

        # 3. Backup current version
        backup_path = self._create_backup(content_path)

        # 4. Write new content
        try:
            with open(content_path, "w", encoding="utf-8") as f:
                f.write(final_content)
        except OSError as e:
            return {"success": False, "error": f"Write failed: {e}"}

        # 5. Post-write validation (format check)
        validation_error = self._validate_skill(content_path, meta)
        if validation_error:
            self._rollback(content_path, backup_path)
            return {"success": False, "error": f"Validation failed: {validation_error}"}

        # 6. Reload skill index to pick up changes
        self.index.invalidate_cache(name)

        logger.info("Skill '%s' updated successfully (backup: %s)", name, backup_path)
        return {"success": True}

    def _create_backup(self, content_path: str) -> str:
        """Create a versioned backup of the skill file."""
        import shutil

        directory = os.path.dirname(content_path)
        basename = os.path.basename(content_path)

        # Find next version number
        version = 1
        while os.path.exists(os.path.join(directory, f"{basename}.v{version}")):
            version += 1

        backup_path = os.path.join(directory, f"{basename}.v{version}")
        shutil.copy2(content_path, backup_path)
        return backup_path

    def _validate_skill(self, content_path: str, meta) -> str | None:
        """Validate skill file format. Returns error string or None if valid."""
        try:
            with open(content_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            return f"Cannot read file: {e}"

        if not content.strip():
            return "Skill content is empty"

        # For format A: no frontmatter validation needed (config.yaml handles metadata)
        if meta.source_format == "config":
            return None

        # For format B/C: validate frontmatter if present
        from llamagent.modules.fs_store.parser import parse_frontmatter
        fm, body = parse_frontmatter(content)
        if fm:
            if not fm.get("name") and not fm.get("description"):
                return "Frontmatter missing both 'name' and 'description'"

        if not body.strip():
            return "Skill body content is empty (only frontmatter)"

        return None

    def _security_scan(self, content: str) -> str | None:
        """Scan content for prompt injection patterns. Returns error or None."""
        for pattern in _INJECTION_PATTERNS:
            match = pattern.search(content)
            if match:
                return f"Potential prompt injection detected: '{match.group()}'"
        return None

    def _rollback(self, content_path: str, backup_path: str):
        """Restore skill file from backup."""
        import shutil

        try:
            shutil.copy2(backup_path, content_path)
            logger.info("Rolled back skill to backup: %s", backup_path)
        except OSError as e:
            logger.error("Rollback failed: %s", e)
