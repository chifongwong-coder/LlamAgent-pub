"""
SkillMeta and SkillIndex: metadata model and directory scanning/lookup.

SkillMeta holds the structured metadata parsed from config.yaml.
SkillIndex scans skill directories, builds an in-memory index,
and provides lookup and tag-matching methods.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from llamagent.modules.skill.matcher import normalize_word, _TOKEN_PATTERN

logger = logging.getLogger(__name__)

# PyYAML is optional; graceful degradation when not installed
try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    logger.warning(
        "PyYAML is not installed, skill module cannot parse config.yaml. "
        "Please run: pip install pyyaml"
    )


@dataclass
class SkillMeta:
    """Skill metadata, parsed from config.yaml."""

    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    invocation: str = "both"  # "user-invocable" / "auto-trigger" / "both"
    skill_dir: str = ""
    priority: str = ""  # "project" / "compat" / "user" / "user-compat" / "custom"
    required_tool_packs: list[str] = field(default_factory=list)  # v1.6: packs to activate


class SkillIndex:
    """
    In-memory index for skill metadata.

    Scans skill directories at startup, parses config.yaml files,
    and provides lookup/matching methods. SKILL.md content is loaded
    lazily on activation (progressive disclosure).
    """

    def __init__(self):
        self._skills: dict[str, SkillMeta] = {}  # name -> meta
        self._alias_map: dict[str, str] = {}  # alias -> name
        self._content_cache: dict[str, str] = {}  # name -> SKILL.md content

    def scan(self, paths: list[tuple[str, str]]) -> list[SkillMeta]:
        """
        Scan skill directories and build the index.

        Args:
            paths: List of (directory_path, priority_label) tuples,
                   ordered from highest to lowest priority.

        Returns:
            List of all indexed SkillMeta objects.
        """
        if not _YAML_AVAILABLE:
            logger.warning("Skipping skill scan: PyYAML is not installed")
            return []

        self._skills.clear()
        self._alias_map.clear()
        self._content_cache.clear()

        for base_dir, priority in paths:
            if not os.path.isdir(base_dir):
                continue

            for entry in sorted(os.listdir(base_dir)):
                skill_dir = os.path.join(base_dir, entry)
                if not os.path.isdir(skill_dir):
                    continue

                config_path = os.path.join(skill_dir, "config.yaml")
                if not os.path.isfile(config_path):
                    # Also check config.yml
                    config_path = os.path.join(skill_dir, "config.yml")
                    if not os.path.isfile(config_path):
                        continue

                meta = self._parse_config(config_path, skill_dir, priority)
                if meta is None:
                    continue

                # Higher priority wins: skip if already indexed
                if meta.name in self._skills:
                    logger.debug(
                        "Skill '%s' already indexed (priority=%s), "
                        "skipping duplicate from %s",
                        meta.name,
                        self._skills[meta.name].priority,
                        skill_dir,
                    )
                    continue

                self._skills[meta.name] = meta
                for alias in meta.aliases:
                    key = alias.lower()
                    # Skip if alias collides with an existing skill name
                    if key in self._skills:
                        continue
                    # First-registered alias wins (higher priority)
                    if key not in self._alias_map:
                        self._alias_map[key] = meta.name

        count = len(self._skills)
        if count > 0:
            logger.info("Skill index built: %d skill(s) indexed", count)
        else:
            logger.debug("Skill index built: no skills found")

        return list(self._skills.values())

    def _parse_config(
        self, config_path: str, skill_dir: str, priority: str
    ) -> SkillMeta | None:
        """Parse a config.yaml file into a SkillMeta object."""
        try:
            with open(config_path, "r", encoding="utf-8-sig") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", config_path, e)
            return None

        if not isinstance(data, dict):
            logger.warning("Invalid config.yaml (not a dict): %s", config_path)
            return None

        # Validate required fields
        name = data.get("name")
        description = data.get("description")
        if not name or not description:
            logger.warning(
                "Skill config missing required fields (name/description): %s",
                config_path,
            )
            return None

        # Validate invocation field
        invocation = data.get("invocation", "both")
        if invocation not in ("user-invocable", "auto-trigger", "both"):
            logger.warning(
                "Skill '%s' has invalid invocation='%s', defaulting to 'both'",
                name,
                invocation,
            )
            invocation = "both"

        # Build tags list (ensure strings, filter None/empty)
        raw_tags = data.get("tags", [])
        tags = [str(t).strip() for t in raw_tags if t is not None] if isinstance(raw_tags, list) else []
        tags = [t for t in tags if t]

        # Build aliases list (ensure strings, filter None/empty)
        raw_aliases = data.get("aliases", [])
        aliases = [str(a).strip() for a in raw_aliases if a is not None] if isinstance(raw_aliases, list) else []
        aliases = [a for a in aliases if a]

        # Build required_tool_packs list
        raw_packs = data.get("required_tool_packs", [])
        required_tool_packs = [str(p).strip() for p in raw_packs if p is not None] if isinstance(raw_packs, list) else []
        required_tool_packs = [p for p in required_tool_packs if p]

        return SkillMeta(
            name=str(name).strip(),
            description=str(description).strip(),
            tags=tags,
            aliases=aliases,
            invocation=invocation,
            skill_dir=skill_dir,
            priority=priority,
            required_tool_packs=required_tool_packs,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, name: str) -> SkillMeta | None:
        """
        Exact lookup by name or alias.

        Name match takes priority over alias match.
        Both name and alias lookups are case-insensitive.

        Args:
            name: Skill name or alias.

        Returns:
            SkillMeta if found, None otherwise.
        """
        # Direct name match (exact)
        if name in self._skills:
            return self._skills[name]
        # Case-insensitive name match
        name_lower = name.lower()
        for skill_name, meta in self._skills.items():
            if skill_name.lower() == name_lower:
                return meta
        # Alias match (case-insensitive)
        real_name = self._alias_map.get(name_lower)
        if real_name and real_name in self._skills:
            return self._skills[real_name]
        return None

    def match_tags(self, tokens: set[str], raw_query: str) -> list[SkillMeta]:
        """
        Match tokenized query against skill tags.

        For ASCII (English) tags: check if the stemmed tag is in the token set.
        For non-ASCII (Chinese etc.) tags: check if the tag is a substring of the raw query.

        Only returns skills whose invocation allows auto-trigger ("auto-trigger" or "both").

        Args:
            tokens: Set of stemmed/lowercased tokens from the query.
            raw_query: Original query string (for Chinese substring matching).

        Returns:
            List of matching SkillMeta objects.
        """
        query_lower = raw_query.lower()
        matched = []

        for meta in self._skills.values():
            # Skip skills that don't allow auto-trigger
            if meta.invocation == "user-invocable":
                continue

            if self._tags_match(meta.tags, tokens, query_lower):
                matched.append(meta)

        return matched

    @staticmethod
    def _tags_match(tags: list[str], tokens: set[str], query_lower: str) -> bool:
        """
        Check if any tag matches the query tokens or raw query.

        English tags: split on non-alphanumeric chars (handles hyphenated tags
        like "db-migration"), normalize each sub-token into variant forms, then
        check exact set intersection against query tokens.

        Non-ASCII tags (Chinese, etc.): substring matching against raw query.
        """
        for tag in tags:
            tag_stripped = tag.strip().lower()
            if not tag_stripped:
                continue

            if tag_stripped.isascii():
                # Split tag into sub-tokens (handles "db-migration" -> ["db", "migration"])
                sub_words = [m.group().lower() for m in _TOKEN_PATTERN.finditer(tag_stripped)]
                if not sub_words:
                    continue
                # Match if ANY sub-token's normalized forms overlap with query tokens
                for sw in sub_words:
                    tag_forms = normalize_word(sw)
                    if tag_forms & tokens:
                        return True
            else:
                # Non-ASCII tag (Chinese, etc.): substring matching
                if tag_stripped in query_lower:
                    return True

        return False

    # ------------------------------------------------------------------
    # Content loading
    # ------------------------------------------------------------------

    def load_content(self, meta: SkillMeta) -> str:
        """
        Lazy-load SKILL.md content for a skill.

        Results are cached to avoid re-reading on subsequent activations.

        Args:
            meta: SkillMeta whose content to load.

        Returns:
            SKILL.md content string, or empty string on error.
        """
        if meta.name in self._content_cache:
            return self._content_cache[meta.name]

        skill_md_path = os.path.join(meta.skill_dir, "SKILL.md")
        if not os.path.isfile(skill_md_path):
            logger.warning("SKILL.md not found: %s", skill_md_path)
            return ""

        try:
            with open(skill_md_path, "r", encoding="utf-8-sig") as f:
                content = f.read()
        except Exception as e:
            logger.warning("Failed to read SKILL.md for skill '%s': %s", meta.name, e)
            return ""

        self._content_cache[meta.name] = content
        return content

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def all_skills(self) -> list[SkillMeta]:
        """Return all indexed skills."""
        return list(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)
