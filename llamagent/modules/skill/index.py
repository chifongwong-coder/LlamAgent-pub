"""
SkillMeta and SkillIndex: metadata model and directory scanning/lookup.

SkillMeta holds the structured metadata for a skill.
SkillIndex scans skill directories, builds an in-memory index,
and provides lookup and tag-matching methods.

Supports three skill formats:
- Format A: config.yaml + SKILL.md (requires PyYAML)
- Format B: SKILL.md with YAML frontmatter (uses fs_store/parser.py)
- Format C: single .md file with optional frontmatter (uses fs_store/parser.py)
"""

from __future__ import annotations

import glob as _glob
import logging
import os
import re
from dataclasses import dataclass, field

from llamagent.modules.skill.matcher import normalize_word, _TOKEN_PATTERN

logger = logging.getLogger(__name__)

# PyYAML is optional; only needed for Format A (config.yaml)
try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    logger.info(
        "PyYAML is not installed; config.yaml skills (Format A) will be skipped. "
        "Frontmatter and plain .md skills still work. "
        "To enable Format A: pip install pyyaml"
    )

# Files to exclude from Format C scan (not skill files)
_EXCLUDED_MD_FILES = {"SKILL.md", "README.md", "CHANGELOG.md"}


@dataclass
class SkillMeta:
    """Skill metadata, parsed from config.yaml, frontmatter, or inferred from .md file."""

    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    invocation: str = "both"  # "user-invocable" / "auto-trigger" / "both"
    skill_dir: str = ""
    priority: str = ""  # "project" / "compat" / "user" / "user-compat" / "custom"
    required_tool_packs: list[str] = field(default_factory=list)  # v1.6: packs to activate
    content_path: str = ""  # Full path to SKILL.md or .md file
    source_format: str = "config"  # "config" / "frontmatter" / "plain_md"
    always: bool = False  # L4: inject every turn


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

        Supports three formats:
        - Format A: subdirectory with config.yaml + SKILL.md (requires PyYAML)
        - Format B: subdirectory with SKILL.md containing frontmatter
        - Format C: single .md file in the base directory

        Args:
            paths: List of (directory_path, priority_label) tuples,
                   ordered from highest to lowest priority.

        Returns:
            List of all indexed SkillMeta objects.
        """
        self._skills.clear()
        self._alias_map.clear()
        self._content_cache.clear()

        for base_dir, priority in paths:
            if not os.path.isdir(base_dir):
                continue

            # 1. Scan subdirectories (Format A / B)
            for entry in sorted(os.listdir(base_dir)):
                skill_dir = os.path.join(base_dir, entry)
                if not os.path.isdir(skill_dir):
                    continue

                config_path = os.path.join(skill_dir, "config.yaml")
                if not os.path.isfile(config_path):
                    config_path = os.path.join(skill_dir, "config.yml")

                skill_path = os.path.join(skill_dir, "SKILL.md")

                meta = None
                if os.path.isfile(config_path) and _YAML_AVAILABLE:
                    # Format A: config.yaml + SKILL.md
                    meta = self._parse_config(config_path, skill_dir, priority)
                elif os.path.isfile(skill_path):
                    # Format B: SKILL.md with frontmatter
                    # Falls back to Format-C-style inference if no frontmatter
                    meta = self._load_from_frontmatter(
                        skill_path, fallback_name=entry
                    )
                    if meta:
                        meta.priority = priority
                else:
                    continue

                if meta is None:
                    continue

                self._register_meta(meta, skill_dir)

            # 2. Scan loose .md files (Format C)
            for md_path in sorted(_glob.glob(os.path.join(base_dir, "*.md"))):
                basename = os.path.basename(md_path)
                if basename in _EXCLUDED_MD_FILES:
                    continue

                meta = self._load_from_md_file(md_path)
                if meta:
                    meta.priority = priority
                    self._register_meta(meta, base_dir)

        count = len(self._skills)
        if count > 0:
            logger.info("Skill index built: %d skill(s) indexed", count)
        else:
            logger.debug("Skill index built: no skills found")

        return list(self._skills.values())

    def _register_meta(self, meta: SkillMeta, skill_dir: str) -> None:
        """Register a SkillMeta in the index (dedup by name, first wins)."""
        if meta.name in self._skills:
            logger.debug(
                "Skill '%s' already indexed (priority=%s), "
                "skipping duplicate from %s",
                meta.name,
                self._skills[meta.name].priority,
                skill_dir,
            )
            return

        self._skills[meta.name] = meta
        for alias in meta.aliases:
            key = alias.lower()
            if key in self._skills:
                continue
            if key not in self._alias_map:
                self._alias_map[key] = meta.name

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

        # content_path: SKILL.md alongside config.yaml
        content_path = os.path.join(skill_dir, "SKILL.md")

        # always flag (L4)
        always = bool(data.get("always", False))

        return SkillMeta(
            name=str(name).strip(),
            description=str(description).strip(),
            tags=tags,
            aliases=aliases,
            invocation=invocation,
            skill_dir=skill_dir,
            priority=priority,
            required_tool_packs=required_tool_packs,
            content_path=content_path,
            source_format="config",
            always=always,
        )

    # ------------------------------------------------------------------
    # Format B/C loaders
    # ------------------------------------------------------------------

    def _load_from_frontmatter(
        self, skill_path: str, fallback_name: str
    ) -> SkillMeta | None:
        """Load skill metadata from a SKILL.md file with optional frontmatter (Format B).

        If frontmatter is empty or missing, falls back to Format-C-style inference
        (name from fallback_name, description from first heading).

        Args:
            skill_path: Full path to the SKILL.md file.
            fallback_name: Name to use when frontmatter has no 'name' field
                           (typically the parent directory name).
        """
        try:
            with open(skill_path, "r", encoding="utf-8-sig") as f:
                raw = f.read()
        except OSError as e:
            logger.warning("Failed to read %s: %s", skill_path, e)
            return None

        from llamagent.modules.fs_store.parser import parse_frontmatter

        meta_dict, body = parse_frontmatter(raw)

        name = str(meta_dict.get("name", fallback_name)).strip()
        if not name:
            name = fallback_name

        # Description: from frontmatter, or first heading, or name
        description = str(meta_dict.get("description", "")).strip()
        if not description:
            description = self._extract_first_heading(body) or name

        # Tags
        raw_tags = meta_dict.get("tags", [])
        tags = self._normalize_list(raw_tags)

        # Aliases
        raw_aliases = meta_dict.get("aliases", [])
        aliases = self._normalize_list(raw_aliases)

        # Invocation
        invocation = str(meta_dict.get("invocation", "both")).strip()
        if invocation not in ("user-invocable", "auto-trigger", "both"):
            invocation = "both"

        # Tool packs
        raw_packs = meta_dict.get("required_tool_packs", [])
        tool_packs = self._normalize_list(raw_packs)

        # Always
        always = bool(meta_dict.get("always", False))

        # source_format: "frontmatter" if we got real frontmatter, "plain_md" otherwise
        source_format = "frontmatter" if meta_dict else "plain_md"

        return SkillMeta(
            name=name,
            description=description,
            tags=tags,
            aliases=aliases,
            invocation=invocation,
            skill_dir=os.path.dirname(skill_path),
            required_tool_packs=tool_packs,
            content_path=skill_path,
            source_format=source_format,
            always=always,
        )

    def _load_from_md_file(self, md_path: str) -> SkillMeta | None:
        """Load skill metadata from a standalone .md file (Format C).

        Name is derived from filename (without .md). If frontmatter is present,
        it is used for metadata; otherwise metadata is inferred.

        Args:
            md_path: Full path to the .md file.
        """
        try:
            with open(md_path, "r", encoding="utf-8-sig") as f:
                raw = f.read()
        except OSError as e:
            logger.warning("Failed to read %s: %s", md_path, e)
            return None

        from llamagent.modules.fs_store.parser import parse_frontmatter

        meta_dict, body = parse_frontmatter(raw)

        # Name: from frontmatter, or filename without extension
        basename = os.path.basename(md_path)
        file_name = os.path.splitext(basename)[0]
        name = str(meta_dict.get("name", file_name)).strip()
        if not name:
            name = file_name

        # Description: from frontmatter, or first heading, or name
        description = str(meta_dict.get("description", "")).strip()
        if not description:
            description = self._extract_first_heading(body) or name

        # Tags
        raw_tags = meta_dict.get("tags", [])
        tags = self._normalize_list(raw_tags)

        # Aliases
        raw_aliases = meta_dict.get("aliases", [])
        aliases = self._normalize_list(raw_aliases)

        # Invocation
        invocation = str(meta_dict.get("invocation", "both")).strip()
        if invocation not in ("user-invocable", "auto-trigger", "both"):
            invocation = "both"

        # Tool packs
        raw_packs = meta_dict.get("required_tool_packs", [])
        tool_packs = self._normalize_list(raw_packs)

        # Always
        always = bool(meta_dict.get("always", False))

        # source_format
        source_format = "frontmatter" if meta_dict else "plain_md"

        return SkillMeta(
            name=name,
            description=description,
            tags=tags,
            aliases=aliases,
            invocation=invocation,
            skill_dir=os.path.dirname(md_path),
            required_tool_packs=tool_packs,
            content_path=md_path,
            source_format=source_format,
            always=always,
        )

    # ------------------------------------------------------------------
    # Shared helpers for B/C format loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_first_heading(text: str) -> str:
        """Extract the text of the first markdown heading (any level)."""
        for line in text.splitlines():
            m = re.match(r"^#+\s+(.+)$", line)
            if m:
                return m.group(1).strip()
        return ""

    @staticmethod
    def _normalize_list(raw) -> list[str]:
        """Normalize a value into a list of non-empty strings."""
        if not isinstance(raw, list):
            return []
        result = [str(item).strip() for item in raw if item is not None]
        return [r for r in result if r]

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
        Lazy-load skill content from content_path.

        For Format A (config), content_path points to SKILL.md with no frontmatter.
        For Format B/C (frontmatter/plain_md), frontmatter is stripped before returning.

        Results are cached to avoid re-reading on subsequent activations.

        Args:
            meta: SkillMeta whose content to load.

        Returns:
            Skill content string, or empty string on error.
        """
        if meta.name in self._content_cache:
            return self._content_cache[meta.name]

        # Determine the file to read
        content_path = meta.content_path
        if not content_path:
            # Legacy fallback: SKILL.md in skill_dir
            content_path = os.path.join(meta.skill_dir, "SKILL.md")

        if not os.path.isfile(content_path):
            logger.warning("Skill content not found: %s", content_path)
            return ""

        try:
            with open(content_path, "r", encoding="utf-8-sig") as f:
                raw = f.read()
        except Exception as e:
            logger.warning("Failed to read content for skill '%s': %s", meta.name, e)
            return ""

        # Format A: SKILL.md has no frontmatter, return as-is
        if meta.source_format == "config":
            content = raw
        else:
            # Format B/C: strip frontmatter, return body only
            from llamagent.modules.fs_store.parser import parse_frontmatter
            _, content = parse_frontmatter(raw)

        self._content_cache[meta.name] = content
        return content

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def invalidate_cache(self, name: str) -> None:
        """Remove cached content for a skill so next load_content reads fresh file."""
        self._content_cache.pop(name, None)

    def all_skills(self) -> list[SkillMeta]:
        """Return all indexed skills."""
        return list(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)
