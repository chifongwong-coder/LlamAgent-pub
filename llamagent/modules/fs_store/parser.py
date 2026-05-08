"""Markdown parsing utilities for FS-based storage.

v3.8.1 R7-#12: frontmatter parsing delegated to ``python-frontmatter``
(industry-standard wrapper around ``yaml.safe_load``; same approach
Hermes / Nanobot / OpenCode all use). The legacy 50-line custom
line-parser is preserved as ``_legacy_parse_frontmatter`` for fallback
when strict YAML parsing fails on user files written under v3.7.x's
permissive subset (description with unquoted ':', tab indentation,
trailing comma in lists, raw '{...}' values, etc.).

Fallback is NOT time-bounded — kept indefinitely so legacy skill files
keep loading. The framework emits a structured ``[skill-migrate]``
warning per file when fallback fires; the ``migrate-skills`` builtin
skill picks up these warnings and offers to repair the files in place.
"""

import re
import logging

import frontmatter as _frontmatter

logger = logging.getLogger(__name__)


def _parse_value(raw: str):
    """Parse a single YAML-like value string into a Python type.

    Supports: str, int, float, bool, and bracket-delimited lists.
    """
    value = raw.strip()

    # Empty value
    if not value:
        return ""

    # Boolean
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False

    # None / null
    if value.lower() in ("null", "~"):
        return None

    # List: [a, b, c]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        items = [item.strip() for item in inner.split(",")]
        return [_parse_value(item) for item in items]

    # Quoted string — strip quotes and return as-is
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # Plain string
    return value


def parse_frontmatter(content: str, *, filepath: str | None = None) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content (v3.8.1 R7-#12).

    Strict YAML parse via ``python-frontmatter`` first; falls back to the
    legacy lenient line-parser when strict fails. Fallback emits a
    ``[skill-migrate]`` warning so the builtin migrate-skills skill can
    offer to repair the file in place when the user asks.

    Args:
        content: full markdown text (may include ``---`` frontmatter markers).
        filepath: optional path for the warning context (skill migrator
            scrapes log lines for paths to repair).

    Returns:
        (metadata_dict, body_text). If no frontmatter, returns ({}, content).
    """
    if not content:
        return {}, ""

    stripped = content.lstrip("\n")
    if not stripped.startswith("---"):
        return {}, content

    try:
        post = _frontmatter.loads(content)
        return dict(post.metadata), post.content
    except Exception as e:
        # Round-4 reviewer: catch broad — yaml.YAMLError, ScannerError,
        # ConstructorError, plus malformed delimiter quirks all surface
        # here. Don't break the agent; log and fall back so legacy
        # skill files keep loading.
        logger.warning(
            "[skill-migrate] frontmatter at %s could not be strict-parsed "
            "(%s: %s). Loaded via legacy lenient parser. Run the "
            "'migrate-skills' builtin skill to fix in place.",
            filepath or "<unknown>", type(e).__name__, str(e)[:80],
        )
        return _legacy_parse_frontmatter(content)


def _legacy_parse_frontmatter(content: str) -> tuple[dict, str]:
    """Legacy lenient line-parser — fallback for v3.7.x permissive files.

    Pre-v3.8.1 this WAS ``parse_frontmatter``. Now retained as the
    fallback for files that strict YAML rejects (description with
    unquoted ``:``, tab indentation, ``[a, b,]`` trailing comma, raw
    ``{role}`` values, etc.). NOT time-bounded — kept indefinitely so
    legacy skill files keep working; users can repair in-place via the
    ``migrate-skills`` builtin skill when convenient.
    """
    stripped = content.lstrip("\n")
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content

    end_line_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_line_idx = i
            break

    if end_line_idx is None:
        return {}, content

    frontmatter_block = "\n".join(lines[1:end_line_idx]).strip()
    body = "\n".join(lines[end_line_idx + 1:]).lstrip("\n")

    metadata = {}
    for line in frontmatter_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        colon_idx = line.find(":")
        if colon_idx == -1:
            continue

        key = line[:colon_idx].strip()
        raw_value = line[colon_idx + 1:]
        metadata[key] = _parse_value(raw_value)

    return metadata, body


def parse_sections(content: str) -> list[dict]:
    """Split markdown content by ``## heading`` into sections.

    Returns a list of ``{"title": str, "content": str}`` dicts.  Content
    between headings (including sub-sections like ### or ####) belongs to the
    nearest ``##`` heading above.  Content before the first ``##`` heading is
    ignored (it is usually the file title or frontmatter).
    """
    if not content:
        return []

    sections: list[dict] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in content.splitlines():
        # Match exactly ## heading (not ### or deeper)
        match = re.match(r"^##(?!#)\s+(.+)$", line)
        if match:
            # Save previous section if any
            if current_title is not None:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines).strip(),
                })
            current_title = match.group(1).strip()
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)

    # Save last section
    if current_title is not None:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines).strip(),
        })

    return sections


def _render_value(value) -> str:
    """Render a Python value into a YAML-like string representation."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        items = ", ".join(_render_value(item) for item in value)
        return f"[{items}]"
    # String — return as-is (no quoting for simple values)
    return str(value)


def render_frontmatter(metadata: dict, body: str = "") -> str:
    """Render a metadata dict and body into markdown with YAML frontmatter.

    Output format::

        ---
        key: value
        ...
        ---

        body
    """
    lines = ["---"]
    for key, value in metadata.items():
        lines.append(f"{key}: {_render_value(value)}")
    lines.append("---")

    result = "\n".join(lines)
    if body:
        result += "\n\n" + body
    return result
