"""Markdown parsing utilities for FS-based storage.

Provides frontmatter parsing, section splitting, and frontmatter rendering.
Uses simple string parsing — no PyYAML dependency.
"""

import re
import logging

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


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content.

    Frontmatter is expected between ``---`` markers at the start of the file.

    Returns:
        (metadata_dict, body_text). If no frontmatter is found, returns
        ({}, content).
    """
    if not content:
        return {}, ""

    stripped = content.lstrip("\n")

    # Must start with --- on its own line
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content

    # Find the closing --- on its own line (not as substring within a value)
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
