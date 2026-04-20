"""
FSLessonStore: file-system based lesson storage using a single lessons.md file.

Provides the same interface as LessonStore so ReflectionModule can use either backend.
Each lesson is stored as a markdown section with structured metadata fields,
separated by horizontal rules.

File format:
    ## {tag}: {task_summary}
    - **lesson_id**: ...
    - **task**: ...
    - **error_description**: ...
    - **root_cause**: ...
    - **improvement**: ...
    - **tags**: ["tag1", "tag2"]
    - **created_at**: ...

    ---
"""

import hashlib
import json
import logging
import re
from datetime import datetime

from llamagent.modules.fs_store.store import FSStore

logger = logging.getLogger(__name__)

_LESSONS_FILE = "lessons.md"

# Regex to extract metadata fields: - **key**: value
_META_RE = re.compile(r"^-\s+\*\*(\w+)\*\*:\s*(.*)$")


def _parse_sections(content: str) -> list[dict]:
    """Parse lessons.md into a list of section dicts.

    Each section dict has:
        - heading: the raw ## heading text
        - meta: dict of key -> value from ``- **key**: value`` lines
    """
    if not content or not content.strip():
        return []

    # Split by ## headings (each lesson starts with ## tag: task_summary)
    # This is more robust than splitting on --- which could appear in content.
    sections = []
    current_heading = ""
    current_lines: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()

        # Top-level heading (# ...) -- skip
        if stripped.startswith("# ") and not stripped.startswith("## "):
            continue

        # New section starts at ## heading
        if stripped.startswith("## "):
            # Save previous section
            if current_heading:
                sec = _parse_single_section(current_heading, current_lines)
                if sec:
                    sections.append(sec)
            current_heading = stripped[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_heading:
        sec = _parse_single_section(current_heading, current_lines)
        if sec:
            sections.append(sec)

    return sections


def _parse_single_section(heading: str, lines: list[str]) -> dict | None:
    """Parse a single lesson section from its heading and content lines."""
    meta: dict = {}

    for line in lines:
        stripped = line.strip()

        # Skip --- separators (visual decoration only)
        if stripped == "---":
            continue

        # Metadata line: - **key**: value
        m = _META_RE.match(stripped)
        if m:
            meta[m.group(1)] = m.group(2).strip()
            continue

    if not meta:
        return None

    return {
        "heading": heading,
        "meta": meta,
    }


def _render_section(meta: dict) -> str:
    """Render a single lesson section as markdown text.

    Args:
        meta: Dictionary with all metadata fields.

    Returns:
        Formatted markdown section string.
    """
    # Build heading: ## {first_tag}: {task_summary}
    tags_raw = meta.get("tags", "[]")
    try:
        tags = json.loads(tags_raw)
    except (json.JSONDecodeError, TypeError):
        tags = []
    tag_label = tags[0] if tags else "uncategorized"

    task = meta.get("task", "unknown")
    # Truncate task for heading readability
    task_summary = task[:60] + "..." if len(task) > 60 else task
    heading = f"## {tag_label}: {task_summary}"

    lines = [heading]

    # Metadata fields in display order
    field_order = [
        "lesson_id", "task", "error_description", "root_cause",
        "improvement", "tags", "related_skill", "created_at",
    ]
    for key in field_order:
        if key in meta:
            lines.append(f"- **{key}**: {meta[key]}")

    return "\n".join(lines)


def _render_file(sections: list[dict]) -> str:
    """Render all sections into a complete lessons.md file."""
    if not sections:
        return ""
    rendered = []
    for sec in sections:
        rendered.append(_render_section(sec["meta"]))
    return "\n\n---\n\n".join(rendered) + "\n\n---\n"


class FSLessonStore:
    """File-system based lesson store using a single lessons.md file.

    Provides the same interface as LessonStore so ReflectionModule can use
    either backend transparently.

    Args:
        base_dir: Directory for the lessons.md file.
    """

    def __init__(self, base_dir: str):
        self._fs = FSStore(base_dir)

    # ============================================================
    # Write operations
    # ============================================================

    def save_lesson(
        self,
        task: str,
        error_description: str,
        root_cause: str,
        improvement: str | None = None,
        tags: list[str] | None = None,
        related_skill: str | None = None,
    ) -> None:
        """Save a lesson to lessons.md.

        Args:
            task:              Task description
            error_description: What went wrong
            root_cause:        Root cause
            improvement:       Improvement strategy (None means no effective improvement)
            tags:              Tag list
            related_skill:     Name of the skill that was active when the lesson was learned
        """
        # Generate unique ID
        lesson_id = hashlib.md5(
            f"{task}:{error_description}:{datetime.now().isoformat()}".encode()
        ).hexdigest()

        # Sanitize newlines to prevent metadata line corruption on round-trip
        def _sanitize(text: str, max_len: int = 500) -> str:
            return text.replace("\n", " ").strip()[:max_len]

        meta = {
            "lesson_id": lesson_id,
            "task": _sanitize(task),
            "error_description": _sanitize(error_description),
            "root_cause": _sanitize(root_cause),
            "improvement": _sanitize(improvement or ""),
            "tags": json.dumps(tags or [], ensure_ascii=False),
            "created_at": datetime.now().isoformat(),
        }
        if related_skill:
            meta["related_skill"] = related_skill

        section_text = _render_section(meta)

        # Append to existing file
        existing = self._fs.read_file(_LESSONS_FILE)
        if existing and existing.strip():
            new_content = existing.rstrip() + "\n\n---\n\n" + section_text + "\n\n---\n"
        else:
            new_content = section_text + "\n\n---\n"

        self._fs.write_file(_LESSONS_FILE, new_content)
        logger.info("Lesson saved: %s", error_description[:60])

    # ============================================================
    # Read operations
    # ============================================================

    def search_lessons(self, query: str, top_k: int = 3) -> list[dict]:
        """Search lessons by keyword matching.

        Returns results in the same format as LessonStore.search_lessons.
        """
        sections = self._read_sections()
        results = []
        query_lower = query.lower()
        query_terms = query_lower.split()

        for sec in sections:
            meta = sec["meta"]

            # Build searchable text from all metadata
            searchable = " ".join([
                meta.get("task", ""),
                meta.get("error_description", ""),
                meta.get("root_cause", ""),
                meta.get("improvement", ""),
                meta.get("tags", ""),
            ]).lower()

            # Simple keyword matching: count how many query terms appear
            hits = sum(1 for term in query_terms if term in searchable)
            if hits == 0:
                continue

            score = hits / max(len(query_terms), 1)

            # Parse tags
            try:
                tags = json.loads(meta.get("tags", "[]"))
            except (json.JSONDecodeError, TypeError):
                tags = []

            results.append({
                "lesson_id": meta.get("lesson_id", ""),
                "task": meta.get("task", ""),
                "error_description": meta.get("error_description", ""),
                "root_cause": meta.get("root_cause", ""),
                "improvement": meta.get("improvement", ""),
                "tags": tags,
                "relevance_score": round(score, 4),
                "created_at": meta.get("created_at", ""),
            })

        # Sort by score descending
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

    def search_lessons_formatted(self, query: str, top_k: int = 3) -> str:
        """Retrieve lessons and return formatted text for context injection.

        Format matches LessonStore.search_lessons_formatted.
        """
        lessons = self.search_lessons(query, top_k)

        if not lessons:
            return ""

        lines = []
        for lesson in lessons:
            lesson_id = lesson.get("lesson_id", "?")
            tags = lesson.get("tags", [])
            tag_text = tags[0] if tags else "Uncategorized"

            improvement = lesson.get("improvement", "")
            if improvement:
                lines.append(
                    f"- [{lesson_id}] {tag_text}: {improvement}"
                )
            else:
                root_cause = lesson.get("root_cause", "Unknown cause")
                lines.append(
                    f"- [{lesson_id}] {tag_text}: {root_cause} (no improvement available)"
                )

        return "\n".join(lines)

    # ============================================================
    # Tool support methods
    # ============================================================

    def list_all_metadata(self) -> str:
        """Return all lessons as formatted metadata text for context injection.

        Format: one line per lesson with lesson_id, tag, task summary, date.
        """
        sections = self._read_sections()
        lines = []
        for sec in sections:
            meta = sec["meta"]
            lesson_id = meta.get("lesson_id", "?")

            try:
                tags = json.loads(meta.get("tags", "[]"))
            except (json.JSONDecodeError, TypeError):
                tags = []
            tag_label = tags[0] if tags else "uncategorized"

            task = meta.get("task", "?")
            task_summary = task[:60] + "..." if len(task) > 60 else task

            created_at = meta.get("created_at", "")
            date_str = created_at[:10] if created_at else "?"

            lines.append(
                f'- [{lesson_id}] {tag_label}: "{task_summary}" ({date_str})'
            )

        if not lines:
            return ""
        return "\n".join(lines)

    def read_lesson(self, lesson_id: str) -> str:
        """Return formatted details for a specific lesson."""
        sections = self._read_sections()
        for sec in sections:
            meta = sec["meta"]
            if meta.get("lesson_id") == lesson_id:
                parts = []
                parts.append(f"Task: {meta.get('task', '?')}")
                parts.append(f"Error: {meta.get('error_description', '?')}")
                parts.append(f"Root cause: {meta.get('root_cause', '?')}")
                improvement = meta.get("improvement", "")
                if improvement:
                    parts.append(f"Improvement: {improvement}")
                else:
                    parts.append("Improvement: (none)")

                try:
                    tags = json.loads(meta.get("tags", "[]"))
                except (json.JSONDecodeError, TypeError):
                    tags = []
                parts.append(f"Tags: {', '.join(tags) if tags else '(none)'}")
                parts.append(f"Created: {meta.get('created_at', '?')}")
                return "\n".join(parts)
        return f"Lesson with id '{lesson_id}' not found."

    def get_lesson(self, lesson_id: str) -> dict | None:
        """Get lesson by ID as a dict.

        Returns a dict with task/error_description/root_cause/improvement/tags/created_at,
        or None if not found.
        """
        sections = self._read_sections()
        for sec in sections:
            meta = sec["meta"]
            if meta.get("lesson_id") == lesson_id:
                try:
                    tags = json.loads(meta.get("tags", "[]"))
                except (json.JSONDecodeError, TypeError):
                    tags = []
                return {
                    "lesson_id": meta.get("lesson_id", ""),
                    "task": meta.get("task", ""),
                    "error_description": meta.get("error_description", ""),
                    "root_cause": meta.get("root_cause", ""),
                    "improvement": meta.get("improvement", ""),
                    "tags": tags,
                    "created_at": meta.get("created_at", ""),
                }
        return None

    def delete_lesson(self, lesson_id: str) -> bool:
        """Delete a lesson by ID.

        Returns True if deleted, False if not found.
        """
        sections = self._read_sections()
        original_count = len(sections)
        sections = [
            sec for sec in sections
            if sec["meta"].get("lesson_id") != lesson_id
        ]
        if len(sections) == original_count:
            return False

        self._write_sections(sections)
        return True

    # ============================================================
    # Skill-related queries
    # ============================================================

    def get_lessons_by_skill(self, skill_name: str) -> list[dict]:
        """Return all lessons related to a specific skill.

        Args:
            skill_name: Name of the skill to filter by.

        Returns:
            List of lesson dicts with the same fields as search_lessons().
        """
        sections = self._read_sections()
        results = []
        for sec in sections:
            meta = sec["meta"]
            if meta.get("related_skill") != skill_name:
                continue

            try:
                tags = json.loads(meta.get("tags", "[]"))
            except (json.JSONDecodeError, TypeError):
                tags = []

            results.append({
                "lesson_id": meta.get("lesson_id", ""),
                "task": meta.get("task", ""),
                "error_description": meta.get("error_description", ""),
                "root_cause": meta.get("root_cause", ""),
                "improvement": meta.get("improvement", ""),
                "tags": tags,
                "related_skill": meta.get("related_skill", ""),
                "created_at": meta.get("created_at", ""),
            })

        return results

    def delete_lessons_by_skill(self, skill_name: str) -> int:
        """Delete all lessons related to a specific skill.

        Returns count of lessons deleted.
        """
        sections = self._read_sections()
        original_count = len(sections)
        sections = [
            sec for sec in sections
            if sec["meta"].get("related_skill") != skill_name
        ]
        deleted = original_count - len(sections)
        if deleted > 0:
            self._write_sections(sections)
        return deleted

    # ============================================================
    # Management
    # ============================================================

    def clear(self) -> None:
        """Clear all lessons by deleting the file."""
        self._fs.delete_file(_LESSONS_FILE)

    def get_stats(self) -> dict:
        """Return lesson statistics."""
        sections = self._read_sections()
        return {"available": True, "count": len(sections)}

    def get_oldest_lesson_id(self) -> str | None:
        """Return the lesson_id of the oldest stored lesson (FIFO).

        Ordering: ``created_at`` ascending. Returns None when the store
        is empty. Used by the cap-enforcement path to evict the oldest
        lesson before saving a new one once the count reaches the cap.
        """
        sections = self._read_sections()
        if not sections:
            return None

        def _order_key(sec: dict) -> str:
            return sec.get("meta", {}).get("created_at", "")

        oldest = min(sections, key=_order_key)
        return oldest.get("meta", {}).get("lesson_id") or None

    # ============================================================
    # Internal helpers
    # ============================================================

    def _read_sections(self) -> list[dict]:
        """Read and parse all sections from lessons.md."""
        content = self._fs.read_file(_LESSONS_FILE)
        if not content:
            return []
        return _parse_sections(content)

    def _write_sections(self, sections: list[dict]) -> None:
        """Rewrite the entire lessons.md from parsed sections."""
        content = _render_file(sections)
        self._fs.write_file(_LESSONS_FILE, content)
