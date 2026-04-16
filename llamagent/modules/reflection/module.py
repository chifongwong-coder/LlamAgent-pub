"""
ReflectionModule: self-reflection and lesson management with read/write mode decoupling.

Write modes (reflection_write_mode):
- off:   No automatic evaluation, no lesson saving
- auto:  on_output evaluates quality; if below threshold, reflects + saves lesson

Read modes (reflection_read_mode):
- off:   No tools registered, no automatic injection
- tool:  Registers list_lessons/read_lesson/delete_lesson tools
- auto:  tool + on_context automatically injects relevant lessons

Backend selection (reflection_backend):
- rag:   ChromaDB vector store (semantic search)
- fs:    File-system markdown store (keyword search)

Pipeline callbacks:
- on_input:   Record current query, reset reflection state
- on_context: Guide injection + auto lesson injection (when read_mode=auto)
- on_output:  Evaluate quality; if below threshold, reflect + save lesson (when write_mode=auto)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llamagent.core.agent import Module
from llamagent.modules.reflection.engine import ReflectionEngine, LessonStore

if TYPE_CHECKING:
    from llamagent.core.agent import LlamAgent

logger = logging.getLogger(__name__)


# ============================================================
# Guide constants (6 variants)
# ============================================================

# Tool mode, read+write (backend-agnostic)
REFLECTION_GUIDE = """\
[Reflection] You have access to a lesson store from past experiences.
- Use list_lessons to search lessons by keyword.
- Use read_lesson to see full details of a specific lesson.
- Use delete_lesson to remove outdated or incorrect lessons."""

# Tool mode, read-only (backend-agnostic)
REFLECTION_GUIDE_READONLY = """\
[Reflection] You have access to a lesson store from past experiences.
- Use list_lessons to search lessons by keyword.
- Use read_lesson to see full details of a specific lesson."""

# RAG auto mode, read+write
REFLECTION_GUIDE_AUTO = """\
[Reflection] You have access to a lesson store. Relevant lessons from past \
experiences are shown below. Use read_lesson to get full details.
- Use delete_lesson to remove outdated or incorrect lessons."""

# RAG auto mode, read-only
REFLECTION_GUIDE_AUTO_READONLY = """\
[Reflection] You have access to a lesson store. Relevant lessons from past \
experiences are shown below. Use read_lesson to get full details."""

# FS auto mode, read+write
REFLECTION_GUIDE_FS_AUTO = """\
[Reflection] You have access to a lesson store. The metadata of all lessons \
is shown below. Use read_lesson to access full details of any relevant lesson.
- Use delete_lesson to remove outdated or incorrect lessons."""

# FS auto mode, read-only
REFLECTION_GUIDE_FS_AUTO_READONLY = """\
[Reflection] You have access to a lesson store. The metadata of all lessons \
is shown below. Use read_lesson to access full details of any relevant lesson."""


class ReflectionModule(Module):
    """
    Reflection module: evaluation + lesson management with read/write mode decoupling.

    Key attributes:
    - engine:        ReflectionEngine instance (None when both modes are off)
    - lesson_store:  LessonStore or FSLessonStore instance (None when backend unavailable)
    - current_query: User query for the current turn
    """

    name: str = "reflection"
    description: str = "Self-reflection: evaluate response quality and learn from mistakes"

    def __init__(self):
        self.engine: ReflectionEngine | None = None
        self.lesson_store = None  # LessonStore | FSLessonStore | None
        self.current_query: str | None = None
        self._write_mode: str = "off"
        self._read_mode: str = "off"
        self._backend: str = "rag"
        self._available: bool = False
        self._pending_skill_check: str | None = None

    def on_attach(self, agent: "LlamAgent"):
        """Initialize reflection engine, lesson store, and register tools."""
        super().on_attach(agent)

        # Parse modes from config
        self._write_mode = getattr(agent.config, "reflection_write_mode", "off")
        self._read_mode = getattr(agent.config, "reflection_read_mode", "off")

        # Validate modes (fallback to off for invalid values)
        if self._write_mode not in ("off", "auto"):
            logger.warning(
                "Unknown reflection_write_mode='%s', falling back to 'off'",
                self._write_mode,
            )
            self._write_mode = "off"
        if self._read_mode not in ("off", "tool", "auto"):
            logger.warning(
                "Unknown reflection_read_mode='%s', falling back to 'off'",
                self._read_mode,
            )
            self._read_mode = "off"

        # Both off -> nothing to do
        if self._write_mode == "off" and self._read_mode == "off":
            self._available = False
            return

        # Initialize engine (stateless evaluation tool, created whenever module is active)
        self.engine = ReflectionEngine(llm=self.llm, config=agent.config)

        # Initialize lesson store backend
        self._backend = getattr(agent.config, "reflection_backend", "rag")
        if self._backend == "fs":
            self._init_fs_backend(agent)
        else:
            self._init_rag_backend(agent)

        # Register tools when read_mode is active
        if self._read_mode != "off":
            self._register_tools()

        self._available = True

    def _init_rag_backend(self, agent: "LlamAgent"):
        """Initialize the RAG (ChromaDB) backend for lesson storage."""
        try:
            from llamagent.modules.rag.factory import create_pipeline
            pipeline = create_pipeline(
                config=agent.config,
                collection_name="lessons",
                enable_lexical=False,
                enable_reranker=False,
            )
            self.lesson_store = LessonStore(pipeline=pipeline)
        except Exception as e:
            logger.info("Lesson store initialization skipped: %s", e)
            self.lesson_store = None

    def _init_fs_backend(self, agent: "LlamAgent"):
        """Initialize the FS backend for lesson storage."""
        import os
        from llamagent.modules.reflection.fs_store import FSLessonStore

        base_dir = getattr(agent.config, "reflection_fs_dir", None)
        if not base_dir:
            base_dir = os.path.join(
                getattr(agent.config, "fs_data_dir", "data/fs"), "lessons"
            )

        self.lesson_store = FSLessonStore(base_dir)

    # ============================================================
    # Tool registration
    # ============================================================

    def _register_tools(self):
        """Register lesson management tools based on mode configuration."""
        # list_lessons: always registered when read_mode != off
        self.agent.register_tool(
            name="list_lessons",
            func=self._tool_list_lessons,
            description=(
                "Search lessons from past experiences by keyword. "
                "Returns matching lessons with their metadata."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords to find relevant lessons",
                    },
                },
                "required": ["query"],
            },
            tier="default",
            safety_level=1,
        )

        # read_lesson: always registered when read_mode != off
        self.agent.register_tool(
            name="read_lesson",
            func=self._tool_read_lesson,
            description=(
                "Read the full details of a specific lesson by its lesson_id. "
                "Use this after finding a lesson via list_lessons."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "lesson_id": {
                        "type": "string",
                        "description": "The lesson_id of the lesson to read",
                    },
                },
                "required": ["lesson_id"],
            },
            tier="default",
            safety_level=1,
        )

        # delete_lesson: only registered when write_mode != off
        if self._write_mode != "off":
            self.agent.register_tool(
                name="delete_lesson",
                func=self._tool_delete_lesson,
                description=(
                    "Delete an outdated or incorrect lesson by its lesson_id."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "lesson_id": {
                            "type": "string",
                            "description": "The lesson_id of the lesson to delete",
                        },
                    },
                    "required": ["lesson_id"],
                },
                tier="default",
                safety_level=2,
            )

    # ============================================================
    # Tool implementations
    # ============================================================

    def _tool_list_lessons(self, query: str) -> str:
        """Search lessons by keyword."""
        if self.lesson_store is None:
            return "Lesson store is currently unavailable."

        try:
            formatted = self.lesson_store.search_lessons_formatted(query)
        except Exception as e:
            return f"Failed to search lessons: {e}"

        if not formatted:
            return "No lessons found matching the query."
        return formatted

    def _tool_read_lesson(self, lesson_id: str) -> str:
        """Read full details of a specific lesson."""
        if self.lesson_store is None:
            return "Lesson store is currently unavailable."

        try:
            # FS backend has read_lesson method
            if hasattr(self.lesson_store, "read_lesson"):
                return self.lesson_store.read_lesson(lesson_id)

            # RAG backend: use get_lesson then format
            lesson = self.lesson_store.get_lesson(lesson_id)
            if lesson is None:
                return f"Lesson with id '{lesson_id}' not found."

            parts = []
            parts.append(f"Task: {lesson.get('task', '?')}")
            parts.append(f"Error: {lesson.get('error_description', '?')}")
            parts.append(f"Root cause: {lesson.get('root_cause', '?')}")
            improvement = lesson.get("improvement", "")
            if improvement:
                parts.append(f"Improvement: {improvement}")
            else:
                parts.append("Improvement: (none)")
            tags = lesson.get("tags", [])
            parts.append(f"Tags: {', '.join(tags) if tags else '(none)'}")
            parts.append(f"Created: {lesson.get('created_at', '?')}")
            return "\n".join(parts)
        except Exception as e:
            return f"Failed to read lesson: {e}"

    def _tool_delete_lesson(self, lesson_id: str) -> str:
        """Delete a lesson by ID."""
        if self.lesson_store is None:
            return "Lesson store is currently unavailable."

        try:
            result = self.lesson_store.delete_lesson(lesson_id)
            if result:
                return f"Lesson '{lesson_id}' deleted successfully."
            else:
                return f"Lesson with id '{lesson_id}' not found."
        except Exception as e:
            return f"Failed to delete lesson: {e}"

    # ============================================================
    # Pipeline Callbacks
    # ============================================================

    def on_input(self, user_input: str) -> str:
        """Record current query, reset reflection state, and check pending skill improvement."""
        # Check for pending skill improvement (delayed from previous turn's on_output)
        if self._pending_skill_check:
            try:
                self._check_skill_improvement(self._pending_skill_check)
            except Exception as e:
                logger.warning("[Reflection] Skill improvement check failed: %s", e)
            finally:
                self._pending_skill_check = None

        self.current_query = user_input
        if self.engine is not None:
            self.engine.reset()
        return user_input

    def on_context(self, query: str, context: str) -> str:
        """Inject reflection guide and auto-inject lessons when read_mode=auto."""
        has_write = self._write_mode != "off"
        has_read = self._read_mode != "off"

        if not has_write and not has_read:
            return context

        # Select guide (read=off means no guide -- write-side on_output doesn't need it)
        guide = self._select_guide(has_write, has_read)

        # Auto injection
        auto_block = ""
        if self._read_mode == "auto" and self.lesson_store is not None:
            if self._backend == "fs":
                metadata = self.lesson_store.list_all_metadata()
                if metadata:
                    auto_block = f"[Reflection] Lesson summaries:\n{metadata}"
            else:
                auto_block = self._do_auto_inject(query)

        # Assemble context
        parts = [p for p in [context, guide, auto_block] if p]
        return "\n\n".join(parts)

    def on_output(self, response: str) -> str:
        """
        Evaluate response quality; if below threshold, analyze root cause and save lesson.

        Only active when write_mode=auto. No retry is performed; response is returned as-is.
        """
        if self._write_mode != "auto":
            return response

        if self.engine is None or not self.current_query:
            return response

        # Skip evaluation if lesson store unavailable (avoids wasted LLM calls)
        if self.lesson_store is None:
            return response

        # Evaluate quality
        try:
            evaluation = self.engine.evaluate_result(
                self.current_query, response
            )
        except Exception as e:
            logger.warning("Quality evaluation failed, skipping reflection: %s", e)
            return response

        # Passed -> return directly
        if evaluation.get("score", 0) >= self._get_threshold():
            return response

        # Below threshold -> reflect and analyze root cause
        try:
            reflection = self.engine.reflect(
                self.current_query, response, evaluation
            )
        except Exception as e:
            logger.warning("Root cause analysis failed: %s", e)
            return response

        # Determine related skill (if SkillModule is loaded and has active skills)
        related_skill = None
        skill_mod = self.agent.modules.get("skill")
        if skill_mod and hasattr(skill_mod, "get_active_skills"):
            active = skill_mod.get_active_skills()
            if active:
                related_skill = active[0].name

        # Save lesson
        if self.lesson_store is not None:
            try:
                self.lesson_store.save_lesson(
                    task=self.current_query,
                    error_description=", ".join(
                        evaluation.get("weaknesses", ["Quality below threshold"])
                    ),
                    root_cause=reflection.get("root_cause", "Unknown cause"),
                    improvement=reflection.get("improvement_strategy"),
                    tags=[reflection.get("failure_type", "unknown")],
                    related_skill=related_skill,
                )
                # Schedule skill improvement check for next turn
                if related_skill:
                    self._pending_skill_check = related_skill
            except Exception as e:
                logger.warning("Failed to save lesson: %s", e)

        # No retry, return as-is
        return response

    # ----------------------------------------------------------
    # Skill improvement (v2.9.8)
    # ----------------------------------------------------------

    def _check_skill_improvement(self, skill_name: str):
        """Check if a skill has accumulated enough lessons to warrant improvement."""
        if not self.agent.interaction_handler:
            return

        skill_mod = self.agent.modules.get("skill")
        if not skill_mod:
            return

        threshold = getattr(self.agent.config, "skill_improve_threshold", 3)
        if threshold <= 0:
            return  # Disabled

        # Count lessons related to this skill
        lessons = self.lesson_store.get_lessons_by_skill(skill_name)
        if len(lessons) < threshold:
            return

        # Look up skill metadata
        skill_meta = skill_mod.get_skill(skill_name)
        if not skill_meta:
            return

        self._propose_skill_improvement(skill_meta, lessons)

    def _propose_skill_improvement(self, skill_meta, lessons):
        """Generate improvement proposal and ask user for confirmation."""
        # Load current skill content
        skill_mod = self.agent.modules.get("skill")
        current_content = skill_mod.activate(skill_meta.name)
        if not current_content:
            return

        # Format lessons (flat fields: root_cause, improvement)
        lesson_text = "\n".join(
            f"- {l.get('improvement', 'N/A')} (cause: {l.get('root_cause', '?')})"
            for l in lessons
        )

        # LLM generates improved skill (uses module.llm, supports per-module model)
        prompt = (
            f"You are improving a skill based on accumulated lessons learned.\n\n"
            f"Current skill content:\n```\n{current_content}\n```\n\n"
            f"Lessons learned ({len(lessons)} issues found):\n{lesson_text}\n\n"
            f"Generate an improved version of the skill that addresses these lessons.\n"
            f"Keep the same structure and format. Only modify parts that need improvement.\n"
            f"Do not include YAML frontmatter (---). Return only the skill body content."
        )

        try:
            improved = self.llm.ask(prompt, temperature=0.3)
        except Exception as e:
            logger.warning("[Reflection] Skill improvement LLM call failed: %s", e)
            return

        if not improved or improved.startswith("[LLM"):
            return

        # Ask user for confirmation via interaction_handler
        self._confirm_and_apply(skill_meta, current_content, improved, lessons)

    def _confirm_and_apply(self, skill_meta, current_content, improved_content, lessons):
        """Ask user to confirm skill improvement via ask_user."""
        if not self.agent.interaction_handler:
            logger.info("[Reflection] No interaction handler, skipping skill improvement confirmation")
            return

        # Build confirmation message (use flat lesson fields)
        message = (
            f"Based on {len(lessons)} lesson(s), I suggest improving the '{skill_meta.name}' skill.\n\n"
            f"Key improvements:\n"
        )
        for l in lessons[:3]:  # Show top 3
            improvement = l.get("improvement", "")
            if improvement:
                message += f"- {improvement}\n"
        message += f"\nApprove this update? (yes/no)"

        try:
            response = self.agent.interaction_handler.ask(message)
            if isinstance(response, str) and response.strip().lower() in ("yes", "y", "ok", "approve"):
                skill_mod = self.agent.modules.get("skill")
                if skill_mod:
                    result = skill_mod.update_skill(skill_meta.name, improved_content)
                    if result.get("success"):
                        logger.info("[Reflection] Skill '%s' improved successfully", skill_meta.name)
                        # Clear related lessons after successful improvement
                        self.lesson_store.delete_lessons_by_skill(skill_meta.name)
                    else:
                        logger.warning("[Reflection] Skill update failed: %s", result.get("error"))
        except Exception as e:
            logger.warning("[Reflection] Skill improvement confirmation failed: %s", e)

    # ----------------------------------------------------------
    # Internal methods
    # ----------------------------------------------------------

    def _select_guide(self, has_write: bool, has_read: bool) -> str:
        """Select the appropriate guide string based on mode and backend."""
        # read=off -> no guide (write-side on_output doesn't need guide in context)
        if not has_read:
            return ""

        if self._backend == "fs":
            if self._read_mode == "auto" and has_write:
                return REFLECTION_GUIDE_FS_AUTO
            elif self._read_mode == "auto":
                return REFLECTION_GUIDE_FS_AUTO_READONLY
            elif has_write:
                return REFLECTION_GUIDE
            else:
                return REFLECTION_GUIDE_READONLY
        else:  # rag
            if self._read_mode == "auto" and has_write:
                return REFLECTION_GUIDE_AUTO
            elif self._read_mode == "auto":
                return REFLECTION_GUIDE_AUTO_READONLY
            elif has_write:
                return REFLECTION_GUIDE
            else:
                return REFLECTION_GUIDE_READONLY

    def _do_auto_inject(self, query: str) -> str:
        """Perform RAG-based semantic search and return formatted lesson block."""
        if self.lesson_store is None:
            return ""

        try:
            lessons_text = self.lesson_store.search_lessons_formatted(query)
        except Exception as e:
            logger.debug("Lesson retrieval failed: %s", e)
            return ""

        if not lessons_text:
            return ""

        return (
            "[Historical Experience] The following are lessons learned from "
            "handling similar problems. Please refer to them:\n"
            f"{lessons_text}"
        )

    def _get_threshold(self) -> float:
        """Get the quality evaluation threshold."""
        if hasattr(self, "agent") and self.agent is not None:
            return getattr(
                self.agent.config, "reflection_score_threshold", 7.0
            )
        return 7.0
