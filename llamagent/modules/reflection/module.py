"""
ReflectionModule: self-reflection and lesson management.

Pipeline callbacks:
- on_input:   Record current query, reset reflection state
- on_context: Semantic retrieval of historical lessons injected into context
- on_output:  Evaluate quality; if below threshold, reflect + save_lesson; no retry, return as-is

Disabled by default (config.reflection_enabled = False).
When enabled, each turn makes an additional LLM call to evaluate response quality, which increases cost.
When disabled, the module does nothing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llamagent.core.agent import Module
from llamagent.modules.reflection.engine import ReflectionEngine, LessonStore

if TYPE_CHECKING:
    from llamagent.core.agent import LlamAgent

logger = logging.getLogger(__name__)


class ReflectionModule(Module):
    """
    Reflection module: evaluation + lesson management, no retries.

    Key attributes:
    - engine:        ReflectionEngine instance
    - lesson_store:  LessonStore instance (None when chromadb is not installed)
    - current_query: User query for the current turn
    """

    name: str = "reflection"
    description: str = "Self-reflection: evaluate response quality and learn from mistakes"

    def __init__(self):
        self.engine: ReflectionEngine | None = None
        self.lesson_store: LessonStore | None = None
        self.current_query: str | None = None

    def on_attach(self, agent: "LlamAgent"):
        """
        Initialize reflection engine and lesson store.

        When chromadb is not installed, lesson_store is None, but evaluation still works normally.
        """
        super().on_attach(agent)

        self.engine = ReflectionEngine(llm=self.llm, config=agent.config)

        # Lesson store: use shared retrieval pipeline via factory
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

    def on_input(self, user_input: str) -> str:
        """
        Record current query and reset reflection state.

        Returns user_input as-is.
        """
        self.current_query = user_input
        if self.engine is not None:
            self.engine.reset()
        return user_input

    def on_context(self, query: str, context: str) -> str:
        """
        Use query for semantic retrieval of historical lessons, appended to context.

        Only effective when reflection_enabled is True and lesson_store is available.
        No content is injected when there are no relevant lessons.
        """
        if not self._is_enabled():
            return context

        if self.lesson_store is None:
            return context

        try:
            lessons_text = self.lesson_store.search_lessons_formatted(query)
        except Exception as e:
            # Lesson retrieval failure should not block the main pipeline
            logger.debug("Lesson retrieval failed: %s", e)
            return context

        if not lessons_text:
            return context

        # Inject in the documented format
        history_section = (
            "[Historical Experience] The following are lessons learned from handling similar problems. Please refer to them:\n"
            f"{lessons_text}"
        )

        if context:
            return f"{context}\n\n{history_section}"
        return history_section

    def on_output(self, response: str) -> str:
        """
        Evaluate response quality; if below threshold, analyze root cause and save lesson.

        No retry is performed; response is returned as-is.
        Quality-based retries are handled by step 6 of the PlanReAct execution strategy.
        """
        if not self._is_enabled():
            return response

        if self.engine is None or not self.current_query:
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
                )
            except Exception as e:
                logger.warning("Failed to save lesson: %s", e)

        # No retry, return as-is
        return response

    # ----------------------------------------------------------
    # Internal methods
    # ----------------------------------------------------------

    def _is_enabled(self) -> bool:
        """Check whether reflection is enabled."""
        if not hasattr(self, "agent") or self.agent is None:
            return False
        return getattr(self.agent.config, "reflection_enabled", False)

    def _get_threshold(self) -> float:
        """Get the quality evaluation threshold."""
        if hasattr(self, "agent") and self.agent is not None:
            return getattr(
                self.agent.config, "reflection_score_threshold", 7.0
            )
        return 7.0
