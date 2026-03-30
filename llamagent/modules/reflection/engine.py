"""
Reflection engine and lesson store.

Core components:
- ReflectionEngine: evaluate response quality, analyze root causes of failures
- LessonStore: lesson persistence via shared RetrievalPipeline (graceful degradation when not available)

Responsibility boundaries:
- The reflection module is only responsible for evaluation and lesson management, not retries
- Quality-based retries are handled by step 6 of the PlanReAct execution strategy
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime

from llamagent.core.llm import LLMClient

logger = logging.getLogger(__name__)


# ============================================================
# Reflection Engine
# ============================================================

class ReflectionEngine:
    """
    Reflection engine: evaluate response quality and analyze root causes of failures.

    Core methods:
    - evaluate_result(query, response) -> dict  Quality scoring
    - reflect(task, response, evaluation) -> dict  Deep root cause analysis
    - reset()  Reset reflection state
    """

    def __init__(self, llm: LLMClient, config=None):
        """
        Args:
            llm:    LLMClient instance
            config: Config object, used to read parameters like reflection_score_threshold
        """
        self.llm = llm

        if config is not None:
            self.score_threshold: float = getattr(
                config, "reflection_score_threshold", 7.0
            )
        else:
            self.score_threshold = 7.0

        # Reflection state (reset per task)
        self._reflection_count: int = 0

    def reset(self):
        """Reset reflection state (called at the start of each new task)."""
        self._reflection_count = 0

    def evaluate_result(self, query: str, response: str) -> dict:
        """
        Evaluate response quality.

        Calls llm.ask_json() to have the model score the response.

        Args:
            query:    User question
            response: AI answer

        Returns:
            {
                "score": int,           # 0-10 score
                "passed": bool,         # score >= threshold
                "strengths": list[str],
                "weaknesses": list[str],
                "failure_type": str | None  # "incomplete" / "inaccurate" / "off_topic" / None
            }
        """
        prompt = f"""Evaluate the quality of the following AI response:

User question: {query}
AI response: {response}

Please return JSON:
{{
    "score": an integer score from 0 to 10,
    "passed": true or false,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "failure_type": "incomplete / inaccurate / off_topic / null"
}}

Scoring criteria:
- 9-10: Accurate and comprehensive, perfect answer
- 7-8: Generally meets requirements, may have minor flaws
- 5-6: Obvious shortcomings, some requirements unmet
- 3-4: Poor, main requirements unmet
- 1-2: Off-topic or factually incorrect

passed threshold: score >= {self.score_threshold}"""

        try:
            evaluation = self.llm.ask_json(prompt, temperature=0.2)

            # Ensure key fields are complete
            score = evaluation.get("score", 0)
            if not isinstance(score, (int, float)):
                score = 0
            evaluation["score"] = score
            evaluation["passed"] = score >= self.score_threshold
            evaluation.setdefault("strengths", [])
            evaluation.setdefault("weaknesses", [])
            evaluation.setdefault("failure_type", None)

            return evaluation

        except Exception as e:
            logger.warning("Quality evaluation failed: %s", e)
            # Return conservative result on failure (assume not passed)
            return {
                "score": 0,
                "passed": False,
                "strengths": [],
                "weaknesses": [f"Error during evaluation: {e}"],
                "failure_type": None,
            }

    def reflect(self, task: str, response: str, evaluation: dict) -> dict:
        """
        Deep analysis of failure causes.

        Args:
            task:       Original task description
            response:   Execution result
            evaluation: Return value from evaluate_result()

        Returns:
            {
                "failure_type": str,
                "root_cause": str,
                "improvement_strategy": str | None  # May be None for complete failures
            }
        """
        self._reflection_count += 1

        weaknesses_text = json.dumps(
            evaluation.get("weaknesses", []), ensure_ascii=False
        )

        prompt = f"""Analyze the shortcomings of the following AI response:

User question: {task}
AI response: {response}
Evaluation results:
- Score: {evaluation.get('score', 'N/A')}/10
- Weaknesses: {weaknesses_text}
- Failure type: {evaluation.get('failure_type', 'N/A')}

Please analyze:
1. Failure type (incomplete / inaccurate / off_topic)
2. Root cause (why this problem occurred)
3. Improvement strategy (what to do next time for similar problems; leave empty if no improvement possible)

Answer concisely, return JSON:
{{
    "failure_type": "failure type",
    "root_cause": "root cause analysis",
    "improvement_strategy": "improvement strategy (null if no improvement possible)"
}}"""

        try:
            reflection = self.llm.ask_json(prompt, temperature=0.3)

            # Ensure fields are complete
            reflection.setdefault("failure_type", "unknown")
            reflection.setdefault("root_cause", "Unknown cause")
            # improvement_strategy is allowed to be None, no default value
            if "improvement_strategy" not in reflection:
                reflection["improvement_strategy"] = None

            return reflection

        except Exception as e:
            logger.warning("Root cause analysis failed: %s", e)
            return {
                "failure_type": "reflection_error",
                "root_cause": f"Error during reflection: {e}",
                "improvement_strategy": None,
            }


# ============================================================
# Lesson Store
# ============================================================

class LessonStore:
    """
    Lesson persistence store via shared retrieval pipeline.

    Not isolated by persona (all roles share the lesson store), collection name is fixed as "lessons".
    Silently degrades when the retrieval layer is not available.
    """

    def __init__(self, pipeline=None):
        """
        Args:
            pipeline: RetrievalPipeline instance (from factory). None = unavailable.
        """
        self._pipeline = pipeline
        self._available = pipeline is not None

    def save_lesson(
        self,
        task: str,
        error_description: str,
        root_cause: str,
        improvement: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """
        Save a lesson.

        Args:
            task:              Task description
            error_description: What went wrong
            root_cause:        Root cause
            improvement:       Improvement strategy (empty means no effective improvement available)
            tags:              Tag list
        """
        if not self._available:
            logger.debug("Lesson store unavailable, skipping save")
            return

        # Build lesson document (as text for vector retrieval)
        lesson_text = f"Task: {task}\nError: {error_description}\nCause: {root_cause}"
        if improvement:
            lesson_text += f"\nImprovement: {improvement}"

        # Generate unique ID
        lesson_id = hashlib.md5(
            f"{task}:{error_description}:{datetime.now().isoformat()}".encode()
        ).hexdigest()

        # Metadata (open dict — all values must be str/int/float/bool for ChromaDB compat)
        metadata = {
            "task": task[:500],
            "error_description": error_description[:500],
            "root_cause": root_cause[:500],
            "improvement": (improvement or "")[:500],
            "tags": json.dumps(tags or [], ensure_ascii=False),
            "created_at": datetime.now().isoformat(),
        }

        try:
            self._pipeline.save(lesson_id, lesson_text, metadata)
            logger.info("Lesson saved: %s", error_description[:60])
        except Exception as e:
            logger.warning("Failed to save lesson: %s", e)

    def search_lessons(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Semantic retrieval of historical lessons related to the current task.

        Args:
            query:  Query text (usually the current task description)
            top_k:  Maximum number of results to return

        Returns:
            List of lessons, each containing task / error_description / root_cause /
            improvement / tags / relevance_score / created_at
        """
        if not self._available:
            return []

        try:
            results = self._pipeline.search(query, top_k, mode="vector")
        except Exception as e:
            logger.warning("Lesson retrieval failed: %s", e)
            return []

        lessons = []
        for r in results:
            metadata = r.get("metadata", {})
            try:
                tags = json.loads(metadata.get("tags", "[]"))
            except (json.JSONDecodeError, TypeError):
                tags = []

            lessons.append({
                "task": metadata.get("task", ""),
                "error_description": metadata.get("error_description", ""),
                "root_cause": metadata.get("root_cause", ""),
                "improvement": metadata.get("improvement", ""),
                "tags": tags,
                "relevance_score": r.get("score", 0),
                "created_at": metadata.get("created_at", ""),
            })

        return lessons

    def search_lessons_formatted(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve lessons and return formatted text, suitable for direct context injection.

        Format:
        - With improvement: "Lesson: {improvement}"
        - Without improvement: "Failure cause: {root_cause}, no effective improvement available"

        Args:
            query:  Query text
            top_k:  Maximum number of results to return

        Returns:
            Formatted lesson text, empty string if no results
        """
        lessons = self.search_lessons(query, top_k)

        if not lessons:
            return ""

        lines = []
        for lesson in lessons:
            tags = lesson.get("tags", [])
            tag_text = tags[0] if tags else "Uncategorized"

            improvement = lesson.get("improvement", "")
            if improvement:
                lines.append(
                    f"- Issue type: {tag_text} | Lesson: {improvement}"
                )
            else:
                root_cause = lesson.get("root_cause", "Unknown cause")
                lines.append(
                    f"- Issue type: {tag_text} | "
                    f"Failure cause: {root_cause}, no effective improvement available"
                )

        return "\n".join(lines)
