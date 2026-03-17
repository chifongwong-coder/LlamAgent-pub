"""
Configuration management: unified configuration center with environment variable overrides.

All LlamAgent runtime parameters are managed centrally in the Config class.
Model auto-detection priority: MODEL_NAME env var > API Key detection (DeepSeek > OpenAI > Anthropic) > Ollama fallback.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Project root directory (module-level, reasonably shared)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env file (module-level, only needs to run once)
load_dotenv(BASE_DIR / ".env")


def _safe_int(key: str, default: int) -> int:
    """Safely convert an environment variable to an integer; returns default and warns on invalid values."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning("Environment variable %s='%s' is not a valid integer, using default %d", key, val, default)
        return default


def _safe_float(key: str, default: float) -> float:
    """Safely convert an environment variable to a float; returns default and warns on invalid values."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning("Environment variable %s='%s' is not a valid float, using default %s", key, val, default)
        return default


class Config:
    """
    Global configuration, each instance holds its own state independently.

    Configuration parameters are grouped by function: model, agent, memory, RAG, persona, tools, planning, reflection, safety, output.
    Parameters that support environment variable overrides have the corresponding env var name noted in comments.
    """

    def __init__(self):
        # ==================== Model ====================
        # Model identifier. Detection priority: MODEL_NAME env > API Key detection > Ollama fallback
        # Environment variable: MODEL_NAME
        self.model: str = os.getenv("MODEL_NAME") or self._detect_model()

        # Model context window size, auto-detected via litellm, can also be manually overridden
        self.max_context_tokens: int = self._detect_max_context_tokens()

        # LLM API call retry count on failure (retries N times after failure, N+1 total calls)
        self.api_retry_count: int = 1

        # ==================== Agent ====================
        # Default identity prompt when no Persona is set
        self.system_prompt: str = (
            "You are LlamAgent, a super capable llama AI assistant. "
            "You are smart, reliable, and happy to help, with an occasional llama-like charm. "
            "Reply concisely and clearly. If you are unsure, be honest about it."
        )

        # Number of conversation turns to retain in context; oldest turns are discarded when exceeded
        self.context_window_size: int = 20

        # Ratio of conversation tokens to max_context_tokens; compression is triggered when exceeded
        self.context_compress_threshold: float = 0.7

        # Number of recent turns to keep uncompressed during partial compression
        self.compress_keep_turns: int = 3

        # Maximum number of ReAct loop steps
        # Environment variable: MAX_REACT_STEPS
        self.max_react_steps: int = _safe_int("MAX_REACT_STEPS", 10)

        # Abort if the same action (identical tool name + arguments) repeats more than this count
        # Environment variable: MAX_DUPLICATE_ACTIONS
        self.max_duplicate_actions: int = _safe_int("MAX_DUPLICATE_ACTIONS", 2)

        # Time limit per ReAct step in seconds; PlanReAct times each step independently
        # Environment variable: REACT_TIMEOUT
        self.react_timeout: float = _safe_float("REACT_TIMEOUT", 210.0)

        # Maximum tokens for a single tool return result; truncated if exceeded
        self.max_observation_tokens: int = 2000

        # ==================== Memory ====================
        # Memory mode: "off" (disabled) / "autonomous" (model-driven) / "hybrid" (autonomous + forced save per turn)
        self.memory_mode: str = os.getenv("MEMORY_MODE", "off")
        if self.memory_mode not in ("off", "autonomous", "hybrid"):
            logger.warning(
                "memory_mode='%s' is invalid, falling back to 'off'. Valid values: off / autonomous / hybrid",
                self.memory_mode,
            )
            self.memory_mode = "off"

        # ==================== RAG ====================
        # ChromaDB persistence directory
        # Environment variable: CHROMA_DIR
        self.chroma_dir: str = os.getenv("CHROMA_DIR", str(BASE_DIR / "data" / "chroma"))

        # Number of retrieval results to return
        # Environment variable: RAG_TOP_K
        self.rag_top_k: int = _safe_int("RAG_TOP_K", 3)

        # Document chunk size
        # Environment variable: CHUNK_SIZE
        self.chunk_size: int = _safe_int("CHUNK_SIZE", 500)

        # ==================== Persona ====================
        # Persona definition JSON file path
        # Environment variable: PERSONA_FILE
        self.persona_file: str = os.getenv("PERSONA_FILE", str(BASE_DIR / "data" / "personas.json"))

        # ==================== Tools ====================
        # Custom tools storage directory
        # Environment variable: AGENT_TOOLS_DIR
        self.agent_tools_dir: str = os.getenv(
            "AGENT_TOOLS_DIR", str(BASE_DIR / "data" / "agent_tools")
        )

        # ==================== Planning ====================
        # Maximum number of plan adjustments during PlanReAct execution
        # Environment variable: MAX_PLAN_ADJUSTMENTS
        self.max_plan_adjustments: int = _safe_int("MAX_PLAN_ADJUSTMENTS", 7)

        # ==================== Reflection ====================
        # Reflection evaluation toggle
        self.reflection_enabled: bool = False

        # Scores below this threshold trigger lesson saving (and re-planning under PlanReAct)
        # Environment variable: REFLECTION_SCORE_THRESHOLD
        self.reflection_score_threshold: float = _safe_float("REFLECTION_SCORE_THRESHOLD", 7.0)

        # ==================== Safety ====================
        # Fallback permission level when no Persona is set (when Persona exists, persona.permission_level takes precedence)
        # Environment variable: PERMISSION_LEVEL
        self.permission_level: int = _safe_int("PERMISSION_LEVEL", 1)

        # ==================== Output ====================
        # File output directory
        # Environment variable: OUTPUT_DIR
        self.output_dir: str = os.getenv("OUTPUT_DIR", str(BASE_DIR / "output"))

    # ------------------------------------------------------------------
    # Model detection
    # ------------------------------------------------------------------

    def _detect_model(self) -> str:
        """
        Auto-detect available models.

        Priority: DEEPSEEK_API_KEY -> OPENAI_API_KEY -> ANTHROPIC_API_KEY -> Ollama fallback.
        """
        if os.getenv("DEEPSEEK_API_KEY"):
            logger.info("Detected DEEPSEEK_API_KEY, using DeepSeek model")
            return "deepseek/deepseek-chat"
        if os.getenv("OPENAI_API_KEY"):
            logger.info("Detected OPENAI_API_KEY, using OpenAI model")
            return "openai/gpt-4o-mini"
        if os.getenv("ANTHROPIC_API_KEY"):
            logger.info("Detected ANTHROPIC_API_KEY, using Anthropic model")
            return "anthropic/claude-sonnet-4-20250514"
        logger.info("No API Key detected, falling back to Ollama local model")
        return "ollama_chat/qwen2.5:7b"

    def _detect_max_context_tokens(self) -> int:
        """
        Auto-detect model context window size.

        Prefers litellm.get_max_tokens(); falls back to a conservative default of 8192 on failure.
        """
        try:
            import litellm
            max_tokens = litellm.get_max_tokens(self.model)
            if max_tokens and max_tokens > 0:
                return max_tokens
        except Exception:
            logger.debug("Unable to get context window size for model %s via litellm, using default", self.model)
        return 8192
