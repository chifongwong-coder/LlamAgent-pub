"""
Configuration management: unified configuration center.

Priority chain: Environment variables > YAML file > Code defaults.

All LlamAgent runtime parameters are managed centrally in the Config class.
Modules access config via flat attributes (config.xxx) regardless of the source.
Default model: local Ollama. Set MODEL_NAME env var to use cloud providers (e.g., MODEL_NAME=openai/gpt-4o-mini).
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


# ======================================================================
# YAML path -> flat attribute mapping
# ======================================================================

# Each entry: (yaml_key_path_tuple, flat_attribute_name, type)
# type is used for basic casting when reading from YAML
_YAML_MAP = [
    (("model", "name"), "model", str),
    (("model", "max_context_tokens"), "max_context_tokens", int),
    (("model", "api_retry_count"), "api_retry_count", int),
    (("llm", "fallback_model"), "fallback_model", str),
    (("llm", "resilience_max_retries"), "resilience_max_retries", int),
    (("llm", "routing_simple_model"), "routing_simple_model", str),
    (("agent", "system_prompt"), "system_prompt", str),
    (("agent", "context_window_size"), "context_window_size", int),
    (("agent", "context_compress_threshold"), "context_compress_threshold", float),
    (("agent", "compress_keep_turns"), "compress_keep_turns", int),
    (("agent", "react", "max_steps"), "max_react_steps", int),
    (("agent", "react", "max_duplicate_actions"), "max_duplicate_actions", int),
    (("agent", "react", "timeout"), "react_timeout", float),
    (("agent", "react", "total_timeout"), "react_total_timeout", float),
    (("agent", "max_observation_tokens"), "max_observation_tokens", int),
    (("agent", "tool_result_persist_threshold"), "tool_result_persist_threshold", int),
    (("changeset", "max_count"), "changeset_max_count", int),
    (("changeset", "max_total_bytes"), "changeset_max_total_bytes", int),
    (("snapshot", "enabled"), "snapshot_enabled", bool),
    (("snapshot", "ignore_gitignore"), "snapshot_ignore_gitignore", bool),
    (("snapshot", "max_size_mb"), "snapshot_max_size_mb", int),
    (("snapshot", "retention_count"), "snapshot_retention_count", int),
    (("snapshot", "dir"), "snapshot_dir", str),
    (("retrieval", "persist_dir"), "retrieval_persist_dir", str),
    (("retrieval", "embedding", "provider"), "embedding_provider", str),
    (("retrieval", "embedding", "model"), "embedding_model", str),
    (("memory", "write_mode"), "memory_mode", str),
    (("memory", "recall_mode"), "memory_recall_mode", str),
    (("memory", "fact_fallback"), "memory_fact_fallback", str),
    (("memory", "auto_recall", "top_k"), "memory_recall_top_k", int),
    (("memory", "auto_recall", "max_inject"), "memory_auto_recall_max_inject", int),
    (("memory", "auto_recall", "threshold"), "memory_auto_recall_threshold", float),
    (("memory", "consolidation_interval"), "memory_consolidation_interval", int),
    (("memory", "consolidation_min_count"), "memory_consolidation_min_count", int),
    (("memory", "compile_mode"), "memory_compile_mode", str),
    (("memory", "dedup_threshold"), "memory_dedup_threshold", int),
    (("memory", "dedup_mode"), "memory_dedup_mode", str),
    (("rag", "top_k"), "rag_top_k", int),
    (("rag", "chunk_size"), "chunk_size", int),
    (("rag", "retrieval_mode"), "rag_retrieval_mode", str),
    (("rag", "rerank_enabled"), "rag_rerank_enabled", bool),
    (("persona", "file"), "persona_file", str),
    (("tools", "agent_tools_dir"), "agent_tools_dir", str),
    (("planning", "max_plan_adjustments"), "max_plan_adjustments", int),
    (("reflection", "write_mode"), "reflection_write_mode", str),
    (("reflection", "read_mode"), "reflection_read_mode", str),
    (("reflection", "score_threshold"), "reflection_score_threshold", float),
    (("reflection", "max_lessons"), "reflection_max_lessons", int),
    (("reflection", "skill_improve_threshold"), "skill_improve_threshold", int),
    (("reflection_backend",), "reflection_backend", str),
    (("reflection_fs_dir",), "reflection_fs_dir", str),
    (("safety", "permission_level"), "permission_level", int),
    (("skill", "dirs"), "skill_dirs", list),
    (("skill", "max_active"), "skill_max_active", int),
    (("job", "default_timeout"), "job_default_timeout", float),
    (("job", "max_active"), "job_max_active", int),
    (("job", "profiles"), "job_profiles", dict),
    (("output", "dir"), "output_dir", str),
    (("edit_root",), "edit_root", str),
    (("web", "search_provider"), "web_search_provider", str),
    (("web", "search_num_results"), "web_search_num_results", int),
    (("web", "search_api_key"), "web_search_api_key", str),
    (("authorization", "mode"), "authorization_mode", str),
    (("authorization", "approval_mode"), "approval_mode", str),
    (("authorization", "auto_approve"), "auto_approve", bool),
    (("authorization", "scopes"), "authorization_scopes", list),
    (("module_models",), "module_models", dict),
    (("retrieval_backend",), "retrieval_backend", str),
    (("memory_backend",), "memory_backend", str),
    (("fs_data_dir",), "fs_data_dir", str),
    (("memory_fs_dir",), "memory_fs_dir", str),
    (("knowledge_dir",), "knowledge_dir", str),
    (("persistence", "enabled"), "persistence_enabled", bool),
    (("persistence", "auto_restore"), "persistence_auto_restore", bool),
    (("persistence_dir",), "persistence_dir", str),
    (("compression", "tool_result_strategy"), "tool_result_strategy", str),
    (("compression", "tool_result_max_chars"), "tool_result_max_chars", int),
    (("compression", "tool_result_head_lines"), "tool_result_head_lines", int),
    (("compression", "strip_thinking"), "strip_thinking", bool),
    (("child_agent", "runner"), "child_agent_runner", str),
    (("child_agent", "max_children"), "child_agent_max_children", int),
    (("child_agent", "role_models"), "child_agent_role_models", dict),
    (("child_agent", "auto_memorize"), "child_agent_auto_memorize", bool),
]

# Build a set of valid YAML key paths for unknown-key detection
_VALID_YAML_PATHS = set()
for _path, _attr, _type in _YAML_MAP:
    for i in range(len(_path)):
        _VALID_YAML_PATHS.add(_path[:i + 1])


class Config:
    """
    Global configuration, each instance holds its own state independently.

    Priority chain: Environment variables > YAML file > Code defaults.

    Usage:
        config = Config()                          # env + defaults
        config = Config(config_path="prod.yaml")   # env > yaml > defaults
    """

    def __init__(self, config_path: str | None = None):
        # Step 1: Set code defaults
        self._set_defaults()

        # Step 2: Load YAML overrides (if available)
        yaml_path = self._resolve_yaml_path(config_path)
        if yaml_path:
            self._load_yaml(yaml_path)

        # Step 3: Apply environment variable overrides (highest priority)
        self._apply_env_overrides()

        # Step 4: Post-processing (validation, aliases, auto-detection)
        self._post_process()

    # ==================================================================
    # Step 1: Code defaults
    # ==================================================================

    def _set_defaults(self):
        """Set all configuration fields to their default values."""
        # Model
        self.model: str = ""  # empty = auto-detect in _post_process
        self.max_context_tokens: int = 0  # 0 = auto-detect in _post_process
        self.api_retry_count: int = 1
        self.fallback_model: str | None = None
        self.resilience_max_retries: int = 3
        self.routing_simple_model: str | None = None

        # Agent
        self.system_prompt: str = (
            "You are LlamAgent, a super capable llama AI assistant. "
            "You are smart, reliable, and happy to help, with an occasional llama-like charm. "
            "Reply concisely and clearly. If you are unsure, be honest about it."
        )
        self.context_window_size: int = 20
        self.context_compress_threshold: float = 0.7
        self.compress_keep_turns: int = 3
        self.max_react_steps: int = 10
        self.max_duplicate_actions: int = 2
        self.react_timeout: float = 210.0
        self.react_total_timeout: float = 0  # 0 = unlimited (ContinuousRunner task_timeout handles it)
        # 32k covers typical web pages, long tool outputs, and most
        # file reads comfortably. When exceeded, the framework persists
        # the full result to a file under the workspace and tells the
        # model to use read_files+ranges for specific sections
        # (see agent.py _truncate_observation + _persist_tool_result).
        self.max_observation_tokens: int = 32000
        self.tool_result_persist_threshold: int = 0  # 0 = follow max_observation_tokens

        # v3.3: changeset LRU caps for ProjectSyncService.
        # Memory grows with each apply_patch / write_files in project zone;
        # cap the count and total pre_image bytes to prevent OOM in long
        # sessions. Eviction is LRU (oldest reverted first, then oldest
        # unreverted); evicted paths are remembered in a small ledger so
        # revert_changes can give a precise error.
        self.changeset_max_count: int = 200
        self.changeset_max_total_bytes: int = 50 * 1024 * 1024  # 50 MB

        # v3.3: Snapshot (D7) — coarse-grained safety net for CI / auto_approve.
        # Lazy: a single tar-free directory copy of agent.write_root captured
        # before the first project-zone write or command invocation.
        self.snapshot_enabled: bool = False
        self.snapshot_ignore_gitignore: bool = True   # skip .git/.gitignore'd files
        self.snapshot_max_size_mb: int = 500           # refuse if write_root exceeds
        self.snapshot_retention_count: int = 5         # keep last N session snapshots
        # Custom snapshot storage directory. Empty string falls back to
        # `<parent of write_root>/.llamagent_snapshots/`. Useful for
        # large project trees where the snapshot would clutter the
        # parent dir, or sharing snapshots across sessions on a SSD.
        self.snapshot_dir: str = ""
        # When auto_approve=True, snapshot is force-enabled (CI safety net),
        # regardless of the explicit setting above. Set explicitly here only
        # to opt INTO snapshot in interactive mode.

        # Retrieval (shared)
        self.embedding_provider: str = "chromadb"
        self.embedding_model: str = ""
        self.retrieval_persist_dir: str = str(BASE_DIR / "data" / "chroma")

        # Memory
        self.memory_mode: str = "off"
        self.memory_recall_mode: str = "tool"
        self.memory_fact_fallback: str = "text"
        self.memory_recall_top_k: int = 5
        self.memory_auto_recall_max_inject: int = 3
        self.memory_auto_recall_threshold: float = 0.35
        self.memory_consolidation_interval: int = 24
        self.memory_consolidation_min_count: int = 20
        # Compile mode: "auto" picks based on memory_backend (rag->structured, fs->raw_text);
        # "structured" forces FactCompiler (JSON atomic facts + triple-key merger);
        # "raw_text" forces subject+body single-entry compile.
        self.memory_compile_mode: str = "auto"
        # Raw-text dedup: 0 disables; when positive, trigger an LLM-based dedup
        # judgement on save once stored entry count >= threshold.
        self.memory_dedup_threshold: int = 0
        # Dedup mode — only consulted when memory_dedup_threshold > 0:
        #   "trigger": run dedup judge; count can grow past threshold on no-match.
        #   "cap":     threshold is a hard ceiling — on no-match, evict the
        #              oldest active entry (LRU by last_accessed_at, fallback
        #              created_at) to keep count bounded.
        self.memory_dedup_mode: str = "trigger"

        # RAG
        self.rag_top_k: int = 3
        self.chunk_size: int = 500
        self.rag_retrieval_mode: str = "hybrid"
        self.rag_rerank_enabled: bool = True

        # Persona
        self.persona_file: str = str(BASE_DIR / "data" / "personas.json")

        # Tools
        self.agent_tools_dir: str = str(BASE_DIR / "data" / "agent_tools")

        # Planning
        self.max_plan_adjustments: int = 7

        # Reflection
        self.reflection_write_mode: str = "off"
        self.reflection_read_mode: str = "off"
        self.reflection_backend: str = "rag"
        self.reflection_fs_dir: str | None = None
        self.reflection_score_threshold: float = 7.0
        self.skill_improve_threshold: int = 3
        # Hard cap on stored lessons. 0 disables. When positive, the oldest
        # active lesson (FIFO by created_at) is evicted on save once the
        # count reaches the cap. Reflection data is lower-stakes than memory
        # (the agent re-learns any evicted lesson next time the pattern
        # recurs), so a cheap FIFO cap is enough — no consolidation loop.
        self.reflection_max_lessons: int = 0

        # Safety
        self.permission_level: int = 1

        # Skill
        self.skill_dirs: list[str] = []
        self.skill_max_active: int = 2

        # Job
        self.job_default_timeout: float = 300.0
        self.job_max_active: int = 10
        self.job_profiles: dict = {}

        # Workspace (runtime field, not encouraged in YAML)
        self.workspace_id: str | None = None

        # v3.3: optional sub-directory of project_dir that the agent is
        # allowed to write into. When unset, the entire project_dir is
        # writable. Invalid values (escape, missing) log a warning and
        # fall back to project_dir. See agent.write_root.
        self.edit_root: str = ""

        # Output
        self.output_dir: str = str(BASE_DIR / "output")

        # Authorization
        self.authorization_mode: str = "interactive"
        self.approval_mode: str = "persistent"  # persistent | temporary
        self.auto_approve: bool = False  # True = preset full-match scope
        self.authorization_scopes: list = []  # Preset scope path list
        self.seed_scopes: list | None = None  # Parsed from YAML, list of dicts

        # Per-module model overrides (module_name -> model_name)
        self.module_models: dict[str, str] = {}

        # Backend selection (FS vs RAG)
        self.retrieval_backend: str = "rag"
        self.memory_backend: str = "rag"
        self.fs_data_dir: str = str(BASE_DIR / "data" / "fs")
        self.memory_fs_dir: str | None = None
        self.knowledge_dir: str | None = None

        # Compression (v2.9.4: tool result compression for history trace)
        self.tool_result_strategy: str = "none"
        self.tool_result_max_chars: int = 2000
        self.tool_result_head_lines: int = 10
        self.strip_thinking: bool = False

        # Persistence
        self.persistence_enabled: bool = False
        self.persistence_auto_restore: bool = True
        self.persistence_dir: str | None = None

        # Child agent
        self.child_agent_runner: str = "inline"
        self.child_agent_max_children: int = 20
        self.child_agent_role_models: dict = {}
        self.child_agent_auto_memorize: bool = True

        # Web
        self.web_search_provider: str = ""  # "" = auto-detect
        self.web_search_num_results: int = 5
        # API key for the selected provider (Tavily, SerpAPI, etc.). Empty
        # string → fall back to the provider's traditional env var
        # (TAVILY_API_KEY, SERPAPI_KEY). Intended for users who prefer
        # config.yaml over env vars.
        self.web_search_api_key: str = ""

        # Hooks (parsed from YAML, not a flat field)
        self.hooks_config: dict | None = None

    # ==================================================================
    # Step 2: YAML loading
    # ==================================================================

    def _resolve_yaml_path(self, explicit_path: str | None) -> str | None:
        """
        Determine which YAML config file to load.

        If explicit_path is given and fails, raise an error (don't silently degrade).
        If auto-discovering, return the first found or None.
        """
        # Explicit path: must work or fail loudly
        if explicit_path is not None:
            if not os.path.isfile(explicit_path):
                raise FileNotFoundError(
                    f"Config file not found: {explicit_path}"
                )
            return explicit_path

        # Environment variable
        env_path = os.getenv("LLAMAGENT_CONFIG")
        if env_path:
            if os.path.isfile(env_path):
                return env_path
            logger.warning("LLAMAGENT_CONFIG='%s' not found, ignoring", env_path)

        # Auto-discover (first found wins, single file, no merge)
        candidates = [
            str(BASE_DIR / "llamagent.yaml"),
            str(BASE_DIR / ".llamagent" / "config.yaml"),
            os.path.expanduser("~/.llamagent/config.yaml"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                logger.info("Using config: %s", candidate)
                return candidate

        return None  # No YAML found, use env + defaults

    def _load_yaml(self, path: str):
        """Load a YAML config file and apply values to flat attributes."""
        try:
            import yaml
        except ImportError:
            if path == os.getenv("LLAMAGENT_CONFIG") or path in [
                str(BASE_DIR / "llamagent.yaml"),
                str(BASE_DIR / ".llamagent" / "config.yaml"),
                os.path.expanduser("~/.llamagent/config.yaml"),
            ]:
                # Auto-discovered: silently skip
                logger.debug("pyyaml not installed, skipping YAML config")
                return
            # Explicit path: fail loudly
            raise ImportError(
                "pyyaml is required to load YAML config files. "
                "Install: pip install pyyaml"
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML config '{path}': {e}") from e

        if not isinstance(data, dict):
            logger.warning("YAML config '%s' is not a dict, ignoring", path)
            return

        # Apply mapped values
        for yaml_path, attr_name, expected_type in _YAML_MAP:
            value = self._get_nested(data, yaml_path)
            if value is not None:
                # Basic type casting for safety
                try:
                    if expected_type == bool and isinstance(value, str):
                        value = value.lower() in ("true", "1", "yes")
                    elif expected_type in (int, float, str) and not isinstance(value, expected_type):
                        value = expected_type(value)
                except (ValueError, TypeError):
                    logger.warning(
                        "YAML config: cannot convert '%s' to %s for '%s', skipping",
                        value, expected_type.__name__, ".".join(yaml_path),
                    )
                    continue
                setattr(self, attr_name, value)

        # Parse hooks section (independent logic, not through _YAML_MAP)
        hooks_raw = data.get("hooks")
        if isinstance(hooks_raw, dict):
            self.hooks_config = hooks_raw

        # Parse seed_scopes from authorization section (nested list, not through _YAML_MAP)
        auth_raw = data.get("authorization")
        if isinstance(auth_raw, dict):
            seed_raw = auth_raw.get("seed_scopes")
            if isinstance(seed_raw, list):
                self.seed_scopes = seed_raw

        # Warn about unknown keys
        self._check_unknown_keys(data)

    @staticmethod
    def _get_nested(data: dict, path: tuple) -> object:
        """Get a value from a nested dict by key path. Returns None if not found."""
        current = data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _check_unknown_keys(self, data: dict, prefix: tuple = ()):
        """Warn about YAML keys not in the mapping table."""
        for key, value in data.items():
            current_path = prefix + (key,)
            # Special sections with their own parsing, skip validation
            if current_path in (("hooks",), ("authorization", "seed_scopes"), ("authorization", "scopes")):
                continue
            if current_path not in _VALID_YAML_PATHS:
                logger.warning(
                    "Unknown YAML config key: '%s' (ignored)",
                    ".".join(current_path),
                )
            elif isinstance(value, dict):
                self._check_unknown_keys(value, current_path)

    # ==================================================================
    # Step 3: Environment variable overrides
    # ==================================================================

    def _apply_env_overrides(self):
        """Apply environment variable overrides (highest priority)."""
        # Model
        env_model = os.getenv("MODEL_NAME")
        if env_model:
            self.model = env_model

        # Agent / React
        self.max_react_steps = _safe_int("MAX_REACT_STEPS", self.max_react_steps)
        self.max_duplicate_actions = _safe_int("MAX_DUPLICATE_ACTIONS", self.max_duplicate_actions)
        self.react_timeout = _safe_float("REACT_TIMEOUT", self.react_timeout)

        # Retrieval
        env_persist = os.getenv("RETRIEVAL_PERSIST_DIR")
        env_chroma = os.getenv("CHROMA_DIR")
        if env_persist:
            self.retrieval_persist_dir = env_persist  # new name wins
        elif env_chroma:
            self.retrieval_persist_dir = env_chroma  # backward compat fallback

        env_embedding_provider = os.getenv("EMBEDDING_PROVIDER")
        if env_embedding_provider:
            self.embedding_provider = env_embedding_provider
        env_embedding_model = os.getenv("EMBEDDING_MODEL")
        if env_embedding_model is not None:
            self.embedding_model = env_embedding_model

        # Memory
        env_memory_mode = os.getenv("MEMORY_MODE")
        if env_memory_mode:
            self.memory_mode = env_memory_mode
        env_recall_mode = os.getenv("MEMORY_RECALL_MODE")
        if env_recall_mode:
            self.memory_recall_mode = env_recall_mode
        env_fact_fallback = os.getenv("MEMORY_FACT_FALLBACK")
        if env_fact_fallback:
            self.memory_fact_fallback = env_fact_fallback
        self.memory_recall_top_k = _safe_int("MEMORY_RECALL_TOP_K", self.memory_recall_top_k)
        self.memory_auto_recall_max_inject = _safe_int("MEMORY_AUTO_RECALL_MAX_INJECT", self.memory_auto_recall_max_inject)
        self.memory_auto_recall_threshold = _safe_float("MEMORY_AUTO_RECALL_THRESHOLD", self.memory_auto_recall_threshold)

        # RAG
        self.rag_top_k = _safe_int("RAG_TOP_K", self.rag_top_k)
        self.chunk_size = _safe_int("CHUNK_SIZE", self.chunk_size)
        env_rag_mode = os.getenv("RAG_RETRIEVAL_MODE")
        if env_rag_mode:
            self.rag_retrieval_mode = env_rag_mode
        env_rerank = os.getenv("RAG_RERANK")
        if env_rerank is not None:
            self.rag_rerank_enabled = env_rerank.lower() in ("true", "1", "yes")

        # Persona
        env_persona = os.getenv("PERSONA_FILE")
        if env_persona:
            self.persona_file = env_persona

        # Tools
        env_tools_dir = os.getenv("AGENT_TOOLS_DIR")
        if env_tools_dir:
            self.agent_tools_dir = env_tools_dir

        # Planning
        self.max_plan_adjustments = _safe_int("MAX_PLAN_ADJUSTMENTS", self.max_plan_adjustments)

        # Reflection
        self.reflection_score_threshold = _safe_float("REFLECTION_SCORE_THRESHOLD", self.reflection_score_threshold)

        # Safety
        self.permission_level = _safe_int("PERMISSION_LEVEL", self.permission_level)

        # Skill
        env_skill_dirs = os.getenv("SKILL_DIRS")
        if env_skill_dirs:
            self.skill_dirs = [d.strip() for d in env_skill_dirs.split(",") if d.strip()]

        # Job
        self.job_default_timeout = _safe_float("JOB_DEFAULT_TIMEOUT", self.job_default_timeout)
        self.job_max_active = _safe_int("JOB_MAX_ACTIVE", self.job_max_active)

        # Web
        env_web_provider = os.getenv("WEB_SEARCH_PROVIDER")
        if env_web_provider:
            self.web_search_provider = env_web_provider
        self.web_search_num_results = _safe_int("WEB_SEARCH_NUM_RESULTS", self.web_search_num_results)
        # Generic provider API key: WEB_SEARCH_API_KEY overrides config.yaml if set.
        env_web_key = os.getenv("WEB_SEARCH_API_KEY")
        if env_web_key:
            self.web_search_api_key = env_web_key

        # Authorization
        env_auth_mode = os.getenv("AUTHORIZATION_MODE")
        if env_auth_mode:
            self.authorization_mode = env_auth_mode

        # Output
        env_output = os.getenv("OUTPUT_DIR")
        if env_output:
            self.output_dir = env_output

    # ==================================================================
    # Step 4: Post-processing
    # ==================================================================

    def _post_process(self):
        """Validation, auto-detection, and alias setup."""
        # Model auto-detection (if not set by YAML or env)
        if not self.model:
            self.model = self._detect_model()

        # Context window auto-detection (if not set)
        if self.max_context_tokens <= 0:
            self.max_context_tokens = self._detect_max_context_tokens()

        # Validation
        if self.memory_mode not in ("off", "autonomous", "hybrid"):
            logger.warning(
                "memory_mode='%s' is invalid, falling back to 'off'",
                self.memory_mode,
            )
            self.memory_mode = "off"

        if self.memory_recall_mode not in ("off", "tool", "auto"):
            logger.warning(
                "memory_recall_mode='%s' is invalid, falling back to 'tool'",
                self.memory_recall_mode,
            )
            self.memory_recall_mode = "tool"

        if self.memory_compile_mode not in ("auto", "structured", "raw_text"):
            logger.warning(
                "memory_compile_mode='%s' is invalid, falling back to 'auto'",
                self.memory_compile_mode,
            )
            self.memory_compile_mode = "auto"

        if self.memory_dedup_threshold < 0:
            logger.warning(
                "memory_dedup_threshold=%d is negative, clamping to 0 (disabled)",
                self.memory_dedup_threshold,
            )
            self.memory_dedup_threshold = 0

        if self.memory_dedup_mode not in ("trigger", "cap"):
            logger.warning(
                "memory_dedup_mode='%s' is invalid, falling back to 'trigger'",
                self.memory_dedup_mode,
            )
            self.memory_dedup_mode = "trigger"

        if self.retrieval_backend not in ("rag", "fs"):
            logger.warning(
                "retrieval_backend='%s' is invalid, falling back to 'rag'",
                self.retrieval_backend,
            )
            self.retrieval_backend = "rag"

        if self.memory_backend not in ("rag", "fs"):
            logger.warning(
                "memory_backend='%s' is invalid, falling back to 'rag'",
                self.memory_backend,
            )
            self.memory_backend = "rag"

        if self.reflection_write_mode not in ("off", "auto"):
            logger.warning(
                "reflection_write_mode='%s' is invalid, falling back to 'off'",
                self.reflection_write_mode,
            )
            self.reflection_write_mode = "off"

        if self.reflection_read_mode not in ("off", "tool", "auto"):
            logger.warning(
                "reflection_read_mode='%s' is invalid, falling back to 'off'",
                self.reflection_read_mode,
            )
            self.reflection_read_mode = "off"

        if self.reflection_max_lessons < 0:
            logger.warning(
                "reflection_max_lessons=%d is negative, clamping to 0 (disabled)",
                self.reflection_max_lessons,
            )
            self.reflection_max_lessons = 0

        if self.reflection_backend not in ("rag", "fs"):
            logger.warning(
                "reflection_backend='%s' is invalid, falling back to 'rag'",
                self.reflection_backend,
            )
            self.reflection_backend = "rag"

        if self.child_agent_runner not in ("inline", "thread", "process"):
            logger.warning(
                "child_agent_runner='%s' is invalid, falling back to 'inline'",
                self.child_agent_runner,
            )
            self.child_agent_runner = "inline"

        if self.tool_result_strategy not in ("none", "placeholder", "head", "llm_summary"):
            logger.warning(
                "tool_result_strategy='%s' is invalid, falling back to 'none'",
                self.tool_result_strategy,
            )
            self.tool_result_strategy = "none"

        # Backward compatibility alias
        self.chroma_dir = self.retrieval_persist_dir

    # ------------------------------------------------------------------
    # Model detection
    # ------------------------------------------------------------------

    def _detect_model(self) -> str:
        """
        Default model selection. Prefers local Ollama to avoid unintended cloud spend.

        To use a cloud provider, set MODEL_NAME explicitly, e.g.:
            MODEL_NAME=openai/gpt-4o-mini
            MODEL_NAME=deepseek/deepseek-chat
            MODEL_NAME=anthropic/claude-3-5-sonnet-latest

        Presence of cloud API keys (OPENAI_API_KEY / DEEPSEEK_API_KEY / ANTHROPIC_API_KEY)
        does NOT change the default — only MODEL_NAME does. This keeps the public default
        predictable and free of unexpected cloud calls.
        """
        logger.info("Using default Ollama local model (set MODEL_NAME env var to use a cloud provider)")
        return "ollama_chat/qwen3.5:latest"

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
