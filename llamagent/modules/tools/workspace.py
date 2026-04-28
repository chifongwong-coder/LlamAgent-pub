"""
WorkspaceService: framework scratch directory management.

Plain service class (not a Module), instantiated by
ToolsModule.on_attach(). Owns the per-session scratch tree under
playground_dir for framework-internal callers (child agents,
sandbox subprocesses, async tool persistence).

Directory layout under playground_dir::

    <playground_dir>/
      sessions/
        <workspace_id>/
          shared/            <-- default workspace (SimpleReAct / plain chat)
          tasks/
            <task_id>/       <-- task-level isolation (PlanReAct / ChildAgent)

v3.3: the model-facing path resolution lives at module scope as
:func:`_resolve_path` + :func:`classify_write` (no ``project:`` prefix,
no ``zone`` parameter). The legacy ``WorkspaceService.resolve_path``
remains as an internal helper used by framework code that writes
into per-session scratch dirs directly.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llamagent.core.agent import LlamAgent

logger = logging.getLogger(__name__)


def _resolve_within(raw: str, *, base: str, allow_absolute: bool = True) -> str:
    """Resolve ``raw`` against ``base``; reject if it escapes ``base``.

    Used by v3.3 file-tool path_extractors as the single, simple
    write-boundary primitive. Distinct from
    :meth:`WorkspaceService.resolve_path` which still understands the
    legacy ``project:`` prefix for backward compatibility.

    Args:
        raw: Caller-supplied raw path string.
        base: Absolute base directory to resolve against.
        allow_absolute: If False, reject absolute paths in ``raw``
            (used in playground mode where the caller must always
            stay relative to the playground root).

    Raises:
        ValueError: If absolute paths are disallowed and ``raw`` is
            absolute, or if the resolved path escapes ``base``.
    """
    if not allow_absolute and os.path.isabs(raw):
        raise ValueError(f"Absolute paths not allowed: {raw}")
    base_real = os.path.realpath(base)
    resolved = os.path.realpath(os.path.join(base_real, raw))
    if not (resolved == base_real or resolved.startswith(base_real + os.sep)):
        raise ValueError(f"Path '{raw}' escapes base directory")
    return resolved


def _resolve_path(raw: str, *, base: str) -> str:
    """Resolve ``raw`` to an absolute realpath. Absolute paths are kept;
    relative paths are joined with ``base``.

    Unlike :func:`_resolve_within` this does **not** reject paths that
    escape ``base`` — the caller (typically :func:`classify_write`)
    decides routing based on the resolved location. macOS HFS+/APFS
    case-insensitivity is handled by the canonical case ``realpath``
    returns when the parent directory exists.
    """
    if os.path.isabs(raw):
        return os.path.realpath(raw)
    return os.path.realpath(os.path.join(base, raw))


def _normcase_path(p: str) -> str:
    """Case-fold a path for prefix comparison.

    ``os.path.normcase`` already lowercases on Windows. On macOS the
    default APFS / HFS+ filesystem is case-insensitive but Python's
    ``normcase`` is a no-op there, so realpath of a mixed-case input
    keeps the input's lexical case (it only canonicalizes when the
    target inode actually exists with a known canonical case). We
    explicitly ``.lower()`` on Darwin so classify_write routes
    ``Llama_Playground/x.txt`` and ``llama_playground/x.txt`` to the
    same zone.
    """
    p = os.path.normcase(p)
    if sys.platform == "darwin":
        p = p.lower()
    return p


def classify_write(resolved_path: str, agent: LlamAgent) -> str:
    """Classify a resolved write path into one of three zones.

    Returns one of:
    - ``"playground"`` — path is under ``agent.playground_dir``. Write
      is allowed but **not** tracked by changeset (ephemeral scratch).
    - ``"project"`` — path is under ``agent.write_root`` and not under
      ``playground_dir``. Write is allowed and tracked.
    - ``"rejected"`` — path is outside both. Caller must reject.

    Order matters: playground is checked first because it physically
    lives inside ``write_root`` (default ``write_root == project_dir``).
    Prefix comparison is case-folded via :func:`_normcase_path` so
    macOS HFS+/APFS case-insensitivity is respected.
    """
    p = _normcase_path(resolved_path)
    pg = _normcase_path(os.path.realpath(agent.playground_dir))
    if p == pg or p.startswith(pg + os.sep):
        return "playground"
    wr = _normcase_path(os.path.realpath(agent.write_root))
    if p == wr or p.startswith(wr + os.sep):
        return "project"
    return "rejected"


class WorkspaceService:
    """Framework scratch directory manager for one LlamAgent instance.

    Provides session-scoped scratch directories under
    ``agent.playground_dir`` for framework-internal use. Model-facing
    path resolution does NOT go through this service; see module-level
    :func:`_resolve_path` and :func:`classify_write` instead.

    Attributes:
        agent: The LlamAgent instance this service is attached to.
    """

    def __init__(self, agent: LlamAgent, workspace_id: str | None = None) -> None:
        """
        Initialize the WorkspaceService.

        Args:
            agent: The LlamAgent instance that owns this workspace.
            workspace_id: Optional external workspace identifier (e.g. API
                session_id). When None, a unique id is lazily generated on
                first access via the ``workspace_id`` property.
        """
        self.agent = agent
        self._workspace_id: str | None = workspace_id

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _sanitize_id(value: str) -> str:
        """Remove path separators and dangerous characters from an ID string."""
        return value.replace("/", "").replace("\\", "").replace("..", "").strip()

    # ================================================================
    # Properties
    # ================================================================

    @property
    def workspace_id(self) -> str:
        """
        Return the workspace identifier, lazily generating one if needed.

        The id is a 32-character hex string (uuid4) when auto-generated.
        Once set (either externally or lazily), it does not change for the
        lifetime of this service instance.
        """
        if self._workspace_id is None:
            self._workspace_id = uuid.uuid4().hex
            logger.info("Workspace id generated: %s", self._workspace_id)
        return self._sanitize_id(self._workspace_id)

    @property
    def workspace_root(self) -> str:
        """
        Return the current workspace directory path, creating it if needed.

        Routing logic:
        - No ``_current_task_id`` on agent -> ``<playground>/sessions/<wid>/shared/``
        - Has ``_current_task_id`` -> ``<playground>/sessions/<wid>/tasks/<task_id>/``

        The directory is created with ``os.makedirs(exist_ok=True)`` so
        callers can assume the returned path exists.
        """
        task_id = getattr(self.agent, "_current_task_id", None)

        if task_id is not None:
            root = os.path.join(
                self.agent.playground_dir,
                "sessions",
                self.workspace_id,
                "tasks",
                self._sanitize_id(task_id),
            )
        else:
            root = os.path.join(
                self.agent.playground_dir,
                "sessions",
                self.workspace_id,
                "shared",
            )

        os.makedirs(root, exist_ok=True)
        return root

    # ================================================================
    # Path resolution
    # ================================================================

    def resolve_path(self, raw_path: str) -> str:
        """
        Resolve a raw path string to an absolute, real filesystem path,
        relative to ``workspace_root`` by default.

        v3.3: legacy ``project:`` prefix support has been removed (B2
        decision); the model-facing tools auto-classify paths via
        :func:`classify_write` instead of selecting via a ``zone``
        kwarg. This method is retained only for framework-internal
        callers writing into per-session scratch directories. New
        code should prefer ``_resolve_path`` or ``_resolve_within``.

        Resolution rules:
        - Absolute path -> used as-is (after realpath)
        - Relative path -> relative to ``workspace_root``

        Args:
            raw_path: The raw path string from a tool argument.

        Returns:
            Absolute, symlink-resolved filesystem path.
        """
        if os.path.isabs(raw_path):
            return os.path.realpath(raw_path)
        return os.path.realpath(
            os.path.join(self.workspace_root, raw_path)
        )

    def resolve_paths(self, raw_paths: list[str]) -> list[str]:
        """
        Batch version of :meth:`resolve_path`.

        Args:
            raw_paths: List of raw path strings.

        Returns:
            List of absolute, symlink-resolved filesystem paths.
        """
        return [self.resolve_path(p) for p in raw_paths]

    # ================================================================
    # Boundary check
    # ================================================================

    def ensure_in_workspace(self, resolved_path: str) -> bool:
        """
        Check whether a resolved path is inside the current workspace root.

        This is used by tools that require source paths to stay within the
        workspace (used by other tools that gate paths to the playground area).

        Args:
            resolved_path: An already-resolved absolute path (typically the
                output of :meth:`resolve_path`).

        Returns:
            True if the path is inside (or equal to) the workspace root,
            False otherwise.
        """
        root = self.workspace_root
        return resolved_path == root or resolved_path.startswith(root + os.sep)

    # ================================================================
    # Workspace-only restriction
    # ================================================================

    # v3.3 commit-13b: resolve_path_workspace_only deleted. Path-fallback
    # tools now share the core 5's resolution: _resolve_path against
    # project_dir + classify_write decides routing. This helper had
    # workspace-first semantics that v3.3 rejects (the model writes to
    # project_dir, not workspace).

    # ================================================================
    # Lifecycle
    # ================================================================

    def cleanup(self) -> None:
        """
        Remove the entire workspace session directory.

        Deletes ``<playground_dir>/sessions/<workspace_id>/`` and all its
        contents.  Called during agent shutdown to free disk space.

        Silently ignores errors (e.g. directory already removed, permission
        issues) so that cleanup never causes a shutdown failure.
        """
        session_dir = os.path.join(
            self.agent.playground_dir,
            "sessions",
            self.workspace_id,
        )
        if os.path.isdir(session_dir):
            try:
                shutil.rmtree(session_dir)
                logger.info("Workspace session cleaned up: %s", session_dir)
            except OSError as e:
                logger.warning(
                    "Failed to clean up workspace session %s: %s",
                    session_dir,
                    e,
                )
