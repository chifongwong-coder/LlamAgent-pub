"""
WorkspaceService: logical workspace management for the v1.5 tool system.

WorkspaceService is a plain service class (not a Module), instantiated by
ToolsModule.on_attach(). It provides:

- Workspace directory management (session-based, task-aware)
- Path resolution with ``project:`` prefix support
- Security boundary enforcement via os.path.realpath()

Directory layout under playground_dir::

    <playground_dir>/
      sessions/
        <workspace_id>/
          shared/            <-- default workspace (SimpleReAct / plain chat)
          tasks/
            <task_id>/       <-- task-level isolation (PlanReAct / ChildAgent)

The workspace always resides inside playground_dir (Zone 1), so all
workspace-internal operations are unrestricted by the zone system.
"""

from __future__ import annotations

import logging
import os
import shutil
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


class WorkspaceService:
    """
    Logical workspace manager for a single LlamAgent instance.

    Provides session-scoped workspace directories, path resolution with
    ``project:`` prefix support, and workspace boundary checks.

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

        v3.3: legacy ``project:`` prefix support has been removed; the 5
        core file tools select between project and playground via the
        ``zone`` parameter and the new :func:`_resolve_within` helper.
        This method is retained for backwards compatibility with
        WorkspaceService internal callers (e.g. ``ensure_in_workspace``)
        and for any third-party code that still uses it; new code should
        prefer ``_resolve_within(raw, base=..., allow_absolute=...)``.

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
        workspace (e.g. ``sync_workspace_to_project`` source validation).

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

    def resolve_path_workspace_only(self, raw_path: str) -> str:
        """
        Resolve a path and verify it stays within workspace root.

        Like resolve_path(), but rejects paths that resolve outside the
        workspace (including project: prefix, project dir absolute paths,
        and external paths). Used by write operations to enforce
        workspace-first workflow.

        Raises:
            ValueError: If the resolved path is outside workspace root.
        """
        resolved = self.resolve_path(raw_path)
        if not self.ensure_in_workspace(resolved):
            raise ValueError(
                f"Write operations are restricted to workspace. "
                f"Path '{raw_path}' resolves outside workspace root. "
                f"Use apply_patch for project modifications."
            )
        return resolved

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
