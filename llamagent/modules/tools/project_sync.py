"""
ProjectSyncService: workspace-to-project synchronization for the v1.5 tool system.

ProjectSyncService is a plain service class (not a Module), instantiated by
ToolsModule.on_attach(). It provides:

- Structured search/replace patching (apply_patch, preview_patch, replace_block)
- Per-write changeset tracking for write_files (record_write_changeset)
- Changeset-based revert for all project modifications (revert_changes)

All project writes are atomic (temp file + fsync + os.rename) and recorded as
changesets with pre-image snapshots for reliable rollback.

Decision references (from v1.5 design doc section 10.8):
- D5: Project sync paths are always relative to project_dir
- B1: apply_patch is atomic (all-or-nothing)
- B5: File-level locks for serialization
- C1: Changeset per successful write, stacked for multi-step revert
- C2: revert marks changeset as reverted (not deleted)
- F1: sync patch mode uses whole-file preimage/postimage compare
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llamagent.core.agent import LlamAgent
    from llamagent.modules.tools.workspace import WorkspaceService

logger = logging.getLogger(__name__)


# ======================================================================
# Changeset data class
# ======================================================================


@dataclass
class Changeset:
    """
    Record of a single project file modification.

    Each successful project write (apply_patch, sync, etc.) creates one
    Changeset.  The pre_image allows exact restoration on revert.

    Attributes:
        changeset_id: Unique identifier for this changeset.
        target_path: Absolute path to the modified project file.
        pre_image: Original file content before modification, or None if
            the file did not exist (new file creation).
        ops: Structured log of operations applied (for audit/preview).
        timestamp: Unix timestamp of when the changeset was created.
        reverted: Whether this changeset has been reverted.
    """

    changeset_id: str
    target_path: str
    pre_image: str | None
    ops: list[dict]
    timestamp: float
    reverted: bool = False


# ======================================================================
# ProjectSyncService
# ======================================================================


class ProjectSyncService:
    """
    Project synchronization service for the v1.5 tool system.

    Manages all writes to the project directory through atomic operations
    with changeset tracking and revert capability.

    Attributes:
        agent: The LlamAgent instance this service is attached to.
        workspace_service: The WorkspaceService for workspace path resolution.
    """

    def __init__(
        self, agent: LlamAgent, workspace_service: WorkspaceService
    ) -> None:
        """
        Initialize the ProjectSyncService.

        Args:
            agent: The LlamAgent instance that owns this service.
            workspace_service: The WorkspaceService used for workspace
                path resolution (used by changeset/write helpers).
        """
        self.agent = agent
        self.workspace_service = workspace_service
        self._changesets: list[Changeset] = []
        # v3.3: paths whose changesets were dropped by the LRU cap;
        # used by revert_changes to surface a precise error.
        self._evicted_paths: set[str] = set()
        self._file_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ================================================================
    # Lock management
    # ================================================================

    def _get_file_lock(self, path: str) -> threading.Lock:
        """
        Return a per-file lock, creating one if it does not exist.

        Thread-safe: uses an internal guard lock to protect the lock dict.

        Args:
            path: Absolute file path to get a lock for.

        Returns:
            A threading.Lock dedicated to the given file path.
        """
        with self._locks_guard:
            if path not in self._file_locks:
                self._file_locks[path] = threading.Lock()
            return self._file_locks[path]

    # ================================================================
    # v3.3: changeset LRU caps
    # ================================================================

    def _changeset_bytes(self, cs: Changeset) -> int:
        """Approximate memory footprint of a Changeset's pre_image."""
        return len(cs.pre_image) if cs.pre_image else 0

    def _enforce_changeset_caps(self) -> None:
        """LRU eviction when ``len(_changesets)`` exceeds count cap or
        ``sum(pre_image bytes)`` exceeds byte cap. Reverted changesets
        are dropped first (they're tombstones); then oldest unreverted.
        Evicted target paths land in ``_evicted_paths`` so
        ``revert_changes`` can surface a precise error.
        """
        cfg = self.agent.config
        max_count = getattr(cfg, "changeset_max_count", 200)
        max_bytes = getattr(cfg, "changeset_max_total_bytes", 50 * 1024 * 1024)
        if max_count <= 0 and max_bytes <= 0:
            return

        def _total_bytes() -> int:
            return sum(self._changeset_bytes(cs) for cs in self._changesets)

        def _over():
            over_count = max_count > 0 and len(self._changesets) > max_count
            over_bytes = max_bytes > 0 and _total_bytes() > max_bytes
            return over_count or over_bytes

        # Pass 1: drop reverted (tombstone) changesets oldest-first.
        if _over():
            keep: list[Changeset] = []
            dropped = 0
            for cs in self._changesets:
                if cs.reverted and _over():
                    self._evicted_paths.add(cs.target_path)
                    logger.info(
                        "changeset evicted (reverted/tombstone): %s",
                        cs.target_path,
                    )
                    dropped += 1
                    # Pretend it's gone for the over-check via a temporary
                    # rebuild after the loop.
                else:
                    keep.append(cs)
            if dropped:
                self._changesets = keep

        # Pass 2: drop oldest unreverted as needed.
        while _over() and self._changesets:
            cs = self._changesets.pop(0)
            self._evicted_paths.add(cs.target_path)
            logger.warning(
                "changeset evicted (cap hit, was unreverted): %s "
                "(count=%d bytes=%d max_count=%d max_bytes=%d)",
                cs.target_path, len(self._changesets) + 1, _total_bytes(),
                max_count, max_bytes,
            )

    def _was_evicted(self, target_path: str) -> bool:
        """Whether a path's changeset was previously dropped by the cap."""
        return target_path in self._evicted_paths

    # ================================================================
    # Path resolution
    # ================================================================

    def resolve_project_path(self, target: str) -> str:
        """
        Resolve a target path for project sync operations.

        Project sync tool paths are ALWAYS relative to project_dir (decision
        D5).  Absolute paths are used as-is after realpath normalization.
        Relative paths are resolved against agent.project_dir.

        Args:
            target: Raw path string from a tool argument.

        Returns:
            Absolute, symlink-resolved filesystem path.
        """
        # v3.3: 'project:' prefix removed (B2 decision). Tools now select
        # between project and playground via the zone parameter; see
        # workspace._resolve_within for the unified resolver.
        if os.path.isabs(target):
            return os.path.realpath(target)
        return os.path.realpath(os.path.join(self.agent.project_dir, target))

    # ================================================================
    # Core: apply_patch
    # ================================================================

    def _maybe_snapshot(self) -> None:
        """v3.3 D7: snapshot trigger lives at the service layer so any
        caller (tool wrapper, child agent, replay code, ...) goes through
        it. Idempotent — the SnapshotService's _taken flag guarantees a
        single capture per session, even if every call site invokes
        ensure_snapshot."""
        agent = self.agent
        if hasattr(agent, "ensure_snapshot"):
            agent.ensure_snapshot()

    def apply_patch(self, target: str, edits: list[dict],
                    record_changeset: bool = True) -> dict:
        """
        Apply structured search/replace edits to a project file atomically.

        Resolution: target is resolved relative to project_dir (decision D5).

        Atomicity (decision B1): all edits are applied in memory first.  If
        every edit succeeds, the result is written via temp file + fsync +
        atomic rename.  If any edit fails, the file is untouched.

        Each successful apply creates a Changeset (decision C1) by default.
        Pass ``record_changeset=False`` to skip changeset recording — used
        by the v3.3 ``zone="playground"`` mode where edits are ephemeral.

        Args:
            target: File path (relative to project_dir, or absolute).
            edits: List of edit operations.  Each dict has keys:
                - ``match`` (str): Text to find in the file.
                - ``replace`` (str): Replacement text.
                - ``expected_count`` (int, optional): Expected number of
                  occurrences of ``match``.  Defaults to 1.
            record_changeset: When False, skip Changeset registration so
                the edit cannot be reverted via ``revert_changes``. Used
                for playground patches in v3.3.

        Returns:
            Result dict with ``status`` ("success" or "error") and details.
        """
        # v3.3 D7: snapshot before mutating project. Skipped for previews
        # and for playground patches (record_changeset=False).
        if record_changeset:
            self._maybe_snapshot()
        resolved = self.resolve_project_path(target)
        lock = self._get_file_lock(resolved)

        with lock:
            # Read current file content
            try:
                with open(resolved, "r", encoding="utf-8") as f:
                    content = f.read()
            except FileNotFoundError:
                return {
                    "status": "error",
                    "target": resolved,
                    "error": f"File not found: {resolved}",
                }
            except OSError as e:
                return {
                    "status": "error",
                    "target": resolved,
                    "error": f"Failed to read file: {e}",
                }
            except UnicodeDecodeError:
                return {
                    "status": "error",
                    "target": resolved,
                    "error": f"Binary file cannot be patched: {resolved}",
                }

            # Save pre-image before any modifications
            pre_image = content

            # Apply edits in memory sequentially
            ops: list[dict] = []
            for i, edit in enumerate(edits):
                try:
                    match = edit["match"]
                    replace = edit["replace"]
                except (KeyError, TypeError) as e:
                    return {
                        "status": "error",
                        "target": resolved,
                        "error": f"Edit {i} has invalid structure: {e}",
                    }
                expected_count = edit.get("expected_count", 1)

                # Reject empty match strings
                if not match:
                    return {
                        "status": "error",
                        "target": resolved,
                        "error": f"Edit {i} has empty match string",
                    }

                actual_count = content.count(match)
                if actual_count != expected_count:
                    # Truncate match text for readable error messages
                    match_preview = match[:80] + "..." if len(match) > 80 else match
                    error_msg = (
                        f"Edit {i} failed: expected {expected_count} "
                        f"occurrences of '{match_preview}', found {actual_count}"
                    )
                    logger.warning(
                        "apply_patch aborted on edit %d for %s: %s",
                        i, resolved, error_msg,
                    )
                    return {
                        "status": "error",
                        "target": resolved,
                        "error": error_msg,
                    }

                content = content.replace(match, replace)
                ops.append({
                    "edit_index": i,
                    "match_preview": match[:80],
                    "replace_preview": replace[:80],
                    "occurrences": actual_count,
                })

            # All edits succeeded — atomic write
            self._atomic_write(resolved, content)

            if not record_changeset:
                # Playground / ephemeral mode: skip changeset registration.
                logger.info(
                    "apply_patch succeeded (no changeset): %s (%d edits)",
                    resolved, len(edits),
                )
                return {
                    "status": "success",
                    "target": resolved,
                    "edits_applied": len(edits),
                    "changeset_id": None,
                }

            # Record changeset
            changeset = Changeset(
                changeset_id=uuid.uuid4().hex,
                target_path=resolved,
                pre_image=pre_image,
                ops=ops,
                timestamp=time.time(),
            )
            self._changesets.append(changeset)
            self._enforce_changeset_caps()

            logger.info(
                "apply_patch succeeded: %s (%d edits, changeset %s)",
                resolved, len(edits), changeset.changeset_id,
            )
            return {
                "status": "success",
                "target": resolved,
                "edits_applied": len(edits),
                "changeset_id": changeset.changeset_id,
            }

    # ================================================================
    # v3.3: write_files changeset registration
    # ================================================================

    def record_write_changeset(self, target: str, pre_image: str | None) -> str:
        """Register a Changeset for an out-of-band write performed by
        ``write_files`` in v3.3.

        ``write_files`` does its own atomic write (it has to support
        ``mode='binary'`` and ``makedirs(exist_ok=True)`` for nested
        paths), so this helper just records the pre-image so that
        ``revert_changes`` can roll back the write.

        Args:
            target: Absolute path that was written.
            pre_image: Prior file content as a string, or ``None`` if
                the file did not exist before the write (revert deletes
                the file).

        Returns:
            The newly-registered Changeset id.
        """
        resolved = os.path.realpath(target)
        changeset = Changeset(
            changeset_id=uuid.uuid4().hex,
            target_path=resolved,
            pre_image=pre_image,
            ops=[{
                "action": "write_file",
                "had_prior_content": pre_image is not None,
            }],
            timestamp=time.time(),
        )
        self._changesets.append(changeset)
        self._enforce_changeset_caps()
        return changeset.changeset_id

    # ================================================================
    # Preview: preview_patch
    # ================================================================

    def preview_patch(self, target: str, edits: list[dict]) -> dict:
        """
        Preview structured search/replace edits without writing to disk.

        Same validation logic as apply_patch but does NOT write to disk
        or create a changeset.

        Args:
            target: File path (relative to project_dir, or absolute).
            edits: List of edit operations (same format as apply_patch).

        Returns:
            Result dict with ``status`` ("preview" or "error") and size info.
        """
        resolved = self.resolve_project_path(target)

        # Read current file content
        try:
            with open(resolved, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            return {
                "status": "error",
                "target": resolved,
                "error": f"File not found: {resolved}",
            }
        except OSError as e:
            return {
                "status": "error",
                "target": resolved,
                "error": f"Failed to read file: {e}",
            }
        except UnicodeDecodeError:
            return {
                "status": "error",
                "target": resolved,
                "error": f"Binary file cannot be patched: {resolved}",
            }

        current_size = len(content)

        # Apply edits in memory (validation only)
        for i, edit in enumerate(edits):
            try:
                match = edit["match"]
                replace = edit["replace"]
            except (KeyError, TypeError) as e:
                return {
                    "status": "error",
                    "target": resolved,
                    "error": f"Edit {i} has invalid structure: {e}",
                }
            if not match:
                return {
                    "status": "error",
                    "target": resolved,
                    "error": f"Edit {i} has empty match string",
                }
            expected_count = edit.get("expected_count", 1)

            actual_count = content.count(match)
            if actual_count != expected_count:
                match_preview = match[:80] + "..." if len(match) > 80 else match
                error_msg = (
                    f"Edit {i} failed: expected {expected_count} "
                    f"occurrences of '{match_preview}', found {actual_count}"
                )
                return {
                    "status": "error",
                    "target": resolved,
                    "error": error_msg,
                }

            content = content.replace(match, replace)

        new_size = len(content)

        logger.debug(
            "preview_patch for %s: current_size=%d, new_size=%d, edits_valid=True",
            resolved, current_size, new_size,
        )
        return {
            "status": "preview",
            "target": resolved,
            "current_size": current_size,
            "new_size": new_size,
            "edits_valid": True,
        }

    # ================================================================
    # Sugar: replace_block
    # ================================================================

    def replace_block(self, file: str, old: str, new: str) -> dict:
        """
        Replace a single block of text in a project file.

        Sugar for ``apply_patch(file, [{"match": old, "replace": new}])``.

        Args:
            file: File path (relative to project_dir, or absolute).
            old: Text block to find.
            new: Replacement text block.

        Returns:
            Result dict (same format as apply_patch).
        """
        return self.apply_patch(file, [{"match": old, "replace": new}])

    # ================================================================
    # Revert: revert_changes
    # ================================================================

    def revert_changes(self, targets: list[str] | None) -> dict:
        """
        Revert project changes by restoring pre-image snapshots.

        When targets is None, reverts the single most recent unreverted
        changeset globally.  When targets is a list, reverts each file's
        most recent unreverted changeset independently so that all listed
        files are rolled back.

        Reverted changesets are marked ``reverted=True`` but not deleted,
        preserving the audit trail (decision C2).

        Args:
            targets: File paths (relative to project_dir) to revert, or
                None to revert the most recent unreverted changeset globally.

        Returns:
            Result dict with reverted file paths and changeset ids.
        """
        if targets is not None:
            # Resolve target paths
            resolved_targets = [
                self.resolve_project_path(t) for t in targets
            ]
        else:
            resolved_targets = None

        # When targets is a list, revert each file's most recent changeset
        # independently; when None, revert one global most-recent changeset.
        if resolved_targets is not None:
            reverted_files: list[str] = []
            changeset_ids: list[str] = []
            errors: list[dict] = []

            for resolved_target in resolved_targets:
                changeset = self._find_revert_candidate([resolved_target])
                if changeset is None:
                    if self._was_evicted(resolved_target):
                        errors.append({
                            "path": resolved_target,
                            "error": (
                                f"Changeset for '{resolved_target}' was evicted "
                                f"(LRU cap hit). Older edits to this file are "
                                f"no longer revertable."
                            ),
                        })
                    else:
                        errors.append({
                            "path": resolved_target,
                            "error": "No unreverted changeset found",
                        })
                    continue

                error = self._revert_single_changeset(changeset)
                if error is not None:
                    errors.append({
                        "path": resolved_target,
                        "error": error,
                    })
                else:
                    reverted_files.append(changeset.target_path)
                    changeset_ids.append(changeset.changeset_id)

            if not reverted_files and errors:
                return {
                    "status": "error",
                    "error": "Failed to revert all targets",
                    "errors": errors,
                }
            result: dict = {
                "status": "success",
                "reverted_files": reverted_files,
                "changeset_ids": changeset_ids,
            }
            if errors:
                result["errors"] = errors
            return result

        # targets is None — revert single most-recent global changeset
        changeset = self._find_revert_candidate(None)
        if changeset is None:
            return {
                "status": "error",
                "error": "No unreverted changeset found for any file",
            }

        error = self._revert_single_changeset(changeset)
        if error is not None:
            return {
                "status": "error",
                "error": error,
            }

        return {
            "status": "success",
            "reverted_files": [changeset.target_path],
            "changeset_id": changeset.changeset_id,
        }

    def _revert_single_changeset(self, changeset: Changeset) -> str | None:
        """
        Revert a single changeset by restoring its pre-image.

        Acquires the file lock, performs the revert, and marks the changeset
        as reverted inside the lock.

        Args:
            changeset: The Changeset to revert.

        Returns:
            None on success, or an error message string on failure.
        """
        resolved = changeset.target_path
        lock = self._get_file_lock(resolved)

        with lock:
            if changeset.pre_image is None:
                # File did not exist before — delete it
                try:
                    os.remove(resolved)
                    logger.info(
                        "revert_changes: deleted %s (file was newly created, "
                        "changeset %s)",
                        resolved, changeset.changeset_id,
                    )
                except FileNotFoundError:
                    logger.warning(
                        "revert_changes: file already gone %s (changeset %s)",
                        resolved, changeset.changeset_id,
                    )
                except OSError as e:
                    return f"Failed to delete file during revert: {e}"
            else:
                # Restore pre-image via atomic write
                try:
                    self._atomic_write(resolved, changeset.pre_image)
                    logger.info(
                        "revert_changes: restored %s from pre-image "
                        "(changeset %s)",
                        resolved, changeset.changeset_id,
                    )
                except OSError as e:
                    return f"Failed to write pre-image during revert: {e}"

            # Mark changeset as reverted inside the lock (decision C2)
            changeset.reverted = True

        return None

    # ================================================================
    # Internal helpers
    # ================================================================

    def _find_revert_candidate(
        self, resolved_targets: list[str] | None
    ) -> Changeset | None:
        """
        Find the most recent unreverted changeset matching the targets.

        Args:
            resolved_targets: List of absolute file paths to match, or
                None to match any file.

        Returns:
            The matching Changeset, or None if not found.
        """
        # Walk changesets from most recent to oldest
        for cs in reversed(self._changesets):
            if cs.reverted:
                continue
            if resolved_targets is None:
                return cs
            if cs.target_path in resolved_targets:
                return cs
        return None

    @staticmethod
    def _atomic_write(path: str, content: str) -> None:
        """
        Write content to a file atomically using temp file + fsync + rename.

        Creates a temporary file in the same directory as the target, writes
        content, calls fsync, then renames (atomic on POSIX).

        Args:
            path: Absolute path of the target file.
            content: String content to write.

        Raises:
            OSError: If the write or rename operation fails.
        """
        target_dir = os.path.dirname(path)
        fd = None
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=target_dir, prefix=".project_sync_", suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = None  # os.fdopen takes ownership of fd
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, path)
            tmp_path = None  # Rename succeeded, nothing to clean up
        finally:
            # Clean up on failure
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
