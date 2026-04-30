"""
ProjectSyncService: project-write tracking and revert for the v3.3 tool system.

Plain service class (not a Module), instantiated by
ToolsModule.on_attach(). Owns the in-memory Changeset journal that
makes every typed project write reversible via revert_changes.

Provides:
- Atomic patch application (apply_patch, preview_patch).
- Changeset registration for the five typed write surfaces:
    * record_write_changeset(target, pre_image)  — write_files
    * apply_patch (registers internally)          — apply_patch
    * record_delete_changeset(target, pre_image)  — delete_path
    * record_move_changeset(src, dst)             — move_path
    * record_copy_changeset(src, dst)             — copy_path
- Changeset-based revert dispatching by action type
  (create / overwrite / patch / delete / move / copy).
- LRU eviction (count + byte cap) with an evicted-paths ledger so
  revert can surface a precise error when the target's changeset
  was dropped under memory pressure.

All project writes go through `_atomic_write` (temp file + fsync +
os.rename, POSIX-atomic). Pre-image snapshots are kept as text;
binary files are refused upstream by write_files / delete_path so
the journal never has to encode bytes.

Key v3.3 decisions baked into this layer:
- All typed writes inside write_root are revertable; playground
  writes are ephemeral and never enter the journal (classify_write
  decides at the tool layer).
- delete_path accepts only single files (MF2): each file gets its
  own delete-action Changeset with the original bytes as pre_image.
- Snapshot (D7) is the cross-changeset safety net for command-tool
  shell mutations that don't go through this service.
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
    from llamagent.modules.tools.scratch import ScratchService

logger = logging.getLogger(__name__)


# ======================================================================
# Changeset data class
# ======================================================================


@dataclass
class Changeset:
    """
    Record of a single project file modification.

    Each successful project write creates one Changeset. The fields
    available depend on the action type:

      action="patch" / "overwrite" / "create":
        target_path holds the file path. pre_image holds the original
        bytes (None for "create"). Inverse: write pre_image back, or
        unlink the file for "create".

      action="delete":
        target_path holds the deleted file's path. pre_image holds the
        bytes that were removed. Inverse: write pre_image back to
        target_path. delete_path tool only accepts files (not dirs),
        so pre_image is always a scalar string.

      action="move":
        src and dst hold the source and destination paths. pre_image is
        None. Inverse: os.rename(dst, src).

      action="copy":
        src holds the source (untouched), dst holds the new copy.
        pre_image is None. Inverse: os.unlink(dst) — src is unchanged
        so no restoration is needed there.

    Attributes:
        changeset_id: Unique identifier for this changeset.
        target_path: Absolute path to the modified project file (for
            patch/overwrite/create/delete; "" or unused for move/copy).
        pre_image: Pre-modification content for patch/overwrite/delete.
            None for create/move/copy.
        ops: Structured log of operations applied (for audit/preview).
        timestamp: Unix timestamp of when the changeset was created.
        reverted: Whether this changeset has been reverted.
        action: Action type — one of "patch" / "overwrite" / "create"
            / "delete" / "move" / "copy". Default "patch" for backwards
            compatibility with pre-13a callers.
        src: Source path for move/copy (None otherwise).
        dst: Destination path for move/copy (None otherwise).
    """

    changeset_id: str
    target_path: str
    pre_image: str | None
    ops: list[dict]
    timestamp: float
    reverted: bool = False
    action: str = "patch"
    src: str | None = None
    dst: str | None = None


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
        scratch_service: The ScratchService for per-session scratch path resolution.
    """

    def __init__(
        self, agent: LlamAgent, scratch_service: ScratchService
    ) -> None:
        """
        Initialize the ProjectSyncService.

        Args:
            agent: The LlamAgent instance that owns this service.
            scratch_service: The ScratchService used for per-session
                scratch path resolution (used by changeset/write helpers).
        """
        self.agent = agent
        self.scratch_service = scratch_service
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
        """Approximate memory footprint of a Changeset.

        v3.3: move/copy actions carry pre_image=None. If we counted only
        pre_image bytes, a long session of moves/copies could swamp the
        in-memory list without ever triggering byte-cap eviction. Add the
        path strings as a floor so move/copy entries take real budget.
        """
        base = len(cs.pre_image) if cs.pre_image else 0
        path_overhead = (
            len(cs.target_path or "")
            + len(cs.src or "")
            + len(cs.dst or "")
        )
        return base + path_overhead

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

        # Pass 1: drop reverted (tombstone) changesets oldest-first,
        # one at a time, re-checking _over() after each drop. Without
        # the per-drop re-check we'd over-evict tombstones whenever the
        # backlog has many of them.
        if _over():
            i = 0
            while i < len(self._changesets) and _over():
                cs = self._changesets[i]
                if cs.reverted:
                    self._evicted_paths.add(cs.target_path)
                    logger.info(
                        "changeset evicted (reverted/tombstone): %s",
                        cs.target_path,
                    )
                    self._changesets.pop(i)
                    # Don't advance i — popped index now points at next.
                else:
                    i += 1

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
        by the v3.3 path-classification logic when the resolved target is
        under ``playground_dir`` (edits there are ephemeral scratch).

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
        # v3.3 D7: snapshot is captured eagerly at agent init, not here.
        # See LlamAgent.__init__ -> ensure_snapshot().
        resolved = self.resolve_project_path(target)
        lock = self._get_file_lock(resolved)

        with lock:
            # Read current file content
            try:
                with open(resolved, "r", encoding="utf-8") as f:
                    content = f.read()
            except FileNotFoundError:
                # v3.3 §3.5: unified file-not-found wording.
                return {
                    "status": "error",
                    "target": resolved,
                    "error": f"file not found: '{target}'",
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
                action="patch",
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
        # Distinguish "create" (no prior content) from "overwrite":
        # both restore by either deleting (create) or writing back
        # pre_image (overwrite). The dataclass-level ``action`` lets
        # revert_changes dispatch without inspecting ``ops``.
        action = "create" if pre_image is None else "overwrite"
        changeset = Changeset(
            changeset_id=uuid.uuid4().hex,
            target_path=resolved,
            pre_image=pre_image,
            ops=[{
                "action": "write_file",
                "had_prior_content": pre_image is not None,
            }],
            timestamp=time.time(),
            action=action,
        )
        self._changesets.append(changeset)
        self._enforce_changeset_caps()
        return changeset.changeset_id

    # ================================================================
    # v3.3 commit-13a: changeset registration for path-fallback tools
    # ================================================================

    def record_delete_changeset(self, target: str, pre_image: str) -> str:
        """Record a delete: target_path holds the deleted file's path,
        pre_image holds its prior bytes. Inverse rewrites pre_image."""
        resolved = os.path.realpath(target)
        changeset = Changeset(
            changeset_id=uuid.uuid4().hex,
            target_path=resolved,
            pre_image=pre_image,
            ops=[{"action": "delete_file"}],
            timestamp=time.time(),
            action="delete",
        )
        self._changesets.append(changeset)
        self._enforce_changeset_caps()
        return changeset.changeset_id

    def record_move_changeset(self, src: str, dst: str) -> str:
        """Record a move: inverse is os.rename(dst, src)."""
        rsrc = os.path.realpath(src)
        rdst = os.path.realpath(dst)
        changeset = Changeset(
            changeset_id=uuid.uuid4().hex,
            target_path=rdst,
            pre_image=None,
            ops=[{"action": "move_file"}],
            timestamp=time.time(),
            action="move",
            src=rsrc,
            dst=rdst,
        )
        self._changesets.append(changeset)
        self._enforce_changeset_caps()
        return changeset.changeset_id

    def record_rmdir_changeset(self, target: str) -> str:
        """Record an empty-directory removal: inverse is os.mkdir(target).

        v3.5: delete_path now accepts empty directories. Since the directory
        was empty, no file pre_images need to be captured — the inverse is
        a single mkdir. action="rmdir"; pre_image=None; ops records the
        rmdir signature so revert can reconstruct.
        """
        resolved = os.path.realpath(target)
        changeset = Changeset(
            changeset_id=uuid.uuid4().hex,
            target_path=resolved,
            pre_image=None,
            ops=[{"action": "rmdir"}],
            timestamp=time.time(),
            action="rmdir",
        )
        self._changesets.append(changeset)
        self._enforce_changeset_caps()
        return changeset.changeset_id

    def record_copy_changeset(self, src: str, dst: str) -> str:
        """Record a copy: inverse is os.unlink(dst). src is unchanged."""
        rsrc = os.path.realpath(src)
        rdst = os.path.realpath(dst)
        changeset = Changeset(
            changeset_id=uuid.uuid4().hex,
            target_path=rdst,
            pre_image=None,
            ops=[{"action": "copy_file"}],
            timestamp=time.time(),
            action="copy",
            src=rsrc,
            dst=rdst,
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
            # v3.3 §3.5: unified file-not-found wording.
            return {
                "status": "error",
                "target": resolved,
                "error": f"file not found: '{target}'",
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
        Revert a single changeset, dispatching by action type.

        Action handlers (v3.3 commit-13a):

        - ``"create"``: pre_image is None → unlink target_path.
        - ``"overwrite"`` / ``"patch"``: rewrite pre_image to target_path.
        - ``"delete"``: rewrite pre_image to target_path (resurrect file).
        - ``"move"``: rename dst back to src.
        - ``"copy"``: unlink dst (src is unchanged).

        Acquires the file lock, performs the revert, and marks the
        changeset as reverted inside the lock.

        Returns:
            None on success, or an error message string on failure.
        """
        # Lock by primary path: target_path for non-move/copy, dst for
        # move/copy (the path that has new content).
        lock_path = (
            changeset.dst
            if changeset.action in ("move", "copy") and changeset.dst
            else changeset.target_path
        )
        lock = self._get_file_lock(lock_path)

        with lock:
            try:
                self._dispatch_revert(changeset)
            except OSError as e:
                return f"Failed to revert ({changeset.action}): {e}"

            # Mark changeset as reverted inside the lock (decision C2)
            changeset.reverted = True

        return None

    def _dispatch_revert(self, changeset: Changeset) -> None:
        """Execute the inverse operation for a Changeset.

        May raise OSError; the caller (``_revert_single_changeset``)
        wraps that into a user-visible error string.
        """
        action = changeset.action
        cid = changeset.changeset_id

        if action in ("patch", "overwrite", "delete"):
            # Rewrite pre_image to target_path.
            if changeset.pre_image is None:
                # Defensive: shouldn't happen for these actions, but
                # tolerate by treating as "create"-style cleanup.
                self._unlink_if_exists(changeset.target_path, cid)
                return
            self._atomic_write(changeset.target_path, changeset.pre_image)
            logger.info(
                "revert_changes: restored %s from pre-image (action=%s, changeset %s)",
                changeset.target_path, action, cid,
            )

        elif action == "create":
            # File didn't exist before — unlink whatever's there now.
            self._unlink_if_exists(changeset.target_path, cid)

        elif action == "move":
            # Inverse: rename dst back to src.
            if not (changeset.src and changeset.dst):
                raise OSError("move changeset missing src/dst")
            os.rename(changeset.dst, changeset.src)
            logger.info(
                "revert_changes: moved back %s -> %s (changeset %s)",
                changeset.dst, changeset.src, cid,
            )

        elif action == "copy":
            # Inverse: unlink the new copy. src is unchanged.
            if not changeset.dst:
                raise OSError("copy changeset missing dst")
            self._unlink_if_exists(changeset.dst, cid)

        elif action == "rmdir":
            # Inverse: recreate the empty directory.
            os.makedirs(changeset.target_path, exist_ok=True)
            logger.info(
                "revert_changes: recreated empty directory %s (changeset %s)",
                changeset.target_path, cid,
            )

        else:
            raise OSError(f"unknown changeset action: {action}")

    def _unlink_if_exists(self, path: str, cid: str) -> None:
        """Best-effort unlink; not finding the file is logged but not raised."""
        try:
            os.remove(path)
            logger.info(
                "revert_changes: removed %s (changeset %s)", path, cid,
            )
        except FileNotFoundError:
            logger.warning(
                "revert_changes: file already gone %s (changeset %s)",
                path, cid,
            )

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
        """Write text content to a file atomically using temp file + fsync + rename.

        v3.3: text-mode only. Binary writes never reach this path because
        write_files refuses to overwrite binaries via text mode and
        delete_path refuses to remove binary files (changeset can't
        record their bytes for revert). If binary revert support is ever
        added, this is the place to grow a `mode` parameter.

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
