"""SnapshotService — coarse-grained safety net for CI / auto_approve mode.

v3.8.2 A1: relocated from ``modules/tools/snapshot.py`` to break the
``core/agent.py → modules/tools/snapshot`` reverse import. SnapshotService
is now a value-object-driven helper at the core layer; agent.py constructs
``SnapshotConfig`` from its config snapshot fields and passes it in.
SnapshotService no longer holds an agent ref — pure data in / pure work out
satisfies P5 single-direction deps.

Captured **eagerly during agent init** (single trigger point) — see
``LlamAgent.__init__`` calling ``ensure_snapshot()`` at the end of
construction. Complements the per-operation Changeset journal (which only
covers typed write tools): even shell-mediated changes via the ``command``
tool are recoverable as long as the snapshot is intact.

Scope:
- Copies ``write_root`` once.
- **Excludes ``playground_dir``** when it lives under ``write_root`` —
  playground is framework ephemeral scratch (tool persistence, child
  shared area, sandbox subprocess outputs). Snapshotting it inflates
  disk and adds no recovery value because the model doesn't put
  intentional state there.

Disabled state (default in interactive mode):
- ``ensure_taken`` is a no-op.
- Auto-enabled when ``config.auto_approve == True`` to give CI users a
  safety net automatically (this is what ``SnapshotConfig.enabled``
  encodes — agent.py OR's the two flags before constructing the config).

LRU cleanup:
- Older session snapshots are pruned on capture (not on shutdown) so
  retention is enforced even if the agent crashes.
"""

from __future__ import annotations

import json as _json
import logging
import os
import shutil
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


_DEFAULT_GITIGNORE_FALLBACK = (
    ".git", ".hg", ".svn",
    "__pycache__", ".pytest_cache",
    "node_modules", ".venv", "venv", "env",
    "dist", "build",
    "*.pyc", "*.pyo",
)


def _read_gitignore_patterns(root: str) -> list[str]:
    """Best-effort parse of a top-level .gitignore. Returns relative
    glob patterns (no fancy negation handling)."""
    path = os.path.join(root, ".gitignore")
    out: list[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("!"):
                    continue
                out.append(line.rstrip("/"))
    except OSError:
        pass
    return out


@dataclass
class SnapshotConfig:
    """Pure value object — every field SnapshotService needs to capture
    a snapshot. Constructed by ``LlamAgent.ensure_snapshot`` from the
    agent's resolved write_root / playground_dir / config snapshot
    fields. SnapshotService receives only these primitives, no agent ref.
    """

    enabled: bool          # config.snapshot_enabled OR config.auto_approve
    max_size_mb: int       # config.snapshot_max_size_mb
    snapshot_dir: str      # config.snapshot_dir or "" (default = parent of write_root)
    ignore_gitignore: bool # config.snapshot_ignore_gitignore
    retention_count: int   # config.snapshot_retention_count
    session_id: str        # short id from ScratchService scratch_id, or pid fallback


class SnapshotService:
    """Per-agent lazy snapshot of write_root.

    Constructed via:
        SnapshotService(write_root=..., playground_dir=..., config=SnapshotConfig(...))
    """

    def __init__(
        self,
        *,
        write_root: str,
        playground_dir: str,
        config: SnapshotConfig,
    ) -> None:
        self._write_root = write_root
        self._playground_dir = playground_dir
        self._config = config
        self._taken: bool = False
        self._snapshot_dir: str | None = None
        self._manifest_path: str | None = None

    def is_enabled(self) -> bool:
        return bool(self._config.enabled)

    def ensure_taken(self) -> str | None:
        """Capture the snapshot if not yet captured AND enabled.
        Returns the snapshot directory path on success, None otherwise.
        Idempotent — subsequent calls return the cached path.
        """
        if self._taken:
            return self._snapshot_dir
        if not self.is_enabled():
            return None
        try:
            return self._capture()
        except Exception as e:
            logger.error("snapshot capture failed: %s", e)
            self._taken = True  # don't keep trying every call
            return None

    def _capture(self) -> str | None:
        write_root = os.path.realpath(self._write_root)
        if not os.path.isdir(write_root):
            logger.warning(
                "snapshot: write_root %r does not exist; nothing to capture",
                write_root,
            )
            self._taken = True
            return None

        size_bytes, file_count = self._estimate_size(write_root)
        max_bytes = self._config.max_size_mb * 1024 * 1024
        if max_bytes > 0 and size_bytes > max_bytes:
            logger.error(
                "snapshot: write_root size %.1f MB exceeds snapshot.max_size_mb (%d MB). "
                "Set a smaller config.edit_root or raise the cap. Snapshot disabled "
                "for this session.",
                size_bytes / (1024 * 1024), max_bytes // (1024 * 1024),
            )
            self._taken = True
            return None

        # Snapshot location:
        # - Custom: config.snapshot_dir (when explicitly set, useful for
        #   large project trees or shared SSD storage)
        # - Default: alongside the project at <parent>/.llamagent_snapshots/
        custom = self._config.snapshot_dir or ""
        if custom:
            base = os.path.realpath(custom)
        else:
            parent = os.path.dirname(write_root)
            # Edge case: write_root is "/" (extremely unlikely in practice
            # but be defensive — falls back to data dir).
            if not parent or parent == os.sep:
                parent = os.path.join(os.path.expanduser("~"), ".llamagent")
            base = os.path.join(parent, ".llamagent_snapshots")
        # Add a 4-char random suffix to defeat same-second collisions
        # when multiple agents start concurrently against the same base.
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_id = self._config.session_id
        rand = "%04x" % (os.getpid() & 0xFFFF)
        snap_dir = os.path.join(base, f"{ts}_{session_id}_{rand}")
        os.makedirs(snap_dir, exist_ok=True)

        ignore = self._build_ignore_callable(write_root)
        target = os.path.join(snap_dir, "tree")
        shutil.copytree(write_root, target, ignore=ignore, symlinks=False, dirs_exist_ok=True)

        manifest = {
            "version": 1,
            "session_id": session_id,
            "write_root": write_root,
            "captured_at": time.time(),
            "size_bytes": size_bytes,
            "file_count": file_count,
            "ignore_gitignore": self._config.ignore_gitignore,
        }
        manifest_path = os.path.join(snap_dir, "MANIFEST.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            _json.dump(manifest, f, indent=2, ensure_ascii=False)

        self._taken = True
        self._snapshot_dir = snap_dir
        self._manifest_path = manifest_path
        logger.info(
            "snapshot captured: %s (%d files, %.1f MB)",
            snap_dir, file_count, size_bytes / (1024 * 1024),
        )
        self._enforce_retention(base)
        return snap_dir

    def _estimate_size(self, root: str) -> tuple[int, int]:
        size = 0
        count = 0
        ignore_dirs = self._gitignore_dirs(root)
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune ignored directories early.
            dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
            for fn in filenames:
                try:
                    st = os.lstat(os.path.join(dirpath, fn))
                    size += st.st_size
                    count += 1
                except OSError:
                    continue
        return size, count

    def _gitignore_dirs(self, root: str) -> set[str]:
        if not self._config.ignore_gitignore:
            return set()
        names = set(_DEFAULT_GITIGNORE_FALLBACK)
        for p in _read_gitignore_patterns(root):
            # Only consume top-level directory names (simple subset).
            if "/" not in p and "*" not in p and "?" not in p:
                names.add(p)
        return names

    def _build_ignore_callable(self, root: str):
        """Returns an ignore function for shutil.copytree honoring the
        gitignore directories AND excluding the playground subdir
        (playground is ephemeral framework scratch, never snapshot it)."""
        ignore_dirs = self._gitignore_dirs(root)

        # Exclude playground subtree if it lives under write_root. When
        # edit_root narrows write_root to a sibling of playground, the
        # abspath comparison short-circuits naturally.
        playground = os.path.realpath(self._playground_dir or "")
        root_real = os.path.realpath(root)
        playground_under_root = bool(
            playground
            and (playground == root_real or playground.startswith(root_real + os.sep))
        )

        if not ignore_dirs and not playground_under_root:
            return None

        def _ignore(dirpath: str, names: list[str]) -> list[str]:
            skipped = [n for n in names if n in ignore_dirs]
            if playground_under_root:
                # Skip the exact playground basename when we're in its parent.
                here = os.path.realpath(dirpath)
                playground_parent = os.path.dirname(playground)
                if here == playground_parent:
                    skipped.append(os.path.basename(playground))
            return skipped

        return _ignore

    def _enforce_retention(self, base: str) -> None:
        """Keep only the most-recent N snapshots under ``base``."""
        keep = self._config.retention_count
        if keep <= 0:
            return
        try:
            entries = []
            for name in os.listdir(base):
                full = os.path.join(base, name)
                if not os.path.isdir(full):
                    continue
                try:
                    entries.append((os.path.getmtime(full), full))
                except OSError:
                    continue
            entries.sort(reverse=True)  # newest first
            for _, full in entries[keep:]:
                try:
                    shutil.rmtree(full, ignore_errors=True)
                    logger.info("snapshot retention: removed old snapshot %s", full)
                except OSError as e:
                    logger.warning("snapshot retention: rmtree failed for %s: %s", full, e)
        except OSError as e:
            logger.warning("snapshot retention: listing %r failed: %s", base, e)
