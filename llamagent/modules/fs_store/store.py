"""File system storage operations with atomic writes.

Provides a simple directory-scoped file store used by Memory FS and
Retrieval FS backends.  Only depends on Python stdlib (os, logging).
"""

import os
import logging

logger = logging.getLogger(__name__)


class FSStore:
    """File system storage operations with atomic writes."""

    def __init__(self, base_dir: str):
        """Initialize the store and create the base directory if needed.

        Args:
            base_dir: Absolute or relative path to the storage directory.
        """
        self._base_dir = os.path.abspath(base_dir)
        os.makedirs(self._base_dir, exist_ok=True)
        logger.debug("FSStore initialized at %s", self._base_dir)

    def _validate_filename(self, filename: str) -> str:
        """v3.8.1 R7-#28: defense-in-depth check that filename stays
        inside ``self._base_dir``. Pre-fix all callers passed
        sanitized filenames internally, but the surface had no
        boundary check — a bug or misuse passing ``"../etc/passwd"``
        would silently write outside the base dir.

        Returns the abspath on success (used for the actual open call);
        raises ValueError when the path would escape base_dir. We
        compare on ``realpath`` (so macOS ``/var`` → ``/private/var``
        symlinks don't trigger false positives) but return the abspath
        because that's what other callers historically used.
        """
        target = os.path.abspath(os.path.join(self._base_dir, filename))
        base_real = os.path.realpath(self._base_dir)
        # Resolve target through any existing symlinks; for non-existing
        # paths realpath returns the input abspath unchanged on POSIX.
        target_real = os.path.realpath(target)
        if not (target_real.startswith(base_real + os.sep) or target_real == base_real):
            raise ValueError(
                f"FSStore filename {filename!r} escapes base_dir "
                f"{self._base_dir!r}; refusing to write."
            )
        return target

    def write_file(self, filename: str, content: str) -> None:
        """Atomic write: write to a temporary file then rename.

        Args:
            filename: File name (not a path) within the base directory.
            content: Text content to write.
        """
        target = self._validate_filename(filename)
        tmp_path = target + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, target)
            logger.debug("Wrote file %s", filename)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def read_file(self, filename: str) -> str | None:
        """Read file content.

        Args:
            filename: File name within the base directory.

        Returns:
            File content as a string, or None if the file does not exist.
        """
        # v3.8.1 R7-#28: same boundary check as write_file (DiD)
        try:
            path = self._validate_filename(filename)
        except ValueError:
            return None
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            logger.warning("Failed to read %s: %s", filename, e)
            return None

    def list_files(self, extension: str = ".md") -> list[str]:
        """List all files with the given extension in the base directory.

        Args:
            extension: File extension filter (default ".md").

        Returns:
            Sorted list of filenames (not full paths).
        """
        try:
            entries = os.listdir(self._base_dir)
        except OSError as e:
            logger.warning("Failed to list directory %s: %s", self._base_dir, e)
            return []
        return sorted(
            name for name in entries
            if os.path.isfile(os.path.join(self._base_dir, name))
            and name.endswith(extension)
        )

    def delete_file(self, filename: str) -> None:
        """Delete a file. Silently ignores if the file does not exist.

        Args:
            filename: File name within the base directory.
        """
        path = os.path.join(self._base_dir, filename)
        try:
            os.remove(path)
            logger.debug("Deleted file %s", filename)
        except FileNotFoundError:
            pass

    def clear(self) -> None:
        """Delete all .md files in the base directory."""
        for filename in self.list_files(".md"):
            self.delete_file(filename)
        logger.debug("Cleared all .md files in %s", self._base_dir)

    def file_exists(self, filename: str) -> bool:
        """Check if a file exists in the base directory.

        Args:
            filename: File name within the base directory.

        Returns:
            True if the file exists, False otherwise.
        """
        return os.path.isfile(os.path.join(self._base_dir, filename))

    @property
    def base_dir(self) -> str:
        """Return the absolute base directory path."""
        return self._base_dir
