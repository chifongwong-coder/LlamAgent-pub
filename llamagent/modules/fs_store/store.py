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

    def write_file(self, filename: str, content: str) -> None:
        """Atomic write: write to a temporary file then rename.

        Args:
            filename: File name (not a path) within the base directory.
            content: Text content to write.
        """
        target = os.path.join(self._base_dir, filename)
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
        path = os.path.join(self._base_dir, filename)
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
