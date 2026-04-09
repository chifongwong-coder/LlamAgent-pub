"""
Markdown validator for FS knowledge base documents.

Usage:
    python -m llamagent.tools.md_validator /path/to/knowledge/
    python -m llamagent.tools.md_validator /path/to/knowledge/ --fix
"""

from llamagent.tools.md_validator.validator import check_file, check_directory, fix_file, main

__all__ = ["check_file", "check_directory", "fix_file", "main"]
