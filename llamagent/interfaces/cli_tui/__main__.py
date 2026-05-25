"""Direct entry point for ``python -m llamagent.interfaces.cli_tui``.

All real launch logic lives in ``cli_tui/__init__.py::run``; this module
is a thin wrapper so the ``python -m`` invocation keeps working
identically to the C0-C7 era while ``cli.py`` and tests share the same
``run()`` function as their entry.
"""
import sys

from llamagent.interfaces.cli_tui import run


if __name__ == "__main__":
    sys.exit(run())
