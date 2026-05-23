"""
LlamAgent CLI entry point — router between the Textual TUI and the
legacy ``input()``-based CLI.

Selection rules (in order):

1. ``--legacy`` flag → ``_legacy_cli.legacy_main(args)``.
2. ``ask`` subcommand → ``_legacy_cli.legacy_main(args)`` (the legacy
   path owns one-shot question handling, JSON output, etc.; the TUI
   doesn't support a question subcommand).
3. ``_terminal_supports_tui()`` returns False (non-TTY stdout, dumb
   terminal, etc.) → ``_legacy_cli.legacy_main(args)``.
4. Otherwise → ``cli_tui.run(args)`` (Textual TUI).

Re-exports ``run_cli`` and ``LlamAgentCLI`` from ``_legacy_cli`` so
``llamagent.main`` (``python -m llamagent --mode cli``) keeps importing
them from this module — no caller has to know about the legacy path
relocation.

This module is intentionally small: argparse + selection + re-exports.
Everything else lives in ``_legacy_cli.py`` (legacy CLI) or
``cli_tui/`` (Textual TUI).
"""

import argparse
import os
import sys

# Re-exports for back-compat: callers that imported these from
# ``llamagent.interfaces.cli`` before the C8 split keep working without
# any change. New code should import from the modules below directly.
from llamagent.interfaces._legacy_cli import (  # noqa: F401
    LlamAgentCLI,
    PRESETS,
    build_agent,
    interactive_setup,
    legacy_main,
    run_cli,
)


def _terminal_supports_tui() -> bool:
    """Return True when stdout is a TTY and the terminal claims more
    than dumb capability.

    Used by ``main()`` to auto-fall-back to the legacy CLI when the
    process is being piped, redirected, or run under a "dumb"
    ``TERM`` (e.g. some CI runners, some IDE consoles). The TUI
    depends on an alt-screen + cursor positioning, neither of which
    works without a real terminal.
    """
    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "").lower()
    if not term or term == "dumb":
        return False
    return True


def _create_parser() -> argparse.ArgumentParser:
    """Argparse layout — preserves the legacy CLI's flag surface so any
    saved invocation keeps working, and adds ``--legacy`` so users (and
    scripts) can force the old path.
    """
    parser = argparse.ArgumentParser(
        prog="llamagent-cli",
        description="LlamAgent CLI — Textual TUI by default; --legacy for the old line-mode CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m llamagent.interfaces.cli                     Default: Textual TUI
  python -m llamagent.interfaces.cli --legacy            Force legacy CLI
  python -m llamagent.interfaces.cli --modules tools     Skip setup; legacy path
  python -m llamagent.interfaces.cli --no-modules        Pure chat; legacy path
  python -m llamagent.interfaces.cli ask "How's the weather"   One-shot question (legacy)
  python -m llamagent.interfaces.cli ask "..." --format json   One-shot JSON output

  python -c "import sys; sys.stdout.isatty()" piped/redirected → auto-falls back to legacy.
        """,
    )

    parser.add_argument(
        "--legacy", action="store_true",
        help="Force the legacy input()-based CLI instead of the Textual TUI",
    )
    parser.add_argument(
        "--modules", type=str, default=None,
        help="Comma-separated list of modules (skips interactive setup; legacy path)",
    )
    parser.add_argument(
        "--no-modules", action="store_true",
        help="Load no modules, pure chat mode (skips interactive setup; legacy path)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("chat", help="Enter interactive chat mode (default)")
    ask_parser = subparsers.add_parser("ask", help="Ask a single question (always uses legacy path)")
    ask_parser.add_argument("question", help="The question to ask")
    ask_parser.add_argument(
        "--format", type=str, choices=["text", "json"], default="text",
        dest="output_format",
        help="Output format (default: text)",
    )

    return parser


def _route(args) -> str:
    """Return which path should handle ``args``:
    ``"legacy"`` or ``"tui"``. Pulled out so unit tests can poke the
    selection logic without launching either UI.
    """
    if args.legacy:
        return "legacy"
    # ``ask`` is one-shot question mode — the TUI is an interactive
    # surface, not appropriate for a single piped question. Always
    # route ``ask`` to the legacy path.
    if getattr(args, "command", None) == "ask":
        return "legacy"
    # --modules / --no-modules currently mean "skip setup and run
    # legacy-style". The TUI has its own SetupScreen for module
    # selection. If the user passed these flags they wanted the
    # legacy direct mode.
    if args.modules is not None or args.no_modules:
        return "legacy"
    if not _terminal_supports_tui():
        return "legacy"
    return "tui"


def main():
    """CLI entry point: argparse + route between TUI and legacy CLI."""
    parser = _create_parser()
    args = parser.parse_args()

    path = _route(args)
    if path == "legacy":
        legacy_main(args)
        return

    # TUI path — delegate to ``cli_tui.run`` so this module stays
    # ignorant of LlamAgentTUI / SetupScreen / etc. cli_tui.run
    # accepts a pre-parsed Namespace (TUI ignores ``--legacy``;
    # honours ``--modules`` / ``--no-modules`` as scripted-startup
    # hints).
    from llamagent.interfaces.cli_tui import run as tui_run
    tui_run(args)


if __name__ == "__main__":
    main()
