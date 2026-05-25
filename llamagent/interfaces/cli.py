"""
LlamAgent CLI entry point — router between the Textual TUI and the
legacy ``input()``-based CLI.

Selection rules (in order):

1. ``--legacy`` flag → ``_legacy_cli.legacy_main(args)``.
2. ``ask`` subcommand → ``_legacy_cli.legacy_main(args)`` (the legacy
   path owns one-shot question handling, JSON output, etc.; the TUI
   doesn't support a question subcommand).
3. ``--modules`` / ``--no-modules`` (legacy "direct mode" flags) →
   ``_legacy_cli.legacy_main(args)``.
4. ``_terminal_supports_tui()`` returns False (non-TTY stdin/stdout,
   dumb terminal, etc.) → ``_legacy_cli.legacy_main(args)``.
5. Otherwise → ``cli_tui.run(args)`` (Textual TUI). This is also the
   path taken by the ``chat`` subcommand — pre-C8 ``chat`` meant
   "enter the legacy interactive loop", post-C8 it means "enter the
   TUI" (the user-facing semantic is the same: open a chat surface).

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
    """Return True when both stdin and stdout are TTYs and the terminal
    claims more than dumb capability.

    Used by ``main()`` to auto-fall-back to the legacy CLI when the
    process is being piped, redirected, or run under a "dumb" ``TERM``
    (e.g. some CI runners, some IDE consoles). The TUI depends on an
    alt-screen + cursor positioning *and* on keyboard events, so both
    stdin and stdout must be a real terminal — round-16 Rev A M1
    caught that ``echo q | python -m ...`` would otherwise launch the
    TUI with a piped stdin, leaving it frozen with no way to type.
    """
    if not sys.stdout.isatty() or not sys.stdin.isatty():
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
  python -m llamagent.interfaces.cli chat                Same as default (TUI)
  python -m llamagent.interfaces.cli --legacy            Force legacy CLI
  python -m llamagent.interfaces.cli --modules tools     Skip setup; legacy path
  python -m llamagent.interfaces.cli --no-modules        Pure chat; legacy path
  python -m llamagent.interfaces.cli ask "How's the weather"   One-shot question (legacy)
  python -m llamagent.interfaces.cli ask "..." --format json   One-shot JSON output

  Piped or non-TTY (echo q | python -m ...) auto-falls back to legacy.
        """,
    )

    parser.add_argument(
        "--legacy", action="store_true",
        help="Force the legacy input()-based CLI instead of the Textual TUI",
    )
    # Round-16 Rev C L1 — mutex group keeps this parser symmetric with
    # ``cli_tui._self_parse``; argparse emits its own clean error when
    # both are passed instead of silently letting --no-modules win.
    scripted_group = parser.add_mutually_exclusive_group()
    scripted_group.add_argument(
        "--modules", type=str, default=None,
        help="Comma-separated list of modules (skips interactive setup; legacy path)",
    )
    scripted_group.add_argument(
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
    #
    # ImportError fallback: if Textual isn't installed in this Python
    # environment (e.g. user picked the wrong interpreter; default
    # python instead of llamagent_env), running into the TUI would
    # crash with a confusing module-not-found traceback. Fall back to
    # the legacy CLI with a one-line install hint instead — the
    # legacy path needs only rich which is already a hard dep.
    try:
        from llamagent.interfaces.cli_tui import run as tui_run
    except ImportError as exc:
        sys.stderr.write(
            f"[Note] Textual not available ({exc}); falling back to legacy CLI.\n"
            f"        Install the TUI with:  pip install -e \".[tui]\"\n"
        )
        legacy_main(args)
        return
    tui_run(args)


if __name__ == "__main__":
    main()
