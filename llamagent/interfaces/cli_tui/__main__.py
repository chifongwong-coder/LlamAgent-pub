"""Production-style launcher for the TUI (C2 entry).

Usage:
    python -m llamagent.interfaces.cli_tui [--modules m1,m2,...] [--no-modules]

Builds a minimal LlamAgent with default Config + Persona, registers the
modules selected via flags (default: same lean set as the dev YAML —
resilience / safety / compression / tools), and hands the agent to
LlamAgentTUI. C3 will replace this with the proper SetupScreen modal
flow; for C2 verification we want the shortest path from CLI to a
working TUI chat with a real agent.

Setup-error mitigation (round-8 HIGH-2): _build_agent runs BEFORE
LlamAgentTUI._handle_exception is in place (App isn't mounted yet),
so any exception from Config / LlamAgent / load_module / on_attach
would dump a Rich-formatted traceback straight to stderr → host
scrollback (same nanozone attack surface as the in-TUI Q6 crash).
main() wraps the build in try/except: full traceback goes to
~/.llamagent/cli_tui.log, only a one-line ASCII notice on stderr.

Smoke / KPI path remains via ``cli_tui.smoke`` — that module never
constructs a real agent and stays mock-only.
"""
import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path


DEFAULT_MODULES = ["resilience", "safety", "compression", "tools"]


def _build_agent(module_names: list[str] | None):
    """Construct a default LlamAgent + load the requested modules."""
    from llamagent.core import Config, LlamAgent, Persona
    from llamagent.main import load_module

    config = Config()
    persona = Persona(
        name="LlamAgent",
        role_description="A helpful AI assistant",
        role="user",
    )
    agent = LlamAgent(config, persona=persona)

    if module_names is None:
        module_names = DEFAULT_MODULES
    for name in module_names:
        mod = load_module(name)
        if mod is not None:
            agent.register_module(mod)

    return agent


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="llamagent.interfaces.cli_tui",
        description="LlamAgent Textual TUI (C2 entry)",
    )
    parser.add_argument(
        "--modules",
        type=str,
        default=None,
        help=f"Comma-separated module names (default: {','.join(DEFAULT_MODULES)})",
    )
    parser.add_argument(
        "--no-modules",
        action="store_true",
        help="Pure chat mode — no modules loaded",
    )
    args = parser.parse_args()

    if args.no_modules:
        modules: list[str] | None = []
    elif args.modules:
        modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    else:
        modules = None  # → DEFAULT_MODULES in _build_agent

    # Round-8 HIGH-2: agent construction runs before App.run() so
    # LlamAgentTUI._handle_exception isn't installed yet. Any exception
    # here would dump a Rich-formatted traceback to stderr → host
    # scrollback (same Q6 attack surface). Write the full traceback to
    # ~/.llamagent/cli_tui.log; emit a single ASCII notice line.
    try:
        agent = _build_agent(modules)
    except Exception as exc:
        log_path = Path.home() / ".llamagent" / "cli_tui.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as f:
                f.write(f"\n=== {datetime.now().isoformat()} setup error ===\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
        sys.stderr.write(
            f"[llamagent setup error] {type(exc).__name__}: {exc}\n"
            f"  full traceback: {log_path}\n"
        )
        return 2

    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    app = LlamAgentTUI(agent=agent)
    app.run()

    # Crash-notice path (Q6 mitigation, plan v11 §2.2)
    notice = getattr(app, "_crash_notice", None)
    if notice:
        sys.stderr.write(notice + "\n")

    return getattr(app, "_return_code", 0) or 0


if __name__ == "__main__":
    sys.exit(main())
