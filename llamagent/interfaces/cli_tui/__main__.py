"""Production launcher for the TUI (C2 entry, C3 SetupScreen integration).

Usage:
    python -m llamagent.interfaces.cli_tui                          # Interactive SetupScreen
    python -m llamagent.interfaces.cli_tui --modules tools,memory   # Scripted: skip SetupScreen
    python -m llamagent.interfaces.cli_tui --no-modules             # Scripted: pure chat

When neither --modules nor --no-modules is given, the App launches
without a preconstructed agent and the SetupScreen (plan §4 C3) runs
inside the TUI to capture persona / preset selection. After the user
clicks Build, ``LlamAgentTUI._on_setup_done`` calls
``build_agent_from_setup`` (exported from this module) to instantiate
the LlamAgent.

When --modules / --no-modules is given, we bypass the SetupScreen and
build the agent up front with default persona — useful for CI / smoke
runs that don't want to drive a modal form.

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

DEFAULT_PERSONA = {
    "persona_name": "LlamAgent",
    "persona_role": "user",
    "persona_desc": "A helpful AI assistant",
}


def build_agent_from_setup(setup: dict):
    """Construct a LlamAgent from a SetupScreen result dict.

    ``setup`` keys (all optional, defaults applied):
    - ``persona_name``: str
    - ``persona_role``: "user" | "admin"
    - ``persona_desc``: str
    - ``modules``: list[str] | None  (None means "load all available")
    """
    from llamagent.core import Config, LlamAgent, Persona
    from llamagent.main import AVAILABLE_MODULES, load_module

    config = Config()
    persona = Persona(
        name=setup.get("persona_name", DEFAULT_PERSONA["persona_name"]),
        role_description=setup.get("persona_desc", DEFAULT_PERSONA["persona_desc"]),
        role=setup.get("persona_role", DEFAULT_PERSONA["persona_role"]),
    )
    agent = LlamAgent(config, persona=persona)

    module_names = setup.get("modules", None)
    if module_names is None:
        module_names = list(AVAILABLE_MODULES.keys())
    for name in module_names:
        mod = load_module(name)
        if mod is not None:
            agent.register_module(mod)

    return agent


def _handle_setup_error(exc: Exception) -> int:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="llamagent.interfaces.cli_tui",
        description="LlamAgent Textual TUI (C3 entry — SetupScreen unless --modules/--no-modules)",
    )
    parser.add_argument(
        "--modules",
        type=str,
        default=None,
        help=f"Comma-separated modules; skips SetupScreen (default scripted: {','.join(DEFAULT_MODULES)})",
    )
    parser.add_argument(
        "--no-modules",
        action="store_true",
        help="Pure chat mode (no modules); skips SetupScreen",
    )
    args = parser.parse_args()

    scripted = args.no_modules or args.modules is not None

    if scripted:
        if args.no_modules:
            modules: list[str] = []
        else:
            modules = [m.strip() for m in args.modules.split(",") if m.strip()]
        setup = {**DEFAULT_PERSONA, "modules": modules}
        try:
            agent = build_agent_from_setup(setup)
        except Exception as exc:
            return _handle_setup_error(exc)
        from llamagent.interfaces.cli_tui.app import LlamAgentTUI
        app = LlamAgentTUI(agent=agent)
    else:
        # Interactive: SetupScreen captures persona + preset inside the TUI
        from llamagent.interfaces.cli_tui.app import LlamAgentTUI
        app = LlamAgentTUI()

    app.run()

    notice = getattr(app, "_crash_notice", None)
    if notice:
        sys.stderr.write(notice + "\n")
    return getattr(app, "_return_code", 0) or 0


if __name__ == "__main__":
    sys.exit(main())
