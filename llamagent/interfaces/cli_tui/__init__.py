"""LlamAgent Textual TUI package.

Public API:
- ``run(args=None) -> int``: launch the TUI. ``args`` is an argparse
  ``Namespace`` (from ``cli.py`` router or ``__main__.py``) with
  ``modules: str|None`` and ``no_modules: bool`` fields; ``None`` means
  "fall back to package-level argparse" so ``python -m
  llamagent.interfaces.cli_tui`` keeps working as a direct entry.
- ``build_agent_from_setup(setup: dict) -> LlamAgent``: shared agent
  factory used by both the scripted (--modules) launch path and the
  interactive SetupScreen path. Mirrors legacy ``cli.build_agent``
  ordering exactly so the two entry points produce equivalent agents
  for the same setup dict.

Package layout: see ``docs/cli-transfer-plan.md §3.3`` for the full
file tree; the public surface stays minimal so external callers
(``cli.py`` router, smoke runner, tests) don't need to know which
sub-module owns the App / widgets / handlers.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llamagent.core.agent import LlamAgent


__all__ = ["run", "build_agent_from_setup", "DEFAULT_PERSONA", "DEFAULT_MODULES"]


DEFAULT_MODULES = ["resilience", "safety", "compression", "tools"]

DEFAULT_PERSONA = {
    "persona_name": "LlamAgent",
    "persona_role": "user",
    "persona_desc": "A helpful AI assistant",
}


def build_agent_from_setup(setup: dict) -> "LlamAgent":
    """Construct a LlamAgent from a SetupScreen result dict.

    Mirrors legacy ``cli.build_agent`` ordering exactly so the two entry
    points produce equivalent agents for the same setup dict:
      1. Resolve module list  (None → all available)
      2. apply_presets(config, modules)  — flips persistence_enabled
         etc. based on which modules will be loaded; MUST run before
         LlamAgent(__init__) reads the config (round-10 BLOCKER C-B1).
      3. LlamAgent(config, persona)
      4. register_module for each name

    ``setup`` keys (all optional, defaults applied):
    - ``persona_name``: str
    - ``persona_role``: "user" | "admin"
    - ``persona_desc``: str
    - ``modules``: list[str] | None  (None means "load all available")
    """
    from llamagent.core import Config, LlamAgent, Persona
    from llamagent.interfaces.presets import apply_presets
    from llamagent.main import AVAILABLE_MODULES, load_module

    config = Config()
    persona = Persona(
        name=setup.get("persona_name", DEFAULT_PERSONA["persona_name"]),
        role_description=setup.get("persona_desc", DEFAULT_PERSONA["persona_desc"]),
        role=setup.get("persona_role", DEFAULT_PERSONA["persona_role"]),
    )

    module_names = setup.get("modules", None)
    if module_names is None:
        module_names = list(AVAILABLE_MODULES.keys())

    # Round-10 BLOCKER C-B1 + HIGH C-H1: apply_presets BEFORE LlamAgent
    # so config.persistence_enabled / memory_backend / etc. land
    # before agent init reads them. Same ordering as cli.build_agent.
    apply_presets(config, module_names)

    agent = LlamAgent(config, persona=persona)
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


def _self_parse() -> argparse.Namespace:
    """Argparse used only when ``run()`` is called without args (i.e.
    direct ``python -m llamagent.interfaces.cli_tui`` invocation).
    Field shape matches the parent ``cli.py`` parser so ``run(args)`` can
    accept either source uniformly.
    """
    parser = argparse.ArgumentParser(
        prog="llamagent.interfaces.cli_tui",
        description="LlamAgent Textual TUI (C3 entry — SetupScreen unless --modules/--no-modules)",
    )
    # Round-10 HIGH C-H2: --modules and --no-modules are mutually exclusive.
    # The mutex group surfaces argparse's own clean error message when
    # both are passed, instead of silently letting one win.
    scripted_group = parser.add_mutually_exclusive_group()
    scripted_group.add_argument(
        "--modules",
        type=str,
        default=None,
        help=f"Comma-separated modules; skips SetupScreen (default scripted: {','.join(DEFAULT_MODULES)})",
    )
    scripted_group.add_argument(
        "--no-modules",
        action="store_true",
        help="Pure chat mode (no modules); skips SetupScreen",
    )
    return parser.parse_args()


def run(args: "argparse.Namespace | None" = None) -> int:
    """Launch the Textual TUI.

    ``args`` shape (when passed by ``cli.py`` router):
    - ``args.modules: str | None``    — comma-separated module names
    - ``args.no_modules: bool``       — pure chat mode

    Other fields on ``args`` (e.g. ``args.legacy``, ``args.command``)
    are ignored — the router has already decided we're the TUI path.

    Behaviour:
    - ``args.no_modules`` or ``args.modules is not None`` → scripted
      mode: build the agent up front from defaults + the module list,
      then launch the App with the agent attached (skips SetupScreen).
    - Otherwise → interactive mode: launch the App without an agent;
      the App pushes SetupScreen on mount and builds the agent through
      the modal's dismiss callback.

    Returns the App's return code (0 success, 1 crash, 2 setup error).
    """
    if args is None:
        args = _self_parse()

    scripted = args.no_modules or args.modules is not None

    # Lazy import: pulling Textual / LlamAgentTUI on package import would
    # force every caller (incl. ``cli.py`` router that may end up routing
    # to legacy) to pay the Textual import cost. Defer to the launch.
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI

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
        app = LlamAgentTUI(agent=agent)
    else:
        # Interactive: SetupScreen captures persona + preset inside the TUI
        app = LlamAgentTUI()

    app.run()

    notice = getattr(app, "_crash_notice", None)
    if notice:
        sys.stderr.write(notice + "\n")
    return getattr(app, "_return_code", 0) or 0
