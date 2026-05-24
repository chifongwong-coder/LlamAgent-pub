"""
LlamAgent entry point.

The "master switch" of the program — decides based on command-line arguments:
1. Which modules to load
2. Which interface to use (CLI / Web / API)
3. Configuration parameters (persona, port, etc.)

Usage:
    python -m llamagent                                    # Textual TUI when terminal supports it, else legacy CLI
    python -m llamagent --mode web                         # Launch Web UI
    python -m llamagent --mode api                         # Launch HTTP API server
    python -m llamagent --modules tools,retrieval,memory    # Specify modules to load (skips TUI SetupScreen)
    python -m llamagent --no-modules                       # Load no modules (pure chat mode)
    python -m llamagent --persona CodeLlama                # Specify persona (skips TUI SetupScreen)
    python -m llamagent --port 9000                        # Specify port (Web/API)

CLI routing: ``python -m llamagent`` and ``python -m llamagent.interfaces.cli``
share the same TUI-vs-legacy dispatch. Piped stdin, dumb terminals, missing
Textual install, or ``ask`` subcommand all auto-fall-back to legacy. For
``--legacy`` / ``ask`` / subcommand variants use the explicit
``python -m llamagent.interfaces.cli`` form.

print() usage: this file is the program entry point. All print() calls
are intentional stdout for banner / error diagnostics / fatal-exit
messages — do NOT replace with logger (most fire before logging is
configured). v3.7.7 cleanup pass categorized library code separately
from CLI entry points.
"""

import argparse
import os
import sys
import traceback

from llamagent.core import LlamAgent, Config, Persona, PersonaManager


# ============================================================
# Available module registry (module name -> import path)
# ============================================================
# All pluggable modules are registered here.
# Dynamic import: only modules specified by the user will be imported,
# preventing startup crashes when optional dependencies (chromadb, mcp, etc.)
# are not installed.

AVAILABLE_MODULES = {
    "resilience": "llamagent.modules.resilience.ResilienceModule",  # First: wraps agent.llm
    "safety": "llamagent.modules.safety.SafetyModule",
    "compression": "llamagent.modules.compression.CompressionModule",
    "persistence": "llamagent.modules.persistence.PersistenceModule",
    "sandbox": "llamagent.modules.sandbox.SandboxModule",       # Before job (provides tool_executor)
    "tools": "llamagent.modules.tools.ToolsModule",             # Before skill (pack reset)
    "job": "llamagent.modules.job.JobModule",                   # After sandbox (hard dependency)
    "retrieval": "llamagent.modules.retrieval.RetrievalModule",
    "memory": "llamagent.modules.memory.MemoryModule",
    "skill": "llamagent.modules.skill.SkillModule",             # After tools (pack activation)
    "reflection": "llamagent.modules.reflection.ReflectionModule",
    "planning": "llamagent.modules.reasoning.PlanningModule",
    "mcp": "llamagent.modules.mcp.MCPModule",
    "child_agent": "llamagent.modules.child_agent.ChildAgentModule",
}


def load_module(name: str):
    """
    Dynamically import and instantiate a module.

    Uses importlib for on-demand loading instead of top-level imports,
    so missing optional dependencies won't cause errors.

    Args:
        name: Module name (e.g., "tools", "retrieval")

    Returns:
        Module instance, or None if loading fails
    """
    if name not in AVAILABLE_MODULES:
        print(f"  [Warning] Unknown module: {name}, available modules: {', '.join(AVAILABLE_MODULES)}")
        return None

    path = AVAILABLE_MODULES[name]
    module_path, class_name = path.rsplit(".", 1)

    try:
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls()
    except ImportError as e:
        print(f"  [Warning] Module {name} dependency not installed: {e}")
        return None
    except Exception as e:
        print(f"  [Warning] Module {name} failed to load: {e}")
        traceback.print_exc()
        return None


def create_agent(
    module_names: list[str] | None = None,
    persona_name: str | None = None,
    config_path: str | None = None,
) -> LlamAgent:
    """
    Create an Agent and load the specified modules.

    This is the shared Agent factory function used by all interfaces (CLI / Web / API).

    Args:
        module_names: List of modules to load.
                      None = load all available modules
                      [] = load no modules (pure chat mode)
        persona_name: Persona name, None uses the default identity

    Returns:
        A configured LlamAgent instance
    """
    config = Config(config_path=config_path)

    # If a persona is specified, try to load it from the persona file
    persona = None
    if persona_name:
        try:
            manager = PersonaManager(config.persona_file)
            # First search by persona_id, then by name
            persona = manager.get(persona_name)
            if not persona:
                # Iterate to find a matching name
                for p in manager.list():
                    if p.name == persona_name:
                        persona = p
                        break
            if persona:
                desc = persona.role_description or persona.name
                print(f"  [Persona] {persona.name}: {desc[:50]}...")
            else:
                print(f"  [Warning] Persona '{persona_name}' not found, using default identity")
        except Exception as e:
            print(f"  [Warning] Failed to load persona: {e}, using default identity")

    agent = LlamAgent(config, persona=persona)

    # Determine the list of modules to load
    if module_names is None:
        module_names = list(AVAILABLE_MODULES.keys())

    print(f"LlamAgent | Model: {agent.config.model}")
    if module_names:
        print(f"Loading modules:")
        for name in module_names:
            mod = load_module(name)
            if mod:
                agent.register_module(mod)
                print(f"  [OK] {name}: {mod.description}")
    else:
        print("Pure chat mode (no modules loaded)")

    print()
    return agent


def _parse_module_names(args) -> list[str] | None:
    """Parse the module list from command-line arguments."""
    if args.no_modules:
        return []
    elif args.modules:
        return [m.strip() for m in args.modules.split(",") if m.strip()]
    else:
        return None  # None = load all


def main():
    """Main entry point: parse command-line arguments and launch the corresponding interface."""
    parser = argparse.ArgumentParser(
        prog="llamagent",
        description="LlamAgent — Modular AI Agent Framework (Textual TUI by default for CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m llamagent                                  TUI by default (legacy fallback when non-TTY / no Textual)
  python -m llamagent --mode web                       Launch Web UI
  python -m llamagent --mode api                       Launch HTTP API
  python -m llamagent --modules tools,retrieval         Load only tools and retrieval
  python -m llamagent --no-modules                     Pure chat mode
  python -m llamagent --persona CodeLlama              Use a specific persona
  python -m llamagent --mode web --port 9000           Specify port

For ``--legacy``, ``ask`` subcommand, or fine-grained CLI flag control:
  python -m llamagent.interfaces.cli --help
        """,
    )

    # Run mode
    parser.add_argument(
        "--mode", type=str, default="cli",
        choices=["cli", "web", "api"],
        help="Run mode: cli (default) / web / api",
    )

    # Module selection
    parser.add_argument(
        "--modules", type=str, default=None,
        help="Comma-separated list of modules, e.g.: tools,retrieval,memory",
    )
    parser.add_argument(
        "--no-modules", action="store_true",
        help="Load no modules (pure chat mode)",
    )

    # Persona
    parser.add_argument(
        "--persona", type=str, default=None,
        help="Specify persona name (must be predefined in the persona file)",
    )

    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: auto-discover llamagent.yaml)",
    )

    # Port (used by Web / API modes)
    parser.add_argument(
        "--port", type=int, default=None,
        help="Listening port for Web UI or API server",
    )

    args = parser.parse_args()
    module_names = _parse_module_names(args)

    # Set config path for downstream Config() calls
    if args.config:
        os.environ["LLAMAGENT_CONFIG"] = args.config

    # Launch the corresponding interface based on mode
    if args.mode == "cli":
        # `python -m llamagent` and `python -m llamagent.interfaces.cli`
        # share the same routing rules — TUI by default when stdin/stdout
        # are a real terminal and Textual is installed; legacy CLI for
        # scripted (--modules / --no-modules), piped stdin, or missing
        # textual. main.py owns --persona / --config so we forward those
        # by pre-building the agent when scripted; otherwise let the
        # TUI's SetupScreen ask interactively.
        from llamagent.interfaces.cli import _route, run_cli

        router_args = argparse.Namespace(
            legacy=False,
            modules=args.modules,
            no_modules=args.no_modules,
            command=None,
            question=None,
            output_format="text",
        )
        scripted = bool(args.modules or args.no_modules or args.persona)

        if _route(router_args) == "tui":
            try:
                from llamagent.interfaces.cli_tui.app import LlamAgentTUI
            except ImportError as exc:
                sys.stderr.write(
                    f"[Note] Textual not available ({exc}); falling back to legacy CLI.\n"
                    f"        Install the TUI with:  pip install textual\n"
                )
                agent = create_agent(module_names, persona_name=args.persona)
                run_cli(agent)
                return
            agent = None
            if scripted:
                agent = create_agent(module_names, persona_name=args.persona)
                app = LlamAgentTUI(agent=agent)
            else:
                app = LlamAgentTUI()
            try:
                app.run()
            finally:
                # Round-18-2 review: shutdown the agent we built so its
                # modules' on_shutdown fires (Chroma close, FTS flush,
                # memory consolidation, child reap). The unscripted path
                # has the App owning the agent — LlamAgentTUI doesn't
                # shut it down on unmount today, so this only covers the
                # scripted case where main.py owns the build. If unset
                # (interactive path), skip — App owns lifecycle.
                if agent is not None:
                    try:
                        agent.shutdown()
                    except Exception as exc:
                        import logging
                        logging.getLogger(__name__).warning(
                            "agent.shutdown() raised at TUI exit: %s", exc
                        )
            notice = getattr(app, "_crash_notice", None)
            if notice:
                sys.stderr.write(notice + "\n")
            sys.exit(getattr(app, "_return_code", 0) or 0)
        else:
            agent = create_agent(module_names, persona_name=args.persona)
            run_cli(agent)

    elif args.mode == "web":
        from llamagent.interfaces.web_ui import create_web_ui, launch_web_ui

        port = args.port or int(os.getenv("WEB_UI_PORT", "7860"))

        try:
            demo = create_web_ui()
        except ImportError as e:
            print(f"Error: {e}")
            sys.exit(1)

        launch_web_ui(demo, port=port)

    elif args.mode == "api":
        from llamagent.interfaces.api_server import launch_api_server

        port = args.port or int(os.getenv("API_PORT", "8000"))
        launch_api_server(
            module_names=module_names,
            persona_name=args.persona,
            port=port,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
