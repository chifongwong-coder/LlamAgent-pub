"""
LlamAgent entry point.

The "master switch" of the program — decides based on command-line arguments:
1. Which modules to load
2. Which interface to use (CLI / Web / API)
3. Configuration parameters (persona, port, etc.)

Usage:
    python -m llamagent                                    # Default CLI interactive chat
    python -m llamagent --mode web                         # Launch Web UI
    python -m llamagent --mode api                         # Launch HTTP API server
    python -m llamagent --modules tools,rag,memory         # Specify modules to load
    python -m llamagent --no-modules                       # Load no modules (pure chat mode)
    python -m llamagent --persona CodeLlama                # Specify persona
    python -m llamagent --port 9000                        # Specify port (Web/API)
"""

import argparse
import sys
import traceback

from llamagent.core import SmartAgent, Config, Persona, PersonaManager


# ============================================================
# Available module registry (module name -> import path)
# ============================================================
# All pluggable modules are registered here.
# Dynamic import: only modules specified by the user will be imported,
# preventing startup crashes when optional dependencies (chromadb, mcp, etc.)
# are not installed.

AVAILABLE_MODULES = {
    "safety": "llamagent.modules.safety.SafetyModule",
    "sandbox": "llamagent.modules.sandbox.SandboxModule",       # Before job (provides tool_executor)
    "tools": "llamagent.modules.tools.ToolsModule",             # Before skill (pack reset)
    "job": "llamagent.modules.job.JobModule",                   # After sandbox (hard dependency)
    "rag": "llamagent.modules.rag.RAGModule",
    "memory": "llamagent.modules.memory.MemoryModule",
    "skill": "llamagent.modules.skill.SkillModule",             # After tools (pack activation)
    "reflection": "llamagent.modules.reflection.ReflectionModule",
    "planning": "llamagent.modules.reasoning.PlanningModule",
    "mcp": "llamagent.modules.mcp.MCPModule",
    "multi_agent": "llamagent.modules.multi_agent.MultiAgentModule",
    "child_agent": "llamagent.modules.child_agent.ChildAgentModule",
}


def load_module(name: str):
    """
    Dynamically import and instantiate a module.

    Uses importlib for on-demand loading instead of top-level imports,
    so missing optional dependencies won't cause errors.

    Args:
        name: Module name (e.g., "tools", "rag")

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
) -> SmartAgent:
    """
    Create an Agent and load the specified modules.

    This is the shared Agent factory function used by all interfaces (CLI / Web / API).

    Args:
        module_names: List of modules to load.
                      None = load all available modules
                      [] = load no modules (pure chat mode)
        persona_name: Persona name, None uses the default identity

    Returns:
        A configured SmartAgent instance
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

    agent = SmartAgent(config, persona=persona)

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
        description="LlamAgent — Modular AI Agent Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m llamagent                                  CLI interactive chat
  python -m llamagent --mode web                       Launch Web UI
  python -m llamagent --mode api                       Launch HTTP API
  python -m llamagent --modules tools,rag              Load only tools and RAG
  python -m llamagent --no-modules                     Pure chat mode
  python -m llamagent --persona CodeLlama              Use a specific persona
  python -m llamagent --mode web --port 9000           Specify port
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
        help="Comma-separated list of modules, e.g.: tools,rag,memory",
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
        from llamagent.interfaces.cli import SmartAgentCLI
        cli = SmartAgentCLI(module_names=module_names, persona_name=args.persona)
        cli.chat_mode()

    elif args.mode == "web":
        from llamagent.interfaces.web_ui import create_web_ui, launch_web_ui

        port = args.port or int(__import__("os").getenv("WEB_UI_PORT", "7860"))

        try:
            demo = create_web_ui()
        except ImportError as e:
            print(f"Error: {e}")
            sys.exit(1)

        launch_web_ui(demo, port=port)

    elif args.mode == "api":
        from llamagent.interfaces.api_server import launch_api_server

        port = args.port or int(__import__("os").getenv("API_PORT", "8000"))
        launch_api_server(
            module_names=module_names,
            persona_name=args.persona,
            port=port,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
