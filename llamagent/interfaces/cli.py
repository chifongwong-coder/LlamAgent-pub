"""
LlamAgent Command-Line Interface (CLI).

Interactive terminal for building and chatting with LlamAgent.

Flow:
    1. Show banner
    2. Interactive setup: select modules, persona role, persona details
    3. Build agent and enter chat mode
    4. Ctrl+C once: exit current agent, return to setup
    5. Ctrl+C at setup: exit program

Usage:
    python -m llamagent                                     # Interactive setup + chat
    python -m llamagent.interfaces.cli                      # Same as above
    python -m llamagent.interfaces.cli ask "How's the weather today"  # Single question
    python -m llamagent.interfaces.cli --modules tools,retrieval  # Skip setup, specify modules
    python -m llamagent.interfaces.cli --no-modules         # Skip setup, pure chat mode
"""

import argparse
import sys

# Rich library: makes terminal output beautiful (colors, tables, panels, progress bars)
# Install: pip install rich
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    # Note: Rich Prompt has issues with CJK input; we use plain input() instead
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ============================================================
# Rich fallback adapter
# ============================================================

if HAS_RICH:
    console = Console()
else:
    import re as _re
    import contextlib as _contextlib

    class _FallbackConsole:
        """Fallback Console when Rich is not installed, maintaining interface compatibility."""

        def print(self, *args, **kwargs):
            text = str(args[0]) if args else ""
            text = _re.sub(r'\[/?[^\]]*\]', '', text)
            print(text)

        def status(self, msg):
            @_contextlib.contextmanager
            def _noop():
                print(msg)
                yield
            return _noop()

    console = _FallbackConsole()


# ============================================================
# Input helpers (plain input() for CJK compatibility)
# ============================================================

def _ask(prompt: str, default: str = "") -> str:
    """Prompt for text input with optional default."""
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        val = ""
    return val or default


def _ask_choice(prompt: str, choices: list[str], default: str = "1") -> str:
    """Prompt for a numbered choice."""
    while True:
        val = _ask(prompt, default)
        if val in choices:
            return val
        print(f"  Invalid choice. Options: {', '.join(choices)}")


def _ask_confirm(prompt: str, default: bool = True) -> bool:
    """Prompt for yes/no confirmation."""
    suffix = "[Y/n]" if default else "[y/N]"
    val = input(f"{prompt} {suffix}: ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes")


# ============================================================
# Version info and welcome banner
# ============================================================
from llamagent import __version__ as VERSION

BANNER = """
[bold cyan]
  _     _                    _                    _
 | |   | |                  / \\   __ _  ___ _ __ | |_
 | |   | | __ _ _ __ ___   / _ \\ / _` |/ _ \\ '_ \\| __|
 | |___| |/ _` | '_ ` _ \\ / ___ \\ (_| |  __/ | | | |_
 |_____|_|\\__,_|_| |_| |_/_/   \\_\\__, |\\___|_| |_|\\__|
                                  |___/
[/bold cyan]
[bold white]  LlamAgent v{version} — Your AI Assistant[/bold white]
[dim]  Type /help for help | Ctrl+C to exit agent[/dim]
""".format(version=VERSION)


# ============================================================
# Available modules for interactive setup
# ============================================================

MODULE_GROUPS = {
    "Core": [
        ("safety", "Input filtering + output sanitization"),
        ("tools", "Four-tier tool system + built-in tools"),
        ("sandbox", "Isolated execution for high-risk tools"),
    ],
    "Intelligence": [
        ("planning", "PlanReAct task decomposition"),
        ("reflection", "Quality evaluation + lesson learning"),
    ],
    "Knowledge": [
        ("retrieval", "Knowledge retrieval over documents"),
        ("memory", "Persistent memory with semantic recall"),
    ],
    "Collaboration": [
        ("multi_agent", "Role-based task delegation"),
        ("child_agent", "Spawn constrained sub-agents"),
        ("mcp", "Model Context Protocol bridge"),
    ],
}

# Preset configurations
PRESETS = {
    "full": "All modules (recommended for first time)",
    "minimal": "Safety + Tools only",
    "chat": "No modules (pure conversation)",
    "custom": "Choose modules manually",
}


# ============================================================
# Interactive setup
# ============================================================

def _load_saved_personas() -> list:
    """Load saved personas from disk. Returns empty list if none exist."""
    from llamagent.core import Config, PersonaManager
    try:
        config = Config()
        manager = PersonaManager(config.persona_file)
        return manager.list()
    except Exception:
        return []


def interactive_setup() -> dict:
    """
    Interactive configuration menu. Returns a dict with:
        modules: list[str] | None
        persona_name: str
        persona_role: str ("admin" | "user")
        persona_desc: str
        save_persona: bool
    """
    console.print("\n[bold cyan]--- Agent Setup ---[/bold cyan]\n")

    # Step 0: Check for saved personas
    saved = _load_saved_personas()
    use_saved = False

    if saved:
        console.print("[bold]Saved personas found:[/bold]\n")
        for i, p in enumerate(saved, 1):
            role_tag = "[red]admin[/red]" if p.role == "admin" else "[green]user[/green]"
            desc = p.role_description or "No description"
            console.print(f"  [bold]{i}[/bold]. {p.name} ({role_tag}) — {desc[:60]}")
        console.print(f"  [bold]{len(saved) + 1}[/bold]. Create new persona")
        console.print()

        p_choice = _ask_choice(
            "Select persona",
            [str(i) for i in range(1, len(saved) + 2)],
            default="1",
        )

        idx = int(p_choice) - 1
        if idx < len(saved):
            # Use saved persona
            p = saved[idx]
            persona_name = p.name
            persona_role = p.role
            persona_desc = p.role_description or "A helpful AI assistant"
            use_saved = True
            console.print(f"\n[green]Using persona: {persona_name}[/green]")

    # Step 1: Module preset
    console.print("\n[bold]Step 1:[/bold] Choose module configuration\n")
    preset_keys = list(PRESETS.keys())
    for i, (key, desc) in enumerate(PRESETS.items(), 1):
        marker = "[cyan]*[/cyan] " if key == "full" else "  "
        console.print(f"  {marker}[bold]{i}[/bold]. {key} — {desc}")

    console.print()
    choice = _ask_choice(
        "Select preset",
        [str(i) for i in range(1, len(PRESETS) + 1)],
        default="1",
    )

    preset = preset_keys[int(choice) - 1]

    if preset == "full":
        module_names = None  # None = load all
    elif preset == "minimal":
        module_names = ["safety", "tools"]
    elif preset == "chat":
        module_names = []
    else:
        module_names = _pick_modules()

    if not use_saved:
        # Step 2: Persona role
        console.print("\n[bold]Step 2:[/bold] Choose persona role\n")
        console.print("  [cyan]*[/cyan] [bold]1[/bold]. user — Standard access")
        console.print("    [bold]2[/bold]. admin — Full access (includes execute_command)")
        console.print()

        role_choice = _ask_choice("Select role", ["1", "2"], default="1")

        persona_role = "admin" if role_choice == "2" else "user"

        # Step 3: Persona details
        console.print("\n[bold]Step 3:[/bold] Persona details\n")

        persona_name = _ask("Agent name", default="LlamAgent")
        persona_desc = _ask("Role description", default="A helpful AI assistant")

        # Ask whether to save
        save_persona = _ask_confirm("Save this persona for next time?")
    else:
        save_persona = False  # Already saved

    return {
        "modules": module_names,
        "persona_name": persona_name,
        "persona_role": persona_role,
        "persona_desc": persona_desc,
        "save_persona": save_persona,
    }


def _pick_modules() -> list[str]:
    """Let user pick individual modules from grouped list."""
    selected = []
    console.print()

    for group_name, modules in MODULE_GROUPS.items():
        console.print(f"\n  [bold cyan]{group_name}[/bold cyan]")
        for mod_name, mod_desc in modules:
            yes = _ask_confirm(f"    Load {mod_name} ({mod_desc})?")
            if yes:
                selected.append(mod_name)

    return selected


def build_agent(setup: dict):
    """Build a LlamAgent from interactive setup results."""
    from llamagent.core import LlamAgent, Config, Persona, PersonaManager
    from llamagent.main import load_module, AVAILABLE_MODULES

    config = Config()

    # Create persona
    persona = Persona(
        name=setup["persona_name"],
        role_description=setup["persona_desc"],
        role=setup["persona_role"],
    )

    # Save persona if requested
    if setup.get("save_persona"):
        try:
            manager = PersonaManager(config.persona_file)
            # Check if already exists
            existing = manager.get(persona.persona_id)
            if not existing:
                manager.create(
                    name=persona.name,
                    role_description=persona.role_description,
                    role=persona.role,
                )
                console.print(f"[green]Persona '{persona.name}' saved![/green]")
            else:
                console.print(f"[dim]Persona '{persona.name}' already exists, skipping save[/dim]")
        except Exception as e:
            console.print(f"[yellow]Failed to save persona: {e}[/yellow]")

    agent = LlamAgent(config, persona=persona)

    # Load modules
    module_names = setup["modules"]
    if module_names is None:
        module_names = list(AVAILABLE_MODULES.keys())

    if module_names:
        console.print(f"\n[dim]Loading modules...[/dim]")
        for name in module_names:
            mod = load_module(name)
            if mod:
                agent.register_module(mod)
                console.print(f"  [green]+[/green] {name}")
    else:
        console.print("\n[dim]Pure chat mode (no modules)[/dim]")

    console.print(f"\n[green]Agent ready![/green] Model: {config.model} | "
                  f"Role: {persona.role} | Modules: {len(agent.modules)}\n")

    return agent


# ============================================================
# CLI main class
# ============================================================

class LlamAgentCLI:
    """
    LlamAgent command-line interface: turns the terminal into a smart chat window.

    Responsibilities:
    1. Handle user input (regular chat vs slash commands)
    2. Call LlamAgent to get responses
    3. Beautify output with Rich (falls back to print when not installed)
    4. Error handling (never crash regardless of user input)
    """

    def __init__(self, agent):
        """
        Initialize CLI with a pre-built agent.

        Args:
            agent: A configured LlamAgent instance
        """
        self.agent = agent
        self._runner = None  # ContinuousRunner instance (if active)

        # Set up confirm_handler for interactive authorization prompts
        from llamagent.core.zone import ConfirmRequest, ConfirmResponse
        def _cli_confirm_handler(req: ConfirmRequest) -> ConfirmResponse:
            console.print(f"\n[yellow]Authorization required: {req.tool_name}[/yellow]")
            console.print(f"  Action: {req.action} | Zone: {req.zone}")
            if req.target_paths:
                console.print(f"  Paths: {', '.join(req.target_paths[:5])}")
            console.print(f"  {req.message}")
            yes = _ask_confirm("  Allow this operation?")
            return ConfirmResponse(allow=yes)

        self.agent.confirm_handler = _cli_confirm_handler

        # Slash command mapping (commands with args handled separately)
        self._slash_commands = {
            "/quit": self._cmd_quit,
            "/exit": self._cmd_quit,
            "/q": self._cmd_quit,
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/modules": self._cmd_modules,
            "/clear": self._cmd_clear,
            "/abort": self._cmd_abort,
        }

    # ============================================================
    # Interactive chat mode (main loop)
    # ============================================================

    def chat_mode(self):
        """
        Enter interactive chat mode.

        Returns:
            "restart" if user pressed Ctrl+C (go back to setup)
            "quit" if user typed /quit
        """
        while True:
            try:
                # Get user input (plain input for CJK compatibility)
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Check if it's a slash command
                if user_input.startswith("/"):
                    cmd_parts = user_input.split(maxsplit=1)
                    cmd = cmd_parts[0].lower()
                    cmd_arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""

                    # /mode takes an argument
                    if cmd == "/mode":
                        result = self._cmd_mode(cmd_arg)
                        if result == "continuous_done":
                            continue  # returned from continuous, resume chat loop
                        continue

                    handler = self._slash_commands.get(cmd)
                    if handler:
                        if handler() is False:
                            return "quit"
                    else:
                        console.print(
                            f"[yellow]Unknown command: {cmd}, "
                            f"type /help for available commands[/yellow]"
                        )
                    continue

                # Regular chat
                self._process_chat(user_input)

            except KeyboardInterrupt:
                # Ctrl+C: exit current agent, return to setup
                console.print("\n\n[dim]Exiting current agent...[/dim]")
                return "restart"
            except EOFError:
                return "quit"

    # ============================================================
    # Single question mode
    # ============================================================

    def ask(self, question: str):
        """Single question mode: ask a question, get an answer, done."""
        try:
            with console.status("[bold cyan]LlamAgent is thinking...[/bold cyan]"):
                response = self.agent.chat(question)

            self._display_response(response)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    # ============================================================
    # Internal methods
    # ============================================================

    def _process_chat(self, user_input: str):
        """Process a single chat turn: stream or non-stream based on agent capability."""
        try:
            if hasattr(self.agent, 'chat_stream') and self.agent.mode == "interactive":
                # Streaming mode
                console.print()
                accumulated = ""
                for chunk in self.agent.chat_stream(user_input):
                    print(chunk, end="", flush=True)
                    accumulated += chunk
                print()  # newline after stream

                # Display contract in special format if detected
                if accumulated.startswith("[Task Contract]"):
                    self._display_response(accumulated)
            else:
                with console.status("[bold cyan]LlamAgent is thinking...[/bold cyan]"):
                    response = self.agent.chat(user_input)
                self._display_response(response)

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            console.print("[dim]Please try again, or type /help for assistance[/dim]")

    def _display_response(self, response: str):
        """Render Agent response with Rich Panel + Markdown, or plain text."""
        is_contract = response.startswith("[Task Contract]")

        if HAS_RICH:
            console.print()
            if is_contract:
                console.print(Panel(
                    response,
                    title="[bold yellow]Task Contract[/bold yellow]",
                    border_style="yellow",
                    padding=(1, 2),
                ))
            else:
                console.print(Panel(
                    Markdown(response),
                    title="[bold cyan]LlamAgent[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                ))
        else:
            if is_contract:
                print(f"\n{'=' * 50}")
                print(response)
                print(f"{'=' * 50}\n")
            else:
                print(f"\nLlamAgent: {response}\n")

    # ============================================================
    # Slash command implementations
    # ============================================================

    def _cmd_quit(self):
        """Exit the conversation."""
        console.print("\n[bold cyan]Goodbye![/bold cyan]")
        return False

    def _cmd_help(self):
        """Display help information."""
        if HAS_RICH:
            help_table = Table(
                title="Available Commands",
                show_header=True,
                header_style="bold cyan",
            )
            help_table.add_column("Command", style="cyan", width=20)
            help_table.add_column("Description", style="white")

            commands = [
                ("/help", "Show this help message"),
                ("/mode [name]", "Show/switch mode (interactive, task, continuous)"),
                ("/abort", "Abort the current task"),
                ("/status", "View Agent runtime status"),
                ("/modules", "View loaded modules"),
                ("/clear", "Clear conversation history"),
                ("/quit", "Exit the conversation (also: /exit, /q)"),
                ("Ctrl+C", "Exit current agent, return to setup"),
            ]
            for cmd, desc in commands:
                help_table.add_row(cmd, desc)

            console.print()
            console.print(help_table)
        else:
            print(
                "\nAvailable Commands:\n"
                "  /help          Show help\n"
                "  /mode [name]   Show/switch mode (interactive, task, continuous)\n"
                "  /abort         Abort the current task\n"
                "  /status        View Agent status\n"
                "  /modules       View loaded modules\n"
                "  /clear         Clear conversation history\n"
                "  /quit          Exit\n"
                "  Ctrl+C         Exit current agent, return to setup\n"
            )

    def _cmd_status(self):
        """Display Agent status."""
        status = self.agent.status()

        if HAS_RICH:
            table = Table(
                title="Agent Status",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Item", style="cyan", width=20)
            table.add_column("Value", style="white")

            table.add_row("Model", status.get("model", "Unknown"))
            table.add_row("Mode", self.agent.mode)
            table.add_row(
                "Persona",
                status.get("persona") or "Default"
            )
            table.add_row(
                "Loaded Modules",
                ", ".join(status.get("modules", {}).keys()) or "None"
            )
            table.add_row(
                "Conversation Turns",
                str(status.get("conversation_turns", 0))
            )

            console.print()
            console.print(table)
        else:
            import json
            print(json.dumps(status, ensure_ascii=False, indent=2))

    def _cmd_modules(self):
        """Display loaded modules."""
        modules = self.agent.modules

        if not modules:
            console.print("[yellow]No modules currently loaded (pure chat mode)[/yellow]")
            return

        if HAS_RICH:
            table = Table(
                title="Loaded Modules",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("#", style="dim", width=4)
            table.add_column("Module", style="cyan")
            table.add_column("Description", style="white")

            for i, (name, mod) in enumerate(modules.items(), 1):
                table.add_row(str(i), name, mod.description or "")

            console.print()
            console.print(table)
            console.print(f"\n[dim]{len(modules)} module(s) loaded[/dim]")
        else:
            print("\nLoaded Modules:")
            for name, mod in modules.items():
                print(f"  {name}: {mod.description}")
            print()

    def _cmd_clear(self):
        """Clear conversation history."""
        self.agent.clear_conversation()
        console.print("[green]Conversation history cleared[/green]")

    def _cmd_abort(self):
        """Abort the current task."""
        self.agent.abort()
        console.print("[yellow]Abort signal sent[/yellow]")

    def _cmd_mode(self, arg: str):
        """Switch or display agent mode."""
        if not arg:
            # Display current mode and config
            mode = self.agent.mode
            config = self.agent.config
            console.print(f"\n[bold]Current mode:[/bold] [cyan]{mode}[/cyan]")
            console.print(f"  max_react_steps:      {config.max_react_steps}")
            console.print(f"  max_duplicate_actions: {config.max_duplicate_actions}")
            console.print(f"  react_timeout:        {config.react_timeout}")
            console.print(f"  max_observation_tokens: {config.max_observation_tokens}")
            return

        target = arg.strip().lower()
        if target not in ("interactive", "task", "continuous"):
            console.print(f"[red]Invalid mode: {target}. Use: interactive, task, continuous[/red]")
            return

        if target == "continuous":
            return self._start_continuous()

        try:
            self.agent.set_mode(target)
            console.print(f"[green]Switched to {target} mode[/green]")
        except RuntimeError as e:
            console.print(f"[red]Cannot switch mode: {e}[/red]")

    def _start_continuous(self):
        """Configure triggers and run ContinuousRunner (blocks until Ctrl+C)."""
        from llamagent.core.runner import ContinuousRunner, TimerTrigger, FileTrigger

        console.print("\n[bold]Continuous Mode Setup[/bold]\n")
        console.print("  [bold]1[/bold]. Timer — run a fixed task at regular intervals")
        console.print("  [bold]2[/bold]. File  — watch a directory for new files")
        trigger_choice = _ask_choice("Select trigger type", ["1", "2"], default="1")

        triggers = []
        if trigger_choice == "1":
            interval = _ask("Interval in seconds", default="60")
            message = _ask("Task message", default="check system status")
            try:
                triggers.append(TimerTrigger(interval=float(interval), message=message))
            except ValueError:
                console.print("[red]Invalid interval[/red]")
                return
        else:
            watch_dir = _ask("Directory to watch", default=".")
            triggers.append(FileTrigger(watch_dir))

        try:
            self.agent.set_mode("continuous")
        except RuntimeError as e:
            console.print(f"[red]Cannot switch mode: {e}[/red]")
            return

        runner = ContinuousRunner(self.agent, triggers, poll_interval=1.0)
        self._runner = runner

        console.print(f"\n[green]Continuous mode started. Press Ctrl+C to stop.[/green]\n")
        try:
            runner.run()
        except KeyboardInterrupt:
            runner.stop()
            console.print("\n[yellow]Continuous mode stopped[/yellow]")
        finally:
            self._runner = None
            try:
                self.agent.set_mode("interactive")
                console.print("[green]Switched back to interactive mode[/green]")
            except Exception:
                pass

        return "continuous_done"


# ============================================================
# Main entry loop
# ============================================================

def _create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="llamagent-cli",
        description="LlamAgent CLI — Terminal Chat Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m llamagent.interfaces.cli                      Interactive setup + chat
  python -m llamagent.interfaces.cli ask "How's the weather"  Single question
  python -m llamagent.interfaces.cli --modules tools,retrieval   Skip setup, load specific modules
  python -m llamagent.interfaces.cli --no-modules          Skip setup, pure chat mode
        """,
    )

    parser.add_argument(
        "--modules", type=str, default=None,
        help="Comma-separated list of modules (skips interactive setup)",
    )
    parser.add_argument(
        "--no-modules", action="store_true",
        help="Load no modules, pure chat mode (skips interactive setup)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("chat", help="Enter interactive chat mode")
    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", help="The question to ask")

    return parser


def main():
    """CLI entry point: interactive setup loop or direct mode."""
    if not HAS_RICH:
        print(
            "[Note] Rich library not installed, using simplified mode. "
            "Install with: pip install rich"
        )

    parser = _create_parser()
    args = parser.parse_args()

    # Direct mode: --modules or --no-modules skips interactive setup
    if args.no_modules or args.modules is not None:
        if args.no_modules:
            module_names = []
        else:
            module_names = [m.strip() for m in args.modules.split(",") if m.strip()]

        setup = {
            "modules": module_names,
            "persona_name": "LlamAgent",
            "persona_role": "user",
            "persona_desc": "A helpful AI assistant",
        }
        agent = build_agent(setup)
        console.print(BANNER)
        cli = LlamAgentCLI(agent)

        try:
            if args.command == "ask":
                cli.ask(args.question)
            else:
                cli.chat_mode()
        finally:
            agent.shutdown()
        return

    # Ask mode with question
    if args.command == "ask":
        setup = {
            "modules": None,
            "persona_name": "LlamAgent",
            "persona_role": "user",
            "persona_desc": "A helpful AI assistant",
        }
        agent = build_agent(setup)
        cli = LlamAgentCLI(agent)
        try:
            cli.ask(args.question)
        finally:
            agent.shutdown()
        return

    # Interactive setup loop
    console.print(BANNER)

    while True:
        try:
            setup = interactive_setup()
        except KeyboardInterrupt:
            console.print("\n\n[bold cyan]Goodbye![/bold cyan]")
            break

        try:
            agent = build_agent(setup)
        except Exception as e:
            console.print(f"\n[red]Failed to build agent: {e}[/red]")
            continue

        cli = LlamAgentCLI(agent)
        result = cli.chat_mode()

        # Cleanup
        try:
            agent.shutdown()
        except Exception:
            pass

        if result == "quit":
            break
        # result == "restart": loop back to setup


if __name__ == "__main__":
    main()
