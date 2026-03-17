"""
LlamAgent Command-Line Interface (CLI).

The "front door" for chatting with LlamAgent in the terminal.
Uses the Rich library to beautify output, giving the command line a polished look.
When Rich is not installed, it automatically falls back to plain text mode — not as pretty, but functional.

Usage:
    python -m llamagent                                     # Interactive chat (default)
    python -m llamagent.interfaces.cli                      # Same as above
    python -m llamagent.interfaces.cli ask "How's the weather today"  # Single question
    python -m llamagent.interfaces.cli --modules tools,rag  # Specify modules
    python -m llamagent.interfaces.cli --no-modules         # Pure chat mode
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
    from rich.prompt import Prompt
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ============================================================
# Rich fallback adapter
# ============================================================
# Runs fine without Rich, just less fancy output.
# Budget Console: strips Rich markup, outputs plain text.

if HAS_RICH:
    console = Console()
else:
    import re as _re
    import contextlib as _contextlib

    class _FallbackConsole:
        """Fallback Console when Rich is not installed, maintaining interface compatibility."""

        def print(self, *args, **kwargs):
            text = str(args[0]) if args else ""
            # Strip Rich markup syntax [bold cyan]...[/bold cyan]
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
# Version info and welcome banner
# ============================================================
VERSION = "1.0.0"

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
[dim]  Type /help for help | /quit to exit[/dim]
""".format(version=VERSION)


# ============================================================
# CLI main class
# ============================================================

class SmartAgentCLI:
    """
    LlamAgent command-line interface: turns the terminal into a smart chat window.

    Responsibilities:
    1. Handle user input (regular chat vs slash commands)
    2. Call SmartAgent to get responses
    3. Beautify output with Rich (falls back to print when not installed)
    4. Error handling (never crash regardless of user input)
    """

    def __init__(
        self,
        module_names: list[str] | None = None,
        persona_name: str | None = None,
        agent=None,
    ):
        """
        Initialize CLI: create Agent instance and set up command mappings.

        Args:
            module_names: List of modules to load, None means load all
            persona_name: Persona name, None uses default identity
            agent: Pass in an already-created SmartAgent instance (highest priority)
        """
        if agent is not None:
            # Use the externally provided Agent directly
            self.agent = agent
        else:
            # Create a new Agent via create_agent
            console.print("\n[dim]Initializing LlamAgent...[/dim]")
            from llamagent.main import create_agent
            self.agent = create_agent(module_names, persona_name=persona_name)

        # Slash command mapping — using a dict instead of if-elif chains, more elegant and extensible
        self._slash_commands = {
            "/quit": self._cmd_quit,
            "/exit": self._cmd_quit,
            "/q": self._cmd_quit,
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/modules": self._cmd_modules,
            "/clear": self._cmd_clear,
        }

        console.print(f"[green]Initialization complete! Model: {self.agent.config.model}[/green]\n")

    # ============================================================
    # Interactive chat mode (main loop)
    # ============================================================

    def chat_mode(self):
        """
        Enter interactive chat mode — the core feature of the CLI.

        Like opening a messaging app to chat with a friend, except this time your
        friend is an AI llama. Supports slash commands (/help, /status, etc.)
        and regular conversation.
        """
        console.print(BANNER)

        while True:
            try:
                # Get user input — Rich's Prompt looks better than input()
                if HAS_RICH:
                    user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
                else:
                    user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Check if it's a slash command
                if user_input.startswith("/"):
                    cmd_parts = user_input.split(maxsplit=1)
                    cmd = cmd_parts[0].lower()

                    handler = self._slash_commands.get(cmd)
                    if handler:
                        # Execute command; returning False means exit
                        if handler() is False:
                            break
                    else:
                        console.print(
                            f"[yellow]Unknown command: {cmd}, "
                            f"type /help for available commands[/yellow]"
                        )
                    continue

                # Regular chat: send to Agent for processing
                self._process_chat(user_input)

            except KeyboardInterrupt:
                # Ctrl+C graceful exit — don't show the user an ugly stack trace
                console.print("\n\n[dim]Exit signal received, goodbye![/dim]")
                break
            except EOFError:
                # End of input stream (pipe mode, etc.)
                break

    # ============================================================
    # Single question mode
    # ============================================================

    def ask(self, question: str):
        """
        Single question mode: ask a question, get an answer, done.

        Suitable for scripted usage:
            python -m llamagent.interfaces.cli ask "Summarize my emails for today"
        """
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
        """
        Process a single chat turn: send to Agent, display the response.

        console.status() shows a spinning animation in the terminal,
        letting the user know LlamAgent is "thinking" — much better than staring at a blank screen.
        """
        try:
            with console.status("[bold cyan]LlamAgent is thinking...[/bold cyan]"):
                response = self.agent.chat(user_input)

            self._display_response(response)

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            console.print("[dim]Please try again, or type /help for assistance[/dim]")

    def _display_response(self, response: str):
        """Render Agent response with Rich Panel + Markdown, or plain text if Rich is unavailable."""
        if HAS_RICH:
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold cyan]LlamAgent[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            ))
        else:
            print(f"\nLlamAgent: {response}\n")

    # ============================================================
    # Slash command implementations
    # ============================================================

    def _cmd_quit(self):
        """Exit the conversation."""
        console.print("\n[bold cyan]Goodbye! Looking forward to chatting with you again.[/bold cyan]")
        return False  # Returning False exits the main loop

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
                ("/status", "View Agent runtime status"),
                ("/modules", "View loaded modules"),
                ("/clear", "Clear conversation history"),
                ("/quit", "Exit the conversation (also: /exit, /q, Ctrl+C)"),
            ]
            for cmd, desc in commands:
                help_table.add_row(cmd, desc)

            console.print()
            console.print(help_table)
        else:
            print(
                "\nAvailable Commands:\n"
                "  /help       Show help\n"
                "  /status     View Agent status\n"
                "  /modules    View loaded modules\n"
                "  /clear      Clear conversation history\n"
                "  /quit       Exit\n"
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


# ============================================================
# Command-line argument parsing (python -m llamagent.interfaces.cli entry)
# ============================================================

def _create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="llamagent-cli",
        description="LlamAgent CLI — Terminal Chat Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m llamagent.interfaces.cli                      Enter interactive chat
  python -m llamagent.interfaces.cli ask "How's the weather"  Single question
  python -m llamagent.interfaces.cli --modules tools,rag   Load only tools and RAG
  python -m llamagent.interfaces.cli --no-modules          Pure chat mode
        """,
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

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # chat command (default)
    subparsers.add_parser("chat", help="Enter interactive chat mode")

    # ask command: single question
    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", help="The question to ask")

    return parser


def main():
    """CLI standalone entry point: parse arguments and execute the corresponding action."""
    if not HAS_RICH:
        print(
            "[Note] Rich library not installed, terminal interface will use simplified mode. "
            "Install with: pip install rich"
        )

    parser = _create_parser()
    args = parser.parse_args()

    # Parse module arguments
    if args.no_modules:
        module_names = []
    elif args.modules:
        module_names = [m.strip() for m in args.modules.split(",") if m.strip()]
    else:
        module_names = None  # None = load all

    # Default to interactive chat mode
    if args.command is None or args.command == "chat":
        cli = SmartAgentCLI(module_names=module_names)
        cli.chat_mode()

    elif args.command == "ask":
        cli = SmartAgentCLI(module_names=module_names)
        cli.ask(args.question)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
