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
import os
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
    try:
        val = input(f"{prompt} {suffix}: ").strip().lower()
    except EOFError:
        return default
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
[bold white]  LlamAgent v{version} — Modular AI Agent Framework[/bold white]
[dim]  Type /help for commands | Ctrl+C to return to setup[/dim]
""".format(version=VERSION)


# ============================================================
# Available modules for interactive setup
# ============================================================

from llamagent.interfaces.presets import MODULE_DESCRIPTIONS, MODULE_GROUPS, apply_presets

# Preset configurations
PRESETS = {
    "full": "All modules — full capabilities (recommended for new users)",
    "minimal": "Safety + Tools — lightweight, no AI planning or memory",
    "chat": "No modules — direct LLM chat, no tools or features",
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


def _check_environment() -> list[str]:
    """Check environment before setup. Returns list of warnings."""
    warnings = []

    from llamagent.core.config import Config
    config = Config()

    # Skip API key check for Ollama models (they don't need API keys)
    if not config.model.startswith("ollama"):
        key_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_HOST"]
        has_key = any(os.environ.get(v) for v in key_vars)
        if not has_key:
            warnings.append(
                "No LLM API key detected. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "or configure OLLAMA_HOST for local models.\n"
                "  Example: export OPENAI_API_KEY=sk-..."
            )

    return warnings


def interactive_setup() -> dict | None:
    """
    Interactive configuration menu. Returns a dict with:
        modules: list[str] | None
        persona_name: str
        persona_role: str ("admin" | "user")
        persona_desc: str
        save_persona: bool

    Returns None if the user decides not to continue after environment warnings.
    """
    # Environment check
    warnings = _check_environment()
    if warnings:
        console.print("\n[yellow]Environment warnings:[/yellow]")
        for w in warnings:
            console.print(f"  [yellow]! {w}[/yellow]")
        if not _ask_confirm("\nContinue anyway?"):
            return None

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
        console.print("    [bold]2[/bold]. admin — Full access (can run system commands)")
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

    for group_name, mod_names in MODULE_GROUPS.items():
        console.print(f"\n  [bold cyan]{group_name}[/bold cyan]")
        for mod_name in mod_names:
            desc = MODULE_DESCRIPTIONS.get(mod_name, mod_name)
            yes = _ask_confirm(f"    Load {desc}?")
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

    # Load modules
    module_names = setup["modules"]
    if module_names is None:
        module_names = list(AVAILABLE_MODULES.keys())

    # Apply smart defaults for selected modules before building agent
    apply_presets(config, module_names)

    agent = LlamAgent(config, persona=persona)

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

    # Capability summary (C5)
    capabilities = []
    if agent.has_module("memory"):
        capabilities.append("remember facts across conversations")
    if agent.has_module("tools"):
        capabilities.append("use tools")
    if agent.has_module("planning"):
        capabilities.append("plan complex tasks")
    if agent.has_module("skill"):
        skill_mod = agent.get_module("skill")
        try:
            count = len(skill_mod.list_skills()) if skill_mod else 0
        except Exception:
            count = 0
        if count:
            capabilities.append(f"follow {count} skill guides")
    if capabilities:
        console.print(f"[dim]Capabilities: {', '.join(capabilities)}. Type /help for commands.[/dim]")

    # Conversation restore hint (C5)
    turns = sum(1 for m in agent.history if m.get("role") == "user")
    if turns:
        console.print(f"[dim]Restored {turns} turns from previous session. Type /clear to start fresh.[/dim]")

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

        # Set up interaction_handler for ask_user tool (C1)
        from llamagent.modules.tools.interaction import UserInteractionHandler

        class _CLIInteractionHandler(UserInteractionHandler):
            def ask(self, question, choices=None):
                console.print(f"\n[yellow]Agent asks:[/yellow] {question}")
                if choices:
                    for i, c in enumerate(choices, 1):
                        console.print(f"  [bold]{i}[/bold]. {c}")
                    val = _ask("Select", default="1")
                    try:
                        idx = int(val) - 1
                        return choices[idx] if 0 <= idx < len(choices) else val
                    except (ValueError, IndexError):
                        return val
                return _ask("Your answer")

        self.agent.interaction_handler = _CLIInteractionHandler()

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
            "/stop": self._cmd_stop,
            "/skills": self._cmd_skills,
            "/memory": self._cmd_memory,
        }

        # Tab completion for slash commands (C4)
        self._setup_readline()

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
                console.print("\n\n[dim]Leaving current agent. Returning to setup...[/dim]")
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
            error_msg = str(e).lower()
            hint = self._get_error_hint(e, error_msg)
            if hint:
                console.print(f"[yellow]Hint: {hint}[/yellow]")
            else:
                console.print("[dim]Try again, or type /help for available commands.[/dim]")

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

    def _get_error_hint(self, error: Exception, error_msg: str) -> str | None:
        """Return a user-friendly recovery hint based on error type."""
        if "api key" in error_msg or "authentication" in error_msg or "401" in error_msg:
            return "Your API key may be invalid or expired. Check your OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
        if "rate limit" in error_msg or "429" in error_msg:
            return "You've hit a rate limit. Wait a moment and try again, or switch to a different model."
        if "context" in error_msg and ("length" in error_msg or "window" in error_msg):
            return "The conversation is too long. Try /clear to start fresh, or enable the compression module."
        if "timeout" in error_msg:
            return "The request timed out. Try a simpler question, or increase react_timeout in config."
        if "connection" in error_msg or "network" in error_msg:
            return "Network error. Check your internet connection and try again."
        if "model" in error_msg and ("not found" in error_msg or "does not exist" in error_msg):
            return "The configured model was not found. Check your model name in config.yaml (e.g., 'openai/gpt-4o')."
        return None

    # ============================================================
    # Slash command implementations
    # ============================================================

    def _cmd_quit(self):
        """Exit the conversation."""
        console.print("\n[bold cyan]Goodbye![/bold cyan]")
        return False

    def _cmd_help(self):
        """Display help information."""
        commands = [
            ("/help", "Show this help message"),
            ("/mode [name]", "Show/switch mode (interactive, task, continuous)"),
            ("/abort", "Cancel the current task"),
            ("/stop", "Stop the background runner"),
            ("/status", "View Agent runtime status"),
            ("/modules", "View loaded modules"),
            ("/skills", "List loaded skills"),
            ("/memory", "Show stored facts and memory usage"),
            ("/clear", "Start a fresh conversation"),
            ("/quit", "Exit the conversation (also: /exit, /q)"),
            ("Ctrl+C", "Exit current agent, return to setup"),
        ]

        if HAS_RICH:
            help_table = Table(
                title="Available Commands",
                show_header=True,
                header_style="bold cyan",
            )
            help_table.add_column("Command", style="cyan", width=20)
            help_table.add_column("Description", style="white")

            for cmd, desc in commands:
                help_table.add_row(cmd, desc)

            console.print()
            console.print(help_table)
        else:
            print("\nAvailable Commands:")
            for cmd, desc in commands:
                print(f"  {cmd:<16} {desc}")
            print()

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
        console.print("[yellow]Abort requested. The current task will stop shortly.[/yellow]")

    def _cmd_mode(self, arg: str):
        """Switch or display agent mode. Selection-based when no argument given."""
        if not arg:
            console.print(f"\n  Current mode: [cyan]{self.agent.mode}[/cyan]\n")
            modes = ["interactive", "task", "continuous"]
            descs = {
                "interactive": "Normal back-and-forth chat",
                "task": "Plan first, confirm, then execute",
                "continuous": "Automated triggers with optional manual input",
            }
            for i, m in enumerate(modes, 1):
                marker = "[cyan]*[/cyan]" if m == self.agent.mode else " "
                console.print(f"  {marker} [bold]{i}[/bold]. {m} -- {descs[m]}")
            console.print()
            choice = _ask_choice("Select mode", ["1", "2", "3"], default="1")
            arg = modes[int(choice) - 1]
            if arg == self.agent.mode:
                console.print(f"[dim]Already in {arg} mode[/dim]")
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
        """Configure triggers and run ContinuousRunner with background monitor + inject loop."""
        from llamagent.core.runner import ContinuousRunner, TimerTrigger, FileTrigger
        import threading

        # Selection-based trigger configuration
        console.print("\n[bold]Continuous Mode Setup[/bold]\n")
        console.print("  [bold]1[/bold]. Timer -- Run a task on a schedule (e.g., every 60 seconds)")
        console.print("  [bold]2[/bold]. File  -- Trigger when new files appear in a directory")
        trigger_choice = _ask_choice("Select trigger", ["1", "2"], default="1")

        triggers = []
        if trigger_choice == "1":
            interval = _ask("Interval seconds", default="60")
            message = _ask("What should the agent do each time?", default="check status")
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

        # Auto-approve during continuous mode to avoid stdin conflict
        from llamagent.core.zone import ConfirmResponse
        saved_handler = self.agent.confirm_handler
        self.agent.confirm_handler = lambda req: ConfirmResponse(allow=True)

        runner = ContinuousRunner(self.agent, triggers, poll_interval=1.0)
        self._runner = runner

        # Runner background thread
        runner_thread = threading.Thread(target=runner.run, daemon=True)
        runner_thread.start()

        # Monitor thread: outputs background task results
        seen_count = [0]
        def _monitor():
            while not runner._stopped.is_set():
                log = runner.get_log()
                for entry in log[seen_count[0]:]:
                    status_mark = "[green]OK[/green]" if entry.status == "completed" else f"[red]{entry.status}[/red]"
                    console.print(f"\n  [dim][{entry.trigger_type}][/dim] {entry.input[:60]} -> {status_mark} ({entry.duration:.1f}s)")
                seen_count[0] = len(log)
                runner._stopped.wait(2)

        monitor_thread = threading.Thread(target=_monitor, daemon=True)
        monitor_thread.start()

        console.print(f"\n[green]Continuous mode active. You can still type messages. Use /stop to end.[/green]\n")

        # Main thread: accept inject input
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input == "/stop":
                    break
                try:
                    response = runner.inject(user_input)
                    self._display_response(response)
                except RuntimeError as e:
                    console.print(f"[red]{e}[/red]")
                    break
            except (KeyboardInterrupt, EOFError):
                break

        # Cleanup
        runner.stop()
        runner_thread.join(timeout=5)
        self._runner = None
        self.agent.confirm_handler = saved_handler
        try:
            self.agent.set_mode("interactive")
            console.print("[green]Switched back to interactive mode[/green]")
        except Exception:
            pass

        return "continuous_done"

    def _cmd_stop(self):
        """Stop the continuous mode runner."""
        if self._runner:
            self._runner.stop()
            console.print("[yellow]Stop signal sent to continuous runner[/yellow]")
        else:
            console.print("[dim]No active continuous runner[/dim]")

    def _cmd_skills(self):
        """List loaded skills."""
        if not self.agent.has_module("skill"):
            console.print("[dim]Skill module is not loaded[/dim]")
            return

        skill_mod = self.agent.get_module("skill")
        skills = skill_mod.list_skills() if skill_mod else []
        if not skills:
            console.print("[dim]No skills available[/dim]")
            return

        if HAS_RICH:
            table = Table(
                title="Available Skills",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("#", style="dim", width=4)
            table.add_column("Skill", style="cyan")
            table.add_column("Description", style="white")

            for i, s in enumerate(skills, 1):
                name = s.get("name", "unnamed") if isinstance(s, dict) else str(s)
                desc = s.get("description", "") if isinstance(s, dict) else ""
                table.add_row(str(i), name, desc)

            console.print()
            console.print(table)
        else:
            print("\nAvailable Skills:")
            for s in skills:
                name = s.get("name", "unnamed") if isinstance(s, dict) else str(s)
                print(f"  - {name}")
            print()

    def _cmd_memory(self):
        """Show memory statistics."""
        if not self.agent.has_module("memory"):
            console.print("[dim]Memory module is not loaded[/dim]")
            return

        memory_mod = self.agent.get_module("memory")
        if not memory_mod:
            console.print("[dim]Memory module is not available[/dim]")
            return

        # Try to get stats from the memory store
        try:
            store = getattr(memory_mod, "store", None)
            if store and hasattr(store, "count"):
                count = store.count()
                console.print(f"\n[bold]Memory Statistics[/bold]")
                console.print(f"  Stored facts: {count}")
            else:
                console.print(f"\n[bold]Memory[/bold]: module loaded, mode={self.agent.config.memory_mode}")
        except Exception:
            console.print(f"\n[bold]Memory[/bold]: module loaded, mode={self.agent.config.memory_mode}")

    def _setup_readline(self):
        """Set up tab completion for slash commands. Wrapped in try/except for portability."""
        try:
            import readline
            commands = list(self._slash_commands.keys()) + ["/mode"]
            def completer(text, state):
                matches = [c for c in commands if c.startswith(text)]
                return matches[state] if state < len(matches) else None
            readline.set_completer(completer)
            readline.parse_and_bind("tab: complete")
        except (ImportError, Exception):
            pass  # readline not available on all platforms


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
    ask_parser.add_argument(
        "--format", type=str, choices=["text", "json"], default="text",
        dest="output_format",
        help="Output format (default: text)",
    )

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
                if getattr(args, 'output_format', 'text') == 'json':
                    import json
                    response = agent.chat(args.question)
                    print(json.dumps({"reply": response, "model": agent.config.model}))
                else:
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
        try:
            if getattr(args, 'output_format', 'text') == 'json':
                import json
                response = agent.chat(args.question)
                print(json.dumps({"reply": response, "model": agent.config.model}))
            else:
                cli = LlamAgentCLI(agent)
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

        if setup is None:
            continue

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
