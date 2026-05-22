"""ModalScreens for cli_tui setup flow (plan v13 §4 C3).

SetupScreen is shown when the App mounts without a preconstructed
agent. It captures the same fields as the legacy ``interactive_setup``
in ``cli.py`` (persona name / role / role-description, plus a module
preset) and returns a dict the App uses to build a real LlamAgent.

C3 scope keeps it minimal — no "saved personas" picker, no "custom"
module checkbox grid. Both are deferrable polish that can land
alongside the persistence module work; the legacy CLI still supports
them via the input() flow. This screen is the bare 95% path that the
plan §4 C3 verification line wants:
"interactive_setup 流程完全 TUI 化；选完点 Build 进 chat".
"""
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static


# Preset name → list of module names (None means "load all modules").
# Names match the legacy cli.PRESETS so module construction is shared.
PRESET_MODULES: dict[str, list[str] | None] = {
    "Full (all modules)": None,
    "Minimal (safety + tools)": ["safety", "tools"],
    "Chat (no modules)": [],
}

PRESET_ORDER: tuple[str, ...] = tuple(PRESET_MODULES.keys())


class SetupScreen(ModalScreen[dict | None]):
    """Setup form pushed at App.on_mount when no agent was preconstructed.

    Dismisses with:
    - ``None`` if the user pressed Cancel — App exits.
    - ``{"persona_name", "persona_role", "persona_desc", "modules"}``
      otherwise. The App's ``_on_setup_done`` callback feeds this into
      ``_build_agent_from_setup`` which constructs the LlamAgent.

    The form runs entirely inside an alt-screen overlay so scrollback
    contribution stays bounded — same C0 KPI invariant as the chat
    surface (plan §9 KPI #1).
    """

    DEFAULT_CSS = """
    SetupScreen {
        align: center middle;
    }

    SetupScreen > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 70;
        height: auto;
    }

    SetupScreen Label {
        margin-top: 1;
    }

    SetupScreen Label.section {
        text-style: bold;
    }

    SetupScreen Horizontal#buttons {
        margin-top: 2;
        align: center middle;
        height: auto;
    }

    SetupScreen Button {
        margin: 0 1;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("[bold cyan]LlamAgent — Setup[/bold cyan]")
            yield Static(
                "[dim]Configure your agent. Press Build to start chatting, "
                "Esc to cancel.[/dim]"
            )

            yield Label("Module preset:", classes="section")
            yield RadioSet(
                *[
                    RadioButton(name, value=(i == 0), id=f"preset-{i}")
                    for i, name in enumerate(PRESET_ORDER)
                ],
                id="preset",
            )

            yield Label("Role:", classes="section")
            yield RadioSet(
                RadioButton("user", value=True, id="role-user"),
                RadioButton("admin", id="role-admin"),
                id="role",
            )

            yield Label("Agent name:", classes="section")
            yield Input(value="LlamAgent", id="name")

            yield Label("Role description:", classes="section")
            yield Input(value="A helpful AI assistant", id="desc")

            with Horizontal(id="buttons"):
                yield Button("Build", variant="primary", id="build")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        # Focus the first interactive widget so keyboard nav works
        # without a mouse click.
        self.query_one("#name", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        if event.button.id != "build":
            return

        preset_idx = self._selected_radio("preset", default=0)
        preset_name = PRESET_ORDER[preset_idx]
        modules = PRESET_MODULES[preset_name]

        role_idx = self._selected_radio("role", default=0)
        role = "user" if role_idx == 0 else "admin"

        name = self.query_one("#name", Input).value.strip() or "LlamAgent"
        desc = (
            self.query_one("#desc", Input).value.strip()
            or "A helpful AI assistant"
        )

        self.dismiss(
            {
                "persona_name": name,
                "persona_role": role,
                "persona_desc": desc,
                "modules": modules,
            }
        )

    def _selected_radio(self, radioset_id: str, default: int = 0) -> int:
        rs = self.query_one(f"#{radioset_id}", RadioSet)
        for i, btn in enumerate(rs.children):
            if isinstance(btn, RadioButton) and btn.value:
                return i
        return default
