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
from typing import Optional

from rich.markup import escape as markup_escape
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

    # Textual's AUTO_FOCUS selector tells the framework which widget
    # should receive initial focus when the screen mounts. Without it
    # the screen attaches but no child widget owns focus, so keyboard
    # events go nowhere (test B 5/23 — "打字 / Esc / Tab 全无反应").
    AUTO_FOCUS = "#name"

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

            # Round-10 MED A-MED-2: placeholder rather than prefilled value
            # so the user doesn't have to select-all-delete to type a real
            # name. The empty-submit fallback below restores the default.
            yield Label("Agent name:", classes="section")
            yield Input(placeholder="LlamAgent", id="name")

            yield Label("Role description:", classes="section")
            yield Input(placeholder="A helpful AI assistant", id="desc")

            with Horizontal(id="buttons"):
                yield Button("Build", variant="primary", id="build")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        # AUTO_FOCUS handles the initial focus declaratively; this
        # explicit focus() is a belt-and-suspenders fallback for
        # Textual versions where AUTO_FOCUS resolves after on_mount.
        self.call_after_refresh(lambda: self.query_one("#name", Input).focus())

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        if event.button.id == "build":
            self._do_build()

    def on_input_submitted(self, event: "Input.Submitted") -> None:
        """Round-10 HIGH A-MED-1: Enter inside an Input triggers Build
        instead of dying as an unhandled event. RadioSet doesn't emit
        Input.Submitted, so the radios stay keyboard-toggleable without
        accidentally submitting the form."""
        self._do_build()

    def _do_build(self) -> None:
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


# ---------------------------------------------------------------------------
# ConfirmModal — authorization prompts (plan §4 C4)
# ---------------------------------------------------------------------------


class ConfirmModal(ModalScreen[Optional[bool]]):
    """Authorization confirm dialog. ``dismiss(True)`` = allow,
    ``dismiss(False)`` = deny, ``dismiss(None)`` = sentinel (Esc/abort).

    The bridge maps the dismiss value to a ``ConfirmResponse``:
    True → allow=True, False/None → allow=False (no separate "abort"
    state, matching plan §1.4 ConfirmResponse schema which has no
    reason field).
    """

    DEFAULT_CSS = """
    ConfirmModal {
        align: center middle;
    }
    ConfirmModal > Vertical {
        background: $surface;
        border: thick $warning;
        padding: 1 2;
        width: 70;
        height: auto;
    }
    ConfirmModal Static.confirm-title {
        text-style: bold;
        color: $warning;
    }
    ConfirmModal Static.confirm-field {
        margin-top: 1;
    }
    ConfirmModal Horizontal#confirm-buttons {
        margin-top: 2;
        align: center middle;
        height: auto;
    }
    """

    BINDINGS = [
        ("escape", "dismiss_sentinel", "Cancel"),
        ("y", "approve", "Allow"),
        ("n", "deny", "Deny"),
    ]

    AUTO_FOCUS = "#confirm-allow"

    def __init__(self, request) -> None:
        super().__init__()
        # ConfirmRequest fields per core/zone.py:73-90 — accessed
        # defensively in case framework adds new optional fields later.
        self.request = request

    def compose(self) -> ComposeResult:
        req = self.request
        with Vertical():
            # Round-11 H2 — kind / mode header so the user can tell a
            # session_authorize from an operation_confirm at a glance.
            kind = getattr(req, "kind", None) or "confirm"
            mode = getattr(req, "mode", None) or "-"
            yield Static(
                f"Authorization required  "
                f"[dim](kind={markup_escape(str(kind))} · mode={markup_escape(str(mode))})[/dim]",
                classes="confirm-title",
            )
            tool = getattr(req, "tool_name", "?")
            action = getattr(req, "action", None) or "-"
            zone = getattr(req, "zone", None) or "-"
            yield Static(
                f"tool: [cyan]{markup_escape(str(tool))}[/cyan]  "
                f"action: [yellow]{markup_escape(str(action))}[/yellow]  "
                f"zone: [magenta]{markup_escape(str(zone))}[/magenta]",
                classes="confirm-field",
            )
            paths = getattr(req, "target_paths", None) or []
            if paths:
                paths_text = "\n  ".join(markup_escape(str(p)) for p in paths[:8])
                more = f"\n  …(+{len(paths) - 8} more)" if len(paths) > 8 else ""
                yield Static(
                    f"paths:\n  {paths_text}{more}",
                    classes="confirm-field",
                )
            # Round-11 H1 — requested_scopes is the v1.9.1 contract
            # scope grant payload (zone / actions / paths / tool_names).
            # Without rendering it, session_authorize prompts ask the
            # user to approve an opaque request — security UX regression
            # vs legacy CLI. Defensive on shape because RequestedScope
            # is a dataclass that may evolve.
            scopes = getattr(req, "requested_scopes", None) or []
            if scopes:
                lines = []
                for s in scopes[:4]:
                    zn = getattr(s, "zone", "?")
                    acts = getattr(s, "actions", None) or []
                    paths_pref = getattr(s, "path_prefixes", None) or []
                    tools = getattr(s, "tool_names", None) or []
                    parts = [f"zone={markup_escape(str(zn))}"]
                    if acts:
                        parts.append(f"actions={markup_escape(','.join(map(str, acts)))}")
                    if paths_pref:
                        parts.append(f"paths={markup_escape(','.join(map(str, paths_pref[:3])))}")
                    if tools:
                        parts.append(f"tools={markup_escape(','.join(map(str, tools[:3])))}")
                    lines.append("  · " + " · ".join(parts))
                more = f"\n  · …(+{len(scopes) - 4} more)" if len(scopes) > 4 else ""
                yield Static(
                    "requested scopes:\n" + "\n".join(lines) + more,
                    classes="confirm-field",
                )
            message = getattr(req, "message", None)
            if message:
                yield Static(
                    f"[dim]{markup_escape(str(message))}[/dim]",
                    classes="confirm-field",
                )
            yield Static(
                "[dim]y / Enter = allow · n / Esc = deny[/dim]",
                classes="confirm-field",
            )
            with Horizontal(id="confirm-buttons"):
                yield Button("Allow", variant="success", id="confirm-allow")
                yield Button("Deny", variant="error", id="confirm-deny")

    def action_dismiss_sentinel(self) -> None:
        self.dismiss(None)

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-allow":
            self.dismiss(True)
        elif event.button.id == "confirm-deny":
            self.dismiss(False)


# ---------------------------------------------------------------------------
# AskUserModal — UserInteractionHandler.ask backing (plan §4 C4)
# ---------------------------------------------------------------------------


class AskUserModal(ModalScreen[Optional[str]]):
    """Free-form or multiple-choice prompt from the ``ask_user`` tool.

    ``dismiss(str)`` = user answer (free-form text or selected choice).
    ``dismiss(None)`` = sentinel (Esc/abort) — bridge returns empty
    string to the tool so it can short-circuit instead of hanging.
    """

    DEFAULT_CSS = """
    AskUserModal {
        align: center middle;
    }
    AskUserModal > Vertical {
        background: $surface;
        border: thick $accent;
        padding: 1 2;
        width: 70;
        height: auto;
    }
    AskUserModal Static.ask-title {
        text-style: bold;
        color: $accent;
    }
    AskUserModal Label.section {
        margin-top: 1;
        text-style: bold;
    }
    AskUserModal Horizontal#ask-buttons {
        margin-top: 2;
        align: center middle;
        height: auto;
    }
    """

    BINDINGS = [("escape", "dismiss_sentinel", "Cancel")]

    AUTO_FOCUS = "#ask-input"

    def __init__(self, question: str, choices: Optional[list] = None) -> None:
        super().__init__()
        self.question = question or "(no prompt)"
        self.choices = list(choices) if choices else []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Agent asks:", classes="ask-title")
            yield Static(markup_escape(self.question))
            if self.choices:
                yield Label("Pick one (or type your own below):", classes="section")
                yield RadioSet(
                    *[
                        RadioButton(markup_escape(str(c)), id=f"ask-choice-{i}")
                        for i, c in enumerate(self.choices)
                    ],
                    id="ask-choices",
                )
            yield Label("Free-form answer:", classes="section")
            yield Input(placeholder="type and press Enter", id="ask-input")
            with Horizontal(id="ask-buttons"):
                yield Button("Submit", variant="primary", id="ask-submit")
                yield Button("Cancel", id="ask-cancel")

    def action_dismiss_sentinel(self) -> None:
        self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._do_submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ask-submit":
            self._do_submit()
        elif event.button.id == "ask-cancel":
            self.dismiss(None)

    def _do_submit(self) -> None:
        text = self.query_one("#ask-input", Input).value.strip()
        if text:
            self.dismiss(text)
            return
        # Empty text — fall back to selected radio choice if any
        if self.choices:
            rs = self.query_one("#ask-choices", RadioSet)
            for i, btn in enumerate(rs.children):
                if isinstance(btn, RadioButton) and btn.value:
                    self.dismiss(str(self.choices[i]))
                    return
        # No text + no radio selected — treat as cancel.
        self.dismiss(None)
