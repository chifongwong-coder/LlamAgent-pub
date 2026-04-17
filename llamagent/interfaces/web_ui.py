"""
LlamAgent Web Interface.

A single-page web interface built with Gradio:
- Top: Configuration panel (dropdowns, checkboxes, radio buttons)
- Bottom: Chat area (activated after agent is built)

Usage:
    python -m llamagent --mode web
    python -m llamagent.interfaces.web_ui
    Then open your browser to http://localhost:7860
"""

import html
import os

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False


# ------------------------------------------------------------
# Structured-event rendering helpers (v3.0.3)
# ------------------------------------------------------------

def _render_segments(segments: list[dict]) -> str:
    """Render a list of segments to Markdown/HTML suitable for gr.Chatbot."""
    parts = []
    for seg in segments:
        kind = seg["type"]
        if kind == "text":
            parts.append(seg["text"])
        elif kind == "tool":
            name = html.escape(seg["name"])
            if seg.get("closed"):
                if seg.get("success"):
                    icon = "✅"
                    suffix = f" · {seg['duration_ms']}ms" if seg.get("duration_ms") is not None else ""
                else:
                    icon = "❌"
                    err = seg.get("error") or "failed"
                    suffix = f" — {html.escape(str(err))}"
                parts.append(
                    f"\n<details><summary>{icon} <code>{name}</code>{suffix}</summary></details>\n"
                )
            else:
                parts.append(
                    f"\n<details open><summary>⏳ Calling <code>{name}</code>...</summary></details>\n"
                )
        elif kind == "status":
            parts.append(f"\n> _{html.escape(seg['message'])}_\n")
        elif kind == "error":
            parts.append(f"\n> **❌ Error:** {html.escape(seg['message'])}\n")
    return "".join(parts).strip() or "(empty response)"


def _apply_event(segments: list[dict], index: dict, event: dict) -> None:
    """Mutate `segments` / `index` in place according to a single event."""
    etype = event["type"]
    if etype == "content":
        if segments and segments[-1]["type"] == "text":
            segments[-1]["text"] += event["text"]
        else:
            segments.append({"type": "text", "text": event["text"]})
    elif etype == "tool_call_start":
        segments.append({
            "type": "tool",
            "call_id": event["call_id"],
            "name": event["name"],
            "closed": False,
        })
        index[event["call_id"]] = len(segments) - 1
    elif etype == "tool_call_end":
        pos = index.get(event["call_id"])
        if pos is not None:
            seg = segments[pos]
            seg["closed"] = True
            seg["success"] = event["success"]
            seg["duration_ms"] = event.get("duration_ms")
            seg["error"] = event.get("error")
    elif etype == "status":
        segments.append({"type": "status", "message": event["message"]})
    elif etype == "error":
        segments.append({"type": "error", "message": event["message"]})
    # "done" triggers no visual change


# ============================================================
# Module definitions for config panel (from shared presets)
# ============================================================

from llamagent.interfaces.presets import MODULE_DESCRIPTIONS, MODULE_GROUPS, apply_presets

MODULE_OPTIONS = [
    (name, MODULE_DESCRIPTIONS.get(name, name))
    for group in MODULE_GROUPS.values()
    for name in group
]

DEFAULT_MODULES = [m[0] for m in MODULE_OPTIONS]

PRESET_CONFIGS = {
    "Full (all modules)": DEFAULT_MODULES,
    "Minimal (safety + tools)": ["safety", "tools"],
    "Chat only (no modules)": [],
}


# ============================================================
# Persona persistence helpers
# ============================================================

def _load_saved_personas() -> list:
    """Load saved personas from disk."""
    from llamagent.core import Config, PersonaManager
    try:
        config = Config()
        manager = PersonaManager(config.persona_file)
        return manager.list()
    except Exception:
        return []


def _save_persona(name, role, desc):
    """Save a persona to disk."""
    from llamagent.core import Config, PersonaManager
    try:
        config = Config()
        manager = PersonaManager(config.persona_file)
        existing = manager.get(name.lower().replace(" ", ""))
        if not existing:
            manager.create(name=name, role_description=desc, role=role)
            return True
    except Exception:
        pass
    return False


def _get_persona_choices():
    """Get persona dropdown choices: saved personas + 'Create new'."""
    saved = _load_saved_personas()
    choices = [f"{p.name} ({p.role})" for p in saved]
    choices.append("+ Create new persona")
    return choices, saved


# ============================================================
# Agent builder
# ============================================================

def _build_agent(modules_list, role, persona_name, persona_desc, mode="interactive"):
    """Build a LlamAgent with the given config and mode."""
    from llamagent.core import LlamAgent, Config, Persona
    from llamagent.core.zone import ConfirmResponse
    from llamagent.main import load_module

    config = Config()

    # Apply smart defaults for selected modules before building agent
    apply_presets(config, modules_list)

    persona = Persona(name=persona_name, role_description=persona_desc, role=role)
    agent = LlamAgent(config, persona=persona)

    for mod_name in modules_list:
        mod = load_module(mod_name)
        if mod:
            agent.register_module(mod)

    # Auto-approve for Web UI (Gradio cannot show mid-flow confirm dialogs;
    # task mode contract provides the confirmation step)
    agent.confirm_handler = lambda req: ConfirmResponse(allow=True)

    if mode != "interactive":
        agent.set_mode(mode)

    return agent


# ============================================================
# Web UI construction
# ============================================================

def create_web_ui() -> "gr.Blocks":
    """Build the Gradio single-page interface with config panel + chat."""
    if not HAS_GRADIO:
        raise ImportError(
            "Gradio is not installed! Please run: pip install gradio\n"
            "Then try again."
        )

    # Shared state
    current_agent = {"agent": None}
    runner_state = {"runner": None, "thread": None}

    # ---- Callback functions ----

    def on_preset_change(preset_name):
        """When preset dropdown changes, update module checkboxes."""
        modules = PRESET_CONFIGS.get(preset_name, DEFAULT_MODULES)
        return modules

    def on_persona_dropdown_change(choice):
        """When persona dropdown changes, fill in details or show create fields."""
        _, saved = _get_persona_choices()
        if choice == "+ Create new persona":
            return "user", "LlamAgent", "A helpful AI assistant", gr.update(visible=True)

        # Find the selected persona
        for p in saved:
            label = f"{p.name} ({p.role})"
            if label == choice:
                return p.role, p.name, p.role_description or "", gr.update(visible=False)

        return "user", "LlamAgent", "A helpful AI assistant", gr.update(visible=True)

    def build_agent_click(modules, role, name, desc, save_check, mode):
        """Build agent from config panel."""
        if not name.strip():
            return (
                gr.update(interactive=False),
                "Please enter an agent name.",
                gr.update(),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        # Stop old runner first (before agent shutdown, since runner calls agent.chat)
        old_runner = runner_state.get("runner")
        if old_runner:
            old_runner.stop()
            old_thread = runner_state.get("thread")
            if old_thread:
                old_thread.join(timeout=5)
            runner_state["runner"] = None
            runner_state["thread"] = None

        # Shutdown old agent after runner is stopped
        old_agent = current_agent.get("agent")
        if old_agent:
            try:
                old_agent.shutdown()
            except Exception:
                pass

        try:
            agent = _build_agent(modules, role, name.strip(), desc.strip(), mode=mode)
            current_agent["agent"] = agent

            # Save persona if requested
            if save_check:
                _save_persona(name.strip(), role, desc.strip())

            mod_count = len(agent.modules)
            status = (
                f"**Agent Ready!**\n\n"
                f"**Name**: {name}\n"
                f"**Role**: {role}\n"
                f"**Mode**: {mode}\n"
                f"**Model**: {agent.config.model}\n"
                f"**Modules**: {mod_count} loaded"
            )

            # Show continuous panel only for continuous mode
            show_continuous = mode == "continuous"
            # Chat input is always enabled: in continuous mode, input routes to inject
            chat_interactive = True

            # Show skills/memory panels if respective modules are loaded
            show_skills = agent.has_module("skill")
            show_memory = agent.has_module("memory")

            return (
                gr.update(interactive=chat_interactive),
                status,
                [],  # Clear chat history
                gr.update(visible=show_continuous),
                gr.update(visible=show_skills),
                gr.update(visible=show_memory),
            )

        except Exception as e:
            return (
                gr.update(interactive=False),
                f"**Build failed**: {e}",
                gr.update(),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

    def chat_respond(message, history):
        """Handle chat messages with streaming support. Routes to inject in continuous mode."""
        if not message.strip():
            yield history or [], ""
            return

        agent = current_agent.get("agent")
        if agent is None:
            history = history or []
            history.append({"role": "assistant", "content": "Please build an agent first using the configuration panel above."})
            yield history, ""
            return

        # W1: Continuous mode -- route to runner.inject()
        runner = runner_state.get("runner")
        if runner and not runner._stopped.is_set():
            try:
                response = runner.inject(message)
            except Exception as e:
                response = f"Error: {e}"
            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            yield history, ""
            return

        history = history or []
        history.append({"role": "user", "content": message})

        if hasattr(agent, 'chat_stream') and agent.mode == "interactive":
            # Streaming mode with structured-event rendering (v3.0.3)
            from llamagent.interfaces.stream_adapter import adapt_stream

            segments: list[dict] = []
            tool_index: dict[str, int] = {}
            any_event = False
            for event in adapt_stream(agent.chat_stream(message)):
                any_event = True
                _apply_event(segments, tool_index, event)
                yield history + [{"role": "assistant", "content": _render_segments(segments)}], ""
            if not any_event or not segments:
                yield history + [{"role": "assistant", "content": "(empty response)"}], ""
        else:
            # Non-streaming fallback
            try:
                response = agent.chat(message)
            except Exception as e:
                response = f"Error: {e}"
            yield history + [{"role": "assistant", "content": response}], ""

    def clear_chat():
        """Clear chat history."""
        if current_agent.get("agent"):
            current_agent["agent"].clear_conversation()
        return [], ""

    def upload_handler(files):
        """Handle document uploads for knowledge base."""
        agent = current_agent.get("agent")
        if not agent:
            return "Please build an agent first."
        if not files:
            return "No files selected."
        if not agent.has_module("retrieval"):
            return "Retrieval module is not loaded."

        results = []
        for file in files:
            try:
                file_path = file.name if hasattr(file, 'name') else str(file)
                filename = os.path.basename(file_path)
                retrieval_module = agent.get_module("retrieval")
                count = retrieval_module.load_documents(file_path)
                results.append(f"OK: {filename} ({count} chunks)")
            except Exception as e:
                results.append(f"Failed: {os.path.basename(str(file))} ({e})")

        return "\n".join(results)

    def start_runner_click(trigger_type, timer_interval, timer_message, file_watch_dir):
        """Start ContinuousRunner with configured trigger."""
        agent = current_agent.get("agent")
        if not agent:
            return "Please build an agent first."

        if runner_state.get("runner"):
            return "Runner is already active. Stop it first."

        import threading
        from llamagent.core.runner import ContinuousRunner, TimerTrigger, FileTrigger

        triggers = []
        if trigger_type == "Timer":
            try:
                interval = float(timer_interval or "60")
            except ValueError:
                return "Invalid interval value."
            triggers.append(TimerTrigger(interval=interval, message=timer_message or "check status"))
        else:
            watch_dir = file_watch_dir or "."
            triggers.append(FileTrigger(watch_dir))

        runner = ContinuousRunner(agent, triggers, poll_interval=1.0)
        runner_state["runner"] = runner

        t = threading.Thread(target=runner.run, daemon=True)
        runner_state["thread"] = t
        t.start()

        return "Runner started. Click 'Refresh' to see results, 'Stop' to end."

    def _log_to_history(log: list):
        """Convert task log entries to chat history format."""
        history = []
        for entry in log:
            history.append({"role": "user", "content": f"[{entry.trigger_type}] {entry.input}"})
            content = entry.output if entry.status == "completed" else f"Error: {entry.error}"
            history.append({"role": "assistant", "content": content})
        return history

    def stop_runner_click():
        """Stop ContinuousRunner."""
        runner = runner_state.get("runner")
        if not runner:
            return [], "No runner active."

        runner.stop()
        t = runner_state.get("thread")
        if t:
            t.join(timeout=5)

        log = runner.get_log()  # thread-safe copy
        history = _log_to_history(log)

        runner_state["runner"] = None
        runner_state["thread"] = None

        # Switch agent back to interactive mode (consistent with CLI/API)
        agent = current_agent.get("agent")
        if agent:
            try:
                agent.set_mode("interactive")
            except Exception:
                pass

        return history, f"Runner stopped. {len(log)} task(s) completed."

    def refresh_runner_click():
        """Refresh chatbot with runner results so far."""
        runner = runner_state.get("runner")
        if not runner:
            return [], "No runner active."

        log = runner.get_log()  # thread-safe copy
        history = _log_to_history(log)
        return history, f"Runner: Active | Tasks completed: {len(log)}"

    def refresh_skills_click():
        """Scan agent's skill module and return skills dataframe."""
        agent = current_agent.get("agent")
        if not agent or not agent.has_module("skill"):
            return [], ""

        skill_mod = agent.get_module("skill")
        if not skill_mod:
            return [], ""

        try:
            skills = skill_mod.list_skills()
        except Exception:
            skills = []

        if not skills:
            return [], "No skills available."

        rows = []
        detail_parts = []
        for s in skills:
            if isinstance(s, dict):
                name = s.get("name", "unnamed")
                desc = s.get("description", "")
                fmt = s.get("format", "")
                rows.append([name, desc, fmt])
                detail_parts.append(f"- {name}: {desc}")
            else:
                rows.append([str(s), "", ""])
                detail_parts.append(f"- {s}")

        detail_text = f"{len(rows)} skill(s) loaded:\n" + "\n".join(detail_parts)
        return rows, detail_text

    def refresh_memory_click():
        """Get memory module statistics."""
        agent = current_agent.get("agent")
        if not agent or not agent.has_module("memory"):
            return "Memory module not loaded."

        memory_mod = agent.get_module("memory")
        if not memory_mod:
            return "Memory module not available."

        try:
            store = getattr(memory_mod, "store", None)
            if store and hasattr(store, "count"):
                count = store.count()
                return f"Stored facts: {count}\nMode: {agent.config.memory_mode}"
            else:
                return f"Memory module loaded. Mode: {agent.config.memory_mode}"
        except Exception:
            return f"Memory module loaded. Mode: {agent.config.memory_mode}"

    def search_memory_click(query):
        """Search memories by keyword."""
        agent = current_agent.get("agent")
        if not agent or not agent.has_module("memory"):
            return "Memory module not loaded."

        if not query or not query.strip():
            return "Please enter a search query."

        memory_mod = agent.get_module("memory")
        if not memory_mod:
            return "Memory module not available."

        try:
            store = getattr(memory_mod, "store", None)
            if store and hasattr(store, "search"):
                results = store.search(query.strip(), top_k=5)
                if not results:
                    return "No matching memories found."
                parts = []
                for i, r in enumerate(results, 1):
                    if isinstance(r, dict):
                        text = r.get("text", r.get("content", str(r)))
                    else:
                        text = str(r)
                    parts.append(f"{i}. {text[:200]}")
                return "\n".join(parts)
            else:
                return "Memory store does not support search."
        except Exception as e:
            return f"Search error: {e}"

    def refresh_sessions_click():
        """Scan persistence directory and return sessions dataframe."""
        agent = current_agent.get("agent")
        if not agent:
            return [], "Build an agent first."

        from llamagent.interfaces.sessions import list_sessions, format_time_ago
        sessions = list_sessions(agent)
        if not sessions:
            return [], "No saved sessions. Enable persistence module."

        rows = []
        persistence_mod = agent.modules.get("persistence")
        current_file = getattr(persistence_mod, "_filename", "") if persistence_mod else ""
        for s in sessions:
            name = s["persona_id"]
            if s["filename"] == current_file:
                name += " (current)"
            age = format_time_ago(s["last_modified"])
            preview = s["summary"][:60] or s["preview"][:60] or ""
            rows.append([name, s["turns"], age, preview])

        return rows, f"{len(sessions)} session(s) found."

    # ---- Build the interface ----

    persona_choices, _ = _get_persona_choices()

    with gr.Blocks(
        title="LlamAgent",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown("# LlamAgent\n**Configure your agent, then start chatting**")

        # ============================================
        # Configuration Panel
        # ============================================
        with gr.Accordion("Setup", open=True):
            with gr.Row():
                # Left: modules
                with gr.Column(scale=2):
                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESET_CONFIGS.keys()),
                        value="Full (all modules)",
                        label="Quick Setup",
                    )
                    module_checkboxes = gr.CheckboxGroup(
                        choices=[f"{m[0]}" for m in MODULE_OPTIONS],
                        value=DEFAULT_MODULES,
                        label="Modules (check to enable)",
                    )
                    preset_dropdown.change(
                        fn=on_preset_change,
                        inputs=[preset_dropdown],
                        outputs=[module_checkboxes],
                    )

                # Right: persona
                with gr.Column(scale=2):
                    persona_dropdown = gr.Dropdown(
                        choices=persona_choices,
                        value=persona_choices[0] if len(persona_choices) > 1 else "+ Create new persona",
                        label="Persona",
                    )
                    role_radio = gr.Radio(
                        choices=["user", "admin"],
                        value="user",
                        label="Role",
                    )
                    with gr.Group(visible=True) as create_fields:
                        persona_name_input = gr.Textbox(
                            value="LlamAgent",
                            label="Agent Name",
                        )
                        persona_desc_input = gr.Textbox(
                            value="A helpful AI assistant",
                            label="Role Description",
                        )
                    save_checkbox = gr.Checkbox(
                        value=True,
                        label="Save persona for next time",
                    )

                    persona_dropdown.change(
                        fn=on_persona_dropdown_change,
                        inputs=[persona_dropdown],
                        outputs=[role_radio, persona_name_input, persona_desc_input, create_fields],
                    )

            # Mode selection
            with gr.Row():
                mode_dropdown = gr.Dropdown(
                    choices=["interactive", "task", "continuous"],
                    value="interactive",
                    label="Agent Mode",
                    scale=1,
                )
                gr.Markdown(
                    "*interactive: per-turn chat | task: prepare/confirm/execute | continuous: trigger-driven*",
                )

            # Build button + status
            with gr.Row():
                build_btn = gr.Button("Build Agent", variant="primary", scale=1)
                status_display = gr.Markdown("*Configure and click Build Agent to start*")

        # ============================================
        # Chat Area
        # ============================================
        chatbot = gr.Chatbot(
            label="Chat",
            height=450,
            type="messages",
        )

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Build your agent first using the panel above, then type here...",
                label="",
                scale=4,
                interactive=False,
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")

        with gr.Row():
            clear_btn = gr.Button("Clear Chat", size="sm")

        # Continuous mode runner panel (hidden by default)
        with gr.Accordion("Background Runner", open=True, visible=False) as continuous_panel:
            with gr.Row():
                trigger_type = gr.Dropdown(
                    choices=["Timer", "File"],
                    value="Timer",
                    label="Trigger Type",
                    scale=1,
                )
                timer_interval = gr.Textbox(value="60", label="Interval (seconds)", scale=1)
                timer_message = gr.Textbox(value="check system status", label="What should the agent do?", scale=2)
            with gr.Row():
                file_watch_dir = gr.Textbox(value=".", label="Watch Directory (for File trigger)", scale=3)
            with gr.Row():
                start_runner_btn = gr.Button("Start Runner", variant="primary", scale=1)
                stop_runner_btn = gr.Button("Stop Runner", variant="stop", scale=1)
                refresh_runner_btn = gr.Button("Refresh Results", scale=1)
            runner_status = gr.Textbox(label="Runner Status", interactive=False, lines=1)

        # Document upload
        with gr.Accordion("Upload Documents", open=False):
            with gr.Row():
                file_upload = gr.File(
                    label="Select Files",
                    file_count="multiple",
                    file_types=[".txt", ".md", ".pdf"],
                )
                upload_btn = gr.Button("Upload", variant="primary")
            upload_result = gr.Textbox(label="Result", interactive=False, lines=2)
            upload_btn.click(fn=upload_handler, inputs=[file_upload], outputs=[upload_result])

        # Skills panel (visible when skill module is loaded)
        with gr.Accordion("Skills", open=False, visible=False) as skills_panel:
            skills_table = gr.Dataframe(
                headers=["Name", "Description", "Format"],
                label="Available Skills",
            )
            skill_detail = gr.Textbox(label="Skill Details", lines=8, interactive=False)
            refresh_skills_btn = gr.Button("Refresh", size="sm")

        # Memory panel (visible when memory module is loaded)
        with gr.Accordion("Memory", open=False, visible=False) as memory_panel:
            memory_stats_display = gr.Textbox(label="Stats", interactive=False, lines=2)
            memory_search_input = gr.Textbox(label="Search", placeholder="Search by keyword or topic...")
            memory_results_display = gr.Textbox(label="Results", interactive=False, lines=5)
            refresh_memory_btn = gr.Button("Refresh Stats", size="sm")
            search_memory_btn = gr.Button("Search", size="sm")

        # Sessions panel
        with gr.Accordion("Sessions", open=False):
            session_table = gr.Dataframe(
                headers=["Name", "Turns", "Last Active", "Preview"],
                label="Saved Sessions",
                interactive=False,
            )
            with gr.Row():
                refresh_sessions_btn = gr.Button("Refresh", size="sm")
            session_status = gr.Textbox(label="", interactive=False, lines=1)

        # ---- Event bindings ----

        build_btn.click(
            fn=build_agent_click,
            inputs=[module_checkboxes, role_radio, persona_name_input, persona_desc_input, save_checkbox, mode_dropdown],
            outputs=[msg_input, status_display, chatbot, continuous_panel, skills_panel, memory_panel],
        )

        send_btn.click(
            fn=chat_respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )
        msg_input.submit(
            fn=chat_respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )
        clear_btn.click(fn=clear_chat, outputs=[chatbot, msg_input])

        # Continuous runner bindings
        start_runner_btn.click(
            fn=start_runner_click,
            inputs=[trigger_type, timer_interval, timer_message, file_watch_dir],
            outputs=[runner_status],
        )
        stop_runner_btn.click(
            fn=stop_runner_click,
            outputs=[chatbot, runner_status],
        )
        refresh_runner_btn.click(
            fn=refresh_runner_click,
            outputs=[chatbot, runner_status],
        )

        # Skills panel bindings
        refresh_skills_btn.click(
            fn=refresh_skills_click,
            outputs=[skills_table, skill_detail],
        )

        # Memory panel bindings
        refresh_memory_btn.click(
            fn=refresh_memory_click,
            outputs=[memory_stats_display],
        )
        search_memory_btn.click(
            fn=search_memory_click,
            inputs=[memory_search_input],
            outputs=[memory_results_display],
        )

        # Sessions panel bindings
        refresh_sessions_btn.click(
            fn=refresh_sessions_click,
            outputs=[session_table, session_status],
        )

        from llamagent import __version__
        gr.Markdown(f"---\n*LlamAgent v{__version__}*")

    return demo


def launch_web_ui(demo: "gr.Blocks", port: int = 7860):
    """Launch the Web UI server."""
    print(f"\n{'='*50}")
    print(f"  LlamAgent Web UI")
    print(f"  URL: http://localhost:{port}")
    print(f"{'='*50}\n")

    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        show_api=False,
    )


def main():
    """python -m llamagent.interfaces.web_ui entry point."""
    if not HAS_GRADIO:
        print(
            "Error: Gradio is not installed!\n"
            "Please run: pip install gradio\n"
            "Then try again."
        )
        return

    port = int(os.getenv("WEB_UI_PORT", "7860"))
    demo = create_web_ui()
    launch_web_ui(demo, port=port)


if __name__ == "__main__":
    main()
