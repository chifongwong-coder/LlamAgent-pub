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

import os

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False


# ============================================================
# Module definitions for config panel
# ============================================================

MODULE_OPTIONS = [
    ("safety", "Safety — Input filtering + output sanitization"),
    ("tools", "Tools — Four-tier tool system + built-ins"),
    ("sandbox", "Sandbox — Isolated execution for high-risk tools"),
    ("planning", "Planning — PlanReAct task decomposition"),
    ("reflection", "Reflection — Quality evaluation + lessons"),
    ("retrieval", "Retrieval — Knowledge retrieval over documents"),
    ("memory", "Memory — Persistent memory with recall"),
    ("multi_agent", "Multi-Agent — Role-based delegation"),
    ("child_agent", "Child Agent — Constrained sub-agents"),
    ("mcp", "MCP — Model Context Protocol bridge"),
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
            # In continuous mode, disable chat input (runner drives chat)
            chat_interactive = mode != "continuous"

            return (
                gr.update(interactive=chat_interactive),
                status,
                [],  # Clear chat history
                gr.update(visible=show_continuous),
            )

        except Exception as e:
            return (
                gr.update(interactive=False),
                f"**Build failed**: {e}",
                gr.update(),
                gr.update(visible=False),
            )

    def chat_respond(message, history):
        """Handle chat messages with streaming support."""
        if not message.strip():
            yield history or [], ""
            return

        agent = current_agent.get("agent")
        if agent is None:
            history = history or []
            history.append({"role": "assistant", "content": "Please build an agent first using the configuration panel above."})
            yield history, ""
            return

        history = history or []
        history.append({"role": "user", "content": message})

        if hasattr(agent, 'chat_stream') and agent.mode == "interactive":
            # Streaming mode
            partial = ""
            for chunk in agent.chat_stream(message):
                partial += chunk
                yield history + [{"role": "assistant", "content": partial}], ""
            if not partial:
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

        return history, f"Runner stopped. {len(log)} task(s) completed."

    def refresh_runner_click():
        """Refresh chatbot with runner results so far."""
        runner = runner_state.get("runner")
        if not runner:
            return [], "No runner active."

        log = runner.get_log()  # thread-safe copy
        history = _log_to_history(log)
        return history, f"Runner: Active | Tasks completed: {len(log)}"

    # ---- Build the interface ----

    persona_choices, _ = _get_persona_choices()

    with gr.Blocks(
        title="LlamAgent",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown("# LlamAgent\n**Build and chat with your AI agent**")

        # ============================================
        # Configuration Panel
        # ============================================
        with gr.Accordion("Agent Configuration", open=True):
            with gr.Row():
                # Left: modules
                with gr.Column(scale=2):
                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESET_CONFIGS.keys()),
                        value="Full (all modules)",
                        label="Module Preset",
                    )
                    module_checkboxes = gr.CheckboxGroup(
                        choices=[f"{m[0]}" for m in MODULE_OPTIONS],
                        value=DEFAULT_MODULES,
                        label="Modules",
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
                    scale=3,
                )

            # Build button + status
            with gr.Row():
                build_btn = gr.Button("Build Agent", variant="primary", scale=1)
                status_display = gr.Markdown("*Configure and click Build Agent to start*", scale=3)

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
                placeholder="Type a message... (build agent first)",
                label="",
                scale=4,
                interactive=False,
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")

        with gr.Row():
            clear_btn = gr.Button("Clear Chat", size="sm")

        # Continuous mode runner panel (hidden by default)
        with gr.Accordion("Continuous Runner", open=True, visible=False) as continuous_panel:
            with gr.Row():
                trigger_type = gr.Dropdown(
                    choices=["Timer", "File"],
                    value="Timer",
                    label="Trigger Type",
                    scale=1,
                )
                timer_interval = gr.Textbox(value="60", label="Interval (seconds)", scale=1)
                timer_message = gr.Textbox(value="check system status", label="Task Message", scale=2)
            with gr.Row():
                file_watch_dir = gr.Textbox(value=".", label="Watch Directory (for File trigger)", scale=3)
            with gr.Row():
                start_runner_btn = gr.Button("Start Runner", variant="primary", scale=1)
                stop_runner_btn = gr.Button("Stop Runner", variant="stop", scale=1)
                refresh_runner_btn = gr.Button("Refresh Results", scale=1)
            runner_status = gr.Textbox(label="Runner Status", interactive=False, lines=1)

        # Document upload
        with gr.Accordion("Document Upload (Retrieval)", open=False):
            with gr.Row():
                file_upload = gr.File(
                    label="Select Files",
                    file_count="multiple",
                    file_types=[".txt", ".md", ".pdf"],
                )
                upload_btn = gr.Button("Upload", variant="primary")
            upload_result = gr.Textbox(label="Result", interactive=False, lines=2)
            upload_btn.click(fn=upload_handler, inputs=[file_upload], outputs=[upload_result])

        # ---- Event bindings ----

        build_btn.click(
            fn=build_agent_click,
            inputs=[module_checkboxes, role_radio, persona_name_input, persona_desc_input, save_checkbox, mode_dropdown],
            outputs=[msg_input, status_display, chatbot, continuous_panel],
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

        from llamagent import __version__
        gr.Markdown(f"---\n*LlamAgent v{__version__} | Built with LiteLLM + Gradio*")

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
