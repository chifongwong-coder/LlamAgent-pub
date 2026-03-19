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
    ("rag", "RAG — Semantic search over documents"),
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

def _build_agent(modules_list, role, persona_name, persona_desc):
    """Build a SmartAgent with the given config."""
    from llamagent.core import SmartAgent, Config, Persona
    from llamagent.main import load_module

    config = Config()
    persona = Persona(name=persona_name, role_description=persona_desc, role=role)
    agent = SmartAgent(config, persona=persona)

    for mod_name in modules_list:
        mod = load_module(mod_name)
        if mod:
            agent.register_module(mod)

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

    def build_agent_click(modules, role, name, desc, save_check):
        """Build agent from config panel."""
        if not name.strip():
            return (
                gr.update(interactive=False),
                "Please enter an agent name.",
                gr.update(),
            )

        try:
            agent = _build_agent(modules, role, name.strip(), desc.strip())
            current_agent["agent"] = agent

            # Save persona if requested
            if save_check:
                _save_persona(name.strip(), role, desc.strip())

            mod_count = len(agent.modules)
            status = (
                f"**Agent Ready!**\n\n"
                f"**Name**: {name}\n"
                f"**Role**: {role}\n"
                f"**Model**: {agent.config.model}\n"
                f"**Modules**: {mod_count} loaded"
            )
            return (
                gr.update(interactive=True),
                status,
                [],  # Clear chat history
            )

        except Exception as e:
            return (
                gr.update(interactive=False),
                f"**Build failed**: {e}",
                gr.update(),
            )

    def chat_respond(message, history):
        """Handle chat messages."""
        if not message.strip():
            return history or [], ""

        agent = current_agent.get("agent")
        if agent is None:
            history = history or []
            history.append({"role": "assistant", "content": "Please build an agent first using the configuration panel above."})
            return history, ""

        try:
            response = agent.chat(message)
        except Exception as e:
            response = f"Error: {e}"

        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, ""

    def clear_chat():
        """Clear chat history."""
        if current_agent.get("agent"):
            current_agent["agent"].clear_conversation()
        return [], ""

    def upload_handler(files):
        """Handle document uploads for RAG."""
        agent = current_agent.get("agent")
        if not agent:
            return "Please build an agent first."
        if not files:
            return "No files selected."
        if not agent.has_module("rag"):
            return "RAG module is not loaded."

        results = []
        for file in files:
            try:
                file_path = file.name if hasattr(file, 'name') else str(file)
                filename = os.path.basename(file_path)
                rag_module = agent.get_module("rag")
                if hasattr(rag_module, 'load_documents'):
                    count = rag_module.load_documents(file_path)
                    results.append(f"OK: {filename} ({count} chunks)")
                else:
                    results.append(f"OK: {filename}")
            except Exception as e:
                results.append(f"Failed: {os.path.basename(str(file))} ({e})")

        return "\n".join(results)

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

        # Document upload
        with gr.Accordion("Document Upload (RAG)", open=False):
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
            inputs=[module_checkboxes, role_radio, persona_name_input, persona_desc_input, save_checkbox],
            outputs=[msg_input, status_display, chatbot],
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

        gr.Markdown("---\n*LlamAgent v1.2.1 | Built with LiteLLM + Gradio*")

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
