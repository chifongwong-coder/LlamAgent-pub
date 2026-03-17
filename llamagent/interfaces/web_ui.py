"""
LlamAgent Web Interface.

A web chat interface built with Gradio — LlamAgent's "pretty face".
Gradio is an optional dependency; importing this module without it installed
will raise an ImportError with installation instructions.

Usage:
    python -m llamagent --mode web
    python -m llamagent.interfaces.web_ui
    Then open your browser to http://localhost:7860

Features:
    - Chat conversation (automatic conversation history management)
    - Document upload (load into RAG knowledge base)
    - Agent status panel
    - Example questions for quick start
"""

import os

# Gradio: optional dependency, provides a friendly message if not installed
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False


# ============================================================
# Web UI construction
# ============================================================

def create_web_ui(agent) -> "gr.Blocks":
    """
    Build the Gradio Web interface.

    Takes an already-created SmartAgent instance and builds a complete
    web interaction interface around it. gr.Blocks is Gradio's "canvas" mode —
    you freely arrange components on it, more flexible than gr.Interface.

    Args:
        agent: SmartAgent instance

    Returns:
        gr.Blocks interface object

    Raises:
        ImportError: Raised when Gradio is not installed
    """
    if not HAS_GRADIO:
        raise ImportError(
            "Gradio is not installed! Please run: pip install gradio\n"
            "Then try again."
        )

    # ---- Callback functions ----
    # These closures capture the agent instance, avoiding global variables

    def chat_handler(message: str, history: list) -> str:
        """Handle user chat messages."""
        if not message.strip():
            return "Please enter your question."
        try:
            return agent.chat(message)
        except Exception as e:
            return f"Sorry, an error occurred: {e}"

    def upload_handler(files) -> str:
        """
        Handle document uploads — let users feed knowledge to LlamAgent via the web.

        Uploaded files are processed by the RAG module (if loaded):
        1. Read file content
        2. Split into chunks
        3. Vectorize and store in ChromaDB
        4. These can be retrieved during subsequent conversations
        """
        if not files:
            return "No files selected."

        if not agent.has_module("rag"):
            return (
                "RAG module is not loaded, cannot process document uploads.\n"
                "Please start with the --modules rag parameter, or load all modules."
            )

        total_chunks = 0
        results = []

        for file in files:
            try:
                file_path = file.name if hasattr(file, 'name') else str(file)
                filename = os.path.basename(file_path)

                rag_module = agent.get_module("rag")
                if hasattr(rag_module, 'load_documents'):
                    count = rag_module.load_documents(file_path)
                    total_chunks += count
                    results.append(f"  [OK] {filename}: {count} chunk(s)")
                else:
                    # Fallback: read file content directly
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    results.append(f"  [OK] {filename}: loaded ({len(content)} characters)")
                    total_chunks += 1

            except Exception as e:
                filename = os.path.basename(str(file))
                results.append(f"  [Failed] {filename}: loading failed ({e})")

        detail = "\n".join(results)
        return f"Loading complete! {total_chunks} document chunk(s) total:\n{detail}"

    def get_status_text() -> str:
        """Get formatted Agent status text — displayed in the sidebar status panel."""
        try:
            status = agent.status()
            modules = status.get("modules", {})

            lines = [
                f"**Model**: {status.get('model', 'Unknown')}",
                f"**Persona**: {status.get('persona') or 'Default'}",
                f"**Conversation Turns**: {status.get('conversation_turns', 0)}",
                "",
                "**Loaded Modules**:",
            ]

            if modules:
                for name, desc in modules.items():
                    lines.append(f"- {name}: {desc}")
            else:
                lines.append("- (No modules)")

            return "\n".join(lines)
        except Exception:
            return "Failed to retrieve status"

    def clear_chat():
        """Clear conversation history."""
        agent.clear_conversation()
        return [], "Conversation cleared."

    # ---- Build the interface ----

    custom_css = """
    .status-panel { font-size: 14px; line-height: 1.6; }
    """

    with gr.Blocks(
        title="LlamAgent",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as demo:

        # Title area
        gr.Markdown(
            """
            # LlamAgent
            **Your AI Assistant** — Supports tool calling / knowledge retrieval / reasoning & planning / multi-agent collaboration
            """
        )

        with gr.Row():
            # ---- Left side: chat + document upload (3/4 width) ----
            with gr.Column(scale=3):

                chatbot = gr.Chatbot(
                    label="LlamAgent Chat",
                    height=500,
                    type="messages",
                )

                # Example questions for guidance
                gr.Examples(
                    examples=[
                        "Hello, introduce yourself",
                        "Help me check the weather in Beijing today",
                        "What is an AI Agent? Explain in simple terms",
                        "Help me analyze the pros and cons of Python vs Go",
                    ],
                    inputs=None,
                    label="Try these questions",
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type a message...",
                        label="",
                        scale=4,
                    )
                    send_btn = gr.Button("Send", scale=1, variant="primary")

                clear_btn = gr.Button("Clear Chat", size="sm")

                # Event bindings
                def respond(message, chat_history):
                    """Process user message and return updated history."""
                    if not message.strip():
                        return chat_history or [], ""

                    response = chat_handler(message, chat_history)
                    chat_history = chat_history or []
                    chat_history.append({"role": "user", "content": message})
                    chat_history.append({"role": "assistant", "content": response})
                    return chat_history, ""

                send_btn.click(
                    fn=respond,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, msg_input],
                )
                msg_input.submit(
                    fn=respond,
                    inputs=[msg_input, chatbot],
                    outputs=[chatbot, msg_input],
                )

                # Document upload area (accordion, collapsed by default since it's less frequently used)
                with gr.Accordion("Document Upload (Load Knowledge Base)", open=False):
                    gr.Markdown(
                        "*Upload .txt / .md / .pdf files, "
                        "and LlamAgent will add them to the knowledge base "
                        "for retrieval during conversations.*"
                    )

                    with gr.Row():
                        file_upload = gr.File(
                            label="Select Files",
                            file_count="multiple",
                            file_types=[".txt", ".md", ".pdf"],
                        )
                        upload_btn = gr.Button("Upload to Knowledge Base", variant="primary")

                    upload_result = gr.Textbox(
                        label="Upload Result",
                        interactive=False,
                        lines=3,
                    )

                    upload_btn.click(
                        fn=upload_handler,
                        inputs=[file_upload],
                        outputs=[upload_result],
                    )

            # ---- Right side: status panel (1/4 width) ----
            with gr.Column(scale=1):
                gr.Markdown("### Agent Status")

                status_display = gr.Markdown(
                    value=get_status_text(),
                    elem_classes=["status-panel"],
                )

                refresh_btn = gr.Button("Refresh Status", size="sm")
                refresh_btn.click(
                    fn=get_status_text,
                    outputs=[status_display],
                )

                gr.Markdown("### Tips")
                gr.Markdown(
                    """
                    **Chat Tips**:
                    - The more specific the question, the more precise the answer
                    - After uploading documents, you can ask questions about them
                    - For complex tasks, describe them step by step

                    **Loaded modules determine LlamAgent's capabilities**:
                    - tools: Tool calling
                    - rag: Knowledge retrieval
                    - memory: Memory system
                    - reasoning: Reasoning & planning
                    - reflection: Self-reflection
                    - safety: Safety guardrails
                    """
                )

        # Clear chat button binding (needs both chatbot and status_display to be created first)
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, status_display],
        )

        # Footer
        gr.Markdown(
            """
            ---
            *LlamAgent | Built with LiteLLM + ChromaDB + Gradio*
            """,
        )

    return demo


def launch_web_ui(demo: "gr.Blocks", port: int = 7860):
    """
    Launch the Web UI server.

    Args:
        demo: The gr.Blocks object returned by create_web_ui()
        port: Listening port
    """
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


# ============================================================
# Standalone entry point
# ============================================================

def main():
    """python -m llamagent.interfaces.web_ui entry point."""
    if not HAS_GRADIO:
        print(
            "Error: Gradio is not installed!\n"
            "Please run: pip install gradio\n"
            "Then try again."
        )
        return

    from llamagent.main import create_agent

    port = int(os.getenv("WEB_UI_PORT", "7860"))
    agent = create_agent()
    demo = create_web_ui(agent)
    launch_web_ui(demo, port=port)


if __name__ == "__main__":
    main()
