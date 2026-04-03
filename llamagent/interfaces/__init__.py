"""
Interface layer: three ways to use LlamAgent.

- cli:        Command-line terminal interface (Rich-enhanced, falls back to plain text if not installed)
- web_ui:     Gradio web chat interface (gradio is an optional dependency)
- api_server: FastAPI RESTful API service (fastapi is an optional dependency)

Three "front desks", one "back kitchen" — the LlamAgent core class.
No matter which door the user comes through, they all use the same Agent engine.
"""

__all__ = ["cli", "web_ui", "api_server"]
