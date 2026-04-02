"""
Platform built-in tools (common tier): available to all llamas out of the box.

Includes:
- ask_user:    Ask the user a question (requires interaction handler)
- web_search:  Web search via pluggable backends (DuckDuckGo / SerpAPI / Tavily)
- web_fetch:   Fetch page content from a specified URL

Registered to global_registry with safety_level assigned per tool characteristics.
"""

import json

from llamagent.modules.tools.registry import tool


# ============================================================
# User interaction
# ============================================================

@tool(
    name="ask_user",
    description="Ask the user a question to get information needed for the current task. "
                "Use this when you need clarification, missing details, or a decision.",
    parameters={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to ask"},
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of choices for the user to pick from",
            },
        },
        "required": ["question"],
    },
    tier="default",
    safety_level=1,
)
def ask_user(question: str, choices: list[str] | None = None) -> str:
    """Ask the user and return their response as a string."""
    handler = getattr(ask_user, "_handler", None)
    if handler is None:
        return "Cannot ask user: no interaction handler configured."
    try:
        return handler.ask(question, choices)
    except Exception as e:
        return f"Failed to get user response: {e}"


# ============================================================
# Web search (real search backends)
# ============================================================

@tool(
    name="web_search",
    description="Search the web for information. Returns titles, URLs, and snippets.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "num_results": {"type": "integer", "description": "Number of results to return (default 5)"},
        },
        "required": ["query"],
    },
    safety_level=1,
    pack="web",
)
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web using the configured search backend."""
    backend = getattr(web_search, "_backend", None)
    if backend is None:
        return json.dumps(
            {"error": "No search backend available. Install: pip install ddgs"},
            ensure_ascii=False,
        )

    try:
        results = backend.search(query, num_results)
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Search failed: {e}"}, ensure_ascii=False)


# ============================================================
# Web page fetching
# ============================================================

@tool(
    name="web_fetch",
    description="Fetch page content from a specified URL and return a text summary",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL of the web page to fetch"}
        },
        "required": ["url"],
    },
    safety_level=1,
    pack="web",
)
def web_fetch(url: str) -> str:
    """Fetch page content from a specified URL. Requires the requests library."""
    try:
        import requests
    except ImportError:
        return json.dumps({"error": "Missing requests library. Please install: pip install requests"}, ensure_ascii=False)

    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; LlamAgent/1.0)"
        })
        resp.raise_for_status()

        # Try to extract plain text (prefer BeautifulSoup)
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                # Remove script and style tags
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
            except ImportError:
                # No BeautifulSoup available, do simple tag removal
                import re
                text = re.sub(r"<[^>]+>", "", resp.text)
                text = re.sub(r"\s+", " ", text).strip()
        else:
            text = resp.text

        # Truncate overly long content
        max_len = 5000
        if len(text) > max_len:
            text = text[:max_len] + f"\n...(content truncated, total {len(resp.text)} characters)"

        return json.dumps({
            "url": url,
            "status_code": resp.status_code,
            "content": text,
            "length": len(text),
        }, ensure_ascii=False)

    except requests.exceptions.Timeout:
        return json.dumps({"error": f"Request timed out: {url}"}, ensure_ascii=False)
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Fetch failed: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Error processing page content: {e}"}, ensure_ascii=False)
