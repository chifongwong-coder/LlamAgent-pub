"""
Platform built-in tools (common tier): available to all llamas out of the box.

Includes:
- web_search:  Web search (LLM-simulated, to be replaced with a real API)
- web_fetch:   Fetch page content from a specified URL

Registered to global_registry with safety_level assigned per tool characteristics.
"""

import json

from llamagent.modules.tools.registry import tool


# ============================================================
# Web search (LLM-simulated)
# ============================================================

@tool(
    name="web_search",
    description="Search the web for the latest information and return a summary of results",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keywords"}
        },
        "required": ["query"],
    },
    safety_level=1,
)
def web_search(query: str) -> str:
    """Web search (LLM-simulated). In production, replace with SerpAPI / Bing Search etc."""
    # _llm is injected by ToolsModule.on_attach()
    llm = getattr(web_search, "_llm", None)
    if llm is None:
        return json.dumps({"error": "Search function not initialized (missing LLM client)"}, ensure_ascii=False)

    try:
        result = llm.ask(
            prompt=f"Search: {query}",
            system=(
                "You are a search engine simulator. Based on the user's search query, "
                "return 3-5 concise search result summaries in JSON array format. "
                "Each result should contain title and snippet fields. "
                "Please return real, accurate information based on your knowledge."
            ),
            temperature=0.3,
        )
        return result
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
