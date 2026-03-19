"""
Platform built-in tools (common tier): available to all llamas out of the box.

Includes:
- web_search:       Web search (LLM-simulated, to be replaced with a real API)
- web_fetch:        Fetch page content from a specified URL
- execute_command:  Execute a shell command (high risk, requires permission)
- read_file:        Read a file (read-only)
- write_file:       Write a file to the output directory (has side effects)

Registered to global_registry with safety_level assigned per tool characteristics.
"""

import os
import json
import subprocess

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


# ============================================================
# Shell command execution (high risk)
# ============================================================

@tool(
    name="execute_command",
    description="Execute a shell command and return its output (admin only)",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to execute"}
        },
        "required": ["command"],
    },
    tier="admin",
    safety_level=2,
)
def execute_command(command: str) -> str:
    """
    Execute a shell command. Admin-only operation (tier=admin).

    Internally calls the safety module's check_command() for command blacklist checking.
    The _agent attribute is injected by ToolsModule.on_attach() to access the safety module.
    """
    # Access the safety module via the agent reference for command blacklist checking
    agent = getattr(execute_command, "_agent", None)
    if agent:
        safety_mod = agent.get_module("safety")
        if safety_mod:
            rejection = safety_mod.check_command(command)
            if rejection:
                return json.dumps({
                    "status": "rejected",
                    "command": command,
                    "reason": rejection,
                }, ensure_ascii=False)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = result.stdout or ""
        error = result.stderr or ""

        # Truncate overly long output
        max_len = 5000
        if len(output) > max_len:
            output = output[:max_len] + f"\n...(output truncated, total {len(result.stdout)} characters)"
        if len(error) > max_len:
            error = error[:max_len] + f"\n...(error output truncated)"

        return json.dumps({
            "status": "success" if result.returncode == 0 else "error",
            "command": command,
            "return_code": result.returncode,
            "stdout": output,
            "stderr": error,
        }, ensure_ascii=False)

    except subprocess.TimeoutExpired:
        return json.dumps({
            "status": "timeout",
            "command": command,
            "reason": "Command execution timed out (30-second limit)",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "command": command,
            "reason": f"Execution failed: {e}",
        }, ensure_ascii=False)


# ============================================================
# File reading (read-only)
# ============================================================

@tool(
    name="read_file",
    description="Read the content of a specified file",
    parameters={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "File path or filename (plain filename reads from the output directory)",
            },
        },
        "required": ["filename"],
    },
    safety_level=1,
)
def read_file(filename: str) -> str:
    """Read file content. Files are resolved relative to the current working directory."""
    filepath = os.path.join(os.getcwd(), filename) if not os.path.isabs(filename) else filename
    try:
        if not os.path.exists(filepath):
            return json.dumps({"error": f"File not found: {filepath}"}, ensure_ascii=False)

        # Safety check: file size limit (10MB)
        file_size = os.path.getsize(filepath)
        if file_size > 10 * 1024 * 1024:
            return json.dumps({"error": f"File too large ({file_size} bytes), exceeds 10MB limit"}, ensure_ascii=False)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Truncate overly long content
        max_len = 10000
        truncated = len(content) > max_len
        if truncated:
            content = content[:max_len] + f"\n...(content truncated, file is {file_size} bytes total)"

        return json.dumps({
            "status": "success",
            "filepath": filepath,
            "content": content,
            "size": file_size,
            "truncated": truncated,
        }, ensure_ascii=False)
    except UnicodeDecodeError:
        return json.dumps({"error": f"File '{filepath}' is not a text file and cannot be read"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Read failed: {e}"}, ensure_ascii=False)


# ============================================================
# File writing (has side effects)
# ============================================================

@tool(
    name="write_file",
    description="Write content to a file (saved under the output directory)",
    parameters={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "Filename, e.g. 'report.md'"},
            "content": {"type": "string", "description": "The content to write to the file"},
        },
        "required": ["filename", "content"],
    },
    safety_level=2,
)
def write_file(filename: str, content: str) -> str:
    """Write a file. Path is resolved relative to the current working directory."""
    try:
        filepath = os.path.join(os.getcwd(), filename) if not os.path.isabs(filename) else filename
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return json.dumps({
            "status": "success",
            "message": f"File saved to {filepath}",
            "path": filepath,
            "size": len(content),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Write failed: {e}"}, ensure_ascii=False)
