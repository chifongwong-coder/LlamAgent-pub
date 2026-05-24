"""
Platform built-in tools (common tier): available to all llamas out of the box.

Includes:
- ask_user:     Ask the user a question (requires interaction handler)
- system_info:  Current time, OS, hostname, cwd, etc. (read-only)
- web_search:   Web search via pluggable backends (DuckDuckGo / SerpAPI / Tavily)
- web_fetch:    Fetch page content from a specified URL

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
    takes_agent=True,
)
def ask_user(agent, question: str, choices: list[str] | None = None) -> str:
    """Ask the user and return their response as a string.

    v3.7.6: ``agent`` is dispatcher-injected (``takes_agent=True``); the
    interaction handler lives on ``agent._tool_state["ask_user_handler"]``,
    written by ``ToolsModule.on_attach``. Per-agent state means two
    sessions in the same process can't clobber each other's handler.
    """
    handler = agent._tool_state.get("ask_user_handler")
    if handler is None:
        return "Cannot ask user: no interaction handler configured."
    try:
        return handler.ask(question, choices)
    except Exception as e:
        return f"Failed to get user response: {e}"


# ============================================================
# System info (current time, host, OS, ...)
# ============================================================

@tool(
    name="system_info",
    description="Get current system info — local + UTC time, weekday, timezone, OS, "
                "hostname, CPU count, Python version, current working directory, and user. "
                "Use this when you need a timestamp or environment context.",
    parameters={"type": "object", "properties": {}, "required": []},
    tier="default",
    safety_level=1,
)
def system_info() -> str:
    """Return a JSON dict of read-only system / time information.

    All fields come from stdlib (datetime / os / platform / socket) so this
    tool has no external dependencies and no side effects. Safe for
    every mode including continuous (no confirm modal, no user prompt).
    """
    import datetime
    import os
    import platform
    import socket

    now_local = datetime.datetime.now().astimezone()
    info = {
        "now_local":     now_local.isoformat(timespec="seconds"),
        "now_utc":       datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "timezone":      str(now_local.tzinfo),
        "weekday":       now_local.strftime("%A"),
        "os":            platform.system(),
        "os_release":    platform.release(),
        "platform":      platform.platform(),
        "hostname":      socket.gethostname(),
        "python_version": platform.python_version(),
        "cpu_count":     os.cpu_count(),
        "cwd":           os.getcwd(),
        "user":          os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
    }
    return json.dumps(info, ensure_ascii=False, indent=2)


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
    takes_agent=True,
)
def web_search(agent, query: str, num_results: int = 5) -> str:
    """Search the web using the configured search backend.

    v3.7.6: ``agent`` is dispatcher-injected (``takes_agent=True``); the
    backend lives on ``agent._tool_state["web_search_backend"]``, written
    by ``ToolsModule.on_attach``. Per-agent state means two sessions in
    the same process don't share or clobber each other's backend.
    """
    backend = agent._tool_state.get("web_search_backend")
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

def _ip_is_private(ip) -> bool:
    """Return True for any loopback / link-local / private / multicast /
    reserved / unspecified IPv4 or IPv6 address."""
    return (
        ip.is_loopback or ip.is_link_local
        or ip.is_private or ip.is_multicast
        or ip.is_reserved or ip.is_unspecified
    )


def _is_private_or_local_host(host: str) -> bool:
    """v3.7.3 + v3.7.8: reject SSRF targets.

    v3.7.3 caught loopback / link-local / RFC1918 IPs and a handful of
    metadata-name shortcuts.

    v3.7.8 C-F6: extends the hostname path to actually DNS-resolve the
    name and reject if any returned address is private. Pre-fix
    ``gitlab.corp.local`` (or any internal hostname that DNS resolves
    to RFC1918) sailed through and was fetched. Failure to resolve is
    treated as deny (conservative; mirrors the existing behavior for
    explicit "metadata" string).
    """
    import ipaddress
    import socket
    try:
        ip = ipaddress.ip_address(host)
        return _ip_is_private(ip)
    except ValueError:
        # Hostname — DNS resolve and reject if any address is private.
        lowered = host.lower()
        if lowered in {"localhost", "metadata.google.internal", "metadata"}:
            return True
        try:
            addrs = socket.getaddrinfo(host, None)
        except (socket.gaierror, socket.herror):
            # Unresolvable — conservative: reject.
            return True
        for addr_info in addrs:
            try:
                ip = ipaddress.ip_address(addr_info[4][0])
            except (ValueError, IndexError):
                continue
            if _ip_is_private(ip):
                return True
        return False


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
        from urllib.parse import urlparse
    except ImportError:
        return json.dumps({"error": "Missing requests library. Please install: pip install requests"}, ensure_ascii=False)

    # v3.7.3: SSRF guard. The bare requests.get(url) accepted any URL
    # including 169.254.169.254 (AWS metadata) and rfc1918 internal IPs.
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return json.dumps(
            {"error": f"web_fetch only supports http/https; got scheme {parsed.scheme!r}"},
            ensure_ascii=False,
        )
    host = parsed.hostname or ""
    if not host:
        return json.dumps({"error": "URL is missing a host component"}, ensure_ascii=False)
    if _is_private_or_local_host(host):
        return json.dumps(
            {"error": f"web_fetch refuses private / loopback / link-local hosts (got {host!r})"},
            ensure_ascii=False,
        )

    # v3.7.3: hard size cap. Without this, a single response could hold an
    # arbitrary amount of memory (10GB pages, infinite-scroll endpoints).
    # We check Content-Length up-front when the server reports it; if the
    # server omits the header, we still cap the post-read body length below.
    MAX_BYTES = 4 * 1024 * 1024  # 4 MiB

    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; LlamAgent/1.0)"
        })
        resp.raise_for_status()
        # Re-validate after redirects landed on the final host: the model
        # could fetch a public URL that 302s to 169.254.169.254. Best-effort
        # — some response objects (esp. mock-built test doubles) may not
        # carry a usable ``url`` attribute.
        final_url = getattr(resp, "url", None) or url
        try:
            final_host = urlparse(final_url).hostname or ""
        except Exception:
            final_host = ""
        if final_host and _is_private_or_local_host(final_host):
            return json.dumps(
                {"error": f"web_fetch refuses redirect to private host {final_host!r}"},
                ensure_ascii=False,
            )
        # Up-front size check via Content-Length when present.
        try:
            content_length = int(resp.headers.get("Content-Length", "") or "0")
        except (TypeError, ValueError):
            content_length = 0
        if content_length and content_length > MAX_BYTES:
            return json.dumps(
                {"error": f"web_fetch refuses response of {content_length} bytes (> {MAX_BYTES})"},
                ensure_ascii=False,
            )

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

        # Post-read size cap (v3.7.3): when servers omit Content-Length
        # we still bound memory by truncating the body string. The
        # framework-side ``_truncate_observation`` further compresses the
        # observation passed to the model, but that runs AFTER web_fetch
        # has already loaded the full text into memory; this cap protects
        # the parent process from a 50MB blob hidden behind a 0-Content-
        # Length header.
        if isinstance(text, str) and len(text) > MAX_BYTES:
            text = text[:MAX_BYTES] + "\n\n[truncated by web_fetch size cap]"

        # v3.3 P2: no internal char truncation. Long pages flow through
        # _truncate_observation (contract A) which persists to disk and
        # emits a uniform hint. Returning plain text (not JSON) keeps the
        # persisted file as raw content — line counts and ranges read
        # back work directly. Header lines surface url/status/length
        # without a JSON wrapper for the model to parse.
        length = len(text)
        return (
            f"URL: {url}\n"
            f"Status: {resp.status_code}\n"
            f"Length: {length}\n\n"
            f"{text}"
        )

    except requests.exceptions.Timeout:
        return json.dumps({"error": f"Request timed out: {url}"}, ensure_ascii=False)
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Fetch failed: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Error processing page content: {e}"}, ensure_ascii=False)
