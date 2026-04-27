"""
Web search backends: pluggable search providers for the web_search tool.

SearchBackend ABC with implementations:
- DuckDuckGoBackend:  Free, no API key required (default fallback)
- SerpAPIBackend:     Requires SERPAPI_KEY
- TavilyBackend:      Requires TAVILY_API_KEY

Factory function create_search_backend() selects the backend based on config
and available API keys, following the same auto-detect pattern as model selection.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SearchBackend(ABC):
    """Abstract search backend interface."""

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> list[dict]:
        """
        Search the web and return results.

        Args:
            query: Search query string
            num_results: Maximum number of results to return

        Returns:
            List of dicts with keys: title, url, snippet
        """
        ...


class DuckDuckGoBackend(SearchBackend):
    """DuckDuckGo search. Supports both 'ddgs' (new) and 'duckduckgo_search' (legacy) packages."""

    def search(self, query: str, num_results: int = 5) -> list[dict]:
        # Support both new (ddgs) and legacy (duckduckgo_search) package names
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = list(DDGS().text(query, max_results=num_results))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]


class SerpAPIBackend(SearchBackend):
    """SerpAPI search. Requires SERPAPI_KEY environment variable."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, num_results: int = 5) -> list[dict]:
        from serpapi import GoogleSearch

        params = {
            "q": query,
            "num": num_results,
            "api_key": self.api_key,
            "engine": "google",
        }
        search = GoogleSearch(params)
        data = search.get_dict()

        results = []
        for r in data.get("organic_results", [])[:num_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "snippet": r.get("snippet", ""),
            })
        return results


class TavilyBackend(SearchBackend):
    """Tavily search. Requires TAVILY_API_KEY environment variable."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, num_results: int = 5) -> list[dict]:
        from tavily import TavilyClient

        client = TavilyClient(api_key=self.api_key)
        response = client.search(query=query, max_results=num_results)

        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
            for r in response.get("results", [])
        ]


def _ddgs_available() -> bool:
    """Check if DuckDuckGo search library is installed (new 'ddgs' or legacy 'duckduckgo_search')."""
    try:
        import ddgs  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import duckduckgo_search  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_api_key(config, provider_env_var: str) -> str:
    """Resolve an API key for the selected provider.

    Priority: ``config.web_search_api_key`` (explicit user config) >
    the provider-specific env var (``TAVILY_API_KEY`` / ``SERPAPI_KEY``).
    Returns empty string when nothing is set.
    """
    cfg_key = getattr(config, "web_search_api_key", "") if config is not None else ""
    if cfg_key:
        return cfg_key
    return os.getenv(provider_env_var, "")


def create_search_backend(config) -> SearchBackend | None:
    """
    Create a search backend based on config and available API keys.

    Priority: explicit config > API key detection > DuckDuckGo fallback.

    Returns:
        SearchBackend instance, or None if no backend is available
        (all imports failed and no API keys configured).
    """
    provider = getattr(config, "web_search_provider", "")

    # Explicit provider specified
    if provider:
        return _create_explicit_backend(provider, config)

    # Auto-detect from API keys
    serpapi_key = _resolve_api_key(config, "SERPAPI_KEY")
    if serpapi_key:
        try:
            import serpapi  # noqa: F401
            logger.info("Using SerpAPI search backend (key configured)")
            return SerpAPIBackend(api_key=serpapi_key)
        except ImportError:
            logger.warning("SerpAPI key configured but serpapi not installed, trying next backend")

    tavily_key = _resolve_api_key(config, "TAVILY_API_KEY")
    if tavily_key:
        try:
            import tavily  # noqa: F401
            logger.info("Using Tavily search backend (key configured)")
            return TavilyBackend(api_key=tavily_key)
        except ImportError:
            logger.warning("Tavily key configured but tavily-python not installed, trying next backend")

    # Fallback to DuckDuckGo (free, no key)
    if _ddgs_available():
        logger.info("Using DuckDuckGo search backend (no API key required)")
        return DuckDuckGoBackend()

    logger.warning(
        "No search backend available. Install one of: "
        "pip install ddgs / pip install google-search-results / pip install tavily-python"
    )
    return None


def _create_explicit_backend(provider: str, config=None) -> SearchBackend | None:
    """Create a backend by explicit provider name. Fails loudly if key is missing."""
    provider = provider.lower()

    if provider == "duckduckgo":
        if _ddgs_available():
            return DuckDuckGoBackend()
        logger.error("web_search_provider='duckduckgo' but ddgs/duckduckgo-search not installed")
        return None

    if provider == "serpapi":
        key = _resolve_api_key(config, "SERPAPI_KEY")
        if not key:
            logger.error(
                "web_search_provider='serpapi' but no API key. Set "
                "config.web_search_api_key or SERPAPI_KEY env var."
            )
            return None
        try:
            import serpapi  # noqa: F401
            return SerpAPIBackend(api_key=key)
        except ImportError:
            logger.error("web_search_provider='serpapi' but serpapi not installed")
            return None

    if provider == "tavily":
        key = _resolve_api_key(config, "TAVILY_API_KEY")
        if not key:
            logger.error(
                "web_search_provider='tavily' but no API key. Set "
                "config.web_search_api_key or TAVILY_API_KEY env var."
            )
            return None
        try:
            import tavily  # noqa: F401
            return TavilyBackend(api_key=key)
        except ImportError:
            logger.error("web_search_provider='tavily' but tavily-python not installed")
            return None

    logger.error("Unknown web_search_provider: '%s' (supported: duckduckgo, serpapi, tavily)", provider)
    return None
