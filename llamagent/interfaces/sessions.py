"""Session browsing utilities for interface layer."""

import json
import os
import time
import logging

logger = logging.getLogger(__name__)


def list_sessions(agent) -> list[dict]:
    """Scan persistence directory for saved sessions.

    Returns list of session info dicts sorted by last modified (newest first).
    Works by reading FSStore files directly — no framework changes needed.
    """
    persistence_mod = agent.modules.get("persistence")
    if not persistence_mod or not getattr(persistence_mod, "_enabled", False):
        return []

    store = getattr(persistence_mod, "_store", None)
    if not store:
        return []

    sessions = []

    for filename in store.list_files(".json"):
        filepath = os.path.join(store.base_dir, filename)
        try:
            raw = store.read_file(filename)
            if raw is None:
                continue
            data = json.loads(raw)
            if data.get("version") != 1:
                continue

            history = data.get("history", [])
            summary = data.get("summary", "")
            user_turns = sum(1 for m in history if m.get("role") == "user")
            persona_id = filename.replace(".json", "")
            # First user message as preview
            first_msg = next(
                (m["content"][:80] for m in history if m.get("role") == "user"), ""
            )
            mtime = os.path.getmtime(filepath)

            sessions.append({
                "persona_id": persona_id,
                "filename": filename,
                "filepath": filepath,
                "turns": user_turns,
                "summary": summary[:120] if summary else "",
                "preview": first_msg,
                "last_modified": mtime,
                "updated_at": data.get("updated_at", ""),
            })
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    sessions.sort(key=lambda s: s["last_modified"], reverse=True)
    return sessions


def format_time_ago(timestamp: float) -> str:
    """Format a timestamp as human-readable relative time."""
    diff = time.time() - timestamp
    if diff < 60:
        return "just now"
    if diff < 3600:
        return f"{int(diff / 60)} min ago"
    if diff < 86400:
        return f"{int(diff / 3600)} hours ago"
    return f"{int(diff / 86400)} days ago"


def delete_session(agent, filename: str) -> bool:
    """Delete a session file. Returns True on success."""
    persistence_mod = agent.modules.get("persistence")
    if not persistence_mod:
        return False
    store = getattr(persistence_mod, "_store", None)
    if not store:
        return False

    # Prevent deleting current session
    current_file = getattr(persistence_mod, "_filename", "")
    if filename == current_file:
        return False  # Can't delete active session

    try:
        store.delete_file(filename)
        return True
    except Exception:
        return False
