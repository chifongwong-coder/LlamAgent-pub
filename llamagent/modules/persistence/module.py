"""
PersistenceModule: auto-save and restore conversation history.

When loaded with persistence_enabled=True, automatically saves the agent's
conversation history and summary to a JSON file after each turn (via on_output)
and on shutdown (via on_shutdown). Optionally restores the previous session on
startup (via on_attach) when persistence_auto_restore=True.

Storage uses FSStore for atomic writes. Each persona gets its own session file;
agents without a persona use "default.json".
"""

import json
import logging
import os
from datetime import datetime

from llamagent.core.agent import Module
from llamagent.modules.fs_store.store import FSStore

logger = logging.getLogger(__name__)


class PersistenceModule(Module):
    """Conversation persistence: auto-save and restore chat history."""

    name = "persistence"
    description = "Conversation persistence: auto-save and restore chat history"

    def on_attach(self, agent):
        super().on_attach(agent)

        if not getattr(agent.config, "persistence_enabled", False):
            self._enabled = False
            return

        self._enabled = True
        self._init_store(agent)

        # Auto-restore previous session
        if getattr(agent.config, "persistence_auto_restore", True):
            self._load(agent)

    def on_output(self, response: str) -> str:
        """Save conversation state after each turn."""
        if self._enabled:
            self._save()
        return response

    def on_shutdown(self) -> None:
        """Final save on agent exit (includes the last turn)."""
        if self._enabled:
            self._save()

    def _init_store(self, agent):
        """Initialize FSStore and determine the session filename."""
        base_dir = getattr(agent.config, "persistence_dir", None)
        if not base_dir:
            base_dir = os.path.join(
                getattr(agent.config, "fs_data_dir", "data/fs"),
                "sessions",
            )
        self._store = FSStore(base_dir)

        if agent.persona:
            self._filename = f"{agent.persona.persona_id}.json"
        else:
            self._filename = "default.json"

        logger.debug(
            "Persistence initialized: dir=%s, file=%s",
            self._store.base_dir,
            self._filename,
        )

    def _save(self):
        """Save current history + summary to a JSON file.

        v3.7.5: schema bumped to v=2 with two extra fields persisted:
        ``delegation_depth`` (so max_delegation_depth checks survive
        restart) and ``active_packs`` (so a follow-up tool-pack stays
        wired up after restore). v=2 files cannot be read by v3.7.4 or
        earlier — see VERSION_CHANGELOG for the downgrade caveat.
        """
        data = {
            "version": 2,
            "updated_at": datetime.now().isoformat(),
            "summary": self.agent.summary,
            "history": self.agent.history,
            "delegation_depth": getattr(self.agent, "_delegation_depth", 0),
            "active_packs": sorted(getattr(self.agent, "_active_packs", set())),
        }
        try:
            self._store.write_file(
                self._filename,
                json.dumps(data, ensure_ascii=False, indent=2),
            )
        except Exception as e:
            logger.warning("Failed to save session '%s': %s", self._filename, e)

    def _load(self, agent):
        """Restore history + summary from a JSON file.

        v3.7.5: accepts both v=1 (history+summary only) and v=2 (adds
        delegation_depth + active_packs). Missing fields fall back to
        sane defaults so a v=1 file restores cleanly and a future v=2
        file with extra unknown keys is also tolerated.
        """
        content = self._store.read_file(self._filename)
        if not content:
            return

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "Corrupt session file '%s', skipping restore: %s",
                self._filename,
                e,
            )
            return

        version = data.get("version")
        if version not in (1, 2):
            logger.warning(
                "Unknown persistence format version %s, skipping restore",
                version,
            )
            return

        agent.history[:] = data.get("history", [])
        agent.summary = data.get("summary")
        # v3.7.5: forward-compat field restore. v=1 files don't have
        # these keys; .get default fills the gap. Parent agents
        # previously had no _delegation_depth attribute (the spawn check
        # used getattr(..., 0)); explicitly setting 0 here just
        # materializes that default and stays consistent with how
        # children (set during spawn) carry the value.
        agent._delegation_depth = data.get("delegation_depth", 0)
        agent._active_packs = set(data.get("active_packs", []))
        logger.info(
            "Restored session '%s': %d messages (schema v=%d)",
            self._filename,
            len(agent.history),
            version,
        )
