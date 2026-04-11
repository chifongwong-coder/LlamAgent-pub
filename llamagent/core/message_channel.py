"""
Messaging infrastructure for agent communication.

MessageChannel: thread-safe async message passing between agents.
AgentRegistry: agent registration and discovery service.
MessageTrigger: ContinuousRunner trigger that fires on incoming messages.

These components are owned by ChildAgentModule (not placed on LlamAgent directly,
per P1 — agent independence). The module creates them in on_attach for continuous
mode agents and wires the messaging tools.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field

from llamagent.core.runner import Trigger


# ======================================================================
# Message
# ======================================================================


@dataclass
class Message:
    """A single message between agents."""
    message_id: str
    from_id: str
    to_id: str          # "*" for broadcast
    content: str
    msg_type: str        # "info" / "alert" / "request" / "response"
    timestamp: float
    replied_to: str | None = None


# ======================================================================
# MessageChannel
# ======================================================================


class MessageChannel:
    """
    Thread-safe async message channel between agents.

    Each registered agent gets a bounded inbox (queue.Queue). Messages are
    delivered by putting them into the recipient's inbox. Bounded queues
    (maxsize=1000) prevent unbounded memory growth in long-running agents;
    when full, the oldest message is discarded to make room.
    """

    def __init__(self, maxsize: int = 1000):
        self._inboxes: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def register(self, agent_id: str) -> None:
        """Create an inbox for the agent. No-op if already registered."""
        with self._lock:
            if agent_id not in self._inboxes:
                self._inboxes[agent_id] = queue.Queue(maxsize=self._maxsize)

    def unregister(self, agent_id: str) -> None:
        """Remove the agent's inbox. No-op if not registered."""
        with self._lock:
            self._inboxes.pop(agent_id, None)

    def send(self, from_id: str, to_id: str, content: str, msg_type: str = "info") -> str:
        """
        Send a message to a specific agent.

        Args:
            from_id: Sender agent ID.
            to_id: Recipient agent ID.
            content: Message content.
            msg_type: Message type (info/alert/request/response).

        Returns:
            The generated message_id.

        Raises:
            KeyError: If to_id is not registered.
        """
        msg = Message(
            message_id=uuid.uuid4().hex[:12],
            from_id=from_id,
            to_id=to_id,
            content=content,
            msg_type=msg_type,
            timestamp=time.time(),
        )
        with self._lock:
            inbox = self._inboxes.get(to_id)
            if inbox is None:
                raise KeyError(f"Agent '{to_id}' is not registered.")
            # Bounded queue: discard oldest if full
            if inbox.full():
                try:
                    inbox.get_nowait()
                except queue.Empty:
                    pass
            inbox.put_nowait(msg)
        return msg.message_id

    def receive(self, agent_id: str, timeout: float = 0) -> list[Message]:
        """
        Receive all pending messages for an agent.

        Args:
            agent_id: The agent whose inbox to drain.
            timeout: Seconds to wait for at least one message.
                     0 = non-blocking (default).

        Returns:
            List of Message objects (may be empty).
        """
        with self._lock:
            inbox = self._inboxes.get(agent_id)
        if inbox is None:
            return []

        messages: list[Message] = []

        # If timeout > 0, block for at most timeout seconds for the first message
        if timeout > 0 and inbox.empty():
            try:
                first = inbox.get(timeout=timeout)
                messages.append(first)
            except queue.Empty:
                return []

        # Drain remaining messages non-blocking
        while True:
            try:
                msg = inbox.get_nowait()
                messages.append(msg)
            except queue.Empty:
                break

        return messages

    def broadcast(self, from_id: str, content: str, msg_type: str = "info") -> str:
        """
        Broadcast a message to all registered agents (except the sender).

        Returns:
            The generated message_id (same ID in all copies).
        """
        message_id = uuid.uuid4().hex[:12]
        ts = time.time()
        with self._lock:
            for agent_id, inbox in self._inboxes.items():
                if agent_id == from_id:
                    continue
                msg = Message(
                    message_id=message_id,
                    from_id=from_id,
                    to_id="*",
                    content=content,
                    msg_type=msg_type,
                    timestamp=ts,
                )
                if inbox.full():
                    try:
                        inbox.get_nowait()
                    except queue.Empty:
                        pass
                inbox.put_nowait(msg)
        return message_id


# ======================================================================
# AgentRegistry
# ======================================================================


class AgentRegistry:
    """
    Agent registration and discovery service.

    Tracks active agents with their metadata (role, mode). Works in tandem
    with MessageChannel: register() creates an inbox, unregister() removes it.
    """

    def __init__(self, channel: MessageChannel):
        self._channel = channel
        self._agents: dict[str, dict] = {}
        self._lock = threading.Lock()

    def register(self, agent_id: str, role: str, mode: str) -> None:
        """Register an agent and create its inbox in the channel."""
        with self._lock:
            self._agents[agent_id] = {"agent_id": agent_id, "role": role, "mode": mode}
        self._channel.register(agent_id)

    def unregister(self, agent_id: str) -> None:
        """Unregister an agent and remove its inbox from the channel."""
        with self._lock:
            self._agents.pop(agent_id, None)
        self._channel.unregister(agent_id)

    def list_agents(self) -> list[dict]:
        """List all registered agents with their metadata."""
        with self._lock:
            return list(self._agents.values())

    def get(self, agent_id: str) -> dict | None:
        """Look up an agent by ID. Returns None if not found."""
        with self._lock:
            return self._agents.get(agent_id)


# ======================================================================
# MessageTrigger
# ======================================================================


class MessageTrigger(Trigger):
    """
    ContinuousRunner trigger that fires when there are pending messages.

    poll() drains the agent's inbox and returns a formatted string of all
    pending messages, or None if the inbox is empty. This integrates with
    the ContinuousRunner's multi-trigger polling architecture.
    """

    def __init__(self, channel: MessageChannel, agent_id: str):
        self._channel = channel
        self._agent_id = agent_id

    def poll(self) -> str | None:
        """Return formatted pending messages, or None if no messages."""
        msgs = self._channel.receive(self._agent_id)
        if not msgs:
            return None
        lines = [f"[Message from {m.from_id}]: {m.content}" for m in msgs]
        return "Incoming messages:\n" + "\n".join(lines)
