"""
User interaction handlers for the ask_user tool.

UserInteractionHandler ABC defines the interface. Implementations are injected
by the caller (interface layer, library user, or custom integration).

Built-in implementations:
- CallbackInteractionHandler:  Delegates to a user-provided callable (primary integration point)
- CLIInteractionHandler:       Convenience example using blocking input()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable

logger = logging.getLogger(__name__)


class UserInteractionHandler(ABC):
    """
    Abstract handler for user interaction during tool execution.

    Implementations decide HOW to present the question and collect the answer.
    The ask_user tool only knows this interface — it is fully decoupled from
    any specific I/O mechanism (CLI, Web, API, message queue, etc.).

    v1.8.2 provides a synchronous ask() method. Future versions may extend
    with async or non-blocking variants for long-running / continuous modes.
    """

    @abstractmethod
    def ask(self, question: str, choices: list[str] | None = None) -> str:
        """
        Present a question to the user and return their response.

        Args:
            question: The question to present
            choices: Optional list of choices (free text if None)

        Returns:
            User's response as a string
        """
        ...


class CallbackInteractionHandler(UserInteractionHandler):
    """
    Delegates to a user-provided callable.

    This is the primary integration point: library users pass their own
    function to handle the interaction however they see fit.

    Usage:
        handler = CallbackInteractionHandler(lambda q, c: input(f"{q}: "))
        agent.interaction_handler = handler
    """

    def __init__(self, callback: Callable[[str, list[str] | None], str]):
        self.callback = callback

    def ask(self, question: str, choices: list[str] | None = None) -> str:
        return self.callback(question, choices)


class CLIInteractionHandler(UserInteractionHandler):
    """
    CLI interaction via blocking input(). A convenience implementation —
    not required, not the only option. Just an example.

    For choices, accepts both numeric index and exact text input.
    """

    def ask(self, question: str, choices: list[str] | None = None) -> str:
        if choices:
            prompt = f"\n[Agent asks] {question}\n"
            for i, c in enumerate(choices, 1):
                prompt += f"  {i}. {c}\n"
            prompt += "Your choice (number or text): "
            val = input(prompt).strip()
            # Number → choice text
            try:
                idx = int(val) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
            except ValueError:
                pass
            return val
        else:
            return input(f"\n[Agent asks] {question}\nYour answer: ").strip()
