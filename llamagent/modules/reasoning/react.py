"""
ReAct execution helper: provides single-step ReAct execution for the PlanReAct strategy.

Note: In the target architecture, the ReAct loop will be moved into core/agent.py (agent.run_react()).
This module is a transitional implementation for PlanReAct to use while run_react has not yet
been moved into core. Once run_react is ready in core, this file can be deleted.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from llamagent.core.llm import LLMClient

logger = logging.getLogger(__name__)


def run_react_loop(
    llm: LLMClient,
    messages: list[dict],
    tools_schema: list[dict],
    tool_dispatch: Callable[[str, dict], str],
    max_steps: int = 10,
    timeout: float = 300.0,
) -> str:
    """
    Execute a single ReAct loop (based on function calling).

    This is a stateless function whose parameters are entirely provided by the caller,
    with no awareness of tool origins.
    Once run_react() in core/agent.py is ready, this function will be replaced.

    Args:
        llm:            LLMClient instance
        messages:       Initial message list (including system prompt)
        tools_schema:   Tool schema list (OpenAI function calling format)
        tool_dispatch:  Tool dispatch function (name, args) -> str
        max_steps:      Maximum number of loop iterations
        timeout:        Timeout in seconds; per-step timing not yet implemented

    Returns:
        The final text response
    """
    import time

    start_time = time.time()
    prev_calls: list[tuple[str, str]] = []  # For duplicate detection

    for step in range(max_steps):
        # Timeout check
        elapsed = time.time() - start_time
        if elapsed > timeout:
            logger.warning("ReAct loop timed out (%.0fs / %.0fs)", elapsed, timeout)
            return _extract_last_text(messages) or f"Execution timed out ({elapsed:.0f}s), task incomplete."

        # Call LLM (with tools)
        try:
            resp = llm.chat(
                messages,
                tools=tools_schema if tools_schema else None,
                tool_choice="auto" if tools_schema else None,
            )
        except Exception as e:
            error_msg = str(e)
            # Context window exceeded -- abort loop
            if "context" in error_msg.lower() and "exceeded" in error_msg.lower():
                logger.warning("Context window exceeded, aborting ReAct loop")
                return _extract_last_text(messages) or "Task too complex, consider breaking it into smaller subtasks."
            logger.error("ReAct LLM call failed: %s", e)
            return _extract_last_text(messages) or f"LLM call error: {e}"

        msg = resp.choices[0].message

        # No tool_calls -> return text directly
        if not msg.tool_calls:
            return msg.content or ""

        # Has tool_calls -> execute one by one
        messages.append(msg)

        for tool_call in msg.tool_calls:
            tc_name = tool_call.function.name
            try:
                tc_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tc_args = {}

            # Duplicate detection: consecutive identical tool name + args
            call_sig = (tc_name, json.dumps(tc_args, sort_keys=True))
            if prev_calls and prev_calls[-1] == call_sig:
                dup_count = sum(1 for c in prev_calls if c == call_sig)
                if dup_count >= 2:
                    logger.warning("Duplicate tool call detected for %s, aborting loop", tc_name)
                    return _extract_last_text(messages) or "Duplicate operation detected, execution stopped."
            prev_calls.append(call_sig)

            # Execute tool
            try:
                result = tool_dispatch(tc_name, tc_args)
            except Exception as e:
                result = f"Tool '{tc_name}' execution error: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

    # Reached maximum steps
    return _extract_last_text(messages) or f"Reached maximum steps ({max_steps}), task incomplete."


def _extract_last_text(messages: list[dict]) -> str:
    """Extract the last assistant text response from the message history."""
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        elif hasattr(msg, "role") and msg.role == "assistant" and msg.content:
            return msg.content
    return ""
