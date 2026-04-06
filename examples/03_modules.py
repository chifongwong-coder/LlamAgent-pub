"""
03 — Modules: Plug In Capabilities

Modules add abilities to the agent through a callback pipeline:
    on_input  (forward) → on_context (forward) → execute → on_output (reverse)

This example shows how to create custom modules and use built-in ones.

Prerequisites:
    pip install -e ".[all]"
"""

from llamagent import LlamAgent, Config, Module


# =============================================================
# Part 1: Build a custom module
# =============================================================

class LoggingModule(Module):
    """A simple module that logs every message passing through the pipeline."""

    name = "logging"
    description = "Logs all pipeline events"

    def on_attach(self, agent):
        super().on_attach(agent)
        print(f"[{self.name}] Attached to agent")

    def on_input(self, user_input: str) -> str:
        print(f"[{self.name}] on_input: {user_input[:50]}...")
        return user_input  # pass through unchanged

    def on_context(self, query: str, context: str) -> str:
        print(f"[{self.name}] on_context: context length = {len(context)}")
        return context  # pass through unchanged

    def on_output(self, response: str) -> str:
        print(f"[{self.name}] on_output: {response[:50]}...")
        return response  # pass through unchanged

    def on_shutdown(self):
        print(f"[{self.name}] Shutting down")


class ContentFilterModule(Module):
    """A module that censors specific words in the output."""

    name = "content_filter"
    description = "Filters sensitive words from output"

    def __init__(self, blocked_words: list[str]):
        self.blocked_words = blocked_words

    def on_output(self, response: str) -> str:
        filtered = response
        for word in self.blocked_words:
            filtered = filtered.replace(word, "[REDACTED]")
        return filtered


def part1_custom_modules():
    print("=" * 60)
    print("Part 1: Custom Modules")
    print("=" * 60)

    config = Config()
    agent = LlamAgent(config)

    # Register modules — order matters!
    # on_input/on_context: called in registration order
    # on_output: called in REVERSE order (onion model)
    agent.register_module(LoggingModule())
    agent.register_module(ContentFilterModule(blocked_words=["password", "secret"]))

    # Check registered modules
    print(f"Modules: {list(agent.modules.keys())}")
    print(f"Has logging? {agent.has_module('logging')}")

    reply = agent.chat("Hello!")
    print(f"Agent: {reply}")

    agent.shutdown()


# =============================================================
# Part 2: Use built-in modules
# =============================================================

def part2_builtin_modules():
    print("\n" + "=" * 60)
    print("Part 2: Built-in Modules")
    print("=" * 60)

    config = Config()
    agent = LlamAgent(config)

    # Load Safety module — should be first (filters dangerous input)
    try:
        from llamagent.modules.safety import SafetyModule
        agent.register_module(SafetyModule())
        print("[OK] Safety module loaded")
    except ImportError:
        print("[SKIP] Safety module (missing dependencies)")

    # Load Tools module — provides built-in tools
    try:
        from llamagent.modules.tools import ToolsModule
        agent.register_module(ToolsModule())
        print("[OK] Tools module loaded")
        tool_schemas = agent.get_all_tool_schemas()
        print(f"     Tools available: {[s['function']['name'] for s in tool_schemas]}")
    except ImportError:
        print("[SKIP] Tools module (missing dependencies)")

    # Load Planning module — upgrades execution from SimpleReAct to PlanReAct
    try:
        from llamagent.modules.reasoning import PlanningModule
        agent.register_module(PlanningModule())
        print("[OK] Planning module loaded")
    except ImportError:
        print("[SKIP] Planning module (missing dependencies)")

    print(f"\nAll modules: {list(agent.modules.keys())}")

    reply = agent.chat("What modules do you have loaded?")
    print(f"Agent: {reply}")

    agent.shutdown()


# =============================================================
# Part 3: Module that registers tools
# =============================================================

class TimeModule(Module):
    """A module that gives the agent the ability to tell time."""

    name = "time"
    description = "Provides current date and time"

    def on_attach(self, agent):
        super().on_attach(agent)
        from datetime import datetime

        def get_current_time() -> str:
            """Get the current date and time."""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        agent.register_tool(
            name="get_current_time",
            func=get_current_time,
            description="Returns the current date and time",
        )


def part3_module_with_tools():
    print("\n" + "=" * 60)
    print("Part 3: Module That Registers Tools")
    print("=" * 60)

    config = Config()
    agent = LlamAgent(config)
    agent.register_module(TimeModule())

    schemas = agent.get_all_tool_schemas()
    print(f"Tools: {[s['function']['name'] for s in schemas]}")

    reply = agent.chat("What time is it right now?")
    print(f"Agent: {reply}")

    agent.shutdown()


if __name__ == "__main__":
    part1_custom_modules()
    part2_builtin_modules()
    part3_module_with_tools()
