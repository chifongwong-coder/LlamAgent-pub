"""
07 — Child Agents: Spawn Constrained Sub-Agents

The ChildAgentModule lets a parent agent spawn child agents for subtasks.
Each child inherits the parent's LLM but operates under strict constraints:
- Role-based tool filtering (researcher, writer, analyst, coder, worker)
- Budget limits (max LLM calls, time, steps)
- No ability to spawn its own children (by default)

The parent uses three tools: spawn_child, list_children, collect_results.

Prerequisites:
    pip install -e .
"""

from llamagent import SmartAgent, Config
from llamagent.modules.child_agent import ChildAgentModule
from llamagent.modules.child_agent.policy import (
    AgentExecutionPolicy,
    ROLE_POLICIES,
)
from llamagent.modules.child_agent.budget import Budget


# =============================================================
# Part 1: Basic child agent spawning
# =============================================================

def part1_basic_spawn():
    print("=" * 60)
    print("Part 1: Basic Child Agent Spawning")
    print("=" * 60)

    config = Config()
    agent = SmartAgent(config)

    # Register some tools that children can inherit
    def search(query: str) -> str:
        """Search for information."""
        data = {
            "AI agents": "AI agents are autonomous systems that perceive and act.",
            "LlamAgent": "LlamAgent is a modular AI agent framework.",
        }
        for key, val in data.items():
            if key.lower() in query.lower():
                return val
        return f"No results for: {query}"

    def summarize(text: str) -> str:
        """Summarize text."""
        words = text.split()
        return " ".join(words[:15]) + ("..." if len(words) > 15 else "")

    agent.register_tool("search", search, "Search for information")
    agent.register_tool("summarize", summarize, "Summarize text")

    # Register the ChildAgentModule
    agent.register_module(ChildAgentModule())

    # The module registers 3 tools for the parent
    schemas = agent.get_all_tool_schemas()
    tool_names = [s["function"]["name"] for s in schemas]
    print(f"Available tools: {tool_names}")
    # Should include: search, summarize, spawn_child, list_children, collect_results

    # Let the agent use child agents via chat
    reply = agent.chat("Spawn a researcher child to search for information about AI agents.")
    print(f"Agent: {reply}")

    agent.shutdown()


# =============================================================
# Part 2: Role-based policies
# =============================================================

def part2_role_policies():
    print("\n" + "=" * 60)
    print("Part 2: Role-Based Policies")
    print("=" * 60)

    # Inspect the preset role policies
    for role, policy in ROLE_POLICIES.items():
        print(f"\n{role}:")
        print(f"  tool_allowlist: {policy.tool_allowlist}")
        print(f"  can_spawn_children: {policy.can_spawn_children}")
        if policy.budget:
            print(f"  budget: max_llm_calls={policy.budget.max_llm_calls}, "
                  f"max_time={policy.budget.max_time_seconds}s")
        if policy.execution_policy:
            print(f"  execution_policy: {type(policy.execution_policy).__name__}")


# =============================================================
# Part 3: Custom policies and budgets
# =============================================================

def part3_custom_policy():
    print("\n" + "=" * 60)
    print("Part 3: Custom Policies and Budgets")
    print("=" * 60)

    # Create a custom budget
    tight_budget = Budget(
        max_llm_calls=5,
        max_time_seconds=30,
        max_steps=3,
    )
    print(f"Budget: max_llm_calls={tight_budget.max_llm_calls}, "
          f"max_time={tight_budget.max_time_seconds}s, "
          f"max_steps={tight_budget.max_steps}")

    # Create a custom policy
    custom_policy = AgentExecutionPolicy(
        tool_allowlist=["search"],        # only allow search
        tool_denylist=None,
        budget=tight_budget,
        can_spawn_children=False,         # no recursive spawning
    )
    print(f"Policy: allowlist={custom_policy.tool_allowlist}, "
          f"can_spawn={custom_policy.can_spawn_children}")


# =============================================================
# Part 4: Programmatic child agent control
# =============================================================

def part4_programmatic():
    print("\n" + "=" * 60)
    print("Part 4: Programmatic Child Agent Control")
    print("=" * 60)

    config = Config()
    agent = SmartAgent(config)

    # Register tools
    def analyze(data: str) -> str:
        """Analyze data and return insights."""
        return f"Analysis of '{data}': 3 key patterns found."

    agent.register_tool("analyze", analyze, "Analyze data for patterns")

    # Register ChildAgentModule
    child_mod = ChildAgentModule()
    agent.register_module(child_mod)

    # Use the tool functions directly (bypass LLM, useful for testing)
    result = child_mod._spawn_child(
        task="Analyze the sales data for Q1 trends",
        role="analyst",
        context="Q1 revenue: $1.2M, Q2 target: $1.5M",
    )
    print(f"Spawn result: {result}")

    # List children
    children = child_mod._list_children()
    print(f"Children: {children}")

    # Collect results
    results = child_mod._collect_results()
    print(f"Results: {results}")

    agent.shutdown()


# =============================================================
# Part 5: Multiple children for parallel subtasks
# =============================================================

def part5_multiple_children():
    print("\n" + "=" * 60)
    print("Part 5: Multiple Children for Parallel Subtasks")
    print("=" * 60)

    config = Config()
    agent = SmartAgent(config)

    # Register tools that different roles can use
    def web_search(query: str) -> str:
        """Search the web."""
        return f"Web results for '{query}': Found 10 relevant articles."

    def read_file(path: str) -> str:
        """Read a file."""
        return f"Contents of {path}: [sample data]"

    def write_file(path: str, content: str) -> str:
        """Write a file."""
        return f"Written {len(content)} chars to {path}"

    agent.register_tool("web_search", web_search, "Search the web")
    agent.register_tool("read_file", read_file, "Read a file")
    agent.register_tool("write_file", write_file, "Write content to a file")

    # Register ChildAgentModule
    child_mod = ChildAgentModule()
    agent.register_module(child_mod)

    # Spawn multiple children with different roles
    tasks = [
        ("Research recent trends in AI agents", "researcher"),
        ("Write a summary report on AI agents", "writer"),
        ("Analyze the competitive landscape", "analyst"),
    ]

    for task, role in tasks:
        result = child_mod._spawn_child(task=task, role=role)
        print(f"[{role}] {result[:80]}...")

    # Collect all results
    print(f"\nTotal children: {child_mod._list_children()}")

    agent.shutdown()


if __name__ == "__main__":
    part1_basic_spawn()
    part2_role_policies()
    part3_custom_policy()
    part4_programmatic()
    part5_multiple_children()
