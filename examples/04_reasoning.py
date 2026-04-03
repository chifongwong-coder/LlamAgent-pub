"""
04 — Reasoning: ReAct Loops and Task Planning

LlamAgent uses two execution strategies:
- SimpleReAct: direct tool-calling loop (default)
- PlanReAct:   plan → execute steps → replan on failure (via PlanningModule)

The agent automatically routes between them based on task complexity.

Prerequisites:
    pip install -e .
"""

from llamagent import LlamAgent, Config


# =============================================================
# Part 1: SimpleReAct — the default tool-calling loop
# =============================================================

def part1_simple_react():
    print("=" * 60)
    print("Part 1: SimpleReAct")
    print("=" * 60)

    config = Config()
    agent = LlamAgent(config)

    # Register some tools for the agent to use
    def search(query: str) -> str:
        """Search for information."""
        return f"Search results for '{query}': LlamAgent is a modular AI agent framework."

    def summarize(text: str) -> str:
        """Summarize a piece of text."""
        words = text.split()
        return " ".join(words[:20]) + "..." if len(words) > 20 else text

    agent.register_tool("search", search, "Search for information by keyword")
    agent.register_tool("summarize", summarize, "Summarize a long text into a short one")

    # With SimpleReAct (default), the agent decides on each step:
    #   1. Call a tool? → execute it, observe result, think again
    #   2. No tool needed? → return final answer
    reply = agent.chat("Search for LlamAgent and summarize what you find.")
    print(f"Agent: {reply}")

    agent.shutdown()


# =============================================================
# Part 2: PlanReAct — task planning for complex requests
# =============================================================

def part2_plan_react():
    print("\n" + "=" * 60)
    print("Part 2: PlanReAct (via PlanningModule)")
    print("=" * 60)

    config = Config()
    agent = LlamAgent(config)

    # Register tools
    def search(query: str) -> str:
        """Search for information."""
        results = {
            "AI agent frameworks": "Popular ones: LangChain, AutoGPT, CrewAI, LlamAgent.",
            "LlamAgent features": "Modular design, ReAct+Planning, multi-LLM support.",
            "LlamAgent vs LangChain": "LlamAgent is simpler and more modular.",
        }
        for key, val in results.items():
            if key.lower() in query.lower():
                return val
        return f"Results for '{query}': No specific data found."

    def write_report(title: str, content: str) -> str:
        """Write a report with the given title and content."""
        return f"Report '{title}' created with {len(content)} characters."

    agent.register_tool("search", search, "Search for information")
    agent.register_tool("write_report", write_report, "Write a structured report")

    # Load the PlanningModule — this upgrades the execution strategy to PlanReAct
    from llamagent.modules.reasoning import PlanningModule
    agent.register_module(PlanningModule())

    # PlanReAct flow:
    #   1. Judge complexity (simple → fallback to SimpleReAct)
    #   2. Generate a multi-step plan (DAG with dependencies)
    #   3. Execute steps in dependency order
    #   4. Replan if a step fails
    #   5. Summarize all results into a final answer
    reply = agent.chat(
        "Research AI agent frameworks, compare LlamAgent with alternatives, "
        "and write a summary report."
    )
    print(f"Agent: {reply}")

    agent.shutdown()


# =============================================================
# Part 3: Understanding the ReAct loop directly
# =============================================================

def part3_react_loop_details():
    print("\n" + "=" * 60)
    print("Part 3: ReAct Loop Internals")
    print("=" * 60)

    config = Config()
    # Customize ReAct behavior
    config.max_react_steps = 5       # max tool-calling iterations
    config.max_duplicate_actions = 2  # stop if same action repeats

    agent = LlamAgent(config)

    call_count = [0]

    def calculator(expression: str) -> str:
        """Evaluate a math expression."""
        call_count[0] += 1
        print(f"  [Tool called #{call_count[0]}] calculator({expression})")
        try:
            return str(eval(expression))  # noqa: S307
        except Exception as e:
            return f"Error: {e}"

    agent.register_tool("calculator", calculator, "Evaluate math expressions")

    # The agent will enter a ReAct loop:
    #   Think → Act (call tool) → Observe (get result) → Think → ...
    reply = agent.chat("What is (2^10 + 3^7) * 17?")
    print(f"Agent: {reply}")
    print(f"Total tool calls: {call_count[0]}")

    agent.shutdown()


if __name__ == "__main__":
    part1_simple_react()
    part2_plan_react()
    part3_react_loop_details()
