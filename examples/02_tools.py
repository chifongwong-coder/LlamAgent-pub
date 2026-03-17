"""
02 — Tools: Register Functions and Let the Agent Call Them

LlamAgent auto-infers JSON schema from your function's type hints.
The agent decides when and how to call tools via the ReAct loop.

Prerequisites:
    pip install -e .
"""

from llamagent import SmartAgent, Config


def create_agent_with_tools():
    config = Config()
    agent = SmartAgent(config)

    # --- Register a simple tool ---
    # Just provide a function, a name, and a description.
    # LlamAgent infers the parameter schema from type hints automatically.
    def calculator(expression: str) -> str:
        """Evaluate a math expression and return the result."""
        try:
            result = eval(expression)  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    agent.register_tool(
        name="calculator",
        func=calculator,
        description="Evaluate a math expression (e.g., '2 + 3 * 4')",
    )

    # --- Register a tool with explicit schema ---
    # For more control, provide the JSON schema yourself.
    def get_weather(city: str, unit: str = "celsius") -> str:
        """Simulated weather lookup."""
        data = {"Beijing": 22, "Tokyo": 18, "Paris": 15, "New York": 12}
        temp = data.get(city, 20)
        if unit == "fahrenheit":
            temp = temp * 9 / 5 + 32
        return f"The weather in {city} is {temp} degrees {unit}."

    agent.register_tool(
        name="get_weather",
        func=get_weather,
        description="Get the current weather for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["city"],
        },
    )

    return agent


def main():
    agent = create_agent_with_tools()

    # --- Verify tools are registered ---
    schemas = agent.get_all_tool_schemas()
    print(f"Registered tools: {[s['function']['name'] for s in schemas]}")

    # --- Call tools directly (for testing) ---
    result = agent.call_tool("calculator", {"expression": "123 * 456"})
    print(f"Direct call result: {result}")

    # --- Let the agent use tools via chat ---
    # The agent will automatically decide whether to call a tool.
    reply = agent.chat("What is 1024 * 768?")
    print(f"Agent: {reply}")

    reply = agent.chat("What's the weather in Tokyo?")
    print(f"Agent: {reply}")

    # --- Safety: pre_call_check ---
    # You can add a check function that intercepts tool calls.
    def block_dangerous(tool_name: str, args: dict) -> str | None:
        """Return a reason string to block, or None to allow."""
        if tool_name == "calculator" and "import" in args.get("expression", ""):
            return "Blocked: import is not allowed in calculator"
        return None

    agent.pre_call_check = block_dangerous

    # This call will be blocked:
    result = agent.call_tool("calculator", {"expression": "__import__('os').listdir('/')"})
    print(f"Blocked call result: {result}")

    agent.shutdown()
    print("Done!")


if __name__ == "__main__":
    main()
