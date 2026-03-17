"""
Example MCP Server: weather query service.

This is a standard MCP Server implementation that any MCP protocol-compatible client can connect to.

How to run:
    python -m llamagent.modules.mcp.server_example

Dependency installation:
    pip install mcp
"""

import json
import random

# mcp package is a required dependency (only for running this server)
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


# ============================================================
# Step 1: Create MCP Server instance
# ============================================================

if MCP_AVAILABLE:
    server = Server("weather-service")

    # ============================================================
    # Step 2: Register tools
    # ============================================================

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Tell the client what tools this server provides."""
        return [
            Tool(
                name="get_weather",
                description="Query weather information for a specified city. Returns temperature, weather conditions, and suggestions.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name to query weather for, e.g.: Beijing, Shanghai, Guangzhou",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit, defaults to celsius",
                            "default": "celsius",
                        },
                    },
                    "required": ["city"],
                },
            ),
            Tool(
                name="get_forecast",
                description="Query the weather forecast for a specified city for the next few days.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name",
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of forecast days (1-7)",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 7,
                        },
                    },
                    "required": ["city"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Execute the corresponding logic based on tool name and arguments, and return results."""
        if name == "get_weather":
            return await _handle_get_weather(arguments)
        elif name == "get_forecast":
            return await _handle_get_forecast(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ============================================================
# Step 3: Implement tool logic (mock data; replace with real API in production)
# ============================================================

MOCK_WEATHER_DATA = {
    "Beijing": {"temp": 22, "condition": "Sunny", "humidity": 35, "wind": "North wind level 3"},
    "Shanghai": {"temp": 26, "condition": "Cloudy", "humidity": 65, "wind": "Southeast wind level 2"},
    "Guangzhou": {"temp": 30, "condition": "Thunderstorm", "humidity": 85, "wind": "South wind level 4"},
    "Shenzhen": {"temp": 29, "condition": "Overcast", "humidity": 75, "wind": "Southwest wind level 2"},
    "Hangzhou": {"temp": 24, "condition": "Light rain", "humidity": 70, "wind": "East wind level 3"},
    "Chengdu": {"temp": 20, "condition": "Cloudy turning sunny", "humidity": 55, "wind": "Breeze"},
}


def _get_suggestion(condition: str, temp: int) -> str:
    """Generate suggestions based on weather conditions."""
    suggestions = []

    if "rain" in condition.lower():
        suggestions.append("Remember to bring an umbrella")
    if "sunny" in condition.lower() and temp > 28:
        suggestions.append("Watch out for sun exposure, stay hydrated")
    if temp < 10:
        suggestions.append("Cold weather, dress warmly")
    if "thunder" in condition.lower():
        suggestions.append("Thunderstorm weather, try to stay indoors")
    if 20 <= temp <= 26 and "sunny" in condition.lower():
        suggestions.append("Nice weather, great for a walk outside")

    return "; ".join(suggestions) if suggestions else "Weather is moderate, normal travel is fine"


async def _handle_get_weather(arguments: dict) -> list:
    """Handle weather query request."""
    city = arguments.get("city", "")
    unit = arguments.get("unit", "celsius")

    if city in MOCK_WEATHER_DATA:
        data = MOCK_WEATHER_DATA[city]
    else:
        # Use random data for unknown cities
        data = {
            "temp": random.randint(5, 35),
            "condition": random.choice(["Sunny", "Cloudy", "Light rain", "Overcast"]),
            "humidity": random.randint(20, 90),
            "wind": random.choice(["North wind level 2", "South wind level 3", "Breeze"]),
        }

    temp = data["temp"]
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32, 1)
        temp_str = f"{temp}F"
    else:
        temp_str = f"{temp}C"

    suggestion = _get_suggestion(data["condition"], data["temp"])

    result = {
        "city": city,
        "temperature": temp_str,
        "condition": data["condition"],
        "humidity": f"{data['humidity']}%",
        "wind": data["wind"],
        "suggestion": suggestion,
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def _handle_get_forecast(arguments: dict) -> list:
    """Handle weather forecast request."""
    city = arguments.get("city", "")
    days = arguments.get("days", 3)

    conditions = ["Sunny", "Cloudy", "Light rain", "Overcast", "Cloudy turning sunny", "Sunny turning cloudy"]
    forecast = []

    base_temp = MOCK_WEATHER_DATA.get(city, {}).get("temp", 22)

    for i in range(days):
        day_temp = base_temp + random.randint(-3, 3)
        forecast.append({
            "day": f"Day {i + 1}",
            "high": day_temp + random.randint(2, 5),
            "low": day_temp - random.randint(2, 5),
            "condition": random.choice(conditions),
        })

    result = {
        "city": city,
        "forecast": forecast,
    }

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


# ============================================================
# Step 4: Start the server (stdio mode)
# ============================================================

async def main():
    """Start the MCP server."""
    if not MCP_AVAILABLE:
        print("Error: mcp package not installed, please run: pip install mcp")
        return

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
