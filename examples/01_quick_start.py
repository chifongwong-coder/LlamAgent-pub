"""
01 — Quick Start: Create an Agent and Chat

This example shows the minimal setup to get a LlamAgent running.
No modules are needed — a bare SmartAgent is already a working chatbot.

Prerequisites:
    pip install -e .
    # Then configure .env (see .env.example)
"""

from llamagent import SmartAgent, Config

# --- Step 1: Create a Config ---
# Config auto-detects your LLM backend:
#   - If DEEPSEEK_API_KEY is set → DeepSeek
#   - If OPENAI_API_KEY is set  → OpenAI
#   - If ANTHROPIC_API_KEY set  → Anthropic
#   - Otherwise                 → Local Ollama (free, no API key needed)
config = Config()
print(f"Model: {config.model}")
print(f"Max context tokens: {config.max_context_tokens}")

# --- Step 2: Create the Agent ---
# A bare SmartAgent with no modules is a fully functional conversational agent.
agent = SmartAgent(config)

# --- Step 3: Chat ---
reply = agent.chat("Hello! What can you do?")
print(f"Agent: {reply}")

# --- Step 4: Multi-turn conversation ---
# The agent maintains conversation history automatically.
reply = agent.chat("Tell me a fun fact about llamas.")
print(f"Agent: {reply}")

reply = agent.chat("Can you elaborate on that?")
print(f"Agent: {reply}")

# --- Step 5: Check conversation status ---
turns = len(agent.history) // 2
print(f"\nConversation turns so far: {turns}")

# --- Cleanup ---
agent.shutdown()
print("Done!")
