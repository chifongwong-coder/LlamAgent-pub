"""
10 — Task Mode: Prepare / Confirm / Execute

Task mode adds a preparation phase before execution. The agent:
1. Plans the task (dry-run, no real writes)
2. Presents a Task Contract showing planned operations
3. Waits for user confirmation (yes/no/more info)
4. Executes only after approval

This is useful for complex tasks where you want to review before execution.

Prerequisites:
    pip install -e ".[all]"
"""

from llamagent import LlamAgent, Config
from llamagent.core.zone import ConfirmResponse


# --- Step 1: Create agent with tools ---
config = Config()
agent = LlamAgent(config)

# Load tools module for file operations
try:
    from llamagent.modules.tools import ToolsModule
    agent.register_module(ToolsModule())
    print("[OK] Tools module loaded")
except ImportError:
    print("[SKIP] Tools module not available")

# --- Step 2: Set up confirm handler ---
# Task mode needs a confirm_handler for operations outside the approved contract.
# For this example, we auto-approve everything.
agent.confirm_handler = lambda req: ConfirmResponse(allow=True)

# --- Step 3: Switch to task mode ---
agent.set_mode("task")
print(f"Mode: {agent.mode}")
print(f"  max_react_steps: {agent.config.max_react_steps}")
print(f"  react_timeout: {agent.config.react_timeout}")

# --- Step 4: Start a task ---
# chat() in task mode runs the prepare phase and returns a Task Contract
print("\n--- Sending task to agent ---")
response = agent.chat("Create a simple hello world Python script")
print(f"\nAgent response:\n{response}")

# --- Step 5: Confirm the contract ---
# The response contains a [Task Contract] with planned operations.
# Reply 'yes' to approve and execute, 'no' to cancel.
if "[Task Contract]" in response:
    print("\n--- Confirming contract ---")
    result = agent.chat("yes")
    print(f"\nExecution result:\n{result}")
else:
    print("\n(No contract presented — task may have been too simple for task mode)")

# --- Step 6: Switch back to interactive ---
agent.set_mode("interactive")
print(f"\nBack to {agent.mode} mode")
print(f"  max_react_steps: {agent.config.max_react_steps}")

# --- Cleanup ---
agent.shutdown()
print("Done!")
