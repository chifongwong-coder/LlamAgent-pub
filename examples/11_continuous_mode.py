"""
11 — Continuous Mode: Trigger-Driven Agent

Continuous mode lets the agent run autonomously, driven by external triggers.
A ContinuousRunner polls triggers and calls agent.chat() for each input.

Built-in triggers:
- TimerTrigger: fires at regular intervals with a fixed message
- FileTrigger: fires when new files appear in a watched directory

Prerequisites:
    pip install -e ".[all]"
"""

import os
import time
import threading
import tempfile

from llamagent import LlamAgent, Config
from llamagent.core.zone import ConfirmResponse
from llamagent.core.runner import ContinuousRunner, TimerTrigger, FileTrigger


# --- Step 1: Create agent ---
config = Config()
agent = LlamAgent(config)

# Load tools module
try:
    from llamagent.modules.tools import ToolsModule
    agent.register_module(ToolsModule())
    print("[OK] Tools module loaded")
except ImportError:
    print("[SKIP] Tools module not available")

# Auto-approve (continuous mode operates without user interaction)
agent.confirm_handler = lambda req: ConfirmResponse(allow=True)

# --- Step 2: Switch to continuous mode ---
agent.set_mode("continuous")
print(f"Mode: {agent.mode}")
print(f"  max_react_steps: {agent.config.max_react_steps} (-1 = unlimited)")
print(f"  max_duplicate_actions: {agent.config.max_duplicate_actions} (-1 = disabled)")
print(f"  react_timeout: {agent.config.react_timeout}")

# --- Step 3: Configure triggers ---

# Example A: TimerTrigger — fires every 5 seconds
timer = TimerTrigger(interval=5.0, message="Report the current time")
print("\nTimerTrigger: fires every 5 seconds")

# Example B: FileTrigger — watches a temp directory
watch_dir = tempfile.mkdtemp(prefix="llamagent_watch_")
file_trigger = FileTrigger(watch_dir, message_template="Process new files: {files}")
print(f"FileTrigger: watching {watch_dir}")

# --- Step 4: Create and run ContinuousRunner ---
runner = ContinuousRunner(
    agent,
    triggers=[timer, file_trigger],
    poll_interval=1.0,       # check triggers every 1 second
    task_timeout=30,          # abort if a single task takes >30s
)

print("\n--- Starting ContinuousRunner (will run for 12 seconds) ---\n")

# Run in a background thread so we can stop it after a delay
t = threading.Thread(target=runner.run)
t.start()

# Simulate file creation after 6 seconds
time.sleep(6)
test_file = os.path.join(watch_dir, "report.txt")
with open(test_file, "w") as f:
    f.write("quarterly report data")
print(f"[Simulated] Created {test_file}")

# Let it run a bit more, then stop
time.sleep(6)
print("\n--- Stopping runner ---")
runner.stop()
t.join(timeout=5)

# --- Step 5: Check results ---
turns = len(agent.history)
print(f"\nRunner completed. Conversation turns recorded: {turns}")

# --- Step 6: Switch back to interactive ---
agent.set_mode("interactive")
print(f"Back to {agent.mode} mode")

# --- Cleanup ---
agent.shutdown()
# Clean up temp directory
import shutil
shutil.rmtree(watch_dir, ignore_errors=True)
print("Done!")
