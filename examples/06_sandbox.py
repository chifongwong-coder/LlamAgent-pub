"""
06 — Sandbox: Isolate High-Risk Tool Execution

The SandboxModule adds execution isolation for dangerous tools.
Tools with safety_level >= 3 are automatically routed through a sandbox
backend (subprocess with timeout, workspace confinement, etc.).

You can also manually assign execution policies to any tool.

Prerequisites:
    pip install -e .
"""

from llamagent import SmartAgent, Config
from llamagent.modules.sandbox import SandboxModule
from llamagent.modules.sandbox.policy import (
    ExecutionPolicy,
    POLICY_LOCAL_SUBPROCESS,
)


# =============================================================
# Part 1: Auto-assign sandbox to high-risk tools
# =============================================================

def part1_auto_sandbox():
    print("=" * 60)
    print("Part 1: Auto-Assign Sandbox to High-Risk Tools")
    print("=" * 60)

    config = Config()
    agent = SmartAgent(config)

    # Register a safe tool (safety_level=1, default)
    def read_data(key: str) -> str:
        """Read data by key (safe, no side effects)."""
        store = {"name": "LlamAgent", "version": "1.2"}
        return store.get(key, "Not found")

    agent.register_tool(
        name="read_data",
        func=read_data,
        description="Read data from the store",
        safety_level=1,
    )

    # Register a high-risk tool (safety_level=3)
    def execute_command(command: str) -> str:
        """Execute a shell command (high risk)."""
        import subprocess
        result = subprocess.run(
            ["sh", "-c", command],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout or result.stderr

    agent.register_tool(
        name="execute_command",
        func=execute_command,
        description="Execute a shell command",
        safety_level=3,
    )

    # Before SandboxModule: no execution policies
    print(f"execute_command policy before: {agent._tools['execute_command'].get('execution_policy')}")

    # Register the SandboxModule — auto_assign=True (default)
    # This will assign POLICY_LOCAL_SUBPROCESS to tools with safety_level >= 3
    agent.register_module(SandboxModule(auto_assign=True))

    # After SandboxModule: high-risk tool gets a sandbox policy
    policy = agent._tools["execute_command"].get("execution_policy")
    print(f"execute_command policy after:  {policy}")
    print(f"  timeout: {policy.timeout_seconds}s")

    # Safe tool remains unaffected
    print(f"read_data policy: {agent._tools['read_data'].get('execution_policy')}")

    # Use the tools via chat — the sandbox routes execution transparently
    reply = agent.chat("What is the project name? Use read_data with key 'name'.")
    print(f"Agent: {reply}")

    agent.shutdown()


# =============================================================
# Part 2: Manual execution policies
# =============================================================

def part2_manual_policy():
    print("\n" + "=" * 60)
    print("Part 2: Manual Execution Policies")
    print("=" * 60)

    config = Config()
    agent = SmartAgent(config)

    # Create a custom policy: 5-second timeout, no network
    strict_policy = ExecutionPolicy(
        runtime="shell",
        isolation="none",
        filesystem="host",
        network="none",
        timeout_seconds=5,
    )

    # Register a tool with an explicit execution policy
    def run_script(command: str) -> str:
        """Run a script in a controlled environment."""
        return f"Executed: {command}"

    agent.register_tool(
        name="run_script",
        func=run_script,
        description="Run a script with strict resource limits",
        safety_level=2,
        execution_policy=strict_policy,
    )

    # Load SandboxModule with auto_assign=False to prevent overwriting manual policies
    agent.register_module(SandboxModule(auto_assign=False))

    # Verify the manual policy is preserved
    policy = agent._tools["run_script"]["execution_policy"]
    print(f"run_script policy:")
    print(f"  runtime: {policy.runtime}")
    print(f"  network: {policy.network}")
    print(f"  timeout: {policy.timeout_seconds}s")

    agent.shutdown()


# =============================================================
# Part 3: Preset policies
# =============================================================

def part3_presets():
    print("\n" + "=" * 60)
    print("Part 3: Preset Execution Policies")
    print("=" * 60)

    from llamagent.modules.sandbox.policy import (
        POLICY_HOST,
        POLICY_READONLY,
        POLICY_UNTRUSTED_CODE,
        POLICY_SHELL_LIMITED,
        POLICY_SANDBOXED_CODER,
        POLICY_LOCAL_SUBPROCESS,
    )

    presets = {
        "POLICY_HOST": POLICY_HOST,
        "POLICY_READONLY": POLICY_READONLY,
        "POLICY_UNTRUSTED_CODE": POLICY_UNTRUSTED_CODE,
        "POLICY_SHELL_LIMITED": POLICY_SHELL_LIMITED,
        "POLICY_SANDBOXED_CODER": POLICY_SANDBOXED_CODER,
        "POLICY_LOCAL_SUBPROCESS": POLICY_LOCAL_SUBPROCESS,
    }

    for name, policy in presets.items():
        print(f"\n{name}:")
        print(f"  runtime={policy.runtime}, isolation={policy.isolation}")
        print(f"  filesystem={policy.filesystem}, network={policy.network}")
        print(f"  timeout={policy.timeout_seconds}s, memory={policy.max_memory_mb}MB")


# =============================================================
# Part 4: Direct ToolExecutor usage
# =============================================================

def part4_direct_executor():
    print("\n" + "=" * 60)
    print("Part 4: Direct ToolExecutor Usage")
    print("=" * 60)

    config = Config()
    agent = SmartAgent(config)

    # Register a tool that runs shell commands
    def run_shell(command: str) -> str:
        """Run a shell command."""
        return f"Direct: {command}"

    agent.register_tool(
        name="run_shell",
        func=run_shell,
        description="Run a shell command in sandbox",
        safety_level=3,
    )

    # Attach sandbox module
    agent.register_module(SandboxModule())

    # Access the executor directly — useful for programmatic execution
    executor = agent.tool_executor
    print(f"Executor type: {type(executor).__name__}")

    # Execute a tool through the executor
    tool_info = agent._tools["run_shell"]
    result = executor.execute(tool_info, {"command": "echo hello"})
    print(f"Executor result: {result}")

    # Clean up
    agent.shutdown()


if __name__ == "__main__":
    part1_auto_sandbox()
    part2_manual_policy()
    part3_presets()
    part4_direct_executor()
