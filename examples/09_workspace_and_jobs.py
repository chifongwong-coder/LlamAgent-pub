"""
09 -- Workspace & Jobs: The v1.6 Tool System

v1.6 workspace-centric tool system with pack-based conditional exposure.
Default surface has 12 tools; additional tools are in packs activated by
skills or runtime state.

Key concepts:
- Workspace: agent's sandbox area under playground_dir, Zone 1 (free zone)
- Project sync: apply_patch / sync_workspace_to_project for safe project modifications
- Jobs: start_job with wait=True (sync) or wait=False (async lifecycle)
- Packs: conditional tool groups activated by skills or state (e.g., job-followup)

Prerequisites:
    pip install -e .
"""

import json
import os
import tempfile

from llamagent import LlamAgent, Config
from llamagent.modules.tools import ToolsModule
from llamagent.modules.job import JobModule
from llamagent.modules.sandbox import SandboxModule


# =============================================================
# Helper: call a registered tool by name
# =============================================================

def call_tool(agent, name, **kwargs):
    """Call a tool registered on the agent and return the parsed result."""
    func = agent._tools[name]["func"]
    raw = func(**kwargs)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


# =============================================================
# Part 1: Workspace Tools
# =============================================================

def part1_workspace_tools():
    print("=" * 60)
    print("Part 1: Workspace Tools")
    print("=" * 60)

    # Create an isolated temp directory as the project root
    tmp_project = tempfile.mkdtemp(prefix="llamagent_ws_demo_")

    # Set up agent with ToolsModule
    config = Config()
    agent = LlamAgent(config)
    agent.project_dir = tmp_project
    agent.playground_dir = os.path.join(tmp_project, "llama_playground")
    os.makedirs(agent.playground_dir, exist_ok=True)

    tools_mod = ToolsModule()
    agent.register_module(tools_mod)

    # Show workspace root path
    ws_root = tools_mod.workspace_service.workspace_root
    print(f"\nWorkspace root: {ws_root}")
    print(f"Project dir:    {agent.project_dir}")

    # --- write_files: create files in the workspace ---
    print("\n-- write_files: create hello.py in workspace --")
    result = call_tool(agent, "write_files", files={
        "hello.py": 'def greet(name):\n    return f"Hello, {name}!"\n',
        "data/config.json": '{"version": "1.5", "debug": false}\n',
    })
    print(f"  Status: {result['status']}")
    print(f"  Written: {result['written']}")

    # --- list_tree: show workspace structure ---
    print("\n-- list_tree: workspace directory tree --")
    result = call_tool(agent, "list_tree")
    print(f"  {result['tree']}")

    # --- read_files: read back the file ---
    print("\n-- read_files: read hello.py from workspace --")
    result = call_tool(agent, "read_files", paths=["hello.py"])
    print(f"  Content:\n{result['files'][0]['content']}")

    # --- read_files with project: prefix ---
    # First create a file in the project directory to read
    project_file = os.path.join(tmp_project, "README.txt")
    with open(project_file, "w") as f:
        f.write("This is the project README.\nVersion: 1.5\n")

    print("\n-- read_files: read a project file with 'project:' prefix --")
    result = call_tool(agent, "read_files", paths=["project:README.txt"])
    print(f"  Content:\n{result['files'][0]['content']}")

    # --- glob_files: search workspace by pattern ---
    print("\n-- glob_files: find all .py files in workspace --")
    result = call_tool(agent, "glob_files", pattern="**/*.py")
    print(f"  Found {result['count']} file(s): {result['files']}")

    # --- search_text: search file content in workspace ---
    print("\n-- search_text: search for 'greet' in workspace --")
    result = call_tool(agent, "search_text", query="greet")
    for match in result["matches"]:
        print(f"  {match['file']}:{match['line']} -> {match['content']}")

    return agent, tools_mod, tmp_project


# =============================================================
# Part 2: Project Sync
# =============================================================

def part2_project_sync(agent, tools_mod, tmp_project):
    print("\n" + "=" * 60)
    print("Part 2: Project Sync (apply_patch / revert)")
    print("=" * 60)

    # Create a project file to patch
    target_file = os.path.join(tmp_project, "app.py")
    original_content = (
        "# app.py\n"
        "VERSION = '1.0'\n"
        "\n"
        "def main():\n"
        "    print('Starting app...')\n"
    )
    with open(target_file, "w") as f:
        f.write(original_content)

    print(f"\nCreated project file: app.py")
    print(f"  Original VERSION = '1.0'")

    # --- apply_patch: modify the project file with structured edits ---
    print("\n-- apply_patch: update VERSION from '1.0' to '1.5' --")
    result = call_tool(agent, "apply_patch",
        target="app.py",
        edits=[{"match": "VERSION = '1.0'", "replace": "VERSION = '1.5'"}],
    )
    print(f"  Status: {result['status']}")
    print(f"  Edits applied: {result.get('edits_applied', 0)}")

    # Show the file changed
    with open(target_file, "r") as f:
        content = f.read()
    print(f"  File now contains: {content.splitlines()[1]}")

    # --- revert_changes: undo the change ---
    print("\n-- revert_changes: undo the patch --")
    result = call_tool(agent, "revert_changes")
    print(f"  Status: {result['status']}")
    print(f"  Reverted: {result.get('reverted_files', [])}")

    # Show the file is restored
    with open(target_file, "r") as f:
        content = f.read()
    print(f"  File now contains: {content.splitlines()[1]}")
    assert "VERSION = '1.0'" in content, "Revert failed!"
    print("  (confirmed: file restored to original)")

    # --- apply_patch with preview=True: dry run without writing ---
    print("\n-- apply_patch (preview=True): dry run a patch --")
    result = call_tool(agent, "apply_patch",
        target="app.py",
        edits=[{"match": "print('Starting app...')", "replace": "print('App v1.6 starting...')"}],
        preview=True,
    )
    print(f"  Status: {result['status']}")
    print(f"  Edits valid: {result.get('edits_valid', False)}")
    print(f"  Current size: {result.get('current_size', '?')} -> New size: {result.get('new_size', '?')}")

    # Verify the file was NOT changed (it's a preview)
    with open(target_file, "r") as f:
        content = f.read()
    assert "Starting app..." in content, "Preview should not modify the file!"
    print("  (confirmed: file unchanged after preview)")


# =============================================================
# Part 3: Job Execution
# =============================================================

def part3_job_execution(agent):
    print("\n" + "=" * 60)
    print("Part 3: Job Execution")
    print("=" * 60)

    # JobModule requires SandboxModule for command execution (secure by default).
    # SandboxModule provides agent.tool_executor which routes commands through
    # the sandbox backend (LocalProcessBackend).
    sandbox_mod = SandboxModule(auto_assign=True)
    agent.register_module(sandbox_mod)

    job_mod = JobModule()
    agent.register_module(job_mod)

    # --- start_job with wait=True: synchronous execution ---
    print("\n-- start_job (wait=True): synchronous 'echo hello' --")
    result = call_tool(agent, "start_job", command="echo hello from llamagent", wait=True)
    print(f"  Status: {result['status']}")
    print(f"  Output: {result.get('output', '').strip()}")

    # --- start_job with wait=False: asynchronous execution ---
    print("\n-- start_job (wait=False): async 'sleep 1 && echo done' --")
    result = call_tool(agent, "start_job",
        command="sleep 1 && echo async job done",
        wait=False,
    )
    job_id = result["job_id"]
    print(f"  Job started, job_id: {job_id}")

    # --- inspect_job: check status + output (non-blocking) ---
    print("\n-- inspect_job: check running job --")
    result = call_tool(agent, "inspect_job", job_id=job_id)
    print(f"  Status: {result['status']}")
    print(f"  Elapsed: {result.get('elapsed_seconds', '?')}s")

    # --- wait_job: wait for completion ---
    print("\n-- wait_job: wait for async job to complete --")
    result = call_tool(agent, "wait_job", job_id=job_id)
    print(f"  Status: {result['status']}")
    print(f"  Return code: {result.get('return_code', '?')}")
    print(f"  Output: {result.get('output', '').strip()}")


# =============================================================
# Part 4: Workspace Lifecycle
# =============================================================

def part4_workspace_lifecycle(agent, tools_mod):
    print("\n" + "=" * 60)
    print("Part 4: Workspace Lifecycle (cleanup on shutdown)")
    print("=" * 60)

    ws_root = tools_mod.workspace_service.workspace_root
    print(f"\n  Workspace root: {ws_root}")
    print(f"  Exists before shutdown: {os.path.isdir(ws_root)}")

    # Shutdown the agent (calls on_shutdown on all modules in reverse order)
    agent.shutdown()
    print("\n  agent.shutdown() called")
    print(f"  Exists after shutdown:  {os.path.isdir(ws_root)}")
    print("  (workspace session directory cleaned up)")


# =============================================================
# Main
# =============================================================

if __name__ == "__main__":
    agent, tools_mod, tmp_project = part1_workspace_tools()
    part2_project_sync(agent, tools_mod, tmp_project)
    part3_job_execution(agent)
    part4_workspace_lifecycle(agent, tools_mod)

    print("\n" + "=" * 60)
    print("Done! v1.6 workspace-first tools with pack-based exposure:")
    print("  - Default surface: read_files, write_files, list_tree, search_text, ...")
    print("  - Project sync:    apply_patch (with preview), revert_changes")
    print("  - Job system:      start_job, inspect_job, wait_job, cancel_job")
    print("=" * 60)
