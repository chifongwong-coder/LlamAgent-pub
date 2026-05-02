"""
JobModule flow tests: v1.6 managed command execution with lifecycle control.

Tests cover tool registration (4 tools), pack declarations, synchronous and
asynchronous job execution, inspect_job unified query, cancellation, pack
activation on start_job, error handling, and shutdown behavior.
"""

import json
import os

from llamagent.modules.job.module import JobModule


# ============================================================
# Helpers
# ============================================================

class _MockToolExecutor:
    """Minimal ToolExecutor with run_command that runs commands via subprocess (for testing)."""

    def run_command(self, command, cwd, timeout=300):
        import subprocess
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=cwd,
            )
            output = result.stdout or ""
            if result.stderr:
                output += result.stderr
            return output
        except subprocess.TimeoutExpired:
            return "[TIMEOUT]"
        except Exception as e:
            return f"Execution error: {e}"


def _make_agent_with_jobs(bare_agent, tmp_path):
    """Set up bare_agent with SandboxModule mock (tool_executor) and register JobModule."""
    playground_dir = os.path.join(str(tmp_path), "llama_playground")
    os.makedirs(playground_dir, exist_ok=True)
    bare_agent.playground_dir = playground_dir

    # JobModule hard-depends on SandboxModule (agent.tool_executor)
    bare_agent.tool_executor = _MockToolExecutor()

    mod = JobModule()
    bare_agent.register_module(mod)
    return mod


def _call_tool(agent, name, **kwargs):
    """Invoke a registered tool by name and return the raw result string.

    v3.6: respects ``takes_agent`` — tools that opt in receive the agent
    as first arg, matching the framework dispatcher contract.
    """
    tool = agent._tools[name]
    func = tool["func"]
    if tool.get("takes_agent"):
        return func(agent, **kwargs)
    return func(**kwargs)


def _call_tool_json(agent, name, **kwargs):
    """Invoke a registered tool by name and return parsed JSON."""
    raw = _call_tool(agent, name, **kwargs)
    return json.loads(raw)


# ============================================================
# Tests
# ============================================================

class TestJobModule:
    """Consolidated JobModule flow tests."""

    def test_registration_and_sync_execution(self, bare_agent, tmp_path):
        """Tool registration (all 4 tools, pack assignment, old tools removed) and sync wait."""
        _make_agent_with_jobs(bare_agent, tmp_path)

        # All 4 job tools appear in agent._tools after registration
        expected_tools = ["start_job", "inspect_job", "wait_job", "cancel_job"]
        for tool_name in expected_tools:
            assert tool_name in bare_agent._tools, f"job tool '{tool_name}' not registered"

        # start_job is in the default public surface (no pack)
        assert bare_agent._tools["start_job"].get("pack") is None

        # inspect_job, wait_job, cancel_job are in the job-followup pack
        for tool_name in ["inspect_job", "wait_job", "cancel_job"]:
            assert bare_agent._tools[tool_name].get("pack") == "job-followup"

        # Old tools (job_status, tail_job, collect_artifacts) are no longer registered
        removed = ["job_status", "tail_job", "collect_artifacts"]
        for tool_name in removed:
            assert tool_name not in bare_agent._tools

        # Synchronous job returns status and stdout
        result = _call_tool_json(bare_agent, "start_job", command="echo hello", wait=True)
        assert result["status"] == "success"
        assert "hello" in result["output"]

    def test_async_and_pack_activation(self, bare_agent, tmp_path):
        """Async start + inspect, and pack activation for both sync and async."""
        _make_agent_with_jobs(bare_agent, tmp_path)

        # --- Async start then inspect ---
        bare_agent._active_packs.add("job-followup")

        start_result = _call_tool_json(
            bare_agent, "start_job", command="echo async_output", wait=False,
        )
        assert start_result["status"] == "started"
        job_id = start_result["job_id"]

        # Wait for completion first
        wait_result = _call_tool_json(bare_agent, "wait_job", job_id=job_id)
        assert wait_result["status"] in ("completed", "success")

        # Then inspect the completed job
        inspect_result = _call_tool_json(bare_agent, "inspect_job", job_id=job_id)
        assert inspect_result["status"] in ("completed", "success")
        assert "async_output" in inspect_result.get("output", "")
        assert "elapsed_seconds" in inspect_result
        assert "artifacts" in inspect_result

        # --- Pack activation: sync ---
        bare_agent2_packs = bare_agent._active_packs
        bare_agent2_packs.clear()
        assert "job-followup" not in bare_agent._active_packs
        _call_tool_json(bare_agent, "start_job", command="echo test", wait=True)
        assert "job-followup" in bare_agent._active_packs

        # --- Pack activation: async ---
        bare_agent._active_packs.clear()
        assert "job-followup" not in bare_agent._active_packs
        result = _call_tool_json(bare_agent, "start_job", command="sleep 1", wait=False)
        assert "job-followup" in bare_agent._active_packs

        # Clean up
        _call_tool(bare_agent, "cancel_job", job_id=result["job_id"])

    def test_cancel_error_and_shutdown(self, bare_agent, tmp_path):
        """Cancellation, error handling (nonexistent job_id), and shutdown."""
        mod = _make_agent_with_jobs(bare_agent, tmp_path)

        # --- Cancellation ---
        start_result = _call_tool_json(
            bare_agent, "start_job", command="sleep 10", wait=False,
        )
        job_id = start_result["job_id"]

        cancel_result = _call_tool_json(bare_agent, "cancel_job", job_id=job_id)
        assert cancel_result["status"] == "cancelled"

        # --- Error handling: nonexistent job_id ---
        result = _call_tool_json(bare_agent, "wait_job", job_id="nonexistent_id_123")
        assert result["status"] == "error"
        assert "not found" in result["reason"]

        # --- Shutdown: start a job then call on_shutdown without crashing ---
        _call_tool_json(
            bare_agent, "start_job", command="sleep 5", wait=False,
        )
        mod.on_shutdown()
