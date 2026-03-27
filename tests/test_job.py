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
    """Invoke a registered tool by name and return the raw result string."""
    func = agent._tools[name]["func"]
    return func(**kwargs)


def _call_tool_json(agent, name, **kwargs):
    """Invoke a registered tool by name and return parsed JSON."""
    raw = _call_tool(agent, name, **kwargs)
    return json.loads(raw)


# ============================================================
# Tool registration
# ============================================================

class TestJobToolRegistration:
    """JobModule registers 4 tools: start_job (default) + 3 pack tools."""

    def test_job_module_registers_all_tools(self, bare_agent, tmp_path):
        """All 4 job tools appear in agent._tools after registration."""
        _make_agent_with_jobs(bare_agent, tmp_path)

        expected_tools = ["start_job", "inspect_job", "wait_job", "cancel_job"]
        for tool_name in expected_tools:
            assert tool_name in bare_agent._tools, f"job tool '{tool_name}' not registered"

    def test_start_job_has_no_pack(self, bare_agent, tmp_path):
        """start_job is in the default public surface (no pack)."""
        _make_agent_with_jobs(bare_agent, tmp_path)
        assert bare_agent._tools["start_job"].get("pack") is None

    def test_followup_tools_have_pack(self, bare_agent, tmp_path):
        """inspect_job, wait_job, cancel_job are in the job-followup pack."""
        _make_agent_with_jobs(bare_agent, tmp_path)

        for tool_name in ["inspect_job", "wait_job", "cancel_job"]:
            assert bare_agent._tools[tool_name].get("pack") == "job-followup"

    def test_old_tools_removed(self, bare_agent, tmp_path):
        """Old tools (job_status, tail_job, collect_artifacts) are no longer registered."""
        _make_agent_with_jobs(bare_agent, tmp_path)

        removed = ["job_status", "tail_job", "collect_artifacts"]
        for tool_name in removed:
            assert tool_name not in bare_agent._tools


# ============================================================
# Synchronous execution
# ============================================================

class TestSyncExecution:
    """start_job with wait=True blocks and returns output."""

    def test_start_job_wait_true(self, bare_agent, tmp_path):
        """Synchronous job returns status and stdout."""
        _make_agent_with_jobs(bare_agent, tmp_path)

        result = _call_tool_json(bare_agent, "start_job", command="echo hello", wait=True)
        assert result["status"] == "success"
        assert "hello" in result["output"]


# ============================================================
# Asynchronous execution + inspect_job
# ============================================================

class TestAsyncExecution:
    """start_job with wait=False returns job_id for later inspection."""

    def test_start_job_wait_false_then_inspect(self, bare_agent, tmp_path):
        """Async start returns job_id; inspect_job returns status and output."""
        _make_agent_with_jobs(bare_agent, tmp_path)

        # Need to activate pack since inspect_job is in job-followup pack
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


# ============================================================
# Pack activation
# ============================================================

class TestJobPackActivation:
    """start_job activates job-followup pack for same-turn visibility."""

    def test_start_job_sync_activates_pack(self, bare_agent, tmp_path):
        """Synchronous start_job adds job-followup to _active_packs."""
        _make_agent_with_jobs(bare_agent, tmp_path)

        assert "job-followup" not in bare_agent._active_packs
        _call_tool_json(bare_agent, "start_job", command="echo test", wait=True)
        assert "job-followup" in bare_agent._active_packs

    def test_start_job_async_activates_pack(self, bare_agent, tmp_path):
        """Asynchronous start_job adds job-followup to _active_packs."""
        _make_agent_with_jobs(bare_agent, tmp_path)

        assert "job-followup" not in bare_agent._active_packs
        result = _call_tool_json(bare_agent, "start_job", command="sleep 1", wait=False)
        assert "job-followup" in bare_agent._active_packs

        # Clean up
        _call_tool(bare_agent, "cancel_job", job_id=result["job_id"])


# ============================================================
# Cancellation
# ============================================================

class TestJobCancellation:
    """cancel_job terminates a running async job."""

    def test_cancel_job(self, bare_agent, tmp_path):
        """Cancelling a running job returns 'cancelled' status."""
        _make_agent_with_jobs(bare_agent, tmp_path)

        start_result = _call_tool_json(
            bare_agent, "start_job", command="sleep 10", wait=False,
        )
        job_id = start_result["job_id"]

        cancel_result = _call_tool_json(bare_agent, "cancel_job", job_id=job_id)
        assert cancel_result["status"] == "cancelled"


# ============================================================
# Error handling
# ============================================================

class TestJobErrorHandling:
    """Error paths: nonexistent job_id, etc."""

    def test_job_not_found(self, bare_agent, tmp_path):
        """wait_job with a fake job_id returns an error."""
        _make_agent_with_jobs(bare_agent, tmp_path)

        result = _call_tool_json(bare_agent, "wait_job", job_id="nonexistent_id_123")
        assert result["status"] == "error"
        assert "not found" in result["reason"]


# ============================================================
# Shutdown
# ============================================================

class TestJobShutdown:
    """on_shutdown terminates running jobs without crashing."""

    def test_job_module_on_shutdown(self, bare_agent, tmp_path):
        """Starting a job then calling on_shutdown does not raise."""
        mod = _make_agent_with_jobs(bare_agent, tmp_path)

        _call_tool_json(
            bare_agent, "start_job", command="sleep 5", wait=False,
        )

        mod.on_shutdown()
