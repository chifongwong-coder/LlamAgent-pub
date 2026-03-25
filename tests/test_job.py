"""
JobModule flow tests: managed command execution with lifecycle control.

Tests cover tool registration, synchronous and asynchronous job execution,
status polling, cancellation, error handling, and shutdown behavior.
"""

import json
import os

from llamagent.modules.job.module import JobModule


# ============================================================
# Helpers
# ============================================================

def _make_agent_with_jobs(bare_agent, tmp_path):
    """Set up bare_agent with playground_dir and register JobModule."""
    playground_dir = os.path.join(str(tmp_path), "llama_playground")
    os.makedirs(playground_dir, exist_ok=True)
    bare_agent.playground_dir = playground_dir

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
    """JobModule registers all six job tools on attach."""

    def test_job_module_registers_all_tools(self, bare_agent, tmp_path):
        """All 6 job tools appear in agent._tools after registration."""
        _make_agent_with_jobs(bare_agent, tmp_path)

        expected_tools = [
            "start_job", "wait_job", "job_status",
            "tail_job", "cancel_job", "collect_artifacts",
        ]
        for tool_name in expected_tools:
            assert tool_name in bare_agent._tools, f"job tool '{tool_name}' not registered"


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
        assert "hello" in result["stdout"]
        assert result["return_code"] == 0


# ============================================================
# Asynchronous execution
# ============================================================

class TestAsyncExecution:
    """start_job with wait=False returns job_id for later polling."""

    def test_start_job_wait_false_then_wait(self, bare_agent, tmp_path):
        """Async start returns job_id; wait_job returns completed output."""
        _make_agent_with_jobs(bare_agent, tmp_path)

        # Start async
        start_result = _call_tool_json(
            bare_agent, "start_job", command="echo async_output", wait=False,
        )
        assert start_result["status"] == "started"
        job_id = start_result["job_id"]
        assert job_id

        # Wait for completion
        wait_result = _call_tool_json(bare_agent, "wait_job", job_id=job_id)
        assert wait_result["status"] == "success"
        assert "async_output" in wait_result["stdout"]

    def test_job_status(self, bare_agent, tmp_path):
        """job_status reports 'running' for a still-active job."""
        _make_agent_with_jobs(bare_agent, tmp_path)

        start_result = _call_tool_json(
            bare_agent, "start_job", command="sleep 2", wait=False,
        )
        job_id = start_result["job_id"]

        # Check status immediately — should be running
        status_result = _call_tool_json(bare_agent, "job_status", job_id=job_id)
        assert status_result["status"] == "running"

        # Clean up: cancel the long-running job
        _call_tool(bare_agent, "cancel_job", job_id=job_id)


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
        assert cancel_result["job_id"] == job_id


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

        # Shutdown should terminate the job gracefully
        mod.on_shutdown()
