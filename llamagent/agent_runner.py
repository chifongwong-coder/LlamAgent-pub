"""
Subprocess entry point for child agent execution.

Usage:
    python -m llamagent.agent_runner --spec <json_file>

Reads a spec JSON file describing the task, configuration, and constraints,
then creates and runs a LlamAgent in isolation. The final result is written
as JSON to stdout; all other output (print, logging) is redirected to stderr
to prevent stdout pollution.

This module is NOT imported by the rest of the framework — it is only invoked
as a subprocess by ProcessRunnerBackend.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time


def main():
    """Entry point for subprocess child agent execution."""
    # ---- stdout swap: redirect all print/logging to stderr ----
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

    # ---- parse arguments ----
    parser = argparse.ArgumentParser(description="LlamAgent subprocess runner")
    parser.add_argument("--spec", required=True, help="Path to spec JSON file")
    args = parser.parse_args()

    start_time = time.time()
    output = None
    agent = None

    try:
        # ---- read spec (inside try so errors produce JSON output) ----
        with open(args.spec, "r", encoding="utf-8") as f:
            spec = json.load(f)

        from llamagent.core.config import Config
        from llamagent.core.agent import LlamAgent

        # Build config from spec
        config = Config()
        spec_config = spec.get("config", {})
        if spec_config.get("model"):
            config.model = spec_config["model"]
        if spec_config.get("max_react_steps"):
            config.max_react_steps = spec_config["max_react_steps"]
        if spec_config.get("react_timeout"):
            config.react_timeout = spec_config["react_timeout"]
        if spec_config.get("system_prompt"):
            config.system_prompt = spec_config["system_prompt"]

        # Create agent
        agent = LlamAgent(config)

        # Set project_dir and playground_dir from spec config (workspace isolation)
        import os
        workspace_mode = spec_config.get("workspace_mode", "sandbox")
        if workspace_mode == "project":
            if spec_config.get("project_dir"):
                agent.project_dir = spec_config["project_dir"]
            if spec_config.get("playground_dir"):
                agent.playground_dir = spec_config["playground_dir"]
        else:
            # Sandbox: create isolated workspace under parent's playground
            parent_playground = spec_config.get("parent_playground_dir") or spec_config.get("playground_dir")
            if parent_playground:
                task_id = os.path.splitext(os.path.basename(args.spec))[0]
                workspace_dir = os.path.join(parent_playground, "children", task_id)
                os.makedirs(workspace_dir, exist_ok=True)
                agent.project_dir = workspace_dir
                agent.playground_dir = os.path.join(workspace_dir, "llama_playground")
                os.makedirs(agent.playground_dir, exist_ok=True)

        # v2.7: import parent scopes for project mode scope inheritance
        parent_scopes = spec.get("parent_scopes")
        if parent_scopes:
            agent._authorization_engine.import_scopes(parent_scopes)

        # 1. Sandbox module (if parent has sandbox) — must register FIRST
        #    so on_attach creates tool_executor before tools are loaded
        if spec.get("sandbox_enabled"):
            try:
                from llamagent.modules.sandbox import SandboxModule
                agent.register_module(SandboxModule())
            except ImportError:
                pass  # Sandbox module not installed

        # 2. Load tools + retrieval modules
        from llamagent.modules.tools import ToolsModule
        from llamagent.modules.retrieval import RetrievalModule

        agent.register_module(ToolsModule())
        agent.register_module(RetrievalModule())

        # 3. Filter tools by allowlist
        # Note: tool_denylist is not applied in process mode (known limitation).
        # Custom tool functions from the parent process are also not available.
        # Use ThreadRunner for scenarios requiring denylist or custom tools.
        tool_allowlist = spec.get("tool_allowlist")
        if tool_allowlist is not None:
            for name in list(agent._tools):
                if name not in tool_allowlist:
                    del agent._tools[name]

        # 4. Apply execution_policy from spec to tools without a policy
        ep_dict = spec.get("execution_policy")
        if ep_dict:
            try:
                from llamagent.modules.sandbox.policy import ExecutionPolicy
                policy = ExecutionPolicy(**ep_dict)
                for tool in agent._tools.values():
                    if tool.get("execution_policy") is None:
                        tool["execution_policy"] = policy
            except ImportError:
                pass  # Sandbox policy module not available

        # Apply BudgetedLLM if budget specified
        budget_spec = spec.get("budget")
        if budget_spec:
            from llamagent.modules.child_agent.budget import (
                Budget,
                BudgetTracker,
                BudgetedLLM,
            )

            budget = Budget(
                max_llm_calls=budget_spec.get("max_llm_calls"),
                max_time_seconds=budget_spec.get("max_time_seconds"),
            )
            tracker = BudgetTracker(budget)
            agent.llm = BudgetedLLM(agent.llm, tracker)

        # Build prompt
        task = spec.get("task", "")
        context = spec.get("context", "")
        if context:
            prompt = f"Context:\n{context}\n\nTask:\n{task}"
        else:
            prompt = task

        # Execute
        result_text = agent.chat(prompt)
        elapsed = time.time() - start_time

        metrics = {"elapsed_seconds": round(elapsed, 2)}
        if hasattr(agent.llm, 'tracker'):
            t = agent.llm.tracker
            metrics["tokens_used"] = t.tokens_used
            metrics["llm_calls"] = t.llm_calls
            metrics["steps_used"] = t.steps_used

        output = {
            "status": "completed",
            "result": result_text,
            "history": list(agent.history),
            "metrics": metrics,
        }

    except SystemExit:
        elapsed = time.time() - start_time
        output = {
            "status": "cancelled",
            "result": "Process terminated",
            "history": list(agent.history) if agent else [],
            "metrics": {"elapsed_seconds": round(elapsed, 2)},
        }

    except BaseException as e:
        elapsed = time.time() - start_time
        output = {
            "status": "failed",
            "result": str(e),
            "history": list(agent.history) if agent else [],
            "metrics": {"elapsed_seconds": round(elapsed, 2)},
        }

    finally:
        # Shutdown agent if it was created
        if agent is not None:
            try:
                agent.shutdown()
            except Exception:
                pass

        # Write JSON result to the real stdout
        if output is not None:
            json.dump(output, real_stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
