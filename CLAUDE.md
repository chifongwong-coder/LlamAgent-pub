# LlamAgent — Project Instructions

## Overview

LlamAgent is a modular AI Agent framework. The core `LlamAgent` is a standalone chatbot; each pluggable module adds a capability (tools, retrieval, memory, reasoning, reflection, multi-agent, MCP, safety, skill, sandbox, child agent, job, compression).

## Directory Structure

```
llamagent/                    # Git root
├── llamagent/                # Python package (PUBLIC)
│   ├── core/                 # Agent, Config, LLM, Persona, Runner
│   ├── modules/              # 13 pluggable modules + shared infrastructure
│   │   ├── rag/              # Shared retrieval backend (embedding, vector, lexical, pipeline, factory, chunker, retriever)
│   │   ├── fs_store/         # Shared FS backend (markdown parser, atomic file store)
│   │   └── job/              # Managed command execution
│   └── interfaces/           # CLI, Web UI, API server
├── examples/                 # Tutorial scripts (PUBLIC)
├── tests/                    # Curated public tests (PUBLIC)
├── tests_internal/           # Full test suite (PRIVATE, gitignored)
├── docs/                     # Design docs (PRIVATE, gitignored)
├── .gitignore
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Language Rules

| Scope | Language | Notes |
|-------|----------|-------|
| Source code (`llamagent/`) | **English only** | Comments, docstrings, prompts, log messages, error strings |
| Examples (`examples/`) | **English only** | All comments and output strings |
| Tests (`tests/`) | **English only** | Docstrings, assertions, test data strings |
| Docs (`docs/`) | Chinese OK | Private, gitignored. But LLM prompt strings inside docs must be English |
| Discussion | Chinese OK | We can discuss in Chinese |
| Git (commits, branches, PRs) | **English only** | Commit messages, branch names, PR titles/descriptions |

## Code Conventions

- Modules are loosely coupled; use `agent.has_module()` to check dependencies
- Optional dependencies (chromadb, mcp, gradio) degrade gracefully when missing
- Tech stack: Python + LiteLLM + ChromaDB + FastAPI + Gradio
- Callback pipeline: on_input (forward) → on_context (forward) → execute → on_output (reverse)
- Event hook system (v1.8): PRE_TOOL_USE / POST_TOOL_USE / TOOL_ERROR + lifecycle events
- Authorization engine (v1.9): ZoneEvaluation + InteractivePolicy / TaskPolicy / ContinuousPolicy
- Four-tier tool system: default / common / admin / agent
- ReAct engine is strategy-agnostic: `run_react(messages, tools_schema, tool_dispatch)`
- Streaming (v2.0.2): `chat_stream()` parallel to `chat()`, strategy decides via `execute_stream()` (None = no support)
- Per-module model (v2.1): `config.module_models` maps module name → model; `register_module()` sets `module.llm` before `on_attach()`
- Compression module (v2.1): extracted from core; `on_input` side-effect checks token threshold; calls `agent.compress_conversation()`
- Backend architecture (v2.2): `modules/rag/` = shared RAG backend infra; `modules/retrieval/` = RetrievalModule (pluggable); `modules/fs_store/` = shared FS backend
- Retrieval/Memory backend switching (v2.2): `config.retrieval_backend` / `config.memory_backend` = `"rag"` or `"fs"`; FS backend uses markdown files with zero external dependencies

## Testing

- **Public tests** (`tests/`): curated flow tests (public, tracked by git, used by CI) — 87 tests
  - `test_react.py` (2) — ReAct loop flow + weak model degradation
  - `test_pipeline.py` (2) — chat pipeline flow + safety/blocked
  - `test_integration.py` (2) — module integration flow + create_agent
  - `test_planning.py` (3) — complexity routing + deadlock/replan + quality evaluation
  - `test_sandbox.py` (4) — executor + security + integration + local process
  - `test_child_agent.py` (4) — integration/budget + roles/task board + zone inheritance + security
  - `test_skill.py` (4) — tag matching/LLM fallback + injection format + pack/slash command + builtins
  - `test_workspace.py` (5) — registration/filtering + context + exploration/file types + write/restrictions + patch/sync
  - `test_job.py` (3) — registration/sync + async/pack + cancel/error/shutdown
  - `test_hooks.py` (4) — tool hooks/skip + matcher + lifecycle/reentry + shell/yaml
  - `test_builtin_tools.py` (2) — ask_user flow + web_search flow
  - `test_authorization.py` (5) — zone/confirm + scope governance/audit + continuous + config-driven init + apply_update
  - `test_task_mode.py` (8) — controller state machine + data flow + dry-run + happy path + cancel/mode switch + scope matching + session scopes + re-prepare/loop protection
  - `test_cross_module.py` (4) — pack-skill + pack-job + workspace-project + hook/context stacking
  - `test_v2_features.py` (35) — v2.0 (mode config, abort, runner, triggers, streaming) + v2.1 (per-module model, compression) + v2.2 (FS parser, FSStore, FSMemoryStore, retrieval/memory backend switching)
  - Run: `llamagent_env/bin/python -m pytest tests/ -v`
- **Internal tests** (`tests_internal/`): detailed tests (private, gitignored) — 512 mock + 14 real
  - v1.1: `test_config.py`, `test_llm_mock.py`, `test_agent_*.py`, `test_persona.py`, `test_step.py`, `test_react_mock.py`, `test_chat_pipeline_mock.py`, `test_integration_mock.py`, `test_plan_react_mock.py`
  - v1.2: `test_sandbox_mock.py`, `test_child_agent_mock.py`
  - v1.8: `test_hooks_mock.py`
  - v1.8.1: `test_web_search_mock.py`
  - v1.8.2: `test_interaction_mock.py`
  - v1.9.0: `test_zone_mock.py`, `test_authorization_mock.py`
  - v2.0-v2.2: `test_v2_features_mock.py`
  - Real: `test_*_real.py` (require local Ollama qwen3:1.7b)
  - Run all mock: `llamagent_env/bin/python -m pytest tests_internal/test_*_mock.py tests_internal/test_config.py tests_internal/test_agent_*.py tests_internal/test_persona.py tests_internal/test_step.py -v`
- conftest.py provides: `mock_llm_client`, `bare_agent`, `make_llm_response`, `make_tool_call`, `make_stream_chunks`, `make_stream_tool_call_chunks`
- **Total: 613 tests** (87 public + 512 internal mock + 14 real)

## Version Update Workflow

Each version follows a strict sequence: **plan → code → test → docs**.

1. **Plan phase**: Write a plan doc (`docs/llamagent-v<X>-plan.md`) that defines scope, design decisions, implementation details, and API surface changes. All ambiguities must be resolved before coding starts. Archive design discussion docs to `docs/archive/` once the plan is finalized.
2. **Code phase**: Implement on a feature branch (`feature/v<X>`). Reference the plan as the source of truth.
3. **Test phase**: Update and run all tests (public + internal). All tests must pass before proceeding.
4. **Docs phase**: Update architecture doc and API reference doc to match the implemented code. Code is the ground truth — docs describe code, not the other way around.

API changes should be identified in the plan (so they can be discussed early), but the actual doc updates happen after code is finalized.

## Git Workflow

- **Never commit directly to `main`**. All development happens on feature branches.
- Branch naming: `feature/<name>`, `fix/<name>`, `refactor/<name>`, etc.
- When work is done and tests pass, the user will merge to `main` and push.
- Do NOT push to remote — the user will push manually.

## Design Principles (v1.9.6+)

Six principles govern all core architecture changes:

- **P1 Agent independence**: Agent is a courier, doesn't understand controller/engine internals
- **P2 Agent doesn't know internals**: Agent routes data without inspecting it
- **P3 Controller pure state machine**: No references to agent/engine, all data via params and returns
- **P4 Engine owns state**: All authorization state managed by engine
- **P5 Single-direction deps**: zone.py(L3) → contract.py(L2) → authorization.py/controller.py(L1) → agent.py(L0)
- **P6 No over-engineering**: Don't implement hypothetical future requirements, but reserve extension interfaces (e.g., callbacks, hooks) for foreseeable needs

### Principle Validation Rules

These rules were learned from v1.9.7 mistakes and must be followed in all future work:

1. **Review output filter**: Every improvement suggestion from expert review must be individually validated against P1-P6 before recording into a backlog or improvement list. Reviewers tend to apply general engineering heuristics (eliminate isinstance, avoid magic strings, provide extension points) that may conflict with project principles — especially P6.

2. **Verify improvement lists against current code**: Improvement lists can become stale. Before starting work on any item, check whether it's already been fixed or is no longer relevant. Don't trust the list blindly.

3. **Evaluate every change against all 6 principles BEFORE implementing**: Not just the skip decisions — every item you plan to implement must be traced through P1-P6. This includes seemingly mechanical changes like renaming or condition simplification.

4. **Analyze each call site individually**: Same-looking code patterns at different call sites may have different semantics. Never batch-apply a change across multiple sites without analyzing why each site has its current form. Example: two `self.mode == "task" and self._controller` checks may serve different purposes — one guards controller-agnostic routing, the other guards access to task-mode-specific state.

## Important Rules

- Do NOT auto-commit without explicit user instruction
- Do NOT push to remote — the user will push manually
- Do NOT add `Co-Authored-By` or any AI attribution in commit messages
- Do NOT change code logic when translating or refactoring text
- When unsure about something, ask first — don't guess and modify
