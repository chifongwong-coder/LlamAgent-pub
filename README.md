<p align="center">
  <h1 align="center">LlamAgent</h1>
  <p align="center"><strong>Start with nothing. Add only what you need.</strong></p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> |
  <a href="#modules">Modules</a> |
  <a href="#examples">Examples</a> |
  <a href="https://github.com/llamagent/llamagent/wiki">Docs</a>
</p>

---

A bare `LlamAgent` is a fully functional AI chatbot. Every capability — tools, memory, planning, safety — is a **pluggable module** that snaps in with one line:

```python
from llamagent import LlamAgent, Config

agent = LlamAgent(Config())
agent.chat("Hello!")  # Works immediately — no setup needed
```

Need more power? Add modules:

```python
agent.register_module(ToolsModule())      # File operations, project sync
agent.register_module(MemoryModule())     # Long-term structured memory
agent.register_module(PlanningModule())   # DAG-based task decomposition
```

## Why LlamAgent?

**Truly modular** — 15 independent modules, zero coupling between them. Load only what your task needs. A chatbot, a code assistant, a research agent — same framework, different module sets.

**Any LLM, zero lock-in** — OpenAI, Anthropic, DeepSeek, Ollama, or any [LiteLLM](https://github.com/BerriAI/litellm)-supported backend. Auto-detects available API keys. No key? Falls back to local Ollama.

**Smart defaults** — Pick capabilities, not config values. Select "Memory" and the framework auto-configures the best settings. Advanced users can still override everything via YAML.

**Resilient LLM calls** — Five-level error classification, exponential backoff retry, model failover with turn-scoped cooldown, and smart routing (simple queries to cheap models). Your agent doesn't crash on a 429.

**Smart tool exposure** — Not all tools need to be visible all the time. The pack system dynamically shows the right tools at the right moment — triggered by task context, skills, or runtime state. Small models see a clean 12-tool surface, not a wall of 40 schemas.

**Two-tier write safety (v3.3)** — Every project write (`write_files`, `apply_patch`, plus path-fallback `move_path` / `copy_path` / `delete_path`) is tracked in a Changeset journal for instant `revert_changes` rollback. In CI / `auto_approve=True` mode, a coarse snapshot of the project is captured at agent startup as a manual-restore safety net. Configure `edit_root` to narrow the agent's write boundary to a sub-directory; the model physically can't write outside it. The framework auto-classifies paths: writes under `llama_playground/` route to ephemeral scratch (no Changeset, never snapshotted), writes inside the write boundary are tracked, anything outside is rejected — no `zone` parameter for the model to choose, just paths.

**Memory that actually works** — Not text blobs in a vector DB. LlamAgent extracts structured facts from conversations, deduplicates them, resolves conflicts when information changes, and auto-recalls relevant memories each turn. Periodic consolidation cleans up outdated or redundant facts automatically.

**Full execution trace** — Tool calls, tool results, and LLM reasoning are persisted in conversation history. The LLM knows what happened in previous turns. Configurable compression strategies (head/placeholder/summary) keep token costs under control.

**Skills, not just prompts** — Separate *what the agent can do* (tools) from *how it should do it* (skills). Four-layer matching (command, tags, LLM index, always-on) with support for external skill formats (YAML frontmatter, plain .md). Skills improve over time through lesson-driven reflection.

**Plan, execute, adapt** — Simple questions get fast single-loop answers. Complex tasks are automatically decomposed into dependency-aware step plans. If a step fails, the planner restructures. If quality is low, reflection triggers replanning. The agent doesn't spin — it adapts.

**Continuous mode with inject** — Run background triggers (timer, file watch) while still accepting user messages. Three-level priority scheduling (urgent > triggers > normal). Non-interruptible tasks stay protected.

**Three interfaces** — Interactive CLI with tab completion, Gradio Web UI with real-time panels, FastAPI server with REST + SSE streaming + WebSocket. All share the same smart module presets.

## Modules

| Module | What it does |
|--------|-------------|
| **Resilience** | LLM call protection — error classification, retry, model failover, smart routing, turn-scoped cooldown. |
| **Safety** | Input filtering, output sanitization, three-zone path protection. |
| **Compression** | Context management — tool result compression (4 strategies), thinking stripping, LLM summarization. |
| **Persistence** | Save and restore conversation history across restarts. |
| **Sandbox** | Isolated execution backends for high-risk tools. Secure by default — no sandbox, no shell. |
| **Tools** | 5 core file tools (read/write/patch/list/revert) + path-fallback pack (move/copy/delete/glob/search/stat/temp) auto-activated when no shell tool. Auto-classified write zones (project / playground / rejected); no `zone` parameter. Changeset-tracked revert for every typed write. |
| **Job** | Run shell commands (sync/async) via sandbox. Inspect, wait, cancel running jobs. |
| **Retrieval** | Load documents (code, markdown, text), hybrid search (vector + BM25), LLM reranking. |
| **Memory** | Structured fact memory — extract, deduplicate, resolve conflicts, auto-recall. Periodic consolidation. |
| **Skill** | Four-layer matching (command/tag/index/always). Supports external formats. Lesson-driven self-improvement. |
| **Reflection** | Score results, extract lessons from failures, trigger replanning. Drives skill improvement. |
| **Planning** | SimpleReAct for quick tasks. PlanReAct with DAG decomposition for complex multi-step work. |
| **MCP** | Connect to external Model Context Protocol servers. |
| **Child Agent** | Spawn constrained sub-agents with budgets, tool allowlists, and independent workspaces. |

Modules share a **retrieval layer** with swappable embedding models and vector backends — no module-level lock-in. A zero-dependency **FS backend** (markdown files) works out of the box.

## Quick Start

### Install

```bash
git clone https://github.com/llamagent/llamagent.git
cd llamagent
pip install -e .

# Optional extras
pip install -e ".[rag]"     # ChromaDB for memory & RAG
pip install -e ".[cli]"     # Rich terminal output
pip install -e ".[web]"     # Gradio web interface
pip install -e ".[api]"     # FastAPI server
pip install -e ".[all]"     # Everything
```

### Configure

```bash
# Option A: Environment variables
cp .env.example .env
# Set MODEL_NAME, or an API key, or leave empty for local Ollama

# Option B: YAML config (layered: env vars > YAML > defaults)
cp llamagent.yaml.example llamagent.yaml
```

### Run

```bash
python -m llamagent                          # Interactive CLI (smart defaults)
python -m llamagent --modules tools,retrieval # Specific modules only
python -m llamagent --mode web               # Gradio Web UI
python -m llamagent --mode api               # FastAPI server (REST + SSE + WS)
python -m llamagent ask "question" --format json  # Single question, JSON output
```

### Use as a library

```python
from llamagent import LlamAgent, Config
from llamagent.modules.tools import ToolsModule
from llamagent.modules.sandbox import SandboxModule
from llamagent.modules.job import JobModule
from llamagent.modules.reasoning import PlanningModule

agent = LlamAgent(Config())
agent.register_module(ToolsModule())
agent.register_module(SandboxModule())
agent.register_module(JobModule())
agent.register_module(PlanningModule())

reply = agent.chat("Analyze the codebase and write a summary report")
```

## How It Works

```
User Input
    |
    v
on_input()           Safety filtering, memory consolidation check
    |
    v
on_context()         Memory recall, knowledge retrieval, skill injection (4-layer)
    |
    v
execute()            ReAct loop or PlanReAct (think -> act -> observe)
    |  (each tool call)
    |   -> PRE_TOOL_USE hook
    |   -> Authorization engine (zone check + scope matching)
    |   -> Execute tool
    |   -> POST_TOOL_USE hook
    |
    v
on_output()          Output masking, reflection scoring, skill improvement check
    |
    v
History              Full execution trace (assistant + tool_calls + tool results)
    |
    v
Response
```

Modules interact through **pipeline callbacks** — forward on input/context, reverse on output. No module imports another. They compose, they don't couple.

**Event hooks** let you intercept every tool call — audit, block, or trigger side effects — without writing Python:

```yaml
# llamagent.yaml
hooks:
  pre_tool_use:
    - matcher: { tool_name: "start_job" }
      shell: "/usr/local/bin/policy-check"   # exit 0 = allow, non-0 = block
  post_tool_use:
    - shell: "echo \"$HOOK_TOOL_NAME ($HOOK_DURATION_MS ms)\" >> /tmp/audit.log"
```

**Three authorization modes** control how much the agent can do on its own:

| Mode | Behavior |
|------|----------|
| `interactive` | Every side-effect operation asks for confirmation (default) |
| `task` | Execute a task with project access, auto-clean scopes on completion. Configure seed scopes to skip the planning phase, or let the agent plan first and confirm once. |
| `continuous` | Run unattended with background triggers + user inject support. Three-level priority scheduling. |

Set the mode in YAML — the agent applies it on startup:

```yaml
authorization:
  mode: continuous
  seed_scopes:
    - zone: project
      actions: [read, write]
      path_prefixes: [src/, docs/]
```

## Examples

See [`examples/`](examples/) for runnable tutorials:

| # | File | What you'll learn |
|---|------|------------------|
| 01 | `quick_start.py` | Create an agent and chat |
| 02 | `tools.py` | Register tools, function calling, zone safety |
| 03 | `modules.py` | Load modules, callback pipeline |
| 04 | `reasoning.py` | ReAct loops, task planning |
| 05 | `persona.py` | Roles, permissions, personas |
| 06 | `sandbox.py` | Isolated execution |
| 07 | `child_agent.py` | Constrained sub-agents |
| 08 | `skill.py` | Task playbooks |
| 09 | `workspace_and_jobs.py` | Workspace tools, project sync, jobs |

## Project Structure

```
llamagent/
├── core/              Agent, Config, LLM, Persona, Hooks, Authorization, Runner
├── modules/
│   ├── resilience/    LLM call protection (retry, failover, routing)
│   ├── compression/   Context compression (tool result strategies, thinking strip)
│   ├── persistence/   Conversation history save/restore
│   ├── rag/           RAG backend (embedding, vector, lexical, chunker, retriever)
│   ├── fs_store/      FS backend (markdown parser, atomic file store)
│   ├── tools/         Workspace tools + project sync + pack system
│   ├── memory/        Structured fact memory + consolidation
│   ├── retrieval/     Knowledge retrieval module (uses RAG/FS backend)
│   ├── job/           Command execution lifecycle
│   ├── skill/         4-layer playbook matching + self-improvement
│   ├── reasoning/     ReAct + PlanReAct strategies
│   ├── reflection/    Quality evaluation + lesson learning + skill reflection
│   ├── safety/        Input/output guards + zone system
│   ├── sandbox/       Execution isolation backends
│   ├── child_agent/   Sub-agent control with budgets + inject + priority
│   └── mcp/           Model Context Protocol
├── interfaces/        CLI, Web UI, API server, Module Presets
├── examples/          Tutorial scripts
└── tests/             820+ tests
```

## License

[MIT](LICENSE)
