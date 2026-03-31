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

A bare `SmartAgent` is a fully functional AI chatbot. Every capability — tools, memory, planning, safety — is a **pluggable module** that snaps in with one line:

```python
from llamagent import SmartAgent, Config

agent = SmartAgent(Config())
agent.chat("Hello!")  # Works immediately — no setup needed
```

Need more power? Add modules:

```python
agent.register_module(ToolsModule())      # File operations, project sync
agent.register_module(MemoryModule())     # Long-term structured memory
agent.register_module(PlanningModule())   # DAG-based task decomposition
```

## Why LlamAgent?

**Truly modular** — 12 independent modules, zero coupling between them. Load only what your task needs. A chatbot, a code assistant, a research agent — same framework, different module sets.

**Any LLM, zero lock-in** — OpenAI, Anthropic, DeepSeek, Ollama, or any [LiteLLM](https://github.com/BerriAI/litellm)-supported backend. Auto-detects available API keys. No key? Falls back to local Ollama.

**Smart tool exposure** — Not all tools need to be visible all the time. The pack system dynamically shows the right tools at the right moment — triggered by task context, skills, or runtime state. Small models see a clean 12-tool surface, not a wall of 40 schemas.

**Workspace-first safety** — The agent works in an isolated workspace. Project modifications go through explicit sync channels with full changeset tracking and one-click revert. Three-zone protection (playground / project / external) prevents accidental damage without blocking productivity.

**Memory that actually works** — Not text blobs in a vector DB. LlamAgent extracts structured facts from conversations, deduplicates them, resolves conflicts when information changes, and auto-recalls relevant memories each turn. Read and write modes are independently configurable.

**Retrieval done right** — Hybrid search combines vector similarity with keyword matching (BM25), then reranks results with an LLM. Structure-aware document chunking understands Markdown headers, code functions, and paragraph boundaries. The entire retrieval layer is backend-agnostic — swap embedding models or vector databases without touching module code.

**Skills, not just prompts** — Separate *what the agent can do* (tools) from *how it should do it* (skills). Skills are natural-language playbooks that activate on demand through tag matching, and they can dynamically unlock additional tool packs when needed.

**Plan, execute, adapt** — Simple questions get fast single-loop answers. Complex tasks are automatically decomposed into dependency-aware step plans. If a step fails, the planner restructures. If quality is low, reflection triggers replanning. The agent doesn't spin — it adapts.

## Modules

| Module | What it does |
|--------|-------------|
| **Tools** | Read, write, search files. Workspace isolation with project sync and changeset tracking. |
| **Memory** | Structured fact memory — extract, deduplicate, resolve conflicts, auto-recall. Read/write decoupling. |
| **RAG** | Load documents (code, markdown, text), hybrid search (vector + BM25), LLM reranking. |
| **Job** | Run shell commands (sync/async) via sandbox. Inspect, wait, cancel running jobs. |
| **Skill** | Task-level playbook injection. Tag matching, LLM fallback, dynamic tool pack activation. |
| **Reasoning** | SimpleReAct for quick tasks. PlanReAct with DAG decomposition for complex multi-step work. |
| **Reflection** | Score results, extract lessons from failures, trigger replanning when quality is low. |
| **Safety** | Input filtering, output sanitization, three-zone path protection. |
| **Sandbox** | Isolated execution backends for high-risk tools. Secure by default — no sandbox, no shell. |
| **Child Agent** | Spawn constrained sub-agents with budgets, tool allowlists, and independent workspaces. |
| **Multi-Agent** | Role-based delegation — writer, coder, analyst, researcher. |
| **MCP** | Connect to external Model Context Protocol servers. |

Modules share a **retrieval layer** with swappable embedding models and vector backends — no module-level lock-in.

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
python -m llamagent                          # Interactive CLI
python -m llamagent --modules tools,rag      # Specific modules only
python -m llamagent --mode web               # Gradio Web UI
python -m llamagent --mode api               # FastAPI server
python -m llamagent --config prod.yaml       # Custom config file
```

### Use as a library

```python
from llamagent import SmartAgent, Config
from llamagent.modules.tools import ToolsModule
from llamagent.modules.sandbox import SandboxModule
from llamagent.modules.job import JobModule
from llamagent.modules.reasoning import PlanningModule

agent = SmartAgent(Config())
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
on_input()      Safety filtering, pack state reset
    |
    v
on_context()    Memory recall, knowledge retrieval, skill injection
    |
    v
execute()       ReAct loop or PlanReAct (think -> act -> observe)
    |
    v
on_output()     Output masking, reflection scoring
    |
    v
Response
```

Modules interact through **hooks** — forward on input/context, reverse on output. No module imports another. They compose, they don't couple.

## Examples

See [`examples/`](examples/) for runnable tutorials:

| # | File | What you'll learn |
|---|------|------------------|
| 01 | `quick_start.py` | Create an agent and chat |
| 02 | `tools.py` | Register tools, function calling, zone safety |
| 03 | `modules.py` | Load modules, hook pipeline |
| 04 | `reasoning.py` | ReAct loops, task planning |
| 05 | `persona.py` | Roles, permissions, personas |
| 06 | `sandbox.py` | Isolated execution |
| 07 | `child_agent.py` | Constrained sub-agents |
| 08 | `skill.py` | Task playbooks |
| 09 | `workspace_and_jobs.py` | Workspace tools, project sync, jobs |

## Project Structure

```
llamagent/
├── core/              Agent, Config, LLM, Persona
├── modules/
│   ├── retrieval/     Shared search infrastructure (embedding, vector, lexical)
│   ├── tools/         Workspace tools + project sync + pack system
│   ├── memory/        Structured fact memory
│   ├── rag/           Document loading + hybrid retrieval
│   ├── job/           Command execution lifecycle
│   ├── skill/         Playbook injection + pack triggering
│   ├── reasoning/     ReAct + PlanReAct strategies
│   ├── reflection/    Quality evaluation + lesson learning
│   ├── safety/        Input/output guards + zone system
│   ├── sandbox/       Execution isolation backends
│   ├── child_agent/   Sub-agent control with budgets
│   ├── multi_agent/   Role-based delegation
│   └── mcp/           Model Context Protocol
├── interfaces/        CLI, Web UI, API server
├── examples/          Tutorial scripts
└── tests/             540+ tests
```

## License

[MIT](LICENSE)
