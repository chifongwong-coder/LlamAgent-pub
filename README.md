# LlamAgent

**The Agent Framework That Grows With You**

Most agent frameworks hand you a monolith and say "good luck." You get a hundred features you didn't ask for, dependencies you can't untangle, and abstractions that fight you the moment you step off the golden path.

LlamAgent takes the opposite approach: **start with nothing, add only what you need.**

A bare `SmartAgent` is a fully functional conversational AI — no modules, no setup, just `agent.chat("hello")` and you're talking. But here's where it gets interesting: every capability is a **pluggable module** that snaps in with a single line of code.

```python
agent.register_module(ToolsModule())      # Now it can call functions
agent.register_module(PlanningModule())    # Now it can decompose complex tasks
agent.register_module(SafetyModule())      # Now it filters dangerous input and masks sensitive output
```

Each module interacts through a **hook pipeline** inspired by the onion model — `on_input` and `on_context` fire forward, `on_output` fires in reverse. Safety catches threats on the way in and scrubs secrets on the way out. RAG injects knowledge. Memory recalls past conversations. Reflection evaluates quality. No module knows about any other module. They compose, they don't couple.

## The Brain: ReAct + PlanReAct

Under the hood, LlamAgent ships with two execution strategies. **SimpleReAct** handles straightforward requests — the classic Think → Act → Observe loop. But when you load the Planning module, the engine upgrades to **PlanReAct**, which automatically judges task complexity via a single LLM call:

- **Simple task?** → Fast-track to ReAct. No overhead.
- **Complex task?** → Decompose into a DAG of 3-8 steps with dependencies, then execute them in topological order.

The planner doesn't just execute blindly. It has **three replan pathways** sharing a single adjustment counter:
1. **Model-initiated** — the agent calls a `replan` tool mid-execution when it realizes the plan needs changing
2. **Failure-triggered** — a step fails, the planner automatically restructures around it
3. **Quality-driven** — after all steps complete, the Reflection engine scores the result; below threshold, it replans and tries again

Deadlock? Detected automatically. Circular dependencies? Caught at plan validation via Kahn's algorithm. The agent doesn't spin — it adapts.

## The Toolbox: Four Tiers of Access Control

Tools aren't just functions you register. LlamAgent implements a **four-tier tool system** — `default`, `common`, `admin`, and `agent` — with visibility controlled by tier and executability controlled by safety level. An intern persona sees read-only tools. An admin persona sees everything. The agent can even **create its own tools** at runtime, with automatic code scanning to assign safety levels.

Three-layer security wraps around every interaction:
- **Input filtering** — blocks injection attacks, jailbreak attempts, and dangerous patterns
- **Permission checking** — compares tool safety level against persona permission level
- **Output sanitization** — masks API keys, credentials, phone numbers, and ID cards before they reach the user

## Any LLM. Any Interface. Zero Lock-in.

LlamAgent talks to **any LLM backend** through [LiteLLM](https://github.com/BerriAI/litellm) — OpenAI, Anthropic, DeepSeek, Mistral, or a free local Ollama model running on your laptop. No API key? No problem. It auto-detects what's available and falls back gracefully.

Ship your agent however you want:
- **CLI** — rich terminal interface with slash commands
- **Web UI** — Gradio-based chat with file upload for RAG
- **API Server** — FastAPI with WebSocket streaming, session management, and Swagger docs

All three interfaces share the same `create_agent()` factory. Same agent, different door.

## Built to Be Understood

LlamAgent isn't a research prototype or a framework-of-frameworks. It's **production-aware code written to be read.** Every module follows the same pattern. Every hook has a clear contract. The entire framework is 41 source files with no magic, no metaclasses, and no surprises.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Interfaces                      │
│            CLI  ·  Web UI  ·  API                │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│                 SmartAgent                        │
│  chat() ──► on_input ──► on_context ──► execute  │
│                                    ──► on_output │
└──┬───────────────────────────────────────────┬──┘
   │              Module Hooks                  │
┌──▼──────────────────────────────────────────▼──┐
│  Tools · RAG · Memory · Reasoning · Reflection  │
│  Multi-Agent · MCP · Safety                     │
└─────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/llamagent/llamagent.git
cd llamagent

# Install with pip
pip install -e .

# Or install with extras
pip install -e ".[all]"     # everything
pip install -e ".[cli]"     # rich terminal output
pip install -e ".[web]"     # Gradio web interface
pip install -e ".[api]"     # FastAPI server
pip install -e ".[rag]"     # ChromaDB for RAG & memory
pip install -e ".[mcp]"     # Model Context Protocol
```

### Configure your LLM

```bash
cp .env.example .env
# Edit .env — pick one:
#   Option A: Set MODEL_NAME directly (e.g., deepseek/deepseek-chat)
#   Option B: Set an API key (DEEPSEEK_API_KEY, OPENAI_API_KEY, etc.)
#   Option C: Leave empty — defaults to local Ollama
```

### Run

```bash
# Interactive CLI (default: loads all available modules)
python -m llamagent

# Web UI
python -m llamagent --mode web

# API server
python -m llamagent --mode api

# Pure chat mode (no modules)
python -m llamagent --no-modules

# Select specific modules
python -m llamagent --modules tools,rag,memory
```

### Use as a library

```python
from llamagent import SmartAgent, Config

# Minimal — just chat
agent = SmartAgent(Config())
reply = agent.chat("What is the capital of France?")
print(reply)

# With modules
from llamagent.modules.tools import ToolsModule
from llamagent.modules.reasoning import PlanningModule

agent.register_module(ToolsModule())
agent.register_module(PlanningModule())
reply = agent.chat("Search for recent AI papers and summarize the top 3")
```

## Modules

| Module | Description | Key Capability |
|--------|-------------|----------------|
| **Tools** | Four-tier tool registry with auto schema inference | `register_tool()`, agent-created tools |
| **RAG** | ChromaDB-based semantic search | `search_knowledge`, document loading |
| **Memory** | Persistent memory with semantic recall | Autonomous / hybrid modes |
| **Reasoning** | ReAct + PlanReAct with DAG-based task planning | Complexity routing, replan on failure |
| **Reflection** | Quality evaluation and lesson learning | Score-based replan trigger |
| **Multi-Agent** | Role-based task delegation | Writer, coder, analyst, researcher |
| **MCP** | Model Context Protocol client | Connect to external MCP servers |
| **Safety** | Three-layer security (input/permission/output) | Injection detection, output masking |

## Module Hook Pipeline

Modules interact with the agent through a hook pipeline:

```
User Input
    │
    ▼
on_input()      ← forward order  (safety filtering, preprocessing)
    │
    ▼
on_context()    ← forward order  (RAG retrieval, memory recall, lesson injection)
    │
    ▼
execute()       ← strategy       (SimpleReAct or PlanReAct)
    │
    ▼
on_output()     ← reverse order  (output masking, reflection)
    │
    ▼
Response
```

## Configuration

All settings can be overridden via environment variables. See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | auto-detect | LLM model identifier |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `MEMORY_MODE` | `off` | Memory mode: `off` / `autonomous` / `hybrid` |
| `PERMISSION_LEVEL` | `1` | Default tool permission level (1-3) |
| `MAX_REACT_STEPS` | `10` | Max ReAct loop iterations |
| `WEB_UI_PORT` | `7860` | Gradio web UI port |
| `API_PORT` | `8000` | FastAPI server port |

## Examples

See the [`examples/`](examples/) directory for runnable tutorials:

- **[01_quick_start.py](examples/01_quick_start.py)** — Create an agent and chat
- **[02_tools.py](examples/02_tools.py)** — Register tools and use function calling
- **[03_modules.py](examples/03_modules.py)** — Load modules and understand the hook pipeline
- **[04_reasoning.py](examples/04_reasoning.py)** — ReAct loops and task planning
- **[05_persona.py](examples/05_persona.py)** — Create personas with roles and permissions

## Project Structure

```
llamagent/
├── llamagent/
│   ├── core/           # Agent, Config, LLM, Persona
│   ├── modules/
│   │   ├── tools/      # Tool registry + built-in tools
│   │   ├── rag/        # Retrieval-Augmented Generation
│   │   ├── memory/     # Persistent memory
│   │   ├── reasoning/  # ReAct + PlanReAct
│   │   ├── reflection/ # Quality evaluation
│   │   ├── multi_agent/# Role-based delegation
│   │   ├── mcp/        # Model Context Protocol
│   │   └── safety/     # Input/permission/output security
│   └── interfaces/     # CLI, Web UI, API server
├── examples/           # Tutorial scripts
├── tests/              # Test suite
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## License

[MIT](LICENSE)
