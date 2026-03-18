"""
Pluggable modules package.

Each subdirectory is an independent module that exports a Module subclass.
Agent gains the corresponding capability after loading a module via register_module().

Available modules:
- safety:       Safety guardrails (input/output filtering, permission checks)
- tools:        Tool registration management + meta-tools
- rag:          RAG knowledge retrieval
- memory:       Long-term memory
- reflection:   Self-reflection and error correction
- planning:     Task planning (PlanReAct strategy)
- mcp:          MCP external system integration
- multi_agent:  Multi-Agent collaboration
- sandbox:      v1.2 sandboxed tool execution (Docker/subprocess isolation)
- child_agent:  v1.2 child Agent spawning for delegated sub-tasks
"""
