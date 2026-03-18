"""
Agent custom tool manager: lets LlamAgent write code to create tools by itself.

Features:
- Isolated storage per persona (JSON persistence)
- AST whitelist validation: rejects code with dangerous constructs (imports, __ access, etc.)
- Restricted execution: compiled in a sandboxed namespace with whitelisted builtins only
- Admin common tools are stored in __common__.json
- Role custom tools are stored in {persona_id}.json

Security note:
    AST whitelist + restricted exec targets LLM-generated code. There is a small
    possibility that advanced techniques could bypass these restrictions; this is
    beyond the current framework's capability. For maximum isolation, use a
    container/microVM sandbox backend.
"""

import ast
import json
import os
from datetime import datetime
from typing import Callable


# Builtins blacklisted from tool code execution.
# These are removed from __builtins__ before exec().
_DANGEROUS_BUILTINS = {
    "exec", "eval", "compile",       # Code execution
    "open",                           # File I/O
    "__import__",                     # Bypass AST import check
    "globals", "locals", "vars",      # Namespace leakage
    "getattr", "setattr", "delattr", # Bypass __ attribute restrictions
    "breakpoint",                     # Debugger
    "input",                          # Blocking stdin
    "exit", "quit",                   # Process termination
    "memoryview",                     # Low-level memory access
}

# Default modules whitelisted for import in tool code (pure computation, no I/O).
_DEFAULT_SAFE_MODULES = {
    "json", "math", "re", "datetime", "string",
    "collections", "itertools", "functools",
    "random", "hashlib", "base64", "copy", "textwrap",
    "difflib", "statistics", "decimal", "fractions",
    "uuid", "dataclasses",
}


class AgentToolManager:
    """
    Persistence manager for agent custom tools.

    Each persona has a JSON file containing tool name, description, code, and parameter schema.
    Tools are validated via AST whitelist and executed in a restricted namespace.

    Args:
        storage_dir: Directory for tool JSON files.
        persona_id: Persona identifier for storage isolation.
        allowed_modules: Modules allowed in tool code. Defaults to _DEFAULT_SAFE_MODULES.
            Pass "*" to allow all modules (developer takes full responsibility).
    """

    def __init__(self, storage_dir: str, persona_id: str,
                 allowed_modules: set[str] | str | None = None):
        os.makedirs(storage_dir, exist_ok=True)
        self.storage_dir = storage_dir
        self.persona_id = persona_id
        self.storage_path = os.path.join(storage_dir, f"{persona_id}.json")
        if allowed_modules == "*":
            self._allowed_modules = None  # None means allow all
        elif allowed_modules is not None:
            self._allowed_modules = _DEFAULT_SAFE_MODULES | set(allowed_modules)
        else:
            self._allowed_modules = _DEFAULT_SAFE_MODULES
        self._tools: list[dict] = []
        self._compiled: dict[str, Callable] = {}
        self._load()

    # ----------------------------------------------------------
    # Create / Query / Delete
    # ----------------------------------------------------------

    def create(
        self,
        name: str,
        description: str,
        code: str,
        parameters: dict | None = None,
    ) -> dict:
        """
        Create and save a new tool.

        Args:
            name: Tool function name (must match the function name in the code)
            description: Functional description
            code: Python function code
            parameters: Parameter definition in JSON Schema format (optional, auto-inferred)

        Returns:
            Tool definition dict

        Raises:
            ValueError: Syntax error, security violation, function name mismatch, or tool name already exists
        """
        # 1. Syntax check
        self._validate_syntax(code)

        # 2. AST security check
        self._validate_safety(code)

        # 3. Duplicate name check
        if any(t["name"] == name for t in self._tools):
            raise ValueError(f"Tool '{name}' already exists, please choose a different name")

        # 4. Compile in restricted namespace
        func = self._compile(code, name)

        tool_def = {
            "name": name,
            "description": description,
            "code": code,
            "parameters": parameters,
            "created_at": datetime.now().isoformat(),
        }
        self._tools.append(tool_def)
        self._compiled[name] = func
        self._save()
        return tool_def

    def get_function(self, name: str) -> Callable | None:
        """Get the compiled tool function."""
        return self._compiled.get(name)

    def list_tools(self) -> list[dict]:
        """List all tools with their names and descriptions."""
        return [
            {"name": t["name"], "description": t["description"]}
            for t in self._tools
        ]

    def delete(self, name: str) -> bool:
        """Delete a tool. Returns True on success, False if not found."""
        for i, t in enumerate(self._tools):
            if t["name"] == name:
                self._tools.pop(i)
                self._compiled.pop(name, None)
                self._save()
                return True
        return False

    def export(self, name: str) -> dict | None:
        """Export a tool definition (for admin review before promoting to common tool)."""
        for t in self._tools:
            if t["name"] == name:
                return dict(t)
        return None

    # ----------------------------------------------------------
    # Code validation and compilation
    # ----------------------------------------------------------

    @staticmethod
    def _validate_syntax(code: str) -> None:
        """Validate that the code syntax is correct."""
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Code syntax error: {e}")

    def _validate_safety(self, code: str) -> None:
        """
        AST validation: reject code containing dangerous constructs.

        Allowed:
        - import of whitelisted modules (configurable, default: json, math, re, etc.)
        - All imports if allowed_modules="*"

        Forbidden:
        - import of non-whitelisted modules (os, subprocess, sys, etc.)
        - Double-underscore attribute access (no __class__, __import__, etc.)
        - Calls to dangerous builtins (open, exec, eval, compile, __import__)
        """
        tree = ast.parse(code)

        for node in ast.walk(tree):
            # Check import statements: only whitelisted modules allowed
            if self._allowed_modules is not None:  # None means allow all
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_module = alias.name.split(".")[0]
                        if root_module not in self._allowed_modules:
                            raise ValueError(
                                f"Security violation: import of '{alias.name}' is not allowed. "
                                f"Allowed modules: {sorted(self._allowed_modules)}"
                            )
                if isinstance(node, ast.ImportFrom):
                    if node.module is None:
                        # Relative imports (from . import x) are not allowed
                        raise ValueError(
                            "Security violation: relative imports are not allowed in tool code"
                        )
                    root_module = node.module.split(".")[0]
                    if root_module not in self._allowed_modules:
                        raise ValueError(
                            f"Security violation: import from '{node.module}' is not allowed. "
                            f"Allowed modules: {sorted(self._allowed_modules)}"
                        )

            # Reject double-underscore attribute access
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise ValueError(
                    f"Security violation: access to '{node.attr}' is not allowed in tool code"
                )

            # Reject double-underscore name access
            if isinstance(node, ast.Name) and node.id.startswith("__") and node.id != "__name__":
                raise ValueError(
                    f"Security violation: access to '{node.id}' is not allowed in tool code"
                )

            # Reject calls to dangerous builtins by name
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("exec", "eval", "compile", "open",
                                     "__import__", "globals", "locals",
                                     "getattr", "setattr", "delattr"):
                    raise ValueError(
                        f"Security violation: call to '{node.func.id}()' is not allowed in tool code"
                    )

    @staticmethod
    def _compile(code: str, name: str) -> Callable:
        """Compile a code string in a restricted namespace and return the callable function."""
        import builtins
        safe_builtins = {k: v for k, v in vars(builtins).items()
                         if k not in _DANGEROUS_BUILTINS}
        namespace = {"__builtins__": safe_builtins}
        try:
            exec(code, namespace)  # noqa: S102 — restricted builtins, AST-validated
        except Exception as e:
            raise ValueError(f"Code compilation failed: {e}")

        if name not in namespace:
            defined = [k for k, v in namespace.items()
                       if callable(v) and not k.startswith("_") and k != "__builtins__"]
            hint = f"Functions defined in the code: {defined}" if defined else "No functions defined in the code"
            raise ValueError(
                f"Function named '{name}' not found in the code. {hint}\n"
                f"Please make sure the code defines def {name}(...)."
            )

        func = namespace[name]
        if not callable(func):
            raise ValueError(f"'{name}' is not a callable function")

        return func

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------

    def _load(self) -> None:
        """Load tool definitions from JSON file and compile them."""
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                self._tools = json.load(f)
            # Compile each tool; skip and log failures
            for t in self._tools:
                try:
                    self._compiled[t["name"]] = self._compile(t["code"], t["name"])
                except Exception as e:
                    print(f"[Tools] Failed to load custom tool '{t['name']}': {e}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"[Tools] Failed to read tool storage file: {e}")
            self._tools = []

    def _save(self) -> None:
        """Persist tool definitions to JSON file."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._tools, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"[Tools] Failed to save tool file: {e}")

    # ----------------------------------------------------------
    # Admin: scan all roles' tools
    # ----------------------------------------------------------

    @staticmethod
    def scan_all(storage_dir: str) -> dict[str, list[dict]]:
        """
        Scan all roles' custom tools in the directory (for admin use).

        Returns:
            {"persona_id": [{"name": ..., "description": ...}, ...], ...}
        """
        result = {}
        if not os.path.isdir(storage_dir):
            return result

        for filename in sorted(os.listdir(storage_dir)):
            if not filename.endswith(".json") or filename == "__common__.json":
                continue
            persona_id = filename[:-5]  # Strip .json
            filepath = os.path.join(storage_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    tools = json.load(f)
                result[persona_id] = [
                    {"name": t["name"], "description": t["description"]}
                    for t in tools
                ]
            except (json.JSONDecodeError, IOError):
                pass

        return result


# Backward compatibility: old name AgentToolStore as an alias for AgentToolManager
AgentToolStore = AgentToolManager
