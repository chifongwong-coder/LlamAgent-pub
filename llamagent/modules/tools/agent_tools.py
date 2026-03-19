"""
Agent custom tool manager: lets LlamAgent write code to create tools by itself.

Features:
- Isolated storage per persona (JSON persistence)
- AST safety checks: rejects __ access and dangerous builtin calls (exec, eval)
- String literal path scanning: rejects code with hardcoded paths outside the project directory
- Restricted execution: compiled in a namespace with dangerous builtins removed
- Admin common tools are stored in __common__.json
- Role custom tools are stored in {persona_id}.json

Security note:
    AST checks + restricted builtins target LLM-generated code. The zone system
    in call_tool() provides runtime path safety for created tools. For maximum
    isolation, use a container/microVM sandbox backend.
"""

import ast
import json
import os
from datetime import datetime
from typing import Callable


# Builtins blacklisted from tool code execution.
# Only block code-nesting primitives (exec/eval). Normal imports are allowed;
# the zone system handles path safety at runtime.
_DANGEROUS_BUILTINS = {"exec", "eval"}


class AgentToolManager:
    """
    Persistence manager for agent custom tools.

    Each persona has a JSON file containing tool name, description, code, and parameter schema.
    Tools are validated via AST checks and executed in a restricted namespace.

    Args:
        storage_dir: Directory for tool JSON files.
        persona_id: Persona identifier for storage isolation.
    """

    def __init__(self, storage_dir: str, persona_id: str):
        os.makedirs(storage_dir, exist_ok=True)
        self.storage_dir = storage_dir
        self.persona_id = persona_id
        self.storage_path = os.path.join(storage_dir, f"{persona_id}.json")
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

    @staticmethod
    def _validate_safety(code: str) -> None:
        """
        AST validation: reject code containing dangerous constructs.

        Forbidden:
        - Double-underscore attribute access (no __class__, __subclasses__, etc.)
        - Double-underscore name access (no __builtins__, etc.)
        - Calls to dangerous builtins (exec, eval)
        - String literal paths outside the current working directory
        """
        tree = ast.parse(code)

        for node in ast.walk(tree):
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
                if node.func.id in _DANGEROUS_BUILTINS:
                    raise ValueError(
                        f"Security violation: call to '{node.func.id}()' is not allowed in tool code"
                    )

        # String literal path scanning
        cwd = os.path.realpath(os.getcwd())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value
                if '/' in val and (val.startswith('/') or val.startswith('~')):
                    resolved = os.path.realpath(os.path.expanduser(val))
                    if not resolved.startswith(cwd + os.sep) and resolved != cwd:
                        raise ValueError(
                            f"Security violation: path '{val}' in code is outside the project directory"
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
