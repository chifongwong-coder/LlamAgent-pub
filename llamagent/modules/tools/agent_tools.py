"""
Agent custom tool manager: lets LlamAgent write code to create tools by itself.

Features:
- Isolated storage per persona (JSON persistence)
- Dynamic compilation: compiles code strings into callable functions
- Admin common tools are stored in __common__.json
- Role custom tools are stored in {persona_id}.json
"""

import ast
import json
import os
from datetime import datetime
from typing import Callable


class AgentToolManager:
    """
    Persistence manager for agent custom tools.

    Each persona has a JSON file containing tool name, description, code, and parameter schema.
    Tools are dynamically compiled into callable functions when loaded.
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
            ValueError: Syntax error, function name mismatch, or tool name already exists
        """
        # Syntax check
        self._validate_syntax(code)

        # Duplicate name check
        if any(t["name"] == name for t in self._tools):
            raise ValueError(f"Tool '{name}' already exists, please choose a different name")

        # Compile into a callable function
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
    def _compile(code: str, name: str) -> Callable:
        """Compile a code string and return the callable function with the specified name."""
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            raise ValueError(f"Code compilation failed: {e}")

        if name not in namespace:
            # List the function names actually defined in the code to help the user debug
            defined = [k for k, v in namespace.items() if callable(v) and not k.startswith("_")]
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
