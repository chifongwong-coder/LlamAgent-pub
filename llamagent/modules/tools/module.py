"""
ToolsModule: four-tier tool system + role-based permission management + v1.5 workspace tools.

Responsibilities:
- Tool registration management: maintains global_registry and agent_registry
- Meta-tools: create_tool / list_my_tools / delete_tool / query_toolbox
- Admin tools: create_common_tool / list_all_agent_tools / promote_tool
- Tool persistence: JSON persistence for role custom tools and admin common tools
- v1.6 workspace tools: WorkspaceService + ProjectSyncService (default + pack)
- v1.6 pack mechanism: state-driven + skill-driven conditional tool exposure
- v1.6 workspace guidelines + capability hint block injection via on_context
"""

import base64
import glob
import json
import logging
import mimetypes
import os
import shutil
import tempfile

from llamagent.core.agent import Module
from llamagent.modules.tools.registry import ToolRegistry, global_registry
from llamagent.modules.tools.agent_tools import AgentToolManager

logger = logging.getLogger(__name__)

# Storage ID for admin-created common tools
COMMON_STORE_ID = "__common__"

# Workspace behavioral guidelines injected via on_context
WORKSPACE_GUIDE = """\
[Workspace Guidelines]
- Work in workspace first, then sync results to project via apply_patch or sync_workspace_to_project.
- Paths in workspace tools are relative to workspace root by default.
  Use "project:" prefix to access project files (e.g., "project:src/main.py").
- For command execution, use start_job. Set wait=True for quick commands, wait=False for long tasks."""

# Known text file extensions for automatic type detection in read_files
_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".csv", ".tsv", ".html", ".xml", ".sql", ".log",
    ".sh", ".c", ".cpp", ".h", ".java", ".go", ".rs", ".rb", ".php",
    ".css", ".scss", ".jsx", ".tsx", ".vue", ".svelte", ".env", ".cfg",
    ".conf", ".properties", ".gradle", ".makefile", ".dockerfile",
    ".bat", ".ps1", ".r", ".scala", ".kt", ".swift", ".m", ".mm",
    ".pl", ".pm", ".lua", ".zig", ".nim", ".ex", ".exs", ".erl",
    ".hs", ".ml", ".tf", ".hcl", ".proto", ".graphql", ".rst",
    ".tex", ".bib", ".gitignore", ".dockerignore", ".editorconfig",
})


def _is_text_file(filepath: str) -> bool:
    """Check if a file is likely text based on extension, then content probe."""
    ext = os.path.splitext(filepath)[1].lower()
    # Known text extension
    if ext in _TEXT_EXTENSIONS:
        return True
    # Known binary extensions (skip probe)
    if ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
               ".pdf", ".zip", ".gz", ".tar", ".bz2", ".7z", ".rar",
               ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a",
               ".woff", ".woff2", ".ttf", ".otf", ".eot",
               ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
               ".sqlite", ".db", ".pyc", ".pyo", ".class", ".jar",
               ".xls", ".xlsx", ".doc", ".docx", ".ppt", ".pptx"):
        return False
    # No extension or unknown: probe first 8KB
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(8192)
        # Null bytes strongly indicate binary
        if b"\x00" in chunk:
            return False
        return True
    except OSError:
        return False


CAPABILITY_HINT_BLOCK = """\
[Available Tool Packs]
These tool packs are hidden by default but can be activated when needed:
- web: Fetch web page content (when task involves URLs or web pages)
- toolsmith: Create and manage custom tools (when explicitly requested)
- workspace-maintenance: Move/copy/delete/glob workspace files (when organizing workspace)
- multi-agent: Lightweight role-based delegation (when collaboration is needed)
- job-followup: Inspect/wait/cancel running jobs (auto-activated when jobs exist)"""


class ToolsModule(Module):
    """Tools module: four-tier tool system + role-based permission management."""

    name = "tools"
    description = "Tool system: core tools + common toolbox + custom tool creation"

    def __init__(self):
        self.common_registry: ToolRegistry | None = None   # Globally shared built-in common tools
        self.agent_registry: ToolRegistry = ToolRegistry()  # Per-instance: meta-tools + custom tools
        self.agent_store: AgentToolManager | None = None
        self._is_admin: bool = False

    def on_attach(self, agent):
        """Initialization logic when module is attached to an Agent."""
        super().on_attach(agent)
        self._is_admin = bool(agent.persona and agent.persona.is_admin)

        # --- 1. Load built-in tools (globally shared: web_search + web_fetch in pack="web") ---
        import llamagent.modules.tools.builtin as builtin
        self.common_registry = global_registry

        # Initialize web search backend (auto-detect or from config)
        from llamagent.modules.tools.web import create_search_backend
        backend = create_search_backend(agent.config)
        if backend is not None:
            builtin.web_search._backend = backend

        # --- 1b. Create v1.5 internal services ---
        from llamagent.modules.tools.workspace import WorkspaceService
        from llamagent.modules.tools.project_sync import ProjectSyncService
        self.workspace_service = WorkspaceService(agent, workspace_id=agent.config.workspace_id)
        self.project_sync_service = ProjectSyncService(agent, self.workspace_service)

        # --- 2. Load admin-created common tools (from __common__.json into common_registry) ---
        try:
            common_store = AgentToolManager(
                storage_dir=agent.config.agent_tools_dir,
                persona_id=COMMON_STORE_ID,
            )
            for tool_info in common_store.list_tools():
                func = common_store.get_function(tool_info["name"])
                if func:
                    self.common_registry.register(
                        name=tool_info["name"], func=func,
                        description=tool_info["description"],
                        tier="common", safety_level=1,
                    )
        except Exception as e:
            print(f"[Tools] Failed to load common tools: {e}")

        # --- 3. Load role custom tools (per-instance) ---
        persona_id = agent.persona.persona_id if agent.persona else "default"
        try:
            self.agent_store = AgentToolManager(
                storage_dir=agent.config.agent_tools_dir,
                persona_id=persona_id,
            )
            for tool_info in self.agent_store.list_tools():
                func = self.agent_store.get_function(tool_info["name"])
                if func:
                    self.agent_registry.register(
                        name=tool_info["name"], func=func,
                        description=tool_info["description"],
                        tier="agent", safety_level=1,
                        creator_id=persona_id,
                    )
        except Exception as e:
            print(f"[Tools] Failed to load role custom tools: {e}")

        # --- 4. Register meta-tools (per-instance, by role) ---
        self._register_meta_tools()

        # --- 4b. Register v1.5 workspace + project sync tools ---
        self._register_workspace_tools()
        self._register_project_sync_tools()

        # --- 5. Bridge all tools to agent._tools (core visibility) ---
        self._bridge_to_core()


    # ============================================================
    # Bridge internal registries → agent._tools (core visibility)
    # ============================================================

    def _bridge_to_core(self):
        """Sync tools from internal registries to agent._tools so the LLM can see and call them."""
        for _name, info in self.common_registry._tools.items():
            if _name in self.agent._tools:
                continue  # Don't overwrite tools registered by other modules
            self.agent.register_tool(
                name=info.name, func=info.func, description=info.description,
                parameters=info.parameters, tier=info.tier,
                safety_level=info.safety_level,
                path_extractor=None,  # web_search/web_fetch have no path extractors
                pack=info.pack,
            )
        for _name, info in self.agent_registry._tools.items():
            if _name in self.agent._tools:
                continue
            self.agent.register_tool(
                name=info.name, func=info.func, description=info.description,
                parameters=info.parameters, tier=info.tier,
                safety_level=info.safety_level,
                creator_id=info.creator_id,
                pack=info.pack,
            )

    # ============================================================
    # Pipeline Callbacks
    # ============================================================

    def on_input(self, user_input: str) -> str:
        """Reset pack state at the start of each turn."""
        self.agent._active_packs.clear()
        return user_input

    def on_context(self, query: str, context: str) -> str:
        """Evaluate state-driven packs and inject guidelines into LLM context."""
        # Step 2: evaluate state-driven packs
        self._evaluate_state_packs()
        # Inject workspace guide + capability hints
        guide = WORKSPACE_GUIDE + "\n\n" + CAPABILITY_HINT_BLOCK
        return f"{context}\n\n{guide}" if context else guide

    def _evaluate_state_packs(self):
        """Activate packs based on current agent state."""
        job_mod = self.agent.get_module("job")
        if job_mod and getattr(job_mod, "service", None) and job_mod.service.list_jobs():
            self.agent._active_packs.add("job-followup")

    def on_shutdown(self) -> None:
        """Clean up workspace session directory on agent shutdown."""
        if hasattr(self, "workspace_service") and self.workspace_service:
            self.workspace_service.cleanup()

    # ============================================================
    # v1.5 workspace tool registration
    # ============================================================

    def _register_workspace_tools(self):
        """Register workspace exploration and modification tools."""
        ws = self.workspace_service

        # --- Exploration tools (safety_level=1) ---

        def _list_tree(root: str = ".", max_depth: int = 3) -> str:
            resolved = ws.resolve_path(root)
            lines = []
            for dirpath, dirnames, filenames in os.walk(resolved):
                depth = dirpath.replace(resolved, "").count(os.sep)
                if depth >= max_depth:
                    dirnames.clear()
                indent = "  " * depth
                lines.append(f"{indent}{os.path.basename(dirpath)}/")
                sub_indent = "  " * (depth + 1)
                for fn in sorted(filenames):
                    lines.append(f"{sub_indent}{fn}")
            return json.dumps({"status": "success", "tree": "\n".join(lines)}, ensure_ascii=False)

        self.agent.register_tool(
            name="list_tree", func=_list_tree,
            description="List directory tree structure. Paths relative to workspace; use 'project:' prefix for project files.",
            parameters={"type": "object", "properties": {
                "root": {"type": "string", "description": "Root directory (default: workspace root). Use 'project:' prefix for project files.", "default": "."},
                "max_depth": {"type": "integer", "description": "Maximum depth (default: 3)", "default": 3},
            }},
            tier="common", safety_level=1,
            path_extractor=lambda args: [ws.resolve_path(args.get("root", "."))],
        )

        def _glob_files(pattern: str, root: str = ".") -> str:
            resolved = ws.resolve_path(root)
            matches = glob.glob(os.path.join(resolved, pattern), recursive=True)
            rel = [os.path.relpath(m, resolved) for m in matches]
            return json.dumps({"status": "success", "files": sorted(rel), "count": len(rel)}, ensure_ascii=False)

        self.agent.register_tool(
            name="glob_files", func=_glob_files,
            description="Search files by glob pattern. Paths relative to workspace; use 'project:' prefix for project files.",
            parameters={"type": "object", "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py')"},
                "root": {"type": "string", "description": "Root directory (default: workspace root)", "default": "."},
            }, "required": ["pattern"]},
            tier="common", safety_level=1,
            pack="workspace-maintenance",
            path_extractor=lambda args: [ws.resolve_path(args.get("root", "."))],
        )

        def _search_text(query: str, paths: list = None, regex: bool = False, case_sensitive: bool = False) -> str:
            import re as _re
            search_root = ws.workspace_root
            target_files = []
            if paths:
                target_files = [ws.resolve_path(p) for p in paths]
            else:
                for dirpath, _, filenames in os.walk(search_root):
                    for fn in filenames:
                        target_files.append(os.path.join(dirpath, fn))

            results = []
            flags = 0 if case_sensitive else _re.IGNORECASE
            for fpath in target_files:
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            matched = (_re.search(query, line, flags) if regex
                                       else (query in line if case_sensitive else query.lower() in line.lower()))
                            if matched:
                                results.append({"file": os.path.relpath(fpath, search_root), "line": i, "content": line.rstrip()[:200]})
                except (OSError, UnicodeDecodeError):
                    continue
            return json.dumps({"status": "success", "matches": results[:100], "total": len(results)}, ensure_ascii=False)

        self.agent.register_tool(
            name="search_text", func=_search_text,
            description="Search files for text content. Searches workspace by default; pass 'project:' prefixed paths to search project files.",
            parameters={"type": "object", "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Paths to search (default: entire workspace)"},
                "regex": {"type": "boolean", "description": "Use regex matching", "default": False},
                "case_sensitive": {"type": "boolean", "description": "Case sensitive search", "default": False},
            }, "required": ["query"]},
            tier="common", safety_level=1,
            path_extractor=lambda args: ws.resolve_paths(args["paths"]) if args.get("paths") else [],
        )

        def _read_files(paths: list, ranges: dict = None, with_line_numbers: bool = True, mode: str = "auto") -> str:
            """Read files with automatic text/binary detection.

            mode: "auto" (detect by extension/content), "text" (force text), "binary" (return base64).
            """
            budget = getattr(self.agent.config, "max_observation_tokens", 2000)
            per_file = max(200, budget // max(len(paths), 1))
            results = []
            for p in paths:
                resolved = ws.resolve_path(p)
                try:
                    # Determine if file is text or binary
                    force_binary = (mode == "binary")
                    is_text = (mode == "text") or (not force_binary and _is_text_file(resolved))

                    if force_binary:
                        # Binary mode: return metadata + base64 content
                        st = os.stat(resolved)
                        max_binary_size = 50 * 1024 * 1024  # 50 MB
                        if st.st_size > max_binary_size:
                            results.append({
                                "path": p, "error": f"File too large for binary read ({st.st_size} bytes, limit {max_binary_size})",
                            })
                            continue
                        mime_type = mimetypes.guess_type(resolved)[0] or "application/octet-stream"
                        with open(resolved, "rb") as f:
                            raw = f.read()
                        results.append({
                            "path": p, "binary": True,
                            "size": st.st_size, "mime": mime_type,
                            "content_base64": base64.b64encode(raw).decode("ascii"),
                        })
                    elif not is_text:
                        # Binary file in auto mode: return metadata only (no content dump)
                        st = os.stat(resolved)
                        mime_type = mimetypes.guess_type(resolved)[0] or "application/octet-stream"
                        results.append({
                            "path": p, "binary": True,
                            "size": st.st_size, "mime": mime_type,
                        })
                    else:
                        # Text file: read with line numbers and truncation
                        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                            all_lines = f.readlines()

                        # Check if a specific range is requested for this file
                        range_spec = (ranges or {}).get(p)
                        if range_spec:
                            parts = range_spec.split("-", 1)
                            start = int(parts[0])
                            end = int(parts[1]) if len(parts) > 1 else len(all_lines)
                            selected = all_lines[max(0, start - 1):end]
                            content_lines = []
                            for i, line in enumerate(selected, start):
                                prefix = f"{i:>4}\t" if with_line_numbers else ""
                                content_lines.append(f"{prefix}{line.rstrip()}")
                            results.append({
                                "path": p, "content": "\n".join(content_lines),
                                "lines": len(all_lines), "range": range_spec,
                                "truncated": False,
                            })
                        else:
                            content_lines = []
                            char_count = 0
                            truncated = False
                            for i, line in enumerate(all_lines, 1):
                                if char_count + len(line) > per_file:
                                    truncated = True
                                    break
                                prefix = f"{i:>4}\t" if with_line_numbers else ""
                                content_lines.append(f"{prefix}{line.rstrip()}")
                                char_count += len(line)
                            results.append({
                                "path": p, "content": "\n".join(content_lines),
                                "lines": len(all_lines), "truncated": truncated,
                            })
                except Exception as e:
                    results.append({"path": p, "error": str(e)})
            return json.dumps({"status": "success", "files": results}, ensure_ascii=False)

        self.agent.register_tool(
            name="read_files", func=_read_files,
            description=(
                "Read one or more files. Text files return content with line numbers; "
                "binary files return metadata (size, mime type). "
                "Paths relative to workspace; use 'project:' prefix for project files. "
                "Use 'ranges' to read specific line ranges. "
                "Use mode='binary' to get base64 content of binary files."
            ),
            parameters={"type": "object", "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "description": "File paths to read"},
                "ranges": {"type": "object", "description": "Optional mapping of file path to 'start-end' line range string (e.g., {'main.py': '10-50'})"},
                "with_line_numbers": {"type": "boolean", "description": "Include line numbers for text files", "default": True},
                "mode": {"type": "string", "description": "Read mode: 'auto' (detect type), 'text' (force text), 'binary' (return base64)", "default": "auto"},
            }, "required": ["paths"]},
            tier="common", safety_level=1,
            path_extractor=lambda args: ws.resolve_paths(args.get("paths", [])),
        )

        def _stat_paths(paths: list) -> str:
            results = []
            for p in paths:
                resolved = ws.resolve_path(p)
                try:
                    st = os.stat(resolved)
                    results.append({
                        "path": p, "size": st.st_size,
                        "mtime": st.st_mtime, "type": "dir" if os.path.isdir(resolved) else "file",
                    })
                except Exception as e:
                    results.append({"path": p, "error": str(e)})
            return json.dumps({"status": "success", "stats": results}, ensure_ascii=False)

        self.agent.register_tool(
            name="stat_paths", func=_stat_paths,
            description="Get file/directory metadata (size, modification time, type).",
            parameters={"type": "object", "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Paths to stat"},
            }, "required": ["paths"]},
            tier="common", safety_level=1,
            pack="workspace-maintenance",
            path_extractor=lambda args: ws.resolve_paths(args.get("paths", [])),
        )

        # --- Modification tools (safety_level=2) ---

        def _write_files(files: dict, mode: str = "text") -> str:
            """Write files to workspace. mode='text' (default) or 'binary' (base64-encoded content)."""
            written = []
            errors = []
            for path, content in files.items():
                try:
                    resolved = ws.resolve_path_workspace_only(path)
                except ValueError as e:
                    errors.append({"path": path, "error": str(e)})
                    continue
                try:
                    os.makedirs(os.path.dirname(resolved), exist_ok=True)
                    if mode == "binary":
                        raw = base64.b64decode(content)
                        with open(resolved, "wb") as f:
                            f.write(raw)
                    else:
                        with open(resolved, "w", encoding="utf-8") as f:
                            f.write(content)
                    written.append(path)
                except Exception as e:
                    errors.append({"path": path, "error": str(e)})
            return json.dumps({"status": "success" if not errors else "partial", "written": written, "errors": errors}, ensure_ascii=False)

        self.agent.register_tool(
            name="write_files", func=_write_files,
            description=(
                "Write one or more files to workspace. Keys are file paths, values are content strings. "
                "Use mode='binary' to write base64-encoded binary content."
            ),
            parameters={"type": "object", "properties": {
                "files": {"type": "object", "description": "Mapping of file path -> content string (or base64 for binary mode)"},
                "mode": {"type": "string", "description": "Write mode: 'text' (default) or 'binary' (base64-encoded)", "default": "text"},
            }, "required": ["files"]},
            tier="common", safety_level=2,
            path_extractor=lambda args: ws.resolve_paths(list(args.get("files", {}).keys())),
        )

        def _create_temp_file(prefix: str = "", suffix: str = "", content: str = "") -> str:
            ws_root = ws.workspace_root
            fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=ws_root)
            # os.fdopen takes ownership of fd; do NOT os.close(fd) after fdopen succeeds
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            return json.dumps({"status": "success", "path": os.path.relpath(path, ws_root)}, ensure_ascii=False)

        self.agent.register_tool(
            name="create_temp_file", func=_create_temp_file,
            description="Create a temporary file in the workspace.",
            parameters={"type": "object", "properties": {
                "prefix": {"type": "string", "description": "Filename prefix", "default": ""},
                "suffix": {"type": "string", "description": "Filename suffix", "default": ""},
                "content": {"type": "string", "description": "File content", "default": ""},
            }},
            tier="common", safety_level=1,
            pack="workspace-maintenance",
        )

        def _move_path(src: str, dst: str) -> str:
            try:
                resolved_src = ws.resolve_path_workspace_only(src)
                resolved_dst = ws.resolve_path_workspace_only(dst)
            except ValueError as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
            try:
                shutil.move(resolved_src, resolved_dst)
                return json.dumps({"status": "success", "src": src, "dst": dst}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

        self.agent.register_tool(
            name="move_path", func=_move_path,
            description="Move a file or directory within workspace.",
            parameters={"type": "object", "properties": {
                "src": {"type": "string", "description": "Source path"},
                "dst": {"type": "string", "description": "Destination path"},
            }, "required": ["src", "dst"]},
            tier="common", safety_level=2,
            pack="workspace-maintenance",
            path_extractor=lambda args: [ws.resolve_path(args.get("src", "")), ws.resolve_path(args.get("dst", ""))],
        )

        def _copy_path(src: str, dst: str) -> str:
            try:
                resolved_src = ws.resolve_path_workspace_only(src)
                resolved_dst = ws.resolve_path_workspace_only(dst)
            except ValueError as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
            try:
                if os.path.isdir(resolved_src):
                    shutil.copytree(resolved_src, resolved_dst)
                else:
                    os.makedirs(os.path.dirname(resolved_dst), exist_ok=True)
                    shutil.copy2(resolved_src, resolved_dst)
                return json.dumps({"status": "success", "src": src, "dst": dst}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

        self.agent.register_tool(
            name="copy_path", func=_copy_path,
            description="Copy a file or directory within workspace.",
            parameters={"type": "object", "properties": {
                "src": {"type": "string", "description": "Source path"},
                "dst": {"type": "string", "description": "Destination path"},
            }, "required": ["src", "dst"]},
            tier="common", safety_level=2,
            pack="workspace-maintenance",
            path_extractor=lambda args: [ws.resolve_path(args.get("src", "")), ws.resolve_path(args.get("dst", ""))],
        )

        def _delete_path(path: str) -> str:
            try:
                resolved = ws.resolve_path_workspace_only(path)
            except ValueError as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
            try:
                if os.path.isdir(resolved):
                    shutil.rmtree(resolved)
                elif os.path.exists(resolved):
                    os.remove(resolved)
                else:
                    return json.dumps({"status": "error", "error": f"Path not found: {path}"}, ensure_ascii=False)
                return json.dumps({"status": "success", "path": path}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

        self.agent.register_tool(
            name="delete_path", func=_delete_path,
            description="Delete a file or directory in workspace.",
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "Path to delete"},
            }, "required": ["path"]},
            tier="common", safety_level=2,
            pack="workspace-maintenance",
            path_extractor=lambda args: [ws.resolve_path(args.get("path", ""))],
        )

    # ============================================================
    # v1.5 project sync tool registration
    # ============================================================

    def _register_project_sync_tools(self):
        """Register project sync tools (apply_patch, sync, revert)."""
        ps = self.project_sync_service

        def _apply_patch(target: str, edits: list, preview: bool = False) -> str:
            if preview:
                result = ps.preview_patch(target, edits)
            else:
                result = ps.apply_patch(target, edits)
            return json.dumps(result, ensure_ascii=False)

        self.agent.register_tool(
            name="apply_patch", func=_apply_patch,
            description="Apply structured search/replace edits to a project file. Atomic: all edits succeed or none are written. Target path is relative to project root. Set preview=true to validate edits without writing.",
            parameters={"type": "object", "properties": {
                "target": {"type": "string", "description": "Target file path relative to project root"},
                "edits": {"type": "array", "items": {"type": "object", "properties": {
                    "match": {"type": "string", "description": "Exact text to find"},
                    "replace": {"type": "string", "description": "Replacement text"},
                    "expected_count": {"type": "integer", "description": "Expected match count (default: 1)", "default": 1},
                }, "required": ["match", "replace"]}, "description": "List of search/replace operations"},
                "preview": {"type": "boolean", "description": "If true, validate edits without writing to disk", "default": False},
            }, "required": ["target", "edits"]},
            tier="common", safety_level=2,
            path_extractor=lambda args: [ps.resolve_project_path(args.get("target", ""))],
        )

        def _sync_workspace_to_project(paths: list = None, mode: str = "auto") -> str:
            result = ps.sync_workspace_to_project(paths, mode)
            return json.dumps(result, ensure_ascii=False)

        self.agent.register_tool(
            name="sync_workspace_to_project", func=_sync_workspace_to_project,
            description="Sync workspace files to project. mode='auto' (default): copy new files, replace changed files. mode='copy': copy only, fail if exists. mode='patch': replace only existing files.",
            parameters={"type": "object", "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Workspace-relative paths to sync (default: all changed files)"},
                "mode": {"type": "string", "description": "Sync mode: 'auto' (default), 'copy', 'patch'", "default": "auto"},
            }},
            tier="common", safety_level=2,
            path_extractor=lambda args: [
                ps.resolve_project_path(p) for p in (args.get("paths") or [])
            ] or [self.agent.project_dir],  # Sentinel: zone-check project_dir when paths=None
        )

        def _revert_changes(targets: list = None) -> str:
            result = ps.revert_changes(targets)
            return json.dumps(result, ensure_ascii=False)

        self.agent.register_tool(
            name="revert_changes", func=_revert_changes,
            description="Revert the most recent project file change. Pass file paths to revert specific files, or omit for the most recent global changeset.",
            parameters={"type": "object", "properties": {
                "targets": {"type": "array", "items": {"type": "string"}, "description": "File paths to revert (relative to project root). Omit to revert most recent changeset."},
            }},
            tier="common", safety_level=2,
            path_extractor=lambda args: [
                ps.resolve_project_path(t) for t in (args.get("targets") or [])
            ] or [self.agent.project_dir],  # Sentinel: zone-check project_dir when targets=None
        )

    # ============================================================
    # Meta-tool registration (all go to agent_registry to avoid global overwrite)
    # ============================================================

    def _register_meta_tools(self):
        """Register meta-tools: shared across all roles + admin-only."""

        # --- Shared across all roles (default tier) ---
        self.agent_registry.register(
            name="create_tool",
            func=self._tool_create,
            description="Create a custom tool. Provide the tool name, description, and Python function code.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Tool function name (in English)"},
                    "description": {"type": "string", "description": "Functional description"},
                    "code": {"type": "string", "description": "Python function code; the function name must match the name parameter"},
                },
                "required": ["name", "description", "code"],
            },
            tier="default",
            safety_level=2,
            pack="toolsmith",
        )

        self.agent_registry.register(
            name="list_my_tools",
            func=self._tool_list_my_tools,
            description="View the list of custom tools you have created.",
            parameters={"type": "object", "properties": {}},
            tier="default",
            safety_level=1,
            pack="toolsmith",
        )

        self.agent_registry.register(
            name="delete_tool",
            func=self._tool_delete,
            description="Delete one of your custom tools.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Tool name"}},
                "required": ["name"],
            },
            tier="default",
            safety_level=2,
            pack="toolsmith",
        )

        self.agent_registry.register(
            name="query_toolbox",
            func=self._tool_query_toolbox,
            description="Query the list of common tools in the toolbox.",
            parameters={"type": "object", "properties": {}},
            tier="default",
            safety_level=1,
            pack="toolsmith",
        )

        # --- Admin-only (admin tier) ---
        if self._is_admin:
            self.agent_registry.register(
                name="create_common_tool",
                func=self._tool_create_common,
                description="[Admin] Create a common tool available to all llamas.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Tool function name"},
                        "description": {"type": "string", "description": "Functional description"},
                        "code": {"type": "string", "description": "Python function code"},
                    },
                    "required": ["name", "description", "code"],
                },
                tier="admin",
                safety_level=2,
                pack="toolsmith",
            )

            self.agent_registry.register(
                name="list_all_agent_tools",
                func=self._tool_list_all_agent_tools,
                description="[Admin] View custom tools created by all llamas.",
                parameters={"type": "object", "properties": {}},
                tier="admin",
                safety_level=1,
                pack="toolsmith",
            )

            self.agent_registry.register(
                name="promote_tool",
                func=self._tool_promote,
                description="[Admin] Promote a llama's custom tool to a common tool.",
                parameters={
                    "type": "object",
                    "properties": {
                        "persona_id": {"type": "string", "description": "Llama ID"},
                        "tool_name": {"type": "string", "description": "Tool name"},
                    },
                    "required": ["persona_id", "tool_name"],
                },
                tier="admin",
                safety_level=2,
                pack="toolsmith",
            )

    # ============================================================
    # Tool implementations: shared across all roles
    # ============================================================

    def _tool_create(self, name: str, description: str, code: str) -> str:
        """Create a role custom tool."""
        if self.agent_store is None:
            return "Tool storage not initialized, cannot create tool."

        try:
            # If the safety module is available, scan the code to determine safety_level
            safety_level = 1
            safety_mod = self.agent.get_module("safety")
            if safety_mod and hasattr(safety_mod, "guard"):
                safety_level = safety_mod.guard.scan_code(code)

            self.agent_store.create(name, description, code)
            func = self.agent_store.get_function(name)
            if func:
                persona_id = self.agent.persona.persona_id if self.agent.persona else "default"
                self.agent_registry.register(
                    name=name, func=func, description=description,
                    tier="agent", safety_level=safety_level,
                    creator_id=persona_id,
                )
                # Bridge to core so LLM can see and call the new tool
                self.agent.register_tool(
                    name=name, func=func, description=description,
                    tier="agent", safety_level=safety_level,
                    creator_id=persona_id,
                )
            return f"Tool '{name}' created successfully! (safety level: {safety_level})"
        except ValueError as e:
            return f"Creation failed: {e}"
        except Exception as e:
            return f"Unexpected error while creating tool: {e}"

    def _tool_list_my_tools(self) -> str:
        """View the list of custom tools created by the current role."""
        if self.agent_store is None:
            return "Tool storage not initialized."

        tools = self.agent_store.list_tools()
        if not tools:
            return "You haven't created any custom tools yet."
        lines = ["Your custom tools:"]
        for t in tools:
            lines.append(f"- {t['name']}: {t['description']}")
        return "\n".join(lines)

    def _tool_delete(self, name: str) -> str:
        """Delete a custom tool of the current role."""
        if self.agent_store is None:
            return "Tool storage not initialized."

        if self.agent_store.delete(name):
            self.agent_registry.remove(name)
            self.agent.remove_tool(name)  # Remove from core
            return f"Tool '{name}' has been deleted."
        return f"No custom tool named '{name}' found."

    def _tool_query_toolbox(self) -> str:
        """View the common toolbox (common-tier tools only)."""
        if self.common_registry is None:
            return "Tool registry not initialized."

        common = self.common_registry.get_by_tier("common")
        if not common:
            return "No common tools in the toolbox yet."
        lines = ["Common tools in the toolbox:"]
        for name, info in common.items():
            lines.append(f"- {name}: {info.description}")
        return "\n".join(lines)

    # ============================================================
    # Tool implementations: admin-only
    # ============================================================

    def _tool_create_common(self, name: str, description: str, code: str) -> str:
        """Create a common tool, registered to global_registry, available to all roles."""
        try:
            common_store = AgentToolManager(
                storage_dir=self.agent.config.agent_tools_dir,
                persona_id=COMMON_STORE_ID,
            )

            # Scan the code to determine safety_level
            safety_level = 1
            safety_mod = self.agent.get_module("safety")
            if safety_mod and hasattr(safety_mod, "guard"):
                safety_level = safety_mod.guard.scan_code(code)

            common_store.create(name, description, code)
            func = common_store.get_function(name)
            if func:
                self.common_registry.register(
                    name=name, func=func, description=description,
                    tier="common", safety_level=safety_level,
                )
                # Bridge to core
                self.agent.register_tool(
                    name=name, func=func, description=description,
                    tier="common", safety_level=safety_level,
                )
            return f"Common tool '{name}' created successfully! Available to all llamas. (safety level: {safety_level})"
        except ValueError as e:
            return f"Creation failed: {e}"
        except Exception as e:
            return f"Unexpected error while creating common tool: {e}"

    def _tool_list_all_agent_tools(self) -> str:
        """View custom tools created by all roles, grouped by persona_id."""
        try:
            all_tools = AgentToolManager.scan_all(self.agent.config.agent_tools_dir)
        except Exception as e:
            return f"Failed to scan tools directory: {e}"

        if not all_tools:
            return "No llamas have created any custom tools yet."
        lines = ["Custom tools from all llamas:"]
        for pid, tools in all_tools.items():
            lines.append(f"\n[{pid}]")
            for t in tools:
                lines.append(f"  - {t['name']}: {t['description']}")
        return "\n".join(lines)

    def _tool_promote(self, persona_id: str, tool_name: str) -> str:
        """Promote a specified role's custom tool to a common tool."""
        try:
            source_store = AgentToolManager(
                storage_dir=self.agent.config.agent_tools_dir,
                persona_id=persona_id,
            )
        except Exception as e:
            return f"Failed to load tool storage for role [{persona_id}]: {e}"

        tool_def = source_store.export(tool_name)
        if not tool_def:
            return f"Tool '{tool_name}' not found in [{persona_id}]."

        try:
            common_store = AgentToolManager(
                storage_dir=self.agent.config.agent_tools_dir,
                persona_id=COMMON_STORE_ID,
            )

            # Scan the code to determine safety_level
            safety_level = 1
            safety_mod = self.agent.get_module("safety")
            if safety_mod and hasattr(safety_mod, "guard"):
                safety_level = safety_mod.guard.scan_code(tool_def.get("code", ""))

            common_store.create(
                name=tool_def["name"],
                description=tool_def["description"],
                code=tool_def["code"],
                parameters=tool_def.get("parameters"),
            )
            func = common_store.get_function(tool_def["name"])
            if func:
                self.common_registry.register(
                    name=tool_def["name"], func=func,
                    description=tool_def["description"],
                    tier="common", safety_level=safety_level,
                )
                # Bridge to core
                self.agent.register_tool(
                    name=tool_def["name"], func=func,
                    description=tool_def["description"],
                    tier="common", safety_level=safety_level,
                )
            return f"Tool '{tool_name}' has been promoted from [{persona_id}] to a common tool! (safety level: {safety_level})"
        except ValueError as e:
            return f"Promotion failed: {e}"
        except Exception as e:
            return f"Unexpected error while promoting tool: {e}"
