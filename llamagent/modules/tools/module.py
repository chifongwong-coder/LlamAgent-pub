"""
ToolsModule: four-tier tool system + role-based permission management + v3.3 file tool surface.

Responsibilities:
- Tool registration management: maintains global_registry and
  agent_registry; exposes the four-tier visibility model (default /
  common / admin / agent).
- Meta-tools (toolsmith pack): create_tool / list_my_tools /
  delete_tool, plus admin-only create_common_tool and promote_tool.
- Tool persistence: JSON persistence for role custom tools and
  admin common tools.
- v3.3 file tool surface (5 model-facing core tools + 7 path-fallback
  pack tools):
    * Default: read_files, write_files, apply_patch, list_tree,
      revert_changes — paths resolve relative to project_dir; the
      framework's classify_write helper routes each write to
      playground (no changeset), project (changeset-tracked), or
      rejected (outside writable root). No `zone` parameter.
    * Pack `path-fallback` (auto-activated when neither `command`
      nor `start_job` is registered): rename_path, move_path,
      copy_path, delete_path, glob_files, search_text, stat_paths,
      create_temp_file. rename_path renames in-place (basename only);
      move_path crosses directories (same-parent rejected). Destructive
      3 (move/copy/delete) register changesets when targeting write_root;
      delete_path rejects directories and binary files. Other packs
      (web / toolsmith / multi-agent / job-followup) follow the same
      pack mechanism.
- FILE_TOOL_GUIDE + CAPABILITY_HINT_BLOCK injection via on_context.
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
from llamagent.modules.tools.scratch import _resolve_within, _resolve_path, classify_write

logger = logging.getLogger(__name__)

# Storage ID for admin-created common tools
COMMON_STORE_ID = "__common__"

# File-tool behavioral guidelines injected via on_context (v3.3 surface).
# This block is the model's primary reference for file-tool semantics; mock
# tests in tests/test_workspace.py assert exact substrings — change carefully.
FILE_TOOL_GUIDE = """\
You can read and write files in the project directory using:
  - read_files(paths)         # read one or more files (relative to project root)
  - write_files(files)        # create or overwrite (relative to project root)
  - apply_patch(target, edits) # surgical match/replace edits
  - list_tree(root)           # browse project structure
  - revert_changes(targets)   # undo recent typed writes (within the project)

File management (path-fallback pack):
  - rename_path(target, new_name)  # rename in-place; new_name must be a filename, not a path
  - move_path(src, dst)            # move across directories (same-parent rejected — use rename_path)
  - copy_path(src, dst)            # copy file or directory
  - delete_path(path)              # delete a single file (directories rejected)

Path conventions:
  - All paths are relative to the project root unless absolute. No prefixes.
  - Some async tools persist large outputs under `llama_playground/...`. When you
    see a path like that in a tool result, read it back with read_files just like
    any other file — it's a real file under the project tree.
  - Symbolic links resolving outside the project root are rejected at write time.

Reading large files:
  read_files truncates per-call to fit context. If a file is too long, the
  result tells you which lines were returned and how to fetch more, e.g.:
    read_files(['big.py'], ranges={'big.py': '200-400'})    # next chunk
    read_files(['llama_playground/tool_results/x.txt'],
               ranges={'llama_playground/tool_results/x.txt': '15000-30000'})  # jump to a section

revert_changes undoes recent writes in reverse order."""

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
- multi-agent: Lightweight role-based delegation (when collaboration is needed)
- job-followup: Inspect/wait/cancel running jobs (auto-activated when jobs exist)
- path-fallback: rename/move/copy/delete/glob/stat files (auto-activated when no shell tool is available)"""


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

        # Initialize user interaction handler (injected by caller)
        if getattr(agent, "interaction_handler", None) is not None:
            builtin.ask_user._handler = agent.interaction_handler

        # --- 1b. Create v1.5 internal services ---
        from llamagent.modules.tools.scratch import ScratchService
        from llamagent.modules.tools.project_sync import ProjectSyncService
        self.scratch_service = ScratchService(agent, scratch_id=agent.config.scratch_id)
        self.project_sync_service = ProjectSyncService(agent, self.scratch_service)

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

        # --- 4b. Register v1.5 file + project sync tools ---
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
                action=getattr(info, "action", None),
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
                action=getattr(info, "action", None),
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
        # Inject file-tool guide + capability hints
        guide = FILE_TOOL_GUIDE + "\n\n" + CAPABILITY_HINT_BLOCK
        return f"{context}\n\n{guide}" if context else guide

    def _evaluate_state_packs(self):
        """Activate packs based on current agent state."""
        job_mod = self.agent.get_module("job")
        if job_mod and getattr(job_mod, "service", None) and job_mod.service.list_jobs():
            self.agent._active_packs.add("job-followup")

        # path-fallback: if neither a registered shell tool (`command`) nor
        # the JobModule's `start_job` is available, expose the path-op
        # tools (rename/move/copy/delete/glob/stat/temp) so the model
        # still has a way to manage files. When a shell tool exists,
        # these stay hidden.
        if not self._shell_tool_available():
            self.agent._active_packs.add("path-fallback")

    def _shell_tool_available(self) -> bool:
        """Whether any general-purpose shell-execution tool is registered.

        Used to decide whether to expose the legacy path-op fallback pack.
        """
        for tool_name in ("command", "start_job"):
            if tool_name in self.agent._tools:
                return True
        return False

    def on_shutdown(self) -> None:
        """Clean up scratch session directory on agent shutdown."""
        if hasattr(self, "scratch_service") and self.scratch_service:
            self.scratch_service.cleanup()

    # ============================================================
    # v1.5 file tool registration
    # ============================================================

    def _register_workspace_tools(self):
        """Register file-tool surface (read/write/patch/list/revert + pack).

        v3.6: every tool registered here uses ``takes_agent=True`` — the
        dispatcher injects the calling agent at call time. Tool funcs
        receive ``agent`` as first positional arg, which shadows the
        closure-captured ``agent = self.agent`` inside their body. This
        rebinds path resolution / write-root / auth_engine to the agent
        actually invoking the tool (parent vs child) instead of the
        agent that was attached at module-registration time.
        """
        ws = self.scratch_service

        def _read_resolve(agent, raw_path: str) -> str:
            """Resolve a read path against the calling agent's project_dir."""
            return _resolve_path(raw_path, base=agent.project_dir)

        def _safe_extract_paths(agent, raw_paths: list[str]) -> list[str]:
            """path_extractor-safe wrapper: drop paths that fail to resolve.
            v3.6: receives ``agent`` from the path_extractor's (args, agent)
            signature so resolution uses the calling agent's project_dir."""
            out = []
            for p in raw_paths:
                if not isinstance(p, str):
                    continue
                try:
                    out.append(_resolve_path(p, base=agent.project_dir))
                except (ValueError, OSError):
                    continue
            return out

        def _writable_root_hint(agent, p: str) -> str:
            """v3.3 §3.5 — uniform error for paths outside writable boundary."""
            return (
                f"Path '{p}' is outside writable root "
                f"'{agent.write_root}'. Choose a path within "
                f"'{agent.write_root}'."
            )

        # --- Exploration tools (safety_level=1) ---

        def _list_tree(agent, root: str = ".", max_depth: int = 3) -> str:
            try:
                resolved = _read_resolve(agent, root)
            except ValueError as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
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
            description=(
                "List directory structure relative to the project root. "
                "Common build/cache dirs are filtered by default."
            ),
            parameters={"type": "object", "properties": {
                "root": {"type": "string", "description": "Root directory (relative to project)", "default": "."},
                "max_depth": {"type": "integer", "description": "Maximum depth (default: 3)", "default": 3},
            }},
            tier="common", safety_level=1,
            path_extractor=lambda args, agent: _safe_extract_paths(agent, [args.get("root", ".")]),
            takes_agent=True,
        )

        def _glob_files(agent, pattern: str, root: str = ".") -> str:
            try:
                resolved = _resolve_path(root, base=agent.project_dir)
            except (ValueError, OSError) as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
            matches = glob.glob(os.path.join(resolved, pattern), recursive=True)
            rel = [os.path.relpath(m, resolved) for m in matches]
            return json.dumps({"status": "success", "files": sorted(rel), "count": len(rel)}, ensure_ascii=False)

        self.agent.register_tool(
            name="glob_files", func=_glob_files,
            description="Search files by glob pattern relative to the project root.",
            parameters={"type": "object", "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py')"},
                "root": {"type": "string", "description": "Root directory relative to project (default: project root)", "default": "."},
            }, "required": ["pattern"]},
            tier="common", safety_level=1,
            pack="path-fallback",
            path_extractor=lambda args, agent: _safe_extract_paths(agent, [args.get("root", ".")]),
            takes_agent=True,
        )

        def _search_text(agent, query: str, paths: list = None, regex: bool = False,
                         case_sensitive: bool = False) -> str:
            import re as _re
            search_root = agent.project_dir
            target_files = []
            if paths:
                try:
                    target_files = [_resolve_path(p, base=search_root) for p in paths]
                except (ValueError, OSError) as e:
                    return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
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
            description="Search files for text content within the project root.",
            parameters={"type": "object", "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Paths to search (default: entire project)"},
                "regex": {"type": "boolean", "description": "Use regex matching", "default": False},
                "case_sensitive": {"type": "boolean", "description": "Case sensitive search", "default": False},
            }, "required": ["query"]},
            tier="common", safety_level=1,
            pack="path-fallback",  # v3.3: shell `command: grep -rn pattern .` is preferred
            path_extractor=lambda args, agent: _safe_extract_paths(agent, args.get("paths") or []),
            takes_agent=True,
        )

        def _read_files(agent, paths: list, ranges: dict = None, with_line_numbers: bool = True,
                        mode: str = "auto") -> str:
            """Read files with automatic text/binary detection.

            mode: "auto" (detect by extension/content), "text" (force text), "binary" (return base64).
            Paths resolve relative to the project root; absolute paths are kept.
            """
            budget = getattr(agent.config, "max_observation_tokens", 2000)
            per_file = max(200, budget // max(len(paths), 1))
            results = []
            for p in paths:
                try:
                    resolved = _resolve_path(p, base=agent.project_dir)
                except (ValueError, OSError) as e:
                    results.append({"path": p, "error": str(e)})
                    continue
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

                        # Check if a specific range is requested for this file.
                        # Schema is dict[str, "start-end"] line-range strings.
                        # Reject non-string values (e.g. tuple/list from a
                        # confused model) cleanly instead of crashing on .split.
                        range_spec = (ranges or {}).get(p)
                        if range_spec is not None and not isinstance(range_spec, str):
                            results.append({
                                "path": p,
                                "error": (
                                    f"ranges value must be a 'start-end' line-range "
                                    f"string (e.g. '10-50'); got {type(range_spec).__name__}"
                                ),
                            })
                            continue
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
                            last_line_shown = 0
                            for i, line in enumerate(all_lines, 1):
                                if char_count + len(line) > per_file:
                                    truncated = True
                                    break
                                prefix = f"{i:>4}\t" if with_line_numbers else ""
                                content_lines.append(f"{prefix}{line.rstrip()}")
                                char_count += len(line)
                                last_line_shown = i
                            entry = {
                                "path": p, "content": "\n".join(content_lines),
                                "lines": len(all_lines), "truncated": truncated,
                            }
                            if truncated:
                                # v3.3 contract B: cap suffix tells the model how
                                # to read more. Format is locked — see plan §3.4.
                                next_end = min(last_line_shown + max(per_file // 50, 50),
                                               len(all_lines))
                                entry["hint"] = (
                                    f"(content truncated at line {last_line_shown} "
                                    f"of {len(all_lines)}; use ranges="
                                    f"{{'{p}': '{last_line_shown + 1}-{next_end}'}} "
                                    f"to read more)"
                                )
                            results.append(entry)
                except FileNotFoundError:
                    # v3.3 §3.5: unified file-not-found wording with quoted user path.
                    results.append({"path": p, "error": f"file not found: '{p}'"})
                except Exception as e:
                    results.append({"path": p, "error": str(e)})
            return json.dumps({"status": "success", "files": results}, ensure_ascii=False)

        self.agent.register_tool(
            name="read_files", func=_read_files,
            description=(
                "Read one or more files relative to the project root. "
                "Supports byte/line ranges and binary mode."
            ),
            parameters={"type": "object", "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "description": "File paths to read"},
                "ranges": {"type": "object", "description": "Optional mapping of file path to 'start-end' line range string (e.g., {'main.py': '10-50'})"},
                "with_line_numbers": {"type": "boolean", "description": "Include line numbers for text files", "default": True},
                "mode": {"type": "string", "description": "Read mode: 'auto' (detect type), 'text' (force text), 'binary' (return base64)", "default": "auto"},
            }, "required": ["paths"]},
            tier="common", safety_level=1,
            path_extractor=lambda args, agent: _safe_extract_paths(agent, args.get("paths") or []),
            takes_agent=True,
        )

        def _stat_paths(agent, paths: list) -> str:
            results = []
            for p in paths:
                try:
                    resolved = _resolve_path(p, base=agent.project_dir)
                except (ValueError, OSError) as e:
                    results.append({"path": p, "error": str(e)})
                    continue
                try:
                    st = os.stat(resolved)
                    results.append({
                        "path": p, "size": st.st_size,
                        "mtime": st.st_mtime,
                        "type": "dir" if os.path.isdir(resolved) else "file",
                    })
                except Exception as e:
                    results.append({"path": p, "error": str(e)})
            return json.dumps({"status": "success", "stats": results}, ensure_ascii=False)

        self.agent.register_tool(
            name="stat_paths", func=_stat_paths,
            description="Get file/directory metadata (size, modification time, type). Paths relative to project root.",
            parameters={"type": "object", "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Paths to stat"},
            }, "required": ["paths"]},
            tier="common", safety_level=1,
            pack="path-fallback",
            truncatable=False,  # short metadata; bypass _truncate_observation
            path_extractor=lambda args, agent: _safe_extract_paths(agent, args.get("paths", [])),
            takes_agent=True,
        )

        # --- Modification tools (safety_level=2) ---

        def _write_files(agent, files: dict, mode: str = "text") -> str:
            """Write files relative to the project root.

            Path classification (via classify_write):
            - Inside playground (e.g. llama_playground/...): write succeeds,
              not tracked (ephemeral scratch).
            - Inside write_root but not playground: write succeeds and is
              tracked by changeset for revert.
            - Outside both: rejected with the §3.5 outside-writable-root error.
            """
            written = []
            errors = []
            for path, content in files.items():
                try:
                    resolved = _resolve_path(path, base=agent.project_dir)
                except (ValueError, OSError) as e:
                    errors.append({"path": path, "error": str(e)})
                    continue
                zone_class = classify_write(resolved, agent)
                if zone_class == "rejected":
                    errors.append({"path": path, "error": _writable_root_hint(agent, path)})
                    continue
                track_changeset = (zone_class == "project")
                # Capture pre-image for changeset if writing to project zone.
                # If the existing file isn't readable as UTF-8 text (i.e. it's
                # binary), refuse to overwrite it via text mode — the changeset
                # journal can't faithfully record the bytes, and a later revert
                # would silently DELETE the binary file (interpreting pre_image
                # =None as "did not exist before"). Surface a clear error so
                # the model retries with mode='binary' or apply_patch.
                pre_image = None
                had_prior = False
                if track_changeset and os.path.isfile(resolved):
                    if mode == "text":
                        try:
                            with open(resolved, "r", encoding="utf-8") as fp:
                                pre_image = fp.read()
                            had_prior = True
                        except UnicodeDecodeError:
                            errors.append({
                                "path": path,
                                "error": (
                                    f"Refusing to overwrite existing binary file "
                                    f"'{path}' with text content. The changeset "
                                    f"journal cannot record binary pre-images, and "
                                    f"reverting would delete the file. Use "
                                    f"mode='binary' to overwrite it as bytes, or "
                                    f"apply_patch if it's actually text in a "
                                    f"different encoding."
                                ),
                            })
                            continue
                        except OSError as e:
                            errors.append({"path": path, "error": str(e)})
                            continue
                    else:
                        # mode='binary': skip pre_image read entirely; revert
                        # will delete the new file on rollback. The user is
                        # accepting bytes-mode semantics by passing mode='binary'.
                        had_prior = False
                        pre_image = None
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
                    if track_changeset:
                        # File didn't exist before -> pre_image=None means
                        # revert deletes it. File existed -> revert restores
                        # the captured pre_image (or deletes if it was binary).
                        self.project_sync_service.record_write_changeset(
                            resolved,
                            pre_image if had_prior else None,
                        )
                except Exception as e:
                    errors.append({"path": path, "error": str(e)})
            response = {
                "status": "success" if not errors else "partial",
                "written": written,
                "errors": errors,
            }
            return json.dumps(response, ensure_ascii=False)

        self.agent.register_tool(
            name="write_files", func=_write_files,
            description=(
                "Create or overwrite files relative to the project root. Each "
                "successful write is tracked for revert. Use mode='binary' for "
                "non-text content."
            ),
            parameters={"type": "object", "properties": {
                "files": {"type": "object", "description": "Mapping of file path -> content string (or base64 for binary mode)"},
                "mode": {"type": "string", "description": "Write mode: 'text' (default) or 'binary' (base64-encoded)", "default": "text"},
            }, "required": ["files"]},
            tier="common", safety_level=2,
            path_extractor=lambda args, agent: _safe_extract_paths(agent, list((args.get("files") or {}).keys())),
            takes_agent=True,
        )

        # v3.3 commit-13b: path-fallback tools share the core 5's path
        # resolution. Destructive ones (move/copy/delete) register
        # changesets via ProjectSyncService when targeting write_root;
        # writes to playground are ephemeral (not tracked).

        def _create_temp_file(agent, prefix: str = "", suffix: str = "", content: str = "") -> str:
            # Always write under playground_dir (framework scratch), no
            # model-supplied path. Returns a path relative to project_dir
            # so the model can read it back via read_files.
            results_dir = agent.playground_dir
            os.makedirs(results_dir, exist_ok=True)
            fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=results_dir)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            rel = os.path.relpath(path, agent.project_dir)
            return json.dumps({"status": "success", "path": rel}, ensure_ascii=False)

        self.agent.register_tool(
            name="create_temp_file", func=_create_temp_file,
            description="Create a temporary file under llama_playground/. Returns its project-relative path.",
            parameters={"type": "object", "properties": {
                "prefix": {"type": "string", "description": "Filename prefix", "default": ""},
                "suffix": {"type": "string", "description": "Filename suffix", "default": ""},
                "content": {"type": "string", "description": "File content", "default": ""},
            }},
            tier="common", safety_level=1,
            pack="path-fallback",
            takes_agent=True,
        )

        def _rename_path(agent, target: str, new_name: str) -> str:
            # new_name must be a plain filename — no path separators allowed.
            if os.sep in new_name or "/" in new_name or "\\" in new_name:
                return json.dumps({
                    "status": "error",
                    "error": (
                        f"rename_path: new_name must be a filename, not a path "
                        f"(got '{new_name}'). Use move_path to move across directories."
                    ),
                }, ensure_ascii=False)
            try:
                rsrc = _resolve_path(target, base=agent.project_dir)
            except (ValueError, OSError) as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
            if classify_write(rsrc, agent) == "rejected":
                return json.dumps({
                    "status": "error",
                    "error": _writable_root_hint(agent, target),
                }, ensure_ascii=False)
            rdst = os.path.realpath(os.path.join(os.path.dirname(rsrc), new_name))
            # Post-realpath guard: new_name must not escape the source directory
            # (e.g. new_name=".." passes the separator check but shifts the dir).
            if os.path.dirname(rdst) != os.path.dirname(rsrc):
                return json.dumps({
                    "status": "error",
                    "error": (
                        f"rename_path: new_name '{new_name}' would move the file "
                        f"outside its directory. Use move_path to change directories."
                    ),
                }, ensure_ascii=False)
            try:
                shutil.move(rsrc, rdst)
                if classify_write(rdst, agent) == "project":
                    self.project_sync_service.record_move_changeset(rsrc, rdst)
                return json.dumps({"status": "success", "target": target, "new_name": new_name}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

        self.agent.register_tool(
            name="rename_path", func=_rename_path,
            description="Rename a file or directory in-place. new_name must be a filename (no slashes); use move_path to move across directories.",
            parameters={"type": "object", "properties": {
                "target": {"type": "string", "description": "Path to rename (relative to project root)"},
                "new_name": {"type": "string", "description": "New filename (basename only, no path separators)"},
            }, "required": ["target", "new_name"]},
            tier="common", safety_level=2,
            pack="path-fallback",
            path_extractor=lambda args, agent: _safe_extract_paths(agent, [args.get("target", "")]),
            takes_agent=True,
        )

        def _move_path(agent, src: str, dst: str) -> str:
            try:
                rsrc = _resolve_path(src, base=agent.project_dir)
                rdst = _resolve_path(dst, base=agent.project_dir)
            except (ValueError, OSError) as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
            # Same-parent rejection: use rename_path for in-place renames.
            if os.path.dirname(rsrc) == os.path.dirname(rdst):
                return json.dumps({
                    "status": "error",
                    "error": (
                        f"move_path: source and destination share the same parent directory. "
                        f"Use rename_path('{src}', '{os.path.basename(dst)}') to rename in-place."
                    ),
                }, ensure_ascii=False)
            for raw, resolved in (("src", rsrc), ("dst", rdst)):
                if classify_write(resolved, agent) == "rejected":
                    return json.dumps({
                        "status": "error",
                        "error": _writable_root_hint(agent, src if raw == "src" else dst),
                    }, ensure_ascii=False)
            try:
                shutil.move(rsrc, rdst)
                # Track changeset only when dst is in write_root (project zone).
                if classify_write(rdst, agent) == "project":
                    self.project_sync_service.record_move_changeset(rsrc, rdst)
                return json.dumps({"status": "success", "src": src, "dst": dst}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

        self.agent.register_tool(
            name="move_path", func=_move_path,
            description="Move a file or directory to a different directory. Rejects same-parent calls — use rename_path to rename in-place.",
            parameters={"type": "object", "properties": {
                "src": {"type": "string", "description": "Source path"},
                "dst": {"type": "string", "description": "Destination path (must be in a different directory than src)"},
            }, "required": ["src", "dst"]},
            tier="common", safety_level=2,
            pack="path-fallback",
            path_extractor=lambda args, agent: _safe_extract_paths(agent, [args.get("src", ""), args.get("dst", "")]),
            takes_agent=True,
        )

        def _copy_path(agent, src: str, dst: str) -> str:
            try:
                rsrc = _resolve_path(src, base=agent.project_dir)
                rdst = _resolve_path(dst, base=agent.project_dir)
            except (ValueError, OSError) as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
            for raw, resolved in (("src", rsrc), ("dst", rdst)):
                if classify_write(resolved, agent) == "rejected":
                    return json.dumps({
                        "status": "error",
                        "error": _writable_root_hint(agent, src if raw == "src" else dst),
                    }, ensure_ascii=False)
            try:
                if os.path.isdir(rsrc):
                    shutil.copytree(rsrc, rdst)
                else:
                    os.makedirs(os.path.dirname(rdst), exist_ok=True)
                    shutil.copy2(rsrc, rdst)
                if classify_write(rdst, agent) == "project":
                    self.project_sync_service.record_copy_changeset(rsrc, rdst)
                return json.dumps({"status": "success", "src": src, "dst": dst}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

        self.agent.register_tool(
            name="copy_path", func=_copy_path,
            description="Copy a file or directory within the project.",
            parameters={"type": "object", "properties": {
                "src": {"type": "string", "description": "Source path"},
                "dst": {"type": "string", "description": "Destination path"},
            }, "required": ["src", "dst"]},
            tier="common", safety_level=2,
            pack="path-fallback",
            path_extractor=lambda args, agent: _safe_extract_paths(agent, [args.get("src", ""), args.get("dst", "")]),
            takes_agent=True,
        )

        def _delete_path(agent, path: str) -> str:
            # v3.5: empty directories are now removable (revert via mkdir).
            # Non-empty directories are still rejected — Changeset can't
            # capture per-file pre_image bytes for revert. Model must
            # delete contents first, then the empty directory.
            try:
                resolved = _resolve_path(path, base=agent.project_dir)
            except (ValueError, OSError) as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
            if classify_write(resolved, agent) == "rejected":
                return json.dumps({
                    "status": "error",
                    "error": _writable_root_hint(agent, path),
                }, ensure_ascii=False)
            if os.path.isdir(resolved):
                # Empty directory: allowed (revert via mkdir).
                if not os.listdir(resolved):
                    track = (classify_write(resolved, agent) == "project")
                    cid = None
                    if track:
                        try:
                            cid = self.project_sync_service.record_rmdir_changeset(resolved)
                        except Exception as e:
                            logger.warning(
                                "delete_path: failed to record rmdir changeset for %s: %s",
                                resolved, e,
                            )
                    try:
                        os.rmdir(resolved)
                    except OSError as e:
                        return json.dumps({
                            "status": "error",
                            "error": f"Failed to remove empty directory '{path}': {e}",
                        }, ensure_ascii=False)
                    return json.dumps({
                        "status": "success",
                        "path": path,
                        "kind": "directory",
                        "changeset_id": cid,
                    }, ensure_ascii=False)
                # Non-empty: still rejected, with hint to delete contents first.
                return json.dumps({
                    "status": "error",
                    "error": (
                        f"Non-empty directory '{path}'. Delete its files first, "
                        f"then call delete_path on the empty dir."
                    ),
                }, ensure_ascii=False)
            if not os.path.exists(resolved):
                return json.dumps({
                    "status": "error",
                    "error": f"file not found: '{path}'",
                }, ensure_ascii=False)
            try:
                # Capture pre_image for changeset (project zone only).
                track = (classify_write(resolved, agent) == "project")
                pre_image = None
                if track:
                    try:
                        with open(resolved, "r", encoding="utf-8") as f:
                            pre_image = f.read()
                    except UnicodeDecodeError:
                        # Binary file: changeset can't faithfully record bytes
                        # in the string-based pre_image. Refuse to delete via
                        # this tool — the model should use the command tool
                        # for binary deletes (which won't be revertable anyway).
                        return json.dumps({
                            "status": "error",
                            "error": (
                                f"delete_path refuses to remove binary file "
                                f"'{path}' because the changeset journal cannot "
                                f"record its bytes for revert. Use the command "
                                f"tool if you accept the no-revert trade-off."
                            ),
                        }, ensure_ascii=False)
                os.remove(resolved)
                if track and pre_image is not None:
                    self.project_sync_service.record_delete_changeset(resolved, pre_image)
                return json.dumps({"status": "success", "path": path}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

        self.agent.register_tool(
            name="delete_path", func=_delete_path,
            description=(
                "Delete a file or empty directory. Non-empty dirs rejected — "
                "delete contents first, then the empty dir."
            ),
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "File or empty-directory path"},
            }, "required": ["path"]},
            tier="common", safety_level=2,
            pack="path-fallback",
            path_extractor=lambda args, agent: _safe_extract_paths(agent, [args.get("path", "")]),
            takes_agent=True,
        )

    # ============================================================
    # v1.5 project sync tool registration
    # ============================================================

    def _register_project_sync_tools(self):
        """Register project sync tools (apply_patch, revert).

        v3.6: tools registered here use ``takes_agent=True``; the helper
        ``_writable_root_hint`` takes ``agent`` so the formatted error
        message reports the calling agent's write_root, not the parent's.
        """
        ps = self.project_sync_service

        def _writable_root_hint(agent, p: str) -> str:
            return (
                f"Path '{p}' is outside writable root "
                f"'{agent.write_root}'. Choose a path within "
                f"'{agent.write_root}'."
            )

        def _apply_patch(agent, target: str, edits: list, preview: bool = False) -> str:
            try:
                resolved = _resolve_path(target, base=agent.project_dir)
            except (ValueError, OSError) as e:
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)
            zone_class = classify_write(resolved, agent)
            if zone_class == "rejected":
                return json.dumps({"status": "error",
                                   "error": _writable_root_hint(agent, target)},
                                  ensure_ascii=False)
            # Snapshot trigger lives in ProjectSyncService.apply_patch
            # (see project_sync.py::_maybe_snapshot). Tool wrapper just
            # forwards the resolved path + record_changeset flag.
            if preview:
                result = ps.preview_patch(resolved, edits)
            else:
                result = ps.apply_patch(
                    resolved, edits, record_changeset=(zone_class == "project"),
                )
            return json.dumps(result, ensure_ascii=False)

        def _safe_apply_patch_extract(args, agent):
            """Exception-safe path extraction for apply_patch (auth layer)."""
            try:
                return [_resolve_path(args.get("target", ""), base=agent.project_dir)]
            except (ValueError, OSError):
                return []

        self.agent.register_tool(
            name="apply_patch", func=_apply_patch,
            description=(
                "Apply surgical match/replace edits to a file relative to the "
                "project root. Each successful edit is tracked for revert. "
                "Prefer this over write_files for partial changes."
            ),
            parameters={"type": "object", "properties": {
                "target": {"type": "string", "description": "Target file path (relative to project)"},
                "edits": {"type": "array", "items": {"type": "object", "properties": {
                    "match": {"type": "string", "description": "Exact text to find"},
                    "replace": {"type": "string", "description": "Replacement text"},
                    "expected_count": {"type": "integer", "description": "Expected match count (default: 1)", "default": 1},
                }, "required": ["match", "replace"]}, "description": "List of search/replace operations"},
                "preview": {"type": "boolean", "description": "If true, validate edits without writing to disk", "default": False},
            }, "required": ["target", "edits"]},
            tier="common", safety_level=2,
            path_extractor=lambda args, agent: _safe_apply_patch_extract(args, agent),
            takes_agent=True,
        )

        def _revert_changes(agent, targets: list = None) -> str:
            if targets:
                resolved_targets = []
                for t in targets:
                    try:
                        resolved_targets.append(
                            _resolve_path(t, base=agent.project_dir))
                    except (ValueError, OSError) as e:
                        return json.dumps({"status": "error", "error": str(e)},
                                          ensure_ascii=False)
                # If a target resolves to playground, surface the §3.5
                # by-design error rather than the generic "no changeset"
                # noise — the model needs to learn this is intentional.
                for resolved, raw in zip(resolved_targets, targets):
                    if classify_write(resolved, agent) == "playground":
                        return json.dumps({
                            "status": "error",
                            "error": (
                                f"No changeset for '{raw}'. Writes under "
                                f"llama_playground/ are not tracked. "
                                f"This is by design; do not retry."
                            ),
                        }, ensure_ascii=False)
                result = ps.revert_changes(resolved_targets)
            else:
                result = ps.revert_changes(None)
            return json.dumps(result, ensure_ascii=False)

        self.agent.register_tool(
            name="revert_changes", func=_revert_changes,
            description="Undo recent typed writes by stack or by path.",
            parameters={"type": "object", "properties": {
                "targets": {"type": "array", "items": {"type": "string"}, "description": "File paths to revert (relative to project). Omit to revert most recent changeset."},
            }},
            tier="common", safety_level=2,
            path_extractor=lambda args, agent: _safe_revert_extract(args, agent),
            takes_agent=True,
        )

        def _safe_revert_extract(args, agent):
            """Exception-safe path extraction for revert_changes."""
            out = []
            for t in (args.get("targets") or []):
                try:
                    out.append(_resolve_path(t, base=agent.project_dir))
                except (ValueError, OSError):
                    continue
            return out or [agent.write_root]  # sentinel for targets=None

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
