You now have access to file-management tools that complement the core
read/write/patch surface.

Available tools (path-fallback pack):
- `glob_files`: find files by pattern (e.g., "**/*.py")
- `search_text`: search file contents by string or regex
- `move_path`: move a file
- `copy_path`: copy a file
- `delete_path`: delete a file (directories and binary files are refused)
- `create_temp_file`: create a temp file under llama_playground/
- `stat_paths`: report file metadata (size, modification time)

Path semantics: paths are relative to the project root. The framework
auto-classifies each path — writes under `llama_playground/` are
ephemeral scratch (not tracked), writes inside the project's
write_root are tracked by the changeset journal and can be undone
with `revert_changes`, and writes outside both are rejected. You do
not specify a zone — only a path.

This pack auto-loads when no shell-execution tool (`command` or
`start_job`) is registered. Prefer `apply_patch` for partial edits to
existing files; use `move_path` / `copy_path` / `delete_path` for
whole-file operations.
