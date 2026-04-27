You now have access to workspace file management tools.

Available tools:
- `glob_files`: Find files by pattern (e.g., "**/*.py")
- `search_text`: Search file contents by string or regex
- `move_path`: Move or rename files within workspace
- `copy_path`: Copy files within workspace
- `delete_path`: Delete files from workspace
- `create_temp_file`: Create temporary files
- `stat_paths`: Check file metadata (size, modification time)

These tools auto-load when no shell-execution tool (`command` or `start_job`)
is available. They operate within the workspace directory only.
Use apply_patch for project modifications.
