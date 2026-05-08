You can fix skill files whose YAML frontmatter is using legacy permissive
syntax that triggers `[skill-migrate]` warnings in the framework log.

When the user reports a migration warning OR proactively asks to repair
skills, do the following:

1. Use `glob_files("**/*.md")` to locate skill files under the user's
   skill directories. Common locations:
   - `data/fs/skills/`
   - `~/.llamagent/skills/`
   - `~/.config/llamagent/skills/`
   - any path the user names

2. For each match, `read_files` to load the content. The frontmatter sits
   between the first `---` marker and the second `---` marker. ONLY this
   region needs editing — never touch the markdown body below.

3. Common legacy patterns to fix (apply minimum-change quoting):

   - `description: Use case: do X` → `description: "Use case: do X"`
     (any value containing `:` outside flow-style mapping)
   - `description: not a # comment` → `description: "not a # comment"`
     (any value containing `#`)
   - mid-frontmatter line lacking `:` → comment it out with `#` or remove
   - tab-indented values → replace tab with two spaces
   - `tags: [a, b,]` → `tags: [a, b]` (drop trailing comma)
   - `template: {role}` → `template: "{role}"` (quote raw flow-style)

4. Use `apply_patch` for the minimum-change rewrite. Fall back to
   `write_files` only when patch fails (e.g. line numbers shifted).

5. After writing, `read_files` again on each fixed file to verify the
   warning is gone. The framework log line for migration warnings ends
   with the file path — any remaining warning means the file still has
   an unhandled pattern.

Be conservative:
- Report each file you fixed and which pattern was matched.
- Refuse to modify files outside the skill directories listed above
  (or whatever directory the user explicitly named).
- If you encounter a frontmatter pattern you don't recognize, ask the
  user before editing.

The `python-frontmatter` library that the framework uses for strict
YAML is the same library Hermes / Nanobot / OpenCode use — your fixes
are aligned with industry-standard YAML frontmatter. Files repaired
through this skill load via the strict path with no warning thereafter.
