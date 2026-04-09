"""
Markdown validator for FS knowledge base documents.

Checks that .md files in a knowledge directory can be correctly parsed
by the FS retrieval backend (list_knowledge, list_entries, read_entry).

Checks:
  1. File is valid UTF-8 text
  2. If frontmatter exists, it is well-formed (opening + closing ---)
  3. Has at least one ## section (needed for list_entries)
  4. Heading levels: # or ### without ## are flagged

Fix mode (--fix):
  - Malformed frontmatter: close unclosed ---
  - No ## sections but has # or ###: normalize to ##
  - Writes fixed files back (originals backed up as .bak)
"""

import os
import re
import shutil
import sys

from llamagent.modules.fs_store.parser import parse_frontmatter, parse_sections


# ============================================================
# Checks
# ============================================================

def check_file(filepath: str) -> list[dict]:
    """Run all checks on a single .md file.

    Returns a list of issue dicts: {"level": "error"|"warning", "message": str}
    """
    issues = []

    # Check 1: UTF-8 readable
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError as e:
        return [{"level": "error", "message": f"Not valid UTF-8: {e}"}]

    if not content.strip():
        return [{"level": "warning", "message": "File is empty"}]

    # Check 2: Frontmatter well-formed (if present)
    stripped = content.lstrip("\n")
    lines = stripped.splitlines()
    if lines and lines[0].strip() == "---":
        has_closing = False
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                has_closing = True
                break
        if not has_closing:
            issues.append({
                "level": "error",
                "message": "Frontmatter opened with --- but never closed",
            })

    # Parse frontmatter
    metadata, body = parse_frontmatter(content)

    # Check 3: Has ## sections
    sections = parse_sections(body)
    if not sections:
        has_h1 = bool(re.search(r"^#(?!#)\s+.+$", body, re.MULTILINE))
        has_h3 = bool(re.search(r"^###\s+.+$", body, re.MULTILINE))

        if has_h1 or has_h3:
            levels = []
            if has_h1:
                levels.append("#")
            if has_h3:
                levels.append("###")
            issues.append({
                "level": "warning",
                "message": f"No ## sections found, but has {'/'.join(levels)} headings. "
                           f"list_entries requires ## headings. Use --fix to normalize.",
            })
        else:
            issues.append({
                "level": "warning",
                "message": "No ## sections found. The entire document will be treated "
                           "as a single block by list_entries.",
            })

    return issues


def check_directory(dirpath: str) -> dict[str, list[dict]]:
    """Check all .md files in a directory.

    Returns {filename: [issues]} for files with issues.
    """
    results = {}
    if not os.path.isdir(dirpath):
        return {"<error>": [{"level": "error", "message": f"Not a directory: {dirpath}"}]}

    for filename in sorted(os.listdir(dirpath)):
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(dirpath, filename)
        if not os.path.isfile(filepath):
            continue
        issues = check_file(filepath)
        if issues:
            results[filename] = issues

    return results


# ============================================================
# Fixes
# ============================================================

def fix_file(filepath: str) -> list[str]:
    """Attempt to fix common issues in a .md file.

    Returns a list of fix descriptions applied. Empty if no fixes needed.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        return ["Cannot fix: file is not valid UTF-8"]

    if not content.strip():
        return []

    fixes = []
    new_content = content

    # Fix 1: Unclosed frontmatter — add closing ---
    stripped = new_content.lstrip("\n")
    lines = stripped.splitlines()
    if lines and lines[0].strip() == "---":
        has_closing = False
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                has_closing = True
                break
        if not has_closing:
            insert_idx = 1
            for i in range(1, len(lines)):
                line = lines[i].strip()
                if not line or (not re.match(r"^\w+\s*:", line) and line != "---"):
                    insert_idx = i
                    break
                insert_idx = i + 1
            lines.insert(insert_idx, "---")
            new_content = "\n".join(lines)
            fixes.append("Added missing frontmatter closing ---")

    # Fix 2: Normalize heading levels to ##
    _, body = parse_frontmatter(new_content)
    sections = parse_sections(body)
    if not sections:
        def normalize_headings(text):
            text = re.sub(r"^###\s+", "## ", text, flags=re.MULTILINE)
            text = re.sub(r"^#(?!#)\s+", "## ", text, flags=re.MULTILINE)
            return text

        normalized_body = normalize_headings(body)
        new_sections = parse_sections(normalized_body)
        if new_sections:
            metadata, _ = parse_frontmatter(new_content)
            if metadata:
                from llamagent.modules.fs_store.parser import render_frontmatter
                new_content = render_frontmatter(metadata, normalized_body)
            else:
                new_content = normalized_body
            fixes.append(f"Normalized heading levels to ## ({len(new_sections)} sections found)")

    if fixes:
        backup_path = filepath + ".bak"
        shutil.copy2(filepath, backup_path)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

    return fixes


# ============================================================
# CLI
# ============================================================

def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m llamagent.tools.md_validator <directory> [--fix]")
        print()
        print("Checks .md files in a knowledge directory for FS retrieval compatibility.")
        print()
        print("Options:")
        print("  --fix    Attempt to auto-fix issues (backs up originals as .bak)")
        sys.exit(1)

    dirpath = sys.argv[1]
    do_fix = "--fix" in sys.argv

    if not os.path.isdir(dirpath):
        print(f"Error: '{dirpath}' is not a directory.")
        sys.exit(1)

    md_files = [f for f in os.listdir(dirpath)
                if f.endswith(".md") and os.path.isfile(os.path.join(dirpath, f))]
    if not md_files:
        print(f"No .md files found in '{dirpath}'.")
        sys.exit(0)

    print(f"Checking {len(md_files)} .md file(s) in '{dirpath}'...")
    print()

    total_issues = 0
    total_fixes = 0

    for filename in sorted(md_files):
        filepath = os.path.join(dirpath, filename)
        issues = check_file(filepath)

        if not issues:
            print(f"  {filename}: OK")
            continue

        total_issues += len(issues)
        for issue in issues:
            marker = "ERROR" if issue["level"] == "error" else "WARN"
            print(f"  {filename}: [{marker}] {issue['message']}")

        if do_fix:
            fixes = fix_file(filepath)
            if fixes:
                total_fixes += len(fixes)
                for fix in fixes:
                    print(f"  {filename}: [FIXED] {fix}")

    print()
    if total_issues == 0:
        print("All files OK.")
    else:
        print(f"{total_issues} issue(s) found across {len(md_files)} file(s).")
        if do_fix:
            print(f"{total_fixes} fix(es) applied. Originals backed up as .bak files.")
        elif total_issues > 0:
            print("Run with --fix to attempt auto-repair.")


if __name__ == "__main__":
    main()
