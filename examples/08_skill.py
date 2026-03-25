"""
08 -- Skills: Task-Level Playbook Injection

The SkillModule lets you define reusable task playbooks that are
automatically injected into the LLM context when relevant.

Skills are NOT tools -- they don't add callable functions. Instead,
they inject step-by-step guidance ("how to do it") into the system
prompt when the query matches the skill's tags.

Each skill is a directory with two files:
- config.yaml  -- metadata (name, description, tags, aliases)
- SKILL.md     -- pure natural language playbook

Three-level activation:
- A: /skill <name>     -- explicit command, always works
- B: Tag matching       -- automatic, based on query words vs tags
- C: LLM fallback       -- optional, sends all metadata to LLM

Prerequisites:
    pip install -e .
    pip install pyyaml
"""

import os
import tempfile

from llamagent import SmartAgent, Config
from llamagent.modules.skill import SkillModule


# =============================================================
# Part 1: Create skill directories and load them
# =============================================================

def part1_basic_skill():
    print("=" * 60)
    print("Part 1: Basic Skill Loading")
    print("=" * 60)

    # Create a temporary skill directory structure
    tmp = tempfile.mkdtemp()
    skills_dir = os.path.join(tmp, ".llamagent", "skills")

    # Skill 1: db-migration
    migration_dir = os.path.join(skills_dir, "db-migration")
    os.makedirs(migration_dir)

    with open(os.path.join(migration_dir, "config.yaml"), "w") as f:
        f.write(
            "name: db-migration\n"
            "description: Guide through database migration workflow\n"
            "tags: [migration, alembic, migrate]\n"
            "aliases: [migrate-db]\n"
        )

    with open(os.path.join(migration_dir, "SKILL.md"), "w") as f:
        f.write(
            "## Goal\n"
            "Guide the agent through a safe database migration.\n\n"
            "## Steps\n"
            "1. Check which migration tool is configured (alembic/django/raw SQL)\n"
            "2. Generate the migration file with up and down operations\n"
            "3. Review the generated SQL\n"
            "4. Ask user for confirmation before applying\n\n"
            "## Tool guidance\n"
            "- Use write_files to create migration files in migrations/ directory\n"
            "- Never auto-apply migrations without confirmation\n"
        )

    # Skill 2: code-review
    review_dir = os.path.join(skills_dir, "code-review")
    os.makedirs(review_dir)

    with open(os.path.join(review_dir, "config.yaml"), "w") as f:
        f.write(
            "name: code-review\n"
            "description: Structured code review checklist\n"
            "tags: [review, pr, pullrequest]\n"
        )

    with open(os.path.join(review_dir, "SKILL.md"), "w") as f:
        f.write(
            "## Goal\n"
            "Perform a thorough code review.\n\n"
            "## Steps\n"
            "1. Check for correctness and logic errors\n"
            "2. Check for security vulnerabilities\n"
            "3. Check naming conventions and readability\n"
            "4. Check test coverage\n"
            "5. Summarize findings with severity levels\n"
        )

    # Create agent with skill_dirs pointing to our temp directory
    config = Config()
    config.skill_dirs = []
    config.skill_max_active = 2
    config.skill_llm_fallback = False

    agent = SmartAgent(config)
    agent.project_dir = tmp  # Point to our temp dir for scanning

    # Register SkillModule
    skill_mod = SkillModule()
    agent.register_module(skill_mod)

    # List loaded skills
    skills = skill_mod.list_skills()
    print(f"\nLoaded {len(skills)} skills:")
    for s in skills:
        print(f"  - {s.name}: {s.description} (tags: {s.tags})")

    return agent, skill_mod


# =============================================================
# Part 2: Explicit activation with /skill command
# =============================================================

def part2_explicit_activation(skill_mod):
    print("\n" + "=" * 60)
    print("Part 2: Explicit /skill Command")
    print("=" * 60)

    # Simulate /skill command
    result = skill_mod.on_input("/skill db-migration create a users table migration")
    print(f"\n/skill command intercepted:")
    print(f"  forced_skill = {skill_mod._forced_skill}")
    print(f"  remaining query = {result!r}")

    # Simulate on_context (would normally be called by chat pipeline)
    context = skill_mod.on_context(result, "")
    print(f"\nInjected context preview (first 200 chars):")
    print(f"  {context[:200]}...")

    # Verify /skill command is one-turn only
    print(f"\nAfter on_context, forced_skill = {skill_mod._forced_skill}")
    print("  (cleared -- /skill is one-turn only)")


# =============================================================
# Part 3: Automatic tag matching
# =============================================================

def part3_tag_matching(skill_mod):
    print("\n" + "=" * 60)
    print("Part 3: Automatic Tag Matching")
    print("=" * 60)

    # This query contains "migration" which matches the db-migration skill's tags
    context = skill_mod.on_context("help me create a migration for the orders table", "")
    if "[Active Skill:" in context:
        # Extract skill name from the block
        start = context.index("[Active Skill: ") + len("[Active Skill: ")
        end = context.index("]", start)
        name = context[start:end]
        print(f"\nQuery 'help me create a migration' matched skill: {name}")
    else:
        print("\nNo skill matched (expected -- depends on tag matching)")

    # This query doesn't match any tags
    context2 = skill_mod.on_context("what is the weather today", "")
    if "[Active Skill:" not in context2:
        print("Query 'what is the weather today' -- no skill matched (correct)")


# =============================================================
# Part 4: External API
# =============================================================

def part4_api(skill_mod):
    print("\n" + "=" * 60)
    print("Part 4: External API")
    print("=" * 60)

    # get_skill by name
    meta = skill_mod.get_skill("db-migration")
    print(f"\nget_skill('db-migration'):")
    print(f"  name: {meta.name}")
    print(f"  tags: {meta.tags}")
    print(f"  aliases: {meta.aliases}")
    print(f"  invocation: {meta.invocation}")

    # get_skill by alias
    meta2 = skill_mod.get_skill("migrate-db")
    print(f"\nget_skill('migrate-db') (alias):")
    print(f"  resolved to: {meta2.name}")

    # activate manually (returns SKILL.md content)
    content = skill_mod.activate("code-review")
    print(f"\nactivate('code-review') content preview:")
    print(f"  {content[:150]}...")

    # reload (re-scan directories)
    count = skill_mod.reload()
    print(f"\nreload() -> {count} skills")


# =============================================================
# Main
# =============================================================

if __name__ == "__main__":
    agent, skill_mod = part1_basic_skill()
    part2_explicit_activation(skill_mod)
    part3_tag_matching(skill_mod)
    part4_api(skill_mod)

    print("\n" + "=" * 60)
    print("Done! Skills are task-level playbooks, not tools.")
    print("They guide the agent on HOW to do things,")
    print("while tools provide WHAT the agent can do.")
    print("=" * 60)
