"""
05 — Personas: Give Your Agent an Identity

Personas define who the agent is — their name, role, system prompt,
and permission level. Different personas get different tool access.

Prerequisites:
    pip install -e .
"""

import tempfile

from llamagent import SmartAgent, Config, Persona, PersonaManager


# =============================================================
# Part 1: Create personas directly
# =============================================================

def part1_direct_persona():
    print("=" * 60)
    print("Part 1: Direct Persona Creation")
    print("=" * 60)

    # Create a persona with a specific identity
    coder = Persona(
        name="CodeLlama",
        role_description="A senior Python developer who writes clean, efficient code.",
        system_prompt=(
            "You are CodeLlama, a senior Python developer. "
            "You write clean, well-tested code. You prefer simplicity over cleverness. "
            "When asked to write code, always include type hints and docstrings."
        ),
        role="user",  # "user" (permission_level=1) or "admin" (permission_level=3)
    )

    print(f"Name: {coder.name}")
    print(f"ID: {coder.persona_id}")          # auto-generated: "codellama"
    print(f"Role: {coder.role}")
    print(f"Permission: {coder.permission_level}")  # auto: 1 for user

    # Use the persona with an agent
    config = Config()
    agent = SmartAgent(config, persona=coder)
    reply = agent.chat("Write a function to check if a number is prime.")
    print(f"Agent: {reply}")

    agent.shutdown()


# =============================================================
# Part 2: Admin vs User permissions
# =============================================================

def part2_tier_visibility():
    print("\n" + "=" * 60)
    print("Part 2: Tier-Based Visibility")
    print("=" * 60)

    config = Config()

    # Admin persona — sees admin-tier tools
    admin = Persona(
        name="Admin",
        role_description="System administrator",
        system_prompt="You are a system administrator with full access.",
        role="admin",
    )

    # Regular user — only sees default + common tier tools
    user = Persona(
        name="Intern",
        role_description="Junior developer",
        system_prompt="You are a helpful assistant.",
        role="user",
    )

    print(f"Admin role: {admin.role}")  # admin
    print(f"User role: {user.role}")    # user

    # Tools with different tiers
    agent = SmartAgent(config, persona=user)

    def read_files(paths: list) -> str:
        """Read files (safe operation)."""
        return f"Contents of {paths}"

    def admin_command(cmd: str) -> str:
        """Run an admin command (admin only)."""
        return f"Executed: {cmd}"

    # tier=common: visible to all; tier=admin: admin only
    agent.register_tool("read_files", read_files, "Read files", tier="common")
    agent.register_tool("admin_command", admin_command, "Admin command", tier="admin")

    # Tool visibility is controlled by tier — visibility equals usability
    schemas = agent.get_all_tool_schemas()
    visible_tools = [s["function"]["name"] for s in schemas]
    print(f"Tools visible to '{user.name}': {visible_tools}")
    # The user won't see admin_command because tier=admin and user is not admin

    agent.shutdown()


# =============================================================
# Part 3: PersonaManager — persist and manage personas
# =============================================================

def part3_persona_manager():
    print("\n" + "=" * 60)
    print("Part 3: PersonaManager (Persistence)")
    print("=" * 60)

    # Use a temp file so this example doesn't leave files around
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        persona_file = f.name
        f.write("{}")

    manager = PersonaManager(persona_file)

    # Create personas (without LLM expansion — just saves as-is)
    writer = manager.create(
        name="TechWriter",
        role_description="Technical documentation writer",
    )
    print(f"Created: {writer.name} (id={writer.persona_id})")

    analyst = manager.create(
        name="DataAnalyst",
        role_description="Data scientist specializing in visualization",
        role="user",
    )
    print(f"Created: {analyst.name} (id={analyst.persona_id})")

    # List all personas
    all_personas = manager.list()
    print(f"Total personas: {len(all_personas)}")
    for p in all_personas:
        print(f"  - {p.name} (role={p.role}, permission={p.permission_level})")

    # Retrieve by ID
    found = manager.get("techwriter")
    print(f"Found by ID: {found.name if found else 'not found'}")

    # Delete
    manager.delete("techwriter")
    print(f"After delete: {len(manager.list())} personas remaining")

    # Clean up temp file
    import os
    os.unlink(persona_file)


if __name__ == "__main__":
    part1_direct_persona()
    part2_tier_visibility()
    part3_persona_manager()
