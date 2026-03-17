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

def part2_permissions():
    print("\n" + "=" * 60)
    print("Part 2: Permission Levels")
    print("=" * 60)

    config = Config()

    # Admin persona — can access all tools (permission_level=3)
    admin = Persona(
        name="Admin",
        role_description="System administrator",
        system_prompt="You are a system administrator with full access.",
        role="admin",
    )

    # Regular user — restricted tool access (permission_level=1)
    user = Persona(
        name="Intern",
        role_description="Junior developer",
        system_prompt="You are a helpful assistant.",
        role="user",
    )

    print(f"Admin permission: {admin.permission_level}")  # 3
    print(f"User permission: {user.permission_level}")    # 1

    # Tools with different safety levels
    agent = SmartAgent(config, persona=user)

    def read_file(path: str) -> str:
        """Read a file (safe operation)."""
        return f"Contents of {path}"

    def delete_file(path: str) -> str:
        """Delete a file (dangerous operation)."""
        return f"Deleted {path}"

    # safety_level=1: anyone can call; safety_level=3: admin only
    agent.register_tool("read_file", read_file, "Read a file", safety_level=1)
    agent.register_tool("delete_file", delete_file, "Delete a file", safety_level=3)

    # Tool schemas are filtered by persona permission
    schemas = agent.get_all_tool_schemas()
    visible_tools = [s["function"]["name"] for s in schemas]
    print(f"Tools visible to '{user.name}': {visible_tools}")
    # The user won't see delete_file because their permission_level (1) < safety_level (3)

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
    part2_permissions()
    part3_persona_manager()
