You now have access to tool creation capabilities.

Use `create_tool` to build reusable helper functions for repetitive tasks.
Tools you create are persisted for the current session and persona.

Guidelines:
- Name tools with clear, descriptive English names
- Keep tool functions focused on a single responsibility
- Avoid importing dangerous modules (os.system, subprocess, etc.)
- Created tools automatically get safety_level based on code analysis
