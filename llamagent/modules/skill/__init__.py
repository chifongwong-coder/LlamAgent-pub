"""
Skill module: task-level playbook injection via on_context.

Tool handles "what can be done", Skill handles "how things should be done".
Skills are loaded from config.yaml + SKILL.md directory pairs and injected
into the LLM context when matched against the user's query.
"""

from llamagent.modules.skill.index import SkillIndex, SkillMeta
from llamagent.modules.skill.module import SkillModule

__all__ = ["SkillModule", "SkillIndex", "SkillMeta"]
