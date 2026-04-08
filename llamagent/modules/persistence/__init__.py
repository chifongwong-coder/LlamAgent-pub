"""
Persistence Module: auto-save and restore conversation history.

Usage:
    from llamagent.modules.persistence import PersistenceModule
    agent.register_module(PersistenceModule())
"""

from llamagent.modules.persistence.module import PersistenceModule

__all__ = ["PersistenceModule"]
