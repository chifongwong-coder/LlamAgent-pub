"""
ResilienceModule: LLM call protection via error classification, smart retry, and model failover.

Wraps agent.llm with ResilientLLM (a LLMClient subclass) on attach. When not loaded,
agent behavior is unchanged. ResilientLLM only overrides chat(); ask/ask_json/count_tokens
are inherited from LLMClient and automatically get resilience.

Should be registered FIRST so subsequent modules get the resilient LLM.
"""

from __future__ import annotations

import logging

from llamagent.core.llm import LLMClient
from llamagent.core.agent import Module
from llamagent.modules.resilience.resilient_llm import ResilientLLM

logger = logging.getLogger(__name__)


class ResilienceModule(Module):
    """LLM call resilience: error classification, smart retry, model failover."""

    name = "resilience"
    description = "LLM call resilience: error classification, smart retry, model failover"

    def on_attach(self, agent):
        super().on_attach(agent)

        fallback_model = getattr(agent.config, "fallback_model", None)
        fallback_llm = LLMClient(fallback_model) if fallback_model else None

        routing_simple_model = getattr(agent.config, "routing_simple_model", None)
        simple_llm = LLMClient(routing_simple_model) if routing_simple_model else None

        max_retries = getattr(agent.config, "resilience_max_retries", 3)

        # Create a new ResilientLLM instance using the model string.
        # The original LLMClient in _llm_cache is NOT modified.
        resilient = ResilientLLM(
            model=agent.llm.model,
            fallback_llm=fallback_llm,
            max_retries=max_retries,
            simple_llm=simple_llm,
        )
        agent.llm = resilient

        logger.info(
            "ResilienceModule attached: max_retries=%d, fallback=%s, simple=%s",
            max_retries,
            fallback_model or "none",
            routing_simple_model or "none",
        )
