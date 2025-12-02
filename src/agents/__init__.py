"""
Agent definitions for Deep Research Agent.

Phase 3: Multi-Agent Research System

This module provides two agent configurations:

1. create_research_agent() - Multi-agent orchestration system
   - Orchestrator: Plans and coordinates
   - Researcher: Gathers information (web search)
   - Critic: Evaluates quality
   - Writer: Creates reports

2. create_simple_research_agent() - Single-agent (Phase 1/2 style)
   - Good for simpler tasks or cost savings
"""

from src.agents.research_agent import (
    # Factory functions
    create_research_agent,
    create_simple_research_agent,
    # Convenience functions
    run_research,
    # System prompts
    ORCHESTRATOR_PROMPT,
    RESEARCHER_PROMPT,
    CRITIC_PROMPT,
    WRITER_PROMPT,
)

__all__ = [
    # Factory functions
    "create_research_agent",
    "create_simple_research_agent",
    # Convenience functions
    "run_research",
    # System prompts
    "ORCHESTRATOR_PROMPT",
    "RESEARCHER_PROMPT",
    "CRITIC_PROMPT",
    "WRITER_PROMPT",
]
