"""
Custom tools for Deep Research Agent.

This module contains tool definitions that extend the DeepAgent's capabilities
beyond the built-in tools (todos, filesystem, subagents).
"""

from src.tools.search import (
    get_search_tools,
    web_search,
    evaluate_research_quality,
)

__all__ = ["web_search", "evaluate_research_quality", "get_search_tools"]
