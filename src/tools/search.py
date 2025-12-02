"""
Web search tools for Deep Research Agent.

This module provides web search capabilities using various providers.
Currently supports Tavily (primary) with Exa planned for future phases.

The search tool is designed to work seamlessly with DeepAgents, providing
LLM-optimized search results with citations and clean content.
"""

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.config import get_settings
from src.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# TOOL INPUT SCHEMAS
# =============================================================================
# Pydantic models define the expected input for tools.
# This provides:
# 1. Type validation
# 2. Clear documentation for the LLM
# 3. Default values


class TavilySearchInput(BaseModel):
    """Input schema for Tavily web search."""
    
    query: str = Field(
        description="The search query to execute. Be specific and descriptive for better results."
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of search results to return (1-10)."
    )
    search_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description=(
            "Search depth: 'basic' for quick searches, "
            "'advanced' for more comprehensive results (costs more credits)."
        )
    )
    include_answer: bool = Field(
        default=True,
        description="Whether to include an AI-generated answer summary."
    )
    include_raw_content: bool = Field(
        default=False,
        description="Whether to include the raw content of each result (increases response size)."
    )


class ResearchQualityInput(BaseModel):
    """Input schema for research quality evaluation."""
    
    original_question: str = Field(
        description="The original research question or topic from the user."
    )
    findings_summary: str = Field(
        description="A brief summary of what was found during research."
    )
    sources_count: int = Field(
        description="Number of unique sources consulted."
    )
    aspects_covered: list[str] = Field(
        description="List of aspects/subtopics that were covered in the research."
    )


# =============================================================================
# TAVILY SEARCH TOOL
# =============================================================================


@tool(args_schema=TavilySearchInput)
def web_search(
    query: str,
    max_results: int = 5,
    search_depth: Literal["basic", "advanced"] = "basic",
    include_answer: bool = True,
    include_raw_content: bool = False,
) -> str:
    """
    Search the web for information on any topic.
    
    Use this tool when you need to:
    - Find current information not in your training data
    - Research a topic thoroughly
    - Verify facts with external sources
    - Gather multiple perspectives on a subject
    
    The tool returns search results with titles, URLs, and content snippets.
    If include_answer is True, it also provides an AI-generated summary.
    
    Args:
        query: The search query. Be specific for better results.
        max_results: Number of results to return (1-10).
        search_depth: 'basic' for quick searches, 'advanced' for thorough research.
        include_answer: Include an AI summary of the results.
        include_raw_content: Include full page content (use sparingly).
    
    Returns:
        Formatted search results with titles, URLs, content, and optional summary.
    """
    settings = get_settings()
    
    # Validate API key is configured
    if not settings.has_tavily:
        error_msg = (
            "Tavily API key not configured. "
            "Please set TAVILY_API_KEY in your .env file. "
            "Get a free key at https://tavily.com/"
        )
        logger.error("Tavily API key missing")
        return f"Error: {error_msg}"
    
    try:
        # Import here to avoid import errors if tavily not installed
        from tavily import TavilyClient
        
        # Initialize client with API key
        client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())
        
        logger.info(
            "Executing web search",
            query=query[:50] + "..." if len(query) > 50 else query,
            max_results=max_results,
            search_depth=search_depth,
        )
        
        # Execute search
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
        )
        
        # Format results for LLM consumption
        formatted_results = _format_tavily_results(response, include_answer)
        
        logger.info(
            "Web search completed",
            num_results=len(response.get("results", [])),
        )
        
        return formatted_results
        
    except ImportError:
        error_msg = "tavily-python package not installed. Run: pip install tavily-python"
        logger.error(error_msg)
        return f"Error: {error_msg}"
        
    except Exception as e:
        error_msg = f"Search failed: {type(e).__name__}: {str(e)}"
        logger.error("Web search failed", error=str(e), error_type=type(e).__name__)
        return f"Error: {error_msg}"


def _format_tavily_results(response: dict, include_answer: bool) -> str:
    """
    Format Tavily API response into a clean, LLM-readable string.
    
    This formatting is important because:
    1. The LLM needs to understand the structure
    2. Citations need to be clear
    3. Content should be scannable
    
    Args:
        response: Raw Tavily API response
        include_answer: Whether to include the AI summary
        
    Returns:
        Formatted string with search results
    """
    parts = []
    
    # Add AI-generated answer if available
    if include_answer and response.get("answer"):
        parts.append("## Summary")
        parts.append(response["answer"])
        parts.append("")  # Empty line for separation
    
    # Add individual results
    results = response.get("results", [])
    if results:
        parts.append("## Search Results")
        parts.append("")
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            content = result.get("content", "No content available")
            
            # Truncate very long content
            if len(content) > 500:
                content = content[:500] + "..."
            
            parts.append(f"### Result {i}: {title}")
            parts.append(f"**URL**: {url}")
            parts.append(f"**Content**: {content}")
            parts.append("")  # Empty line between results
    else:
        parts.append("No results found for this query.")
    
    return "\n".join(parts)


# =============================================================================
# RESEARCH QUALITY EVALUATION TOOL
# =============================================================================


@tool(args_schema=ResearchQualityInput)
def evaluate_research_quality(
    original_question: str,
    findings_summary: str,
    sources_count: int,
    aspects_covered: list[str],
) -> str:
    """
    Evaluate the quality and completeness of your research.
    
    Use this tool AFTER completing your research to assess whether you have
    gathered enough information to provide a comprehensive answer.
    
    This helps you decide whether to:
    - Continue with synthesis (if quality is sufficient)
    - Do more research (if gaps are identified)
    
    Args:
        original_question: The research question you're trying to answer.
        findings_summary: Brief summary of what you found.
        sources_count: How many unique sources you consulted.
        aspects_covered: List of subtopics/aspects you researched.
    
    Returns:
        Quality assessment with recommendations.
    """
    logger.info(
        "Evaluating research quality",
        sources=sources_count,
        aspects=len(aspects_covered),
    )
    
    # Build assessment
    assessment_parts = []
    issues = []
    score = 100  # Start with perfect score, deduct for issues
    
    # Check source count
    if sources_count < 3:
        issues.append("- Few sources consulted (aim for 3+ diverse sources)")
        score -= 20
    elif sources_count >= 5:
        assessment_parts.append("✓ Good number of sources consulted")
    
    # Check aspects coverage
    if len(aspects_covered) < 2:
        issues.append("- Limited aspects covered (consider broadening research)")
        score -= 20
    elif len(aspects_covered) >= 4:
        assessment_parts.append("✓ Multiple aspects thoroughly covered")
    
    # Check findings summary
    if len(findings_summary) < 100:
        issues.append("- Findings summary is brief (may need more depth)")
        score -= 15
    elif len(findings_summary) >= 300:
        assessment_parts.append("✓ Substantial findings documented")
    
    # Build final assessment
    result_parts = ["## Research Quality Assessment", ""]
    
    # Score interpretation
    if score >= 80:
        result_parts.append(f"**Overall Quality: GOOD** (Score: {score}/100)")
        result_parts.append("")
        result_parts.append("Your research appears comprehensive enough to proceed with synthesis.")
    elif score >= 60:
        result_parts.append(f"**Overall Quality: ADEQUATE** (Score: {score}/100)")
        result_parts.append("")
        result_parts.append("Research is acceptable but could be improved. Consider addressing the issues below.")
    else:
        result_parts.append(f"**Overall Quality: NEEDS IMPROVEMENT** (Score: {score}/100)")
        result_parts.append("")
        result_parts.append("Consider doing additional research before synthesis.")
    
    result_parts.append("")
    
    # Positives
    if assessment_parts:
        result_parts.append("### Strengths")
        result_parts.extend(assessment_parts)
        result_parts.append("")
    
    # Issues
    if issues:
        result_parts.append("### Areas for Improvement")
        result_parts.extend(issues)
        result_parts.append("")
    
    # Specific recommendations
    result_parts.append("### Recommendations")
    if score >= 80:
        result_parts.append("- Proceed to synthesize your findings")
        result_parts.append("- Create a final report with citations")
    else:
        result_parts.append("- Consider searching for additional perspectives")
        result_parts.append("- Look for primary sources if using secondary sources")
        result_parts.append("- Check if any key aspects were missed")
    
    return "\n".join(result_parts)


# =============================================================================
# TOOL FACTORY FUNCTION
# =============================================================================


def get_search_tools() -> list:
    """
    Get the list of search tools to provide to the agent.
    
    This factory function allows us to:
    1. Add multiple search tools in the future (Exa, etc.)
    2. Configure tools based on settings
    3. Keep the agent creation code clean
    
    Returns:
        List of search tool functions
    """
    settings = get_settings()
    tools = []
    
    # Always include Tavily if configured
    if settings.has_tavily:
        tools.append(web_search)
        logger.debug("Added Tavily web_search tool")
    else:
        logger.warning(
            "No search tools available - TAVILY_API_KEY not configured"
        )
    
    # Always include quality evaluation tool
    tools.append(evaluate_research_quality)
    logger.debug("Added evaluate_research_quality tool")
    
    # Future: Add Exa here when we implement Phase 4+
    # if settings.has_exa:
    #     tools.append(exa_search)
    
    return tools


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["web_search", "evaluate_research_quality", "get_search_tools"]