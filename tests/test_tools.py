"""Tests for tools module."""

import pytest

from src.tools import get_search_tools, web_search, evaluate_research_quality


class TestGetSearchTools:
    """Test get_search_tools factory function."""

    def test_returns_list(self):
        """get_search_tools should return a list."""
        tools = get_search_tools()
        assert isinstance(tools, list)

    def test_contains_evaluate_research_quality(self):
        """Tools should always include evaluate_research_quality."""
        tools = get_search_tools()
        tool_names = [t.name for t in tools]
        assert "evaluate_research_quality" in tool_names

    def test_tools_have_names(self):
        """All tools should have names."""
        tools = get_search_tools()
        for tool in tools:
            assert hasattr(tool, "name")
            assert tool.name is not None


class TestWebSearch:
    """Test web_search tool."""

    def test_web_search_exists(self):
        """web_search tool should exist."""
        assert web_search is not None

    def test_web_search_has_name(self):
        """web_search should have correct name."""
        assert web_search.name == "web_search"

    def test_web_search_has_description(self):
        """web_search should have a description."""
        assert web_search.description is not None
        assert len(web_search.description) > 0

    def test_web_search_without_api_key(self, monkeypatch):
        """web_search should return error message without API key."""
        # This test checks graceful handling when API key is missing
        from src.config import get_settings
        get_settings.cache_clear()
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        # Reload settings without API key
        from src.config.settings import Settings
        settings = Settings()

        if not settings.has_tavily:
            # If no API key, calling web_search should return an error message
            result = web_search.invoke({"query": "test query"})
            assert "Error" in result or "error" in result.lower()

        get_settings.cache_clear()


class TestEvaluateResearchQuality:
    """Test evaluate_research_quality tool."""

    def test_tool_exists(self):
        """evaluate_research_quality tool should exist."""
        assert evaluate_research_quality is not None

    def test_tool_has_name(self):
        """Tool should have correct name."""
        assert evaluate_research_quality.name == "evaluate_research_quality"

    def test_tool_has_description(self):
        """Tool should have a description."""
        assert evaluate_research_quality.description is not None

    def test_high_quality_research(self):
        """High quality research should score >= 80."""
        result = evaluate_research_quality.invoke({
            "original_question": "What is Python programming?",
            "findings_summary": "Python is a high-level, interpreted programming language known for its readability and versatility. It was created by Guido van Rossum and released in 1991. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            "sources_count": 5,
            "aspects_covered": ["history", "features", "use cases", "syntax"]
        })
        assert "GOOD" in result or "Score: 100" in result or "Score: 85" in result

    def test_low_quality_research(self):
        """Low quality research should score < 80."""
        result = evaluate_research_quality.invoke({
            "original_question": "What is Python?",
            "findings_summary": "Python is a language.",
            "sources_count": 1,
            "aspects_covered": ["basics"]
        })
        assert "NEEDS IMPROVEMENT" in result or "ADEQUATE" in result

    def test_returns_recommendations(self):
        """Tool should always return recommendations."""
        result = evaluate_research_quality.invoke({
            "original_question": "Test question",
            "findings_summary": "Test findings with enough content to pass the length check for the evaluation tool.",
            "sources_count": 3,
            "aspects_covered": ["aspect1", "aspect2"]
        })
        assert "Recommendations" in result
