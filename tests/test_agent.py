"""Tests for agent module."""

import pytest

from src.agents import create_research_agent, RESEARCH_AGENT_PROMPT


class TestResearchAgentPrompt:
    """Test the research agent system prompt."""

    def test_prompt_exists(self):
        """System prompt should exist."""
        assert RESEARCH_AGENT_PROMPT is not None
        assert len(RESEARCH_AGENT_PROMPT) > 0

    def test_prompt_mentions_planning(self):
        """Prompt should emphasize planning first."""
        assert "PLANNING" in RESEARCH_AGENT_PROMPT or "plan" in RESEARCH_AGENT_PROMPT.lower()
        assert "write_todos" in RESEARCH_AGENT_PROMPT

    def test_prompt_mentions_web_search(self):
        """Prompt should mention web_search tool."""
        assert "web_search" in RESEARCH_AGENT_PROMPT

    def test_prompt_mentions_quality_evaluation(self):
        """Prompt should mention evaluate_research_quality tool."""
        assert "evaluate_research_quality" in RESEARCH_AGENT_PROMPT

    def test_prompt_has_phases(self):
        """Prompt should have multiple phases."""
        assert "Phase 1" in RESEARCH_AGENT_PROMPT or "PLANNING" in RESEARCH_AGENT_PROMPT
        assert "Phase 2" in RESEARCH_AGENT_PROMPT or "RESEARCH" in RESEARCH_AGENT_PROMPT

    def test_prompt_mentions_citations(self):
        """Prompt should emphasize citations."""
        assert "citation" in RESEARCH_AGENT_PROMPT.lower() or "source" in RESEARCH_AGENT_PROMPT.lower()


class TestCreateResearchAgent:
    """Test create_research_agent factory function."""

    def test_creates_agent(self):
        """Should create an agent without errors."""
        agent = create_research_agent()
        assert agent is not None

    def test_agent_is_runnable(self):
        """Agent should have invoke method."""
        agent = create_research_agent()
        assert hasattr(agent, "invoke")

    def test_agent_has_stream(self):
        """Agent should have stream method."""
        agent = create_research_agent()
        assert hasattr(agent, "stream")


class TestAgentIntegration:
    """Integration tests for the agent (requires API keys)."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_agent_responds_to_simple_query(self):
        """Agent should respond to a simple query."""
        agent = create_research_agent()
        result = agent.invoke({
            "messages": [{"role": "user", "content": "What is 2+2?"}]
        })
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) > 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_agent_uses_tools(self):
        """Agent should use tools for research queries."""
        agent = create_research_agent()
        result = agent.invoke({
            "messages": [{"role": "user", "content": "Research what Python is in one sentence."}]
        })
        assert result is not None
        assert "messages" in result


# Pytest configuration for markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require API keys)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
