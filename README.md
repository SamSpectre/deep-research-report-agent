# Deep Research Agent

A production-grade multi-agent research system built with [LangChain DeepAgents](https://github.com/langchain-ai/deepagents).

## Overview

This project implements a sophisticated research agent that can:

- **Plan**: Break down complex research tasks into manageable steps
- **Search**: Query the web for relevant information
- **Validate**: Cross-check and verify findings
- **Synthesize**: Generate comprehensive research reports

Built on the DeepAgents framework, which implements patterns from production agents like Claude Code and Manus.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RESEARCH ORCHESTRATOR                     │
│                    (Main DeepAgent)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Planner   │  │  Researcher │  │   Writer    │         │
│  │  Sub-Agent  │  │  Sub-Agent  │  │  Sub-Agent  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Built-in Tools: todos, filesystem, task delegation         │
│  Custom Tools: web_search                                   │
├─────────────────────────────────────────────────────────────┤
│  Middleware: Planning, Summarization, HITL, Caching         │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **DeepAgents Architecture**: Planning, sub-agents, filesystem, detailed prompts
- **Multi-Model Support**: Claude (Anthropic) and GPT (OpenAI)
- **Web Search**: Tavily integration for AI-optimized search
- **Production Ready**: Proper logging, configuration, error handling
- **LangSmith Integration**: Tracing and observability

## Quick Start

### Prerequisites

- Python 3.11 or higher
- API keys for:
  - Anthropic (Claude) OR OpenAI (GPT)
  - Tavily (web search)
  - LangSmith (optional, for tracing)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deep-research-agent.git
cd deep-research-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# Required: ANTHROPIC_API_KEY or OPENAI_API_KEY
# Required: TAVILY_API_KEY
# Optional: LANGSMITH_API_KEY
```

### Usage

```bash
# Run the research agent
python main.py

# Or with a specific topic
python main.py "Your research topic here"
```

## Project Structure

```
deep-research-agent/
├── src/
│   ├── agents/          # DeepAgent definitions
│   ├── tools/           # Custom tools (web search, etc.)
│   ├── config/          # Configuration management
│   └── utils/           # Utilities (logging, etc.)
├── tests/               # Test files
├── docs/                # Documentation
├── pyproject.toml       # Dependencies
├── .env.example         # Environment template
└── main.py              # Entry point
```

## Learning Path

This project is designed as a learning journey through DeepAgents:

1. **Phase 1**: Basic agent with web search
2. **Phase 2**: Planning and task management
3. **Phase 3**: Multi-agent orchestration
4. **Phase 4**: Quality and iteration loops
5. **Phase 5**: Human-in-the-loop and persistence
6. **Phase 6**: Report generation

## Built With

- [DeepAgents](https://github.com/langchain-ai/deepagents) - Agent harness
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent runtime
- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [Tavily](https://tavily.com/) - AI search API

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- LangChain team for DeepAgents and LangGraph
- Anthropic for Claude Code (architectural inspiration)
- The AI agent community