"""Deep Research Agent - Main Entry Point."""

import sys

from dotenv import load_dotenv

# Load .env file into environment (must be before other imports)
load_dotenv()

from src.agents import create_research_agent
from src.config import get_settings
from src.utils import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def main(topic: str) -> dict:
    """Run research on a topic."""
    settings = get_settings()

    logger.info("Starting research", topic=topic[:50], model=settings.default_model)

    # Create agent
    agent = create_research_agent()

    # Run agent
    result = agent.invoke({
        "messages": [{"role": "user", "content": topic}]
    })

    # Print final response
    if result and "messages" in result:
        final_message = result["messages"][-1]
        if hasattr(final_message, "content"):
            print("\n" + "=" * 50)
            print("RESEARCH RESULTS")
            print("=" * 50)
            print(final_message.content)

    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = "What is DeepAgents framework?"

    main(topic)
