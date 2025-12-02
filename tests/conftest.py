"""Pytest configuration and fixtures."""

import pytest
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env for tests
from dotenv import load_dotenv
load_dotenv()


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require API keys)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Reset settings cache before each test."""
    from src.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
