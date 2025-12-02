"""Tests for configuration module."""

import pytest
from pydantic import SecretStr

from src.config import Settings, get_settings


class TestSettings:
    """Test Settings class."""

    def test_settings_loads(self):
        """Settings should load without errors."""
        settings = get_settings()
        assert settings is not None

    def test_settings_has_default_model(self):
        """Settings should have a default model."""
        settings = get_settings()
        assert settings.default_model is not None
        assert "anthropic" in settings.default_model or "openai" in settings.default_model

    def test_settings_has_log_level(self):
        """Settings should have a log level."""
        settings = get_settings()
        assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]

    def test_settings_api_key_properties(self):
        """Settings should have API key check properties."""
        settings = get_settings()
        # These should return bool, not raise errors
        assert isinstance(settings.has_anthropic, bool)
        assert isinstance(settings.has_openai, bool)
        assert isinstance(settings.has_tavily, bool)
        assert isinstance(settings.has_langsmith, bool)

    def test_settings_singleton(self):
        """get_settings should return cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_api_keys_are_secret_str(self):
        """API keys should be SecretStr type when set."""
        settings = get_settings()
        if settings.anthropic_api_key is not None:
            assert isinstance(settings.anthropic_api_key, SecretStr)
        if settings.tavily_api_key is not None:
            assert isinstance(settings.tavily_api_key, SecretStr)


class TestSettingsFromEnv:
    """Test Settings loads from environment."""

    def test_loads_from_dotenv(self, monkeypatch):
        """Settings should load values from environment."""
        # Clear cache to force reload
        get_settings.cache_clear()

        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        settings = Settings()
        assert settings.log_level == "DEBUG"

        # Reset cache
        get_settings.cache_clear()
