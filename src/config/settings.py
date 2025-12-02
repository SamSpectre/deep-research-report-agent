"""
Settings configuration for Deep Research Agent.

This module uses Pydantic Settings to manage configuration from environment
variables with validation and type safety.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Pydantic Settings automatically:
    - Loads values from .env file
    - Validates types
    - Provides defaults
    - Keeps secrets secure (SecretStr masks values in logs)
    
    Usage:
        settings = get_settings()
        api_key = settings.anthropic_api_key.get_secret_value()
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra env vars
    )
    
    # =========================================================================
    # LLM PROVIDER API KEYS
    # =========================================================================
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key for Claude models"
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key for GPT models"
    )
    
    # =========================================================================
    # SEARCH API KEYS
    # =========================================================================
    tavily_api_key: SecretStr | None = Field(
        default=None,
        description="Tavily API key for web search"
    )
    
    # =========================================================================
    # LANGSMITH (OBSERVABILITY)
    # =========================================================================
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        description="LangSmith API key for tracing"
    )
    langsmith_project: str = Field(
        default="deep-research-agent",
        description="LangSmith project name"
    )
    langsmith_tracing: bool = Field(
        default=False,
        description="Enable LangSmith tracing"
    )
    
    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================
    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default LLM model for the orchestrator agent"
    )
    
    # Sub-agent model overrides (Phase 3)
    # These allow cost optimization by using different models per role
    researcher_model: str | None = Field(
        default=None,
        description=(
            "Model for researcher sub-agent. "
            "Defaults to 'openai:gpt-4o-mini' if not set. "
            "Use format 'provider:model' (e.g., 'openai:gpt-4o-mini', 'anthropic:claude-haiku-4-5-20251001')"
        )
    )
    critic_model: str | None = Field(
        default=None,
        description=(
            "Model for critic sub-agent. "
            "Defaults to 'openai:gpt-4o' if not set. "
            "A more capable model is recommended for thorough evaluation."
        )
    )
    writer_model: str | None = Field(
        default=None,
        description=(
            "Model for writer sub-agent. "
            "Defaults to 'anthropic:claude-haiku-4-5-20251001' if not set. "
            "Optimized for fast, high-quality writing."
        )
    )
    
    # =========================================================================
    # APPLICATION SETTINGS
    # =========================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment"
    )
    
    # =========================================================================
    # VALIDATORS
    # =========================================================================
    @field_validator("default_model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate that the model is a known model."""
        known_models = {
            # Anthropic models
            "claude-sonnet-4-20250514",
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-20250514",
            # OpenAI models
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
        }
        if v not in known_models:
            # Allow unknown models but log a warning
            import warnings
            warnings.warn(
                f"Model '{v}' is not in the known models list. "
                f"Known models: {known_models}"
            )
        return v
    
    # =========================================================================
    # HELPER PROPERTIES
    # =========================================================================
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return self.anthropic_api_key is not None
    
    @property
    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return self.openai_api_key is not None
    
    @property
    def has_tavily(self) -> bool:
        """Check if Tavily API key is configured."""
        return self.tavily_api_key is not None
    
    @property
    def has_langsmith(self) -> bool:
        """Check if LangSmith is configured and enabled."""
        return self.langsmith_api_key is not None and self.langsmith_tracing
    
    def get_model_provider(self) -> Literal["anthropic", "openai"]:
        """
        Determine the model provider based on the default model.
        
        Returns:
            The provider name ('anthropic' or 'openai')
        """
        if self.default_model.startswith("claude"):
            return "anthropic"
        elif self.default_model.startswith("gpt"):
            return "openai"
        else:
            raise ValueError(f"Unknown model provider for model: {self.default_model}")
    
    def validate_required_keys(self) -> None:
        """
        Validate that all required API keys are present.
        
        Raises:
            ValueError: If required keys are missing
        """
        errors = []
        
        # Check for at least one LLM provider
        if not self.has_anthropic and not self.has_openai:
            errors.append(
                "No LLM provider configured. "
                "Set either ANTHROPIC_API_KEY or OPENAI_API_KEY"
            )
        
        # Check for search capability
        if not self.has_tavily:
            errors.append(
                "No search API configured. "
                "Set TAVILY_API_KEY for web search capability"
            )
        
        # Check that the configured model has its provider key
        provider = self.get_model_provider()
        if provider == "anthropic" and not self.has_anthropic:
            errors.append(
                f"Model '{self.default_model}' requires ANTHROPIC_API_KEY"
            )
        elif provider == "openai" and not self.has_openai:
            errors.append(
                f"Model '{self.default_model}' requires OPENAI_API_KEY"
            )
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are loaded only once and reused.
    This is a common pattern for configuration management.
    
    Returns:
        Settings instance with loaded configuration
    """
    return Settings()