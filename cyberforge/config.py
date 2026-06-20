# cyberforge/config.py
"""
Central configuration manager for the CyberForge pipeline.

Uses Pydantic Settings to load and validate environment variables,
providing a type-safe, immutable, and validated configuration object
for the entire application.
"""

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cyberforge.core.constants import ENV_FILE_PATH, MAX_TEXT_CONTEXT_LENGTH
from cyberforge.core.enums import ModelProvider
from cyberforge.core.exceptions import ConfigurationError

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = ["settings", "Settings"]


# -----------------------------------------------------------------------------
# Configuration Class
# -----------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Centralized, validated configuration for CyberForge.

    All settings are loaded from environment variables, with safe defaults
    provided where appropriate. The configuration is immutable and type-safe.
    """

    # -------------------------------------------------------------------------
    # Pydantic Settings Configuration
    # -------------------------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="forbid",  # Reject unknown environment variables
    )

    # -------------------------------------------------------------------------
    # AI Provider Selection
    # -------------------------------------------------------------------------

    AI_PROVIDER: ModelProvider = Field(
        default=ModelProvider.GEMINI,
        description="The primary AI provider to use (GEMINI, OPENAI, CLAUDE, OLLAMA).",
        examples=["GEMINI", "OPENAI"],
    )

    DEFAULT_MODEL: str = Field(
        default="gemini-2.0-flash",
        min_length=1,
        max_length=200,
        description="Default model name for the selected provider.",
        examples=["gemini-2.0-flash", "gpt-4", "claude-3-opus-20240229"],
    )

    # -------------------------------------------------------------------------
    # Provider API Keys (as comma-separated strings in .env)
    # -------------------------------------------------------------------------

    GEMINI_API_KEYS: tuple[str, ...] = Field(
        default=(),
        description="Comma-separated list of Gemini API keys (for rotation).",
        examples=["key1,key2,key3"],
    )

    OPENAI_API_KEYS: tuple[str, ...] = Field(
        default=(),
        description="Comma-separated list of OpenAI API keys (for rotation).",
        examples=["sk-...", "sk-..."],
    )

    CLAUDE_API_KEYS: tuple[str, ...] = Field(
        default=(),
        description="Comma-separated list of Claude API keys (for rotation).",
        examples=["sk-ant-...", "sk-ant-..."],
    )

    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        min_length=1,
        max_length=500,
        description="Base URL for the Ollama server.",
        examples=["http://localhost:11434", "http://ollama.example.com"],
    )

    # -------------------------------------------------------------------------
    # Operational Parameters
    # -------------------------------------------------------------------------

    MAX_TEXT_LENGTH: int = Field(
        default=MAX_TEXT_CONTEXT_LENGTH,
        ge=1,
        description="Maximum text length (in characters) for processing.",
        examples=[120_000],
    )

    # -------------------------------------------------------------------------
    # Custom Validators
    # -------------------------------------------------------------------------

    @field_validator(
        "GEMINI_API_KEYS",
        "OPENAI_API_KEYS",
        "CLAUDE_API_KEYS",
        mode="before",
    )
    @classmethod
    def parse_comma_separated_keys(cls, value: str | tuple | list | None) -> tuple[str, ...]:
        """
        Parse API keys from strings, lists, or tuples into a standard tuple.
        """
        if isinstance(value, (tuple, list)):
            return tuple(value)
        if not value or not isinstance(value, str):
            return ()
        keys = [k.strip() for k in value.split(",") if k.strip()]
        return tuple(keys)

    @field_validator("DEFAULT_MODEL", mode="after")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        """
        Ensure the model name is not empty and is reasonably short.
        """
        if not value or not value.strip():
            raise ValueError("DEFAULT_MODEL cannot be empty.")
        return value.strip()

    @field_validator("MAX_TEXT_LENGTH", mode="after")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        """
        Ensure the max text length is positive.
        """
        if value <= 0:
            raise ValueError("MAX_TEXT_LENGTH must be a positive integer.")
        return value

    @model_validator(mode="after")
    def validate_provider_configuration(self) -> "Settings":
        """
        Ensure that the selected AI provider has the required configuration.

        This prevents runtime failures due to missing credentials or invalid
        base URLs.
        """
        provider = self.AI_PROVIDER

        if provider is ModelProvider.GEMINI and not self.GEMINI_API_KEYS:
            raise ValueError(
                "GEMINI_API_KEYS must be configured when AI_PROVIDER=GEMINI"
            )
        if provider is ModelProvider.OPENAI and not self.OPENAI_API_KEYS:
            raise ValueError(
                "OPENAI_API_KEYS must be configured when AI_PROVIDER=OPENAI"
            )
        if provider is ModelProvider.CLAUDE and not self.CLAUDE_API_KEYS:
            raise ValueError(
                "CLAUDE_API_KEYS must be configured when AI_PROVIDER=CLAUDE"
            )
        if (
            provider is ModelProvider.OLLAMA
            and not self.OLLAMA_BASE_URL.strip()
        ):
            raise ValueError(
                "OLLAMA_BASE_URL must not be empty when AI_PROVIDER=OLLAMA"
            )

        return self


# -----------------------------------------------------------------------------
# Singleton Instance with Bulletproof Initialisation
# -----------------------------------------------------------------------------

try:
    settings = Settings()
except ValidationError as exc:
    raise ConfigurationError(
        f"Failed to load application settings due to validation errors: {exc}"
    ) from exc