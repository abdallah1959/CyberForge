# cyberforge/ai/models.py
"""
AI Provider models and options data contracts.
Defines the structures for request configuration, runtime contexts, and provider responses.

These models are used by:
    - AIParserService: for creating provider contexts and parsing responses.
    - GeminiProvider: for returning structured responses with telemetry.
    - ProviderFactory: for instantiating providers with the correct options.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
from cyberforge.core.enums import ModelProvider

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    "GenerationOptions",
    "ProviderContext",
    "ProviderResponse",
    "HealthCheckMode",
]


class GenerationOptions(BaseModel):
    """
    Configuration options for tuning language model generation.

    These options control randomness, output length, and sampling strategies.
    All fields have safe defaults that work for most extraction tasks.
    """

    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Controls randomness (0.0 = deterministic, 1.0 = creative)."
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum number of tokens to generate. If None, provider default is used."
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter (alternative to temperature)."
    )
    top_k: Optional[int] = Field(
        default=None,
        gt=0,
        description="Top-k sampling parameter (limits token pool to top k)."
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalises repetition of existing tokens."
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalises frequent tokens globally."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible, deterministic outputs."
    )


class ProviderContext(BaseModel):
    """
    Runtime execution context for a provider request.

    This encapsulates the authentication key and per-request timeouts.
    It is passed to the provider's `generate_parsed_response` method.
    """

    api_key: str = Field(
        ...,
        description="The API key to authenticate the request."
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Optional base URL for the provider API, required by some providers."
    )
    timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Request timeout in seconds."
    )


class ProviderResponse(BaseModel):
    """
    Standardised wrapper for the output returned by any abstract AI provider.

    This model is used to pass the parsed data, raw response, and usage metrics
    from the provider layer back to the AI Parser Service.

    It includes all fields required by `AIParserService._validate_response()`
    and `AIParserService._record_telemetry()`.
    """

    provider: ModelProvider = Field(
        ...,
        description="The provider that served the request."
    )
    model_name: str = Field(
        ...,
        description="The model version used (e.g., 'gemini-2.0-flash')."
    )

    data: dict[str, Any] = Field(
        ...,
        description="The parsed JSON data conforming to the requested schema."
    )

    raw_response: Optional[str] = Field(
        default=None,
        description="The raw text response from the provider, for debugging/fallback."
    )

    prompt_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of tokens in the input prompt."
    )
    completion_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of tokens in the generated completion."
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens used (prompt + completion)."
    )

    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Request duration in seconds."
    )

class HealthCheckMode(str, Enum):
    """Defines the rigor of the provider health check."""
    SHALLOW = "shallow"  # Local validation only (keys exist)
    CONNECTIVITY = "connectivity"        # Actual API ping to verify connectivity