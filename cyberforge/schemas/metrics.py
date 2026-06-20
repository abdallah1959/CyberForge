# cyberforge/schemas/metrics.py
"""
Schema definition for AI pipeline metrics.

Tracks execution time, token usage, provider behavior, and cost estimation
for observability, monitoring, and analytics purposes.
"""

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, computed_field

from cyberforge.core.enums import ModelProvider

# Public API contract
__all__ = ["PipelineMetrics"]


# -----------------------------------------------------------------------------
# Constrained String Types
# -----------------------------------------------------------------------------

ModelName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=200,
        strip_whitespace=True,
    ),
]


# -----------------------------------------------------------------------------
# Schema Definition
# -----------------------------------------------------------------------------

class PipelineMetrics(BaseModel):
    """
    Telemetry data for a single AI parsing operation.

    This schema captures all relevant operational metrics for a provider
    interaction, including execution duration, token consumption, success
    status, and estimated cost. It is designed to be easily serialized
    for logging, dashboards, and cost analysis.

    All numeric fields are strictly non‑negative, and timestamps are
    normalised to UTC.
    """

    # -------------------------------------------------------------------------
    # Pydantic v2 Configuration
    # -------------------------------------------------------------------------

    model_config = ConfigDict(
        extra="forbid",               # Reject unknown fields
        frozen=True,                  # Make instances immutable
        strict=True,                  # Disallow implicit type coercion
        str_strip_whitespace=True,    # Trim leading/trailing whitespace
    )

    # -------------------------------------------------------------------------
    # Provider & Model Identification
    # -------------------------------------------------------------------------

    provider_used: ModelProvider = Field(
        ...,
        description="The AI provider that handled this transaction.",
        examples=["OPENAI", "GEMINI"],
    )

    model_name: ModelName = Field(
        ...,
        description="The specific model version used (e.g., 'gpt-4', 'gemini-2.0-flash').",
        examples=["gpt-4", "gemini-2.0-flash", "claude-3-opus-20240229"],
    )

    # -------------------------------------------------------------------------
    # Performance & Resource Consumption
    # -------------------------------------------------------------------------

    duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total time taken for the AI provider to return the response, in seconds.",
        examples=[1.234, 0.567],
    )

    prompt_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of tokens consumed by the input prompt.",
        examples=[1500, 2048],
    )

    completion_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of tokens generated in the completion response.",
        examples=[256, 512],
    )

    # -------------------------------------------------------------------------
    # Outcome & Cost
    # -------------------------------------------------------------------------

    successful_parse: bool = Field(
        ...,
        description="Indicates whether the response successfully passed Pydantic validation.",
        examples=[True, False],
    )

    estimated_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        description="Estimated cost of this transaction in USD, if pricing data is available.",
        examples=[0.0123, 0.0005],
    )

    # -------------------------------------------------------------------------
    # Temporal Tracking
    # -------------------------------------------------------------------------

    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the metrics were recorded.",
        examples=["2026-06-19T12:34:56Z"],
    )

    # -------------------------------------------------------------------------
    # Computed Fields (Derived from other fields)
    # -------------------------------------------------------------------------

    @computed_field
    @property
    def total_tokens(self) -> int:
        """
        Total tokens used in this transaction (prompt + completion).

        This is a computed property and is not stored separately in the database,
        but it is included in the serialized output for convenience.
        """
        return self.prompt_tokens + self.completion_tokens
