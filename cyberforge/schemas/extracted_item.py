# cyberforge/schemas/extracted_item.py
"""
Schema definition for raw extracted items.

Represents the initial data payload gathered from external sources (e.g., CTFTime, HackerOne)
before it is parsed and structured by the AI models. This schema serves as the ingress
boundary for all untrusted data entering the CyberForge pipeline.
"""

from datetime import UTC, datetime, timezone
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, field_validator

# Public API contract
__all__ = ["ExtractedItem"]


# -----------------------------------------------------------------------------
# Constrained String Types
# -----------------------------------------------------------------------------

RawContent = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=500_000,
        strip_whitespace=True,
    ),
]

AuthorName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=200,
        strip_whitespace=True,
    ),
]

# Strongly typed content hash for deduplication and integrity checks.
# Ensures format: "sha256:" followed by exactly 64 hex characters.
ContentHash = Annotated[
    str,
    StringConstraints(
        pattern=r"^sha256:[a-fA-F0-9]{64}$",
        strip_whitespace=True,
    ),
]


# -----------------------------------------------------------------------------
# Schema Definition
# -----------------------------------------------------------------------------

class ExtractedItem(BaseModel):
    """
    The raw data artifact collected by an extractor module.

    This schema represents the very first point of data ingestion into CyberForge.
    It validates and structures the raw, untrusted content scraped from external
    sources before any AI processing or parsing occurs.

    All text fields are length-constrained, and the model is immutable to ensure
    data integrity from the moment of ingestion.
    """

    # -------------------------------------------------------------------------
    # Pydantic v2 Configuration – Strict & Safe by Default
    # -------------------------------------------------------------------------

    model_config = ConfigDict(
        extra="forbid",               # Reject unknown fields
        frozen=True,                  # Make instances immutable
        strict=True,                  # Disallow implicit type coercion
        str_strip_whitespace=True,    # Trim leading/trailing whitespace
    )

    # -------------------------------------------------------------------------
    # Source & Identification Fields
    # -------------------------------------------------------------------------

    source_name: str = Field(
        ...,
        description="The platform from which the data was extracted.",
        examples=["CTFTIME", "HACKERONE"],
    )

    url: HttpUrl = Field(
        ...,
        description="The original URL of the writeup or vulnerability report.",
        examples=["https://ctftime.org/writeup/12345"],
    )

    raw_content: RawContent = Field(
        ...,
        description="The complete, unstructured text content scraped from the source. "
                    "Must be non-empty and limited to 500,000 characters.",
        examples=[
            "Vulnerability: SQL Injection\nDescription: ...",
        ],
    )

    author: AuthorName | None = Field(
        default=None,
        description="The author or researcher who published the writeup. "
                    "Limited to 200 characters.",
        examples=["John Doe", "security_researcher"],
    )

    # -------------------------------------------------------------------------
    # Temporal & Tracking Fields
    # -------------------------------------------------------------------------

    published_date: datetime | None = Field(
        default=None,
        description="The publication date of the original report, if available. "
                    "Will be normalized to UTC.",
        examples=["2026-06-19T10:30:00Z"],
    )

    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when this item was extracted from the source.",
        examples=["2026-06-19T10:30:00Z"],
    )

    content_hash: ContentHash | None = Field(
        default=None,
        description="Hash of the raw content for deduplication and integrity checks. "
                    "Must be in the format 'sha256:' followed by 64 hex characters.",
        examples=["sha256:abc123..."],
    )

    # -------------------------------------------------------------------------
    # Custom Validators
    # -------------------------------------------------------------------------

    @field_validator("source_name", mode="after")
    @classmethod
    def validate_source_not_unknown(cls, value: str) -> str:
        """
        Prevent usage of UNKNOWN at the ingestion boundary.

        While UNKNOWN is useful for classification in later stages, at the
        extraction point we want explicit knowledge of the source platform.
        This ensures extractor bugs are detected early.

        Args:
            value: The source platform string.

        Returns:
            The validated string.

        Raises:
            ValueError: If the source is UNKNOWN.
        """
        if value.strip().upper() == "UNKNOWN":
            raise ValueError(
                "Source platform cannot be UNKNOWN at the extraction stage. "
                "Please specify a valid platform or update the extractor."
            )
        return value

    @field_validator("published_date", mode="before")
    @classmethod
    def normalize_published_date(cls, value: datetime | None) -> datetime | None:
        """
        Normalize the published_date to UTC.

        If the datetime is naive, it is assumed to be in UTC and converted.
        If it is aware, it is converted to UTC.
        If the value is None, it is returned as-is.

        Args:
            value: The datetime value to normalize.

        Returns:
            A UTC-aware datetime, or None.

        Raises:
            ValueError: If the value is not a datetime or cannot be converted.
        """
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise ValueError("published_date must be a datetime object")
        if value.tzinfo is None:
            # Assume naive datetime is UTC
            return value.replace(tzinfo=UTC)
        # Convert to UTC
        return value.astimezone(UTC)