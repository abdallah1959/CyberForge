# cyberforge/schemas/evidence.py
"""
Schema definition for vulnerability evidence.
Validates the proof-of-concept artifacts extracted from writeups.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints

from cyberforge.core.enums import EvidenceType

# Public API contract
__all__ = ["Evidence"]


# Define constrained string types for reuse and clarity.
EvidenceContent = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=100_000,
        strip_whitespace=True,
    ),
]


class Evidence(BaseModel):
    """
    Represents an artifact proving the vulnerability's existence.

    This schema serves as a security boundary for all incoming evidence data.
    It enforces strict typing, length limits, and rejects any extra fields
    to ensure data quality and prevent injection of unexpected content.
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
    # Fields
    # -------------------------------------------------------------------------

    evidence_type: EvidenceType = Field(
        ...,
        description="Standardized category of the evidence artifact. "
                    "Must be one of the predefined types from the core enum.",
        examples=["HTTP_REQUEST", "CODE_SNIPPET", "STACK_TRACE"],
    )

    content: EvidenceContent = Field(
        ...,
        description="The raw content of the evidence (logs, headers, code snippets, etc.). "
                    "Must be non-empty and limited to 100,000 characters.",
        examples=[
            "GET /admin HTTP/1.1\nHost: example.com",
            "def exploit():\n    return 'payload'",
        ],
    )

    source_url: HttpUrl | None = Field(
        default=None,
        description="Direct link to an external resource containing the evidence, "
                    "such as a screenshot, pastebin, or hosted log file.",
        examples=[
            "https://example.com/screenshot.png",
            "https://pastebin.com/raw/abc123",
        ],
    )
