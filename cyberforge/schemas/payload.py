# cyberforge/schemas/payload.py
"""
Schema definition for attack payloads.
Validates the malicious input used to trigger a vulnerability.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from cyberforge.core.enums import VulnerabilityType

# Public API contract
__all__ = ["Payload"]


class Payload(BaseModel):
    """
    Represents a specific attack vector or malicious payload.

    This schema serves as a security boundary for all incoming payload data.
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
        # validate_assignment=True removed: redundant with frozen=True
    )

    # -------------------------------------------------------------------------
    # Fields
    # -------------------------------------------------------------------------

    vulnerability_type: VulnerabilityType = Field(
        ...,
        description="Standardized category of the vulnerability. "
                    "Must be one of the predefined types from the core enum.",
        examples=["SQLI", "XSS", "RCE"],
    )

    injection_string: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="The exact payload or code snippet used for the exploit. "
                    "Must be non-empty and limited to 20,000 characters.",
        examples=["' OR 1=1 --", "<script>alert(1)</script>"],
    )

    description: Optional[str] = Field(
        default=None,
        max_length=2_000,
        description="Brief explanation of how the payload manipulates the target system.",
        examples=["SQL injection using tautology to bypass authentication."],
    )
