# cyberforge/schemas/parsed_writeup.py
"""
Master schema for AI-parsed vulnerability writeups.

This serves as the definitive data contract for the AI output and storage layers.
It aggregates all extracted information, including payloads, evidence, metadata,
and tracking fields, forming the central data model for the CyberForge pipeline.
"""

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints

from cyberforge.core.enums import ExtractionStatus, SeverityLevel, VulnerabilityType
from cyberforge.schemas.evidence import Evidence
from cyberforge.schemas.payload import Payload

# -----------------------------------------------------------------------------
# Public API Contract
# -----------------------------------------------------------------------------

__all__ = ["ParsedWriteup"]


# -----------------------------------------------------------------------------
# Constrained String Types for Reuse and Clarity
# -----------------------------------------------------------------------------

WriteupTitle = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=500,
        strip_whitespace=True,
    ),
]

TargetSystem = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=300,
        strip_whitespace=True,
    ),
]

ExecutiveSummary = Annotated[
    str,
    StringConstraints(
        min_length=10,
        max_length=10_000,
        strip_whitespace=True,
    ),
]

CVEIdentifier = Annotated[
    str,
    StringConstraints(
        pattern=r"^CVE-\d{4}-\d{4,7}$",
        strip_whitespace=True,
    ),
]


# -----------------------------------------------------------------------------
# Schema Definition
# -----------------------------------------------------------------------------

class ParsedWriteup(BaseModel):
    """
    The structured representation of a fully analyzed cybersecurity writeup.

    This schema serves as the central data contract for the entire pipeline,
    combining vulnerability metadata, attack payloads, supporting evidence,
    and pipeline tracking fields. It enforces strict validation to ensure
    data quality and consistency across all downstream systems.

    All text fields are length-constrained, and the model is immutable
    once created, making it safe for storage and analysis.
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
    # Core Identification & Metadata
    # -------------------------------------------------------------------------

    writeup_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for the writeup (e.g., source ID, hash, or UUID).",
        examples=["WRP-12345", "ctf-2024-001"],
    )

    source_url: HttpUrl | None = Field(
        default=None,
        description="Original URL of the writeup or CTF challenge source.",
        examples=["https://example.com/blog/cve-2024-1234"],
    )

    title: WriteupTitle = Field(
        ...,
        description="Standardized title of the vulnerability or CTF challenge.",
        examples=["SQL Injection in Product X", "Buffer Overflow in Service Y"],
    )

    target_system: TargetSystem = Field(
        ...,
        description="Software, platform, or hardware targeted by the exploit.",
        examples=["Apache Tomcat 9.0.1", "WordPress 5.8", "Linux Kernel 5.4"],
    )

    # -------------------------------------------------------------------------
    # Vulnerability Classification
    # -------------------------------------------------------------------------

    severity: SeverityLevel = Field(
        default=SeverityLevel.UNKNOWN,
        description="Assessed severity level of the vulnerability.",
        examples=["CRITICAL", "HIGH"],
    )

    vulnerability_types: set[VulnerabilityType] = Field(
        default_factory=set,
        description=(
            "Set of vulnerability categories applicable to this writeup. "
            "While individual payloads may have their own types, this field "
            "provides a high-level classification for the entire writeup. "
            "Duplicates are automatically eliminated."
        ),
        examples=[["SQLI", "XSS"], ["RCE"]],  # List syntax for JSON Schema compatibility
    )

    cve_id: CVEIdentifier | None = Field(
        default=None,
        description=(
            "Common Vulnerabilities and Exposures (CVE) identifier, if applicable. "
            "Must match the standard CVE format: CVE-YYYY-NNNNN."
        ),
        examples=["CVE-2024-12345", "CVE-2023-0001"],
    )

    summary: ExecutiveSummary = Field(
        ...,
        description="Concise, technical executive summary of the exploit methodology.",
        examples=[
            "SQL injection via unsanitized user input in the login parameter.",
            "Buffer overflow in the HTTP parser leading to RCE.",
        ],
    )

    # -------------------------------------------------------------------------
    # Extracted Artifacts
    # -------------------------------------------------------------------------

    payloads: list[Payload] = Field(
        default_factory=list,
        description="List of isolated attack payloads extracted from the text.",
        examples=[
            [{"vulnerability_type": "SQLI", "injection_string": "' OR 1=1 --"}],
        ],
    )

    evidences: list[Evidence] = Field(
        default_factory=list,
        description="List of proof-of-concept artifacts supporting the exploit.",
        examples=[
            [
                {
                    "evidence_type": "HTTP_REQUEST",
                    "content": "GET /admin?id=1' OR '1'='1",
                }
            ],
        ],
    )

    # -------------------------------------------------------------------------
    # Pipeline Tracking & Lifecycle
    # -------------------------------------------------------------------------

    extraction_status: ExtractionStatus = Field(
        default=ExtractionStatus.PENDING,
        description="Current pipeline processing state for this writeup.",
        examples=["PENDING", "VALIDATED", "FAILED"],
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp (UTC) when this record was created in the pipeline.",
        examples=["2026-06-19T10:30:00Z"],
    )
