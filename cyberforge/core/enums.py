# cyberforge/core/enums.py (إضافة جديدة)
"""
Core enumerations for the CyberForge pipeline.
Ensures type safety and standardizes categorical variables across all modules.
"""

from enum import StrEnum, unique

# Public API contract – explicitly declares what this module exports.
__all__ = [
    "ModelProvider",
    "SeverityLevel",
    "ExtractionStatus",
    "VulnerabilityType",
    "EvidenceType",
    "SourcePlatform",  # جديد
]


@unique
class ModelProvider(StrEnum):
    """
    Supported AI model providers.

    This enum is used to standardize provider names across the platform,
    ensuring type safety when interacting with external AI/LLM APIs.
    """
    GEMINI = "GEMINI"
    OPENAI = "OPENAI"
    OLLAMA = "OLLAMA"
    CLAUDE = "CLAUDE"


@unique
class SeverityLevel(StrEnum):
    """
    Standardized vulnerability severity levels.

    Maps to industry-standard CVSS qualitative severity ratings.
    """
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"
    UNKNOWN = "UNKNOWN"


@unique
class ExtractionStatus(StrEnum):
    """
    Tracking states for the data ingestion and parsing pipeline.

    The typical lifecycle flow is:
        PENDING -> EXTRACTED -> PARSED -> VALIDATED -> COMPLETED

    Any state can transition to FAILED if an unrecoverable error occurs.
    """
    PENDING = "PENDING"
    EXTRACTED = "EXTRACTED"
    PARSED = "PARSED"
    VALIDATED = "VALIDATED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    @property
    def is_terminal(self) -> bool:
        """
        Indicates whether the current status represents a terminal state.

        Terminal states are those where no further processing is expected
        for the record, either due to successful completion or failure.

        Uses object identity comparison (``is``) because each enum member
        is a singleton. This guarantees semantic correctness and eliminates
        the need for value-based lookups.

        Returns:
            bool: True if the status is COMPLETED or FAILED, False otherwise.
        """
        return self is self.COMPLETED or self is self.FAILED


# -----------------------------------------------------------------------------
# Vulnerability Classification Enum
# -----------------------------------------------------------------------------

@unique
class VulnerabilityType(StrEnum):
    """
    Standardized vulnerability categories used across the platform.

    This enum ensures that all vulnerability classifications are consistent
    between the extractor, parser, and storage layers. It covers the most
    common vulnerability types found in bug bounty reports, CVE databases,
    and CTF writeups.

    The inclusion of ``UNKNOWN`` ensures the pipeline can gracefully handle
    novel or uncategorised vulnerabilities without failing.
    """
    # Web & API Vulnerabilities
    SQLI = "SQLI"
    XSS = "XSS"
    XXE = "XXE"
    SSTI = "SSTI"
    CSRF = "CSRF"
    SSRF = "SSRF"
    OPEN_REDIRECT = "OPEN_REDIRECT"

    # Access Control & Authentication
    BROKEN_ACCESS_CONTROL = "BROKEN_ACCESS_CONTROL"
    IDOR = "IDOR"
    AUTHENTICATION_BYPASS = "AUTHENTICATION_BYPASS"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"

    # Injection & Execution
    RCE = "RCE"
    COMMAND_INJECTION = "COMMAND_INJECTION"
    LFI = "LFI"
    RFI = "RFI"

    # File & Resource Handling
    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    FILE_UPLOAD = "FILE_UPLOAD"
    INSECURE_DESERIALIZATION = "INSECURE_DESERIALIZATION"

    # Configuration & Information Leak
    INSECURE_CONFIGURATION = "INSECURE_CONFIGURATION"
    INFORMATION_DISCLOSURE = "INFORMATION_DISCLOSURE"

    # Availability
    DENIAL_OF_SERVICE = "DENIAL_OF_SERVICE"

    # Fallback
    UNKNOWN = "UNKNOWN"


# -----------------------------------------------------------------------------
# Evidence Type Enum
# -----------------------------------------------------------------------------

@unique
class EvidenceType(StrEnum):
    """
    Standardized evidence artifact types used across the platform.

    This enum categorises the type of proof-of-concept artifacts extracted
    from vulnerability writeups, enabling consistent filtering, analysis,
    and storage of evidence data.
    """
    HTTP_REQUEST = "HTTP_REQUEST"
    HTTP_RESPONSE = "HTTP_RESPONSE"
    LOG_OUTPUT = "LOG_OUTPUT"
    CODE_SNIPPET = "CODE_SNIPPET"
    STACK_TRACE = "STACK_TRACE"
    SCREENSHOT = "SCREENSHOT"
    TERMINAL_OUTPUT = "TERMINAL_OUTPUT"
    CONFIGURATION_FILE = "CONFIGURATION_FILE"

    # Fallback for uncategorised evidence
    UNKNOWN = "UNKNOWN"


# -----------------------------------------------------------------------------
# Source Platform Enum (New)
# -----------------------------------------------------------------------------

@unique
class SourcePlatform(StrEnum):
    """
    Standardized source platforms for extracted data.

    This enum categorises the origin of the raw extracted items, enabling
    consistent processing and analytics across different data sources.
    """
    CTFTIME = "CTFTIME"
    HACKERONE = "HACKERONE"
    EXPLOITDB = "EXPLOITDB"
    UNKNOWN = "UNKNOWN"
