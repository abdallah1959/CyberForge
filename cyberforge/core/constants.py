# cyberforge/core/constants.py
"""
Global core constants for the CyberForge pipeline.

This module contains only fundamental, environment-agnostic constants
that are safe to hardcode. Operational settings (timeouts, retries, etc.)
and domain-specific prompts belong in separate configuration or prompt layers.
"""

from pathlib import Path

# Public API contract – explicitly declares what this module exports.
__all__ = [
    "PROJECT_ROOT",
    "ENV_FILE_PATH",
    "BYTES_PER_MB",
    "MAX_PAYLOAD_SIZE_BYTES",
    "MAX_TEXT_CONTEXT_LENGTH",
    "STANDARD_COOLDOWN_SECONDS",
    "EXTENDED_COOLDOWN_SECONDS",
]


# -----------------------------------------------------------------------------
# Path Resolutions
# -----------------------------------------------------------------------------

# Absolute path to the project root directory.
# This resolution assumes the standard structure: project_root / cyberforge / core / constants.py
# If the directory structure changes, this should be updated or overridden by settings.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Path to the environment configuration file (e.g., .env) located at the project root.
ENV_FILE_PATH: Path = PROJECT_ROOT / ".env"


# -----------------------------------------------------------------------------
# Data Size Units & Limits
# -----------------------------------------------------------------------------

# Convenience constant for byte conversion (1 MiB).
BYTES_PER_MB: int = 1024 * 1024

# Maximum payload size accepted by the system (5 MiB).
# This is a hard limit for any single request payload.
MAX_PAYLOAD_SIZE_BYTES: int = 5 * BYTES_PER_MB

# Maximum length of text context processed in a single operation.
# This is a hard safety limit, not a tunable performance parameter.
MAX_TEXT_CONTEXT_LENGTH: int = 120_000


# -----------------------------------------------------------------------------
# Cooldown Timings
# -----------------------------------------------------------------------------

# System-wide default cooldown period (in seconds) between operations.
# This value serves as a safe baseline and may be overridden by operational
# settings if configured for specific environments or providers.
STANDARD_COOLDOWN_SECONDS: int = 60

# Extended cooldown period for heavy or rate-limited operations.
# Like the standard cooldown, this is a safe default that may be tuned
# at runtime via the settings layer.
EXTENDED_COOLDOWN_SECONDS: int = 120

# -----------------------------------------------------------------------------
# AI Prompt Constants
# -----------------------------------------------------------------------------

DEFAULT_AI_SYSTEM_PROMPT = """You are an elite Cybersecurity Data Engineer and CTF Analyst. 
Your task is to analyze the provided CTF writeup or security article and extract structured intelligence.
You MUST output STRICTLY VALID JSON that perfectly matches the provided JSON schema.
Do not include any Markdown formatting (like ```json), just the raw JSON object.
Focus on identifying exact tools used, specific vulnerabilities (e.g., SQLi, XSS), and clear technical summaries."""

# -----------------------------------------------------------------------------
# Network & Timeout Constants
# -----------------------------------------------------------------------------

DEFAULT_HEALTHCHECK_TIMEOUT_SECONDS = 10
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 60