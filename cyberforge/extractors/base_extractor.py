# cyberforge/extractors/base_extractor.py
"""
Abstract Base Extractor for the CyberForge Ingestion Layer.

Defines the enterprise contract for scraping and fetching data from various
external cybersecurity sources (e.g., HackerOne, CTFTime, ExploitDB).
Ensures all raw data is strictly standardized before entering the AI parser.

This contract is designed to be:
    - Async-native (all I/O operations are asynchronous)
    - Stream-friendly (batch extraction yields items as they arrive)
    - Observable (health checks raise explicit exceptions)
    - Extensible (supports context and options for runtime configuration)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import AsyncIterator

from cyberforge.core.exceptions import ExtractionError
from cyberforge.schemas.extracted_item import ExtractedItem

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    "BaseExtractor",
    "SourceType",
    "ExtractorContext",
    "ExtractionOptions",
]


# -----------------------------------------------------------------------------
# Source Type Enum
# -----------------------------------------------------------------------------


class SourceType(StrEnum):
    """
    Standardised types of data sources for extraction.

    This enum ensures that source_type is consistently typed across all
    extractors, preventing mismatched strings and enabling pattern-based
    routing (e.g., API vs HTML parsing strategies).
    """

    API = "API"
    HTML = "HTML"
    RSS = "RSS"
    STATIC = "STATIC"


# -----------------------------------------------------------------------------
# Extraction Context & Options
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractorContext:
    """
    Runtime context for an extractor operation.

    This object encapsulates configuration that may vary per request,
    such as authentication, headers, proxy settings, and timeouts.

    Future extensions:
        - authentication (API keys, tokens)
        - custom_headers (dict[str, str])
        - proxy (str)
    """

    user_agent: str | None = None
    timeout: float = 30.0
    max_retries: int = 3
    rate_limit_per_second: float | None = None


@dataclass(frozen=True)
class ExtractionOptions:
    """
    Fine-tuning options for extraction behaviour.

    These options control how the extractor processes data,
    such as pagination limits, date filtering, and logging verbosity.
    """

    limit: int | None = None
    since_date: datetime | None = None
    include_metadata: bool = True
    verbose: bool = False


# -----------------------------------------------------------------------------
# Base Extractor Contract
# -----------------------------------------------------------------------------


class BaseExtractor(ABC):
    """
    Universal async interface for all data extraction modules.

    Any new platform integrated into CyberForge must implement this contract
    to guarantee pipeline symmetry and type safety.

    Design Decisions:
        - Each extractor is stateless; configuration is passed via
          ExtractorContext or ExtractionOptions at request time.
        - Health checks raise ExtractionError rather than returning bool,
          providing clear failure reasons for monitoring.
        - Batch extraction continues on individual failures, ensuring
          high throughput and resilience.

    Example:
        >>> class HackerOneExtractor(BaseExtractor):
        ...     @property
        ...     def source_name(self) -> str:
        ...         return "HackerOne"
        ...
        ...     @property
        ...     def source_type(self) -> SourceType:
        ...         return SourceType.API
        ...
        ...     async def extract_single(self, url: str) -> ExtractedItem:
        ...         # Implementation
        ...         ...
        ...
        ...     async def extract_batch(self, urls: list[str]) -> AsyncIterator[ExtractedItem]:
        ...         # Implementation
        ...         ...
        ...
        ...     async def health_check(self) -> None:
        ...         # Implementation
        ...         ...
    """

    # -------------------------------------------------------------------------
    # Extractor Identity
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def source_name(self) -> str:
        """
        The standardized identifier for the source platform (e.g., 'HackerOne').

        Used for logging, routing, and database indexing.
        """
        ...

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """
        The type of the source platform (e.g., API, HTML, RSS, STATIC).

        This enables the system to apply different strategies for rate limiting,
        caching, and retry logic.
        """
        ...

    # -------------------------------------------------------------------------
    # Core Extraction Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    async def extract_single(
        self,
        url: str,
        context: ExtractorContext | None = None,
        options: ExtractionOptions | None = None,
    ) -> ExtractedItem:
        """
        Extracts a single vulnerability report or writeup from the given URL.

        Args:
            url: The exact target URL to scrape or fetch via API.
            context: Optional runtime configuration (timeout, headers, etc.).
            options: Optional fine-tuning options for extraction.

        Returns:
            An ExtractedItem Pydantic model containing the raw unstructured data.

        Raises:
            ExtractionError: If the URL is invalid, the network request fails,
                             or the target page structure has fundamentally changed.
        """
        ...

    @abstractmethod
    async def extract_batch(
        self,
        urls: list[str],
        context: ExtractorContext | None = None,
        options: ExtractionOptions | None = None,
    ) -> AsyncIterator[ExtractedItem]:
        """
        Asynchronously yields extracted items from a list of URLs.

        Designed for high-throughput ingestion and memory-efficient streaming.

        Behaviour on failure:
            If a single URL fails, the extractor SHOULD log the error and
            continue processing the remaining URLs. The batch should only
            halt if a critical, unrecoverable error occurs (e.g., authentication
            failure, network outage). Individual failures are reported via
            logging and monitoring, but the pipeline continues.

        Args:
            urls: A list of target URLs to process.
            context: Optional runtime configuration.
            options: Optional fine-tuning options.

        Yields:
            ExtractedItem models as they are successfully retrieved.

        Raises:
            ExtractionError: If a critical failure occurs that prevents
                             the entire batch from continuing.
        """
        ...

    # -------------------------------------------------------------------------
    # Health & Connectivity
    # -------------------------------------------------------------------------

    @abstractmethod
    async def health_check(
        self,
        context: ExtractorContext | None = None,
    ) -> None:
        """
        Verifies connectivity to the target platform and ensures that
        anti-scraping mechanisms (like Cloudflare) aren't blocking the pipeline.

        This method should be lightweight and quick to execute. It is used for:
            - Startup validation
            - Health probes in production
            - Diagnostic debugging

        Args:
            context: Optional runtime configuration.

        Raises:
            ExtractionError: If the platform is unreachable, returns an error,
                             or blocks the request. The error message should
                             provide a clear reason for the failure.

        Example:
            >>> await extractor.health_check()
            >>> # If healthy, returns None.
            >>> # If unhealthy, raises ExtractionError with details.
        """
        ...
